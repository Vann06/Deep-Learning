from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.anomaly import (
    anomaly_metrics,
    error_by_feature_group,
    get_anomaly_score,
    reconstruction_scores,
    select_threshold,
)
from src.models.autoencoder import SequenceAutoencoder, fit_autoencoder, masked_reconstruction_error
from src.models.encoder import EncoderConfig, load_encoder, save_encoder
from src.utils import set_seed


def synthetic_batch(
    batch_size: int = 8, sequence_length: int = 32, num_features: int = 49, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lengths = rng.integers(3, sequence_length + 1, size=batch_size).astype(np.int64)
    mask = np.arange(sequence_length)[None, :] < lengths[:, None]
    X = rng.normal(size=(batch_size, sequence_length, num_features)).astype(np.float32)
    X = X * mask[:, :, None]
    return X, mask, lengths


class EncoderContractTests(unittest.TestCase):
    def test_forward_exposes_hidden_states_and_latent(self) -> None:
        X, mask, lengths = synthetic_batch()
        config = EncoderConfig(num_features=49, hidden_size=16, latent_size=8)
        model = SequenceAutoencoder(config)
        x_hat, hidden_states, latent = model(torch.from_numpy(X), torch.from_numpy(lengths))
        self.assertEqual(tuple(hidden_states.shape), (8, 32, 16))
        self.assertEqual(tuple(latent.shape), (8, 8))
        self.assertEqual(tuple(x_hat.shape), (8, 32, 49))

    def test_encoder_round_trip_preserves_outputs(self) -> None:
        X, mask, lengths = synthetic_batch()
        config = EncoderConfig(num_features=49, hidden_size=16, latent_size=8)
        model = SequenceAutoencoder(config)
        x_t, l_t = torch.from_numpy(X), torch.from_numpy(lengths)
        _, hidden_before, latent_before = model(x_t, l_t)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "encoder.pt"
            save_encoder(path, model.encoder, config)
            loaded, loaded_config = load_encoder(path)
        hidden_after, latent_after = loaded(x_t, l_t)
        self.assertEqual(loaded_config, config)
        self.assertTrue(torch.equal(hidden_before, hidden_after))
        self.assertTrue(torch.equal(latent_before, latent_after))


class MaskedLossTests(unittest.TestCase):
    def test_padding_does_not_affect_masked_loss(self) -> None:
        sequence_length, num_features = 32, 49
        x = torch.zeros(2, sequence_length, num_features)
        x_hat = torch.zeros(2, sequence_length, num_features)
        mask = torch.zeros(2, sequence_length, dtype=torch.bool)
        mask[:, :5] = True
        x[:, :5] = 1.0
        x_hat[:, :5] = 1.0
        x_hat[:, 5:] = 999.0  # error grande fuera de la máscara
        loss = masked_reconstruction_error(x, x_hat, mask)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_per_sequence_loss_matches_scalar_average(self) -> None:
        X, mask, lengths = synthetic_batch(batch_size=4)
        config = EncoderConfig(num_features=49, hidden_size=16, latent_size=8)
        model = SequenceAutoencoder(config)
        x_t = torch.from_numpy(X)
        mask_t = torch.from_numpy(mask)
        x_hat, _, _ = model(x_t, torch.from_numpy(lengths))
        per_sequence = masked_reconstruction_error(x_t, x_hat, mask_t, per_sequence=True)
        scalar = masked_reconstruction_error(x_t, x_hat, mask_t)
        weights = mask.sum(axis=1)
        weighted_mean = (per_sequence.detach().numpy() * weights).sum() / weights.sum()
        self.assertAlmostEqual(float(scalar.item()), float(weighted_mean), places=4)


class TrainingLoopTests(unittest.TestCase):
    def test_fit_autoencoder_reduces_validation_loss(self) -> None:
        set_seed(0)
        X, mask, lengths = synthetic_batch(batch_size=64, seed=1)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X), torch.from_numpy(lengths), torch.from_numpy(mask)),
            batch_size=16,
            shuffle=True,
        )
        config = EncoderConfig(num_features=49, hidden_size=16, latent_size=8)
        model = SequenceAutoencoder(config)
        history = fit_autoencoder(
            model, loader, loader, epochs=5, lr=1e-2, patience=5, device=torch.device("cpu")
        )
        self.assertGreaterEqual(len(history["train_loss"]), 1)
        self.assertLessEqual(history["val_loss"][-1], history["val_loss"][0] + 1e-6)


class EvaluationTests(unittest.TestCase):
    def test_reconstruction_scores_and_threshold_selection(self) -> None:
        X, mask, lengths = synthetic_batch(batch_size=50, seed=2)
        labels = (np.arange(50) % 10 == 0).astype(np.int8)
        config = EncoderConfig(num_features=49, hidden_size=16, latent_size=8)
        model = SequenceAutoencoder(config)
        scores = reconstruction_scores(model, X, mask, lengths, device=torch.device("cpu"))
        self.assertEqual(scores.shape, (50,))
        metrics = anomaly_metrics(scores, labels)
        self.assertIn("roc_auc", metrics)
        self.assertIn("pr_auc", metrics)
        for method, kwargs in (
            ("f1_max", {}),
            ("budget", {"budget": 0.2}),
            ("percentile", {"reference_scores": scores[labels == 0], "percentile": 95}),
        ):
            candidate = select_threshold(scores, labels, method=method, **kwargs)
            self.assertIn("threshold", candidate)
            self.assertIn("f1", candidate)

    def test_error_by_feature_group_covers_all_features(self) -> None:
        X, mask, lengths = synthetic_batch(batch_size=10, seed=3)
        feature_names = (
            ["log_amount_paid", "log_amount_received", "log_amount_abs_difference",
             "log_time_gap_minutes", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"]
            + ["same_bank", "same_currency", "destination_changed", "new_destination"]
            + [f"payment_format_{i}" for i in range(7)]
            + [f"payment_currency_{i}" for i in range(15)]
            + [f"receiving_currency_{i}" for i in range(15)]
        )
        self.assertEqual(len(feature_names), 49)
        x_hat = X + 0.1
        groups = error_by_feature_group(X, x_hat, mask, feature_names)
        self.assertEqual(set(groups), {"continuous", "binary", "one_hot"})
        for value in groups.values():
            self.assertGreaterEqual(value, 0.0)


class PublicApiTests(unittest.TestCase):
    def test_get_anomaly_score_round_trip(self) -> None:
        config = EncoderConfig(num_features=49, hidden_size=16, latent_size=8)
        model = SequenceAutoencoder(config)
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "stage_a_model.pt"
            threshold_path = Path(directory) / "anomaly_threshold.json"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": {
                        "num_features": 49,
                        "hidden_size": 16,
                        "latent_size": 8,
                        "num_layers": 1,
                        "dropout": 0.0,
                    },
                },
                model_path,
            )
            threshold_path.write_text(
                json.dumps(
                    {
                        "normal_train_error_stats": {"mean": 1.0, "std": 0.1},
                        "selected": {"method": "f1_max", "value": 1.2},
                    }
                ),
                encoding="utf-8",
            )
            sequence = np.random.default_rng(4).normal(size=(32, 49)).astype(np.float32)
            result = get_anomaly_score(
                sequence, 15, model_path=model_path, threshold_path=threshold_path
            )
        self.assertIn("score", result)
        self.assertIn("z_score", result)
        self.assertIn("is_anomalous", result)


if __name__ == "__main__":
    unittest.main()
