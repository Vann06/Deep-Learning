from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.models.attention import AdditiveAttention
from src.models.classifier import (
    AMLClassifier,
    ClassifierConfig,
    build_classifier,
    fit_classifier,
    load_classifier,
    make_supervised_loader,
    predict_aml,
    save_classifier,
)
from src.models.encoder import EncoderConfig, SequenceEncoder, save_encoder
from src.utils import set_seed

CPU = torch.device("cpu")


def small_config(num_features: int = 49, hidden_size: int = 16, latent_size: int = 8) -> ClassifierConfig:
    encoder = EncoderConfig(
        num_features=num_features, hidden_size=hidden_size, latent_size=latent_size
    )
    return ClassifierConfig(encoder=encoder, attention_size=8, head_hidden=8, dropout=0.0)


def synthetic_batch(
    batch_size: int = 8, sequence_length: int = 32, num_features: int = 49, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lengths = rng.integers(3, sequence_length + 1, size=batch_size).astype(np.int64)
    mask = np.arange(sequence_length)[None, :] < lengths[:, None]
    X = rng.normal(size=(batch_size, sequence_length, num_features)).astype(np.float32)
    X = X * mask[:, :, None]
    return X, mask, lengths


class AttentionTests(unittest.TestCase):
    def test_weights_sum_to_one_and_ignore_padding(self) -> None:
        set_seed(0)
        _, mask, _ = synthetic_batch(batch_size=6)
        hidden = torch.randn(6, 32, 16)
        attention = AdditiveAttention(hidden_size=16, attention_size=8)
        with torch.no_grad():
            context, weights = attention(hidden, torch.from_numpy(mask))
        self.assertEqual(tuple(context.shape), (6, 16))
        self.assertEqual(tuple(weights.shape), (6, 32))
        # exactamente cero fuera de la máscara: el heatmap no debe señalar padding
        self.assertEqual(float(weights[~torch.from_numpy(mask)].abs().max()), 0.0)
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones(6), atol=1e-6))

    def test_extra_padding_does_not_change_context(self) -> None:
        set_seed(0)
        attention = AdditiveAttention(hidden_size=16, attention_size=8)
        hidden_short = torch.randn(3, 10, 16)
        lengths = torch.tensor([4, 7, 10])
        mask_short = torch.arange(10)[None, :] < lengths[:, None]
        hidden_short = hidden_short * mask_short[:, :, None]

        hidden_long = torch.zeros(3, 32, 16)
        hidden_long[:, :10] = hidden_short
        mask_long = torch.arange(32)[None, :] < lengths[:, None]

        context_short, _ = attention(hidden_short, mask_short)
        context_long, _ = attention(hidden_long, mask_long)
        self.assertTrue(torch.allclose(context_short, context_long, atol=1e-6))


class ClassifierContractTests(unittest.TestCase):
    def test_forward_shapes(self) -> None:
        X, mask, lengths = synthetic_batch()
        model = AMLClassifier(small_config())
        logits, weights = model(
            torch.from_numpy(X), torch.from_numpy(lengths), torch.from_numpy(mask)
        )
        self.assertEqual(tuple(logits.shape), (8,))
        self.assertEqual(tuple(weights.shape), (8, 32))

    def test_pretrained_arm_loads_stage_a_weights_and_scratch_does_not(self) -> None:
        config = small_config()
        set_seed(123)
        source = SequenceEncoder(config.encoder)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "encoder.pt"
            save_encoder(path, source, config.encoder)
            pretrained = build_classifier(config, pretrained_encoder_path=path)
            scratch = build_classifier(config, pretrained_encoder_path=None)

        reference = source.state_dict()
        loaded = pretrained.encoder.state_dict()
        self.assertEqual(sorted(loaded), sorted(reference))
        for key, value in reference.items():
            self.assertTrue(torch.equal(value, loaded[key]), f"{key} no coincide")

        random_init = scratch.encoder.state_dict()
        differs = any(not torch.equal(reference[key], random_init[key]) for key in reference)
        self.assertTrue(differs, "El brazo scratch no debería heredar los pesos de la Etapa A")

    def test_build_classifier_rejects_mismatched_encoder_config(self) -> None:
        config = small_config()
        other = EncoderConfig(num_features=49, hidden_size=32, latent_size=8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "encoder.pt"
            save_encoder(path, SequenceEncoder(other), other)
            with self.assertRaises(ValueError):
                build_classifier(config, pretrained_encoder_path=path)


class TrainingTests(unittest.TestCase):
    def _separable_split(self, n: int, seed: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        sequence_length, num_features = 12, 6
        lengths = rng.integers(3, sequence_length + 1, size=n).astype(np.int16)
        mask = np.arange(sequence_length)[None, :] < lengths[:, None]
        y = (rng.random(n) < 0.5).astype(np.int8)
        X = rng.normal(scale=0.1, size=(n, sequence_length, num_features)).astype(np.float32)
        X[:, :, 0] += y[:, None] * 4.0  # señal separable en la primera feature
        X = (X * mask[:, :, None]).astype(np.float32)
        return {"X": X, "y": y, "lengths": lengths, "mask": mask}

    def test_fit_classifier_improves_validation_pr_auc(self) -> None:
        set_seed(0)
        train = self._separable_split(240, seed=1)
        val = self._separable_split(120, seed=2)
        config = small_config(num_features=6, hidden_size=8, latent_size=4)
        model = build_classifier(config)
        history = fit_classifier(
            model,
            make_supervised_loader(train, batch_size=32, shuffle=True),
            make_supervised_loader(val, batch_size=32),
            epochs=15,
            patience=15,
            pos_weight=1.0,
            device=CPU,
        )
        self.assertEqual(len(history["train_loss"]), history["epochs_run"])
        self.assertGreater(history["best_val_pr_auc"], 0.9)


class PublicApiTests(unittest.TestCase):
    def test_predict_aml_round_trip(self) -> None:
        set_seed(7)
        config = small_config()
        model = build_classifier(config)
        model.eval()
        X, mask, lengths = synthetic_batch(batch_size=1, seed=5)

        with torch.no_grad():
            logits, weights = model(
                torch.from_numpy(X), torch.from_numpy(lengths), torch.from_numpy(mask)
            )
        expected_probability = float(torch.sigmoid(logits)[0].item())
        real_length = int(lengths[0])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage_b_model.pt"
            save_classifier(path, model, config)
            reloaded, reloaded_config = load_classifier(path)
            self.assertEqual(reloaded_config, config)
            probability, attention = predict_aml(X[0], real_length, model_path=path, device=CPU)

        self.assertAlmostEqual(probability, expected_probability, places=5)
        # los pesos llegan recortados a la longitud real y siguen sumando 1
        self.assertEqual(attention.shape, (real_length,))
        self.assertAlmostEqual(float(attention.sum()), 1.0, places=5)
        self.assertTrue(
            np.allclose(attention, weights[0, :real_length].numpy(), atol=1e-6)
        )


if __name__ == "__main__":
    unittest.main()
