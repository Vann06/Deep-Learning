"""Evaluación de anomalías: scores, métricas, umbral y la API pública del MVP."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from src.models.autoencoder import SequenceAutoencoder, masked_reconstruction_error
from src.models.encoder import EncoderConfig
from src.utils import get_device

BINARY_FEATURE_NAMES = {"same_bank", "same_currency", "destination_changed", "new_destination"}
ONE_HOT_PREFIXES = ("payment_format_", "payment_currency_", "receiving_currency_")


def reconstruction_scores(
    model: SequenceAutoencoder,
    X: np.ndarray,
    mask: np.ndarray,
    lengths: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Error de reconstrucción por secuencia sobre un split completo, en batches."""
    model.eval()
    model.to(device)
    scores = np.empty(len(X), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            end = start + batch_size
            x = torch.from_numpy(X[start:end]).to(device)
            m = torch.from_numpy(mask[start:end]).to(device)
            l = torch.from_numpy(lengths[start:end].astype(np.int64))
            x_hat, _, _ = model(x, l)
            batch_scores = masked_reconstruction_error(x, x_hat, m, per_sequence=True)
            scores[start:end] = batch_scores.cpu().numpy()
    return scores


def anomaly_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Métricas agnósticas al umbral: apropiadas con 4.76% de positivos (accuracy no sirve)."""
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "n_sequences": int(len(scores)),
        "n_positive": int(labels.sum()),
    }


def _precision_recall_f1_at(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    predicted = (scores >= threshold).astype(np.int8)
    true_positive = int(((predicted == 1) & (labels == 1)).sum())
    false_positive = int(((predicted == 1) & (labels == 0)).sum())
    false_negative = int(((predicted == 0) & (labels == 1)).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "alert_rate": float(predicted.mean()),
    }


def select_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    method: str,
    budget: float = 0.05,
    reference_scores: np.ndarray | None = None,
    percentile: float = 95.0,
) -> dict:
    """Un candidato de umbral, con su justificación y sus métricas sobre `scores`/`labels`.

    - "f1_max": recorre la curva precision-recall de VALIDATION y toma el punto
      de F1 máximo. Es el criterio principal: con clases tan desbalanceadas,
      ROC es optimista y accuracy es inútil.
    - "budget": fija cuántas alertas por revisar caben en un presupuesto
      operativo (ej. el 5% de mayor score). Traduce directamente el problema
      de negocio de falsos positivos del enunciado.
    - "percentile": referencia clásica de detección de anomalías, el percentil
      P de los errores sobre secuencias normales de TRAIN.
    """
    if method == "f1_max":
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
        best_index = int(np.nanargmax(f1[:-1])) if len(thresholds) else 0
        threshold = float(thresholds[best_index]) if len(thresholds) else float(scores.max())
    elif method == "budget":
        threshold = float(np.quantile(scores, 1.0 - budget))
    elif method == "percentile":
        if reference_scores is None:
            raise ValueError("method='percentile' requiere reference_scores")
        threshold = float(np.percentile(reference_scores, percentile))
    else:
        raise ValueError(f"Método de umbral desconocido: {method}")

    result = _precision_recall_f1_at(scores, labels, threshold)
    result["method"] = method
    return result


def _feature_groups(feature_names: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {"continuous": [], "binary": [], "one_hot": []}
    for index, name in enumerate(feature_names):
        if name in BINARY_FEATURE_NAMES:
            groups["binary"].append(index)
        elif name.startswith(ONE_HOT_PREFIXES):
            groups["one_hot"].append(index)
        else:
            groups["continuous"].append(index)
    return groups


def error_by_feature_group(
    x: np.ndarray, x_hat: np.ndarray, mask: np.ndarray, feature_names: list[str]
) -> dict[str, float]:
    """MSE promedio por grupo de features, solo sobre pasos reales.

    Las 37 columnas one-hot pueden dominar una MSE plana y ahogar las 8
    columnas continuas (montos y tiempos), que es donde vive la señal AML.
    Esta descomposición es la evidencia para decidir si hace falta ponderar.
    """
    groups = _feature_groups(feature_names)
    squared_error = (x - x_hat) ** 2  # [N, T, F]
    result: dict[str, float] = {}
    for group_name, indices in groups.items():
        if not indices:
            continue
        group_error = squared_error[:, :, indices]  # [N, T, len(indices)]
        group_mask = np.broadcast_to(mask[:, :, None], group_error.shape)
        result[group_name] = float(group_error[group_mask].mean())
    return result


_STAGE_A_CACHE: dict[tuple[str, str], tuple[SequenceAutoencoder, dict]] = {}


def _load_stage_a(model_path: Path, threshold_path: Path, device: torch.device) -> tuple:
    key = (str(model_path), str(threshold_path))
    if key not in _STAGE_A_CACHE:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        config = EncoderConfig(**checkpoint["config"])
        model = SequenceAutoencoder(config)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        model.to(device)
        threshold_info = json.loads(Path(threshold_path).read_text(encoding="utf-8"))
        _STAGE_A_CACHE[key] = (model, threshold_info)
    return _STAGE_A_CACHE[key]


def get_anomaly_score(
    sequence: np.ndarray,
    length: int | None = None,
    *,
    model_path: str | Path = "artifacts/stage_a_model.pt",
    threshold_path: str | Path = "artifacts/anomaly_threshold.json",
    device: torch.device | None = None,
) -> dict:
    """API pública de la Etapa A (contrato de `docs/DIVISION.md`).

    Recibe una secuencia `[T, F]` (o `[1, T, F]`) y su longitud real de
    transacciones (si no se indica, asume que todos los pasos son reales).
    Carga el autoencoder y el umbral una sola vez por proceso.

    Devuelve un dict con:
    - `score`: error de reconstrucción crudo (lo que exige la interfaz mínima).
    - `z_score`: el mismo error expresado en desviaciones sobre la media de
      los remitentes normales de TRAIN, para la explicación en lenguaje
      natural del MVP ("2.3σ sobre el comportamiento normal").
    - `threshold` / `is_anomalous`: el umbral elegido en el notebook y si la
      secuencia lo supera.
    """
    device = device or get_device()
    model, threshold_info = _load_stage_a(Path(model_path), Path(threshold_path), device)

    x = np.asarray(sequence, dtype=np.float32)
    if x.ndim == 2:
        x = x[None, ...]
    total_length = x.shape[1]
    real_length = total_length if length is None else int(length)
    mask = (np.arange(total_length)[None, :] < real_length)

    x_t = torch.from_numpy(x).to(device)
    mask_t = torch.from_numpy(mask).to(device)
    lengths_t = torch.tensor([real_length], dtype=torch.int64)

    with torch.no_grad():
        x_hat, _, _ = model(x_t, lengths_t)
        raw_error = masked_reconstruction_error(x_t, x_hat, mask_t, per_sequence=True)[0].item()

    stats = threshold_info["normal_train_error_stats"]
    z_score = (raw_error - stats["mean"]) / max(stats["std"], 1e-8)
    threshold = threshold_info["selected"]["value"]
    return {
        "score": raw_error,
        "z_score": float(z_score),
        "threshold": float(threshold),
        "is_anomalous": bool(raw_error >= threshold),
    }
