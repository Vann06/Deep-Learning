"""Métricas del clasificador y ejecución reproducible de la ablación (Etapa B).

Las métricas agnósticas al umbral (`anomaly_metrics`) y la selección de umbral
(`select_threshold`) se reutilizan desde `src/evaluation/anomaly.py`: operan
sobre cualquier score numérico, así que sirven igual para probabilidades del
clasificador. Usar las mismas funciones en ambas etapas mantiene una única
metodología de umbral en todo el proyecto en vez de dos criterios paralelos.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.evaluation.anomaly import anomaly_metrics
from src.models.classifier import (
    ClassifierConfig,
    build_classifier,
    fit_classifier,
    make_supervised_loader,
    predict_scores,
)
from src.utils import set_seed


def classification_metrics(
    probabilities: np.ndarray, labels: np.ndarray, threshold: float
) -> dict:
    """Métricas dependientes del umbral, con la matriz de confusión desglosada.

    Se reporta `alert_rate` porque en AML el costo real es cuántos expedientes
    hay que revisar, no solo qué tan buena es la precisión.
    """
    predicted = (probabilities >= threshold).astype(np.int8)
    labels = labels.astype(np.int8)
    true_positive = int(((predicted == 1) & (labels == 1)).sum())
    false_positive = int(((predicted == 1) & (labels == 0)).sum())
    false_negative = int(((predicted == 0) & (labels == 1)).sum())
    true_negative = int(((predicted == 0) & (labels == 0)).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "alert_rate": float(predicted.mean()),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }


def run_ablation(
    splits: dict[str, dict[str, np.ndarray]],
    config: ClassifierConfig,
    *,
    arms: dict[str, str | Path | None],
    seeds: Sequence[int],
    device: torch.device,
    batch_size: int = 256,
    **fit_kwargs,
) -> tuple[pd.DataFrame, dict[tuple[str, int], dict]]:
    """Entrena cada `(brazo, semilla)` y devuelve `(tabla de resultados, corridas)`.

    `arms` mapea el nombre del brazo a la ruta de `encoder.pt` (brazo
    preentrenado) o a `None` (brazo entrenado desde cero). Cada corrida llama a
    `set_seed` antes de construir el modelo, de modo que la inicialización de la
    atención y la cabeza, el orden de los batches y el dropout son idénticos
    entre brazos para una misma semilla: la única diferencia real es de dónde
    parte el encoder.

    `corridas[(brazo, semilla)]` trae `{"state_dict", "history"}`. Se devuelven
    los pesos para que el notebook se quede con el mejor según VALIDATION
    (elegirlo por TEST sería leakage) y el historial para graficar las curvas
    del brazo principal sin tener que reentrenarlo aparte.
    """
    val_loader = make_supervised_loader(splits["val"], batch_size=batch_size)
    test_loader = make_supervised_loader(splits["test"], batch_size=batch_size)

    rows: list[dict] = []
    runs: dict[tuple[str, int], dict] = {}

    for arm_name, encoder_path in arms.items():
        for seed in seeds:
            set_seed(seed)
            train_loader = make_supervised_loader(
                splits["train"], batch_size=batch_size, shuffle=True
            )
            model = build_classifier(
                config, pretrained_encoder_path=encoder_path, map_location=device
            )
            started = time.perf_counter()
            history = fit_classifier(
                model, train_loader, val_loader, device=device, **fit_kwargs
            )
            minutes = (time.perf_counter() - started) / 60

            val_probabilities, _, val_labels = predict_scores(model, val_loader, device)
            test_probabilities, _, test_labels = predict_scores(model, test_loader, device)
            val_metrics = anomaly_metrics(val_probabilities, val_labels)
            test_metrics = anomaly_metrics(test_probabilities, test_labels)

            rows.append(
                {
                    "arm": arm_name,
                    "seed": int(seed),
                    "epochs_run": history["epochs_run"],
                    "best_epoch": history["best_epoch"],
                    "val_roc_auc": val_metrics["roc_auc"],
                    "val_pr_auc": val_metrics["pr_auc"],
                    "test_roc_auc": test_metrics["roc_auc"],
                    "test_pr_auc": test_metrics["pr_auc"],
                    "train_minutes": float(minutes),
                }
            )
            runs[(arm_name, int(seed))] = {
                "state_dict": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
                "history": history,
            }

    return pd.DataFrame(rows), runs


def summarize_ablation(
    results: pd.DataFrame,
    *,
    metrics: Sequence[str] = ("val_pr_auc", "test_pr_auc", "test_roc_auc"),
) -> pd.DataFrame:
    """Media y desviación estándar por brazo.

    Con una sola semilla por brazo no se puede distinguir una mejora real del
    ruido de inicialización; por eso la conclusión de la ablación se lee sobre
    esta tabla y no sobre una corrida suelta.
    """
    summary = results.groupby("arm")[list(metrics)].agg(["mean", "std"])
    return summary.sort_values((metrics[0], "mean"), ascending=False)
