"""Splits reproducibles por remitente, sin compartir entidades."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SPLIT_NAMES = ("train", "val", "test")


def split_and_sample_senders(
    sender_summary: pd.DataFrame,
    *,
    min_transactions: int,
    normal_to_positive_ratio: int,
    random_state: int,
    train_size: float = 0.70,
    val_size: float = 0.15,
) -> tuple[pd.DataFrame, dict]:
    """Separa remitentes estratificados y submuestrea solo la clase normal.

    Todos los remitentes positivos elegibles se conservan. El split se realiza
    antes del submuestreo para garantizar conjuntos disjuntos por identidad.
    """
    required = {"sender_id", "transaction_count", "positive_transactions"}
    missing = required.difference(sender_summary.columns)
    if missing:
        raise ValueError(f"Faltan columnas en sender_summary: {sorted(missing)}")
    if train_size <= 0 or val_size <= 0 or train_size + val_size >= 1:
        raise ValueError("train_size y val_size deben dejar una fracción positiva para test")

    eligible = sender_summary.loc[
        sender_summary["transaction_count"].ge(min_transactions)
    ].copy()
    eligible["label"] = eligible["positive_transactions"].gt(0).astype("int8")
    if eligible["label"].nunique() != 2:
        raise ValueError("Se requieren remitentes de ambas clases para estratificar")

    train, remainder = train_test_split(
        eligible,
        train_size=train_size,
        random_state=random_state,
        stratify=eligible["label"],
    )
    relative_val = val_size / (1.0 - train_size)
    val, test = train_test_split(
        remainder,
        train_size=relative_val,
        random_state=random_state + 1,
        stratify=remainder["label"],
    )

    rng = np.random.default_rng(random_state)
    selected_parts: list[pd.DataFrame] = []
    split_stats: dict[str, dict] = {}
    for split_name, frame in zip(SPLIT_NAMES, (train, val, test), strict=True):
        positives = frame.loc[frame["label"].eq(1)]
        normals = frame.loc[frame["label"].eq(0)]
        normal_target = min(len(normals), normal_to_positive_ratio * len(positives))
        chosen_normal_idx = rng.choice(normals.index.to_numpy(), normal_target, replace=False)
        chosen = pd.concat([positives, normals.loc[chosen_normal_idx]], axis=0)
        chosen = chosen.sort_values("sender_id", kind="stable").copy()
        chosen["split"] = split_name
        selected_parts.append(chosen)
        split_stats[split_name] = {
            "eligible_before_sampling": int(len(frame)),
            "positive_senders": int(len(positives)),
            "normal_senders_available": int(len(normals)),
            "normal_senders_selected": int(normal_target),
            "selected_senders": int(len(chosen)),
        }

    selected = pd.concat(selected_parts, ignore_index=True)
    if selected["sender_id"].duplicated().any():
        raise AssertionError("Un remitente fue asignado a más de un split")

    stats = {
        "eligible_senders": int(len(eligible)),
        "eligible_positive_senders": int(eligible["label"].sum()),
        "eligible_positive_rate": float(eligible["label"].mean()),
        "normal_to_positive_ratio_target": int(normal_to_positive_ratio),
        "splits": split_stats,
    }
    return selected, stats


def assert_disjoint_senders(split_to_senders: dict[str, np.ndarray]) -> None:
    """Falla si cualquier identidad aparece en dos splits."""
    sets = {name: set(values.tolist()) for name, values in split_to_senders.items()}
    for index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[index + 1 :]:
            overlap = sets[left].intersection(sets[right])
            if overlap:
                raise AssertionError(
                    f"Leakage: {len(overlap)} remitentes compartidos entre {left} y {right}"
                )
