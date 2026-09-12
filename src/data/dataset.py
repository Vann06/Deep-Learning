"""Contrato de lectura/escritura de los artefactos compartidos."""

from __future__ import annotations

from pathlib import Path

import numpy as np


REQUIRED_ARRAYS = (
    "X",
    "y",
    "sender_id",
    "lengths",
    "mask",
    "transaction_y",
    "row_id",
)


def save_split(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = set(REQUIRED_ARRAYS).difference(arrays)
    if missing:
        raise ValueError(f"Faltan arrays requeridos: {sorted(missing)}")
    np.savez_compressed(path, **{name: arrays[name] for name in REQUIRED_ARRAYS})


def load_split(path: str | Path, *, validate: bool = True) -> dict[str, np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = set(REQUIRED_ARRAYS).difference(archive.files)
        if missing:
            raise ValueError(f"{path} no cumple el contrato: {sorted(missing)}")
        arrays = {name: archive[name] for name in REQUIRED_ARRAYS}
    if validate:
        validate_split(arrays, source=str(path))
    return arrays


def validate_split(arrays: dict[str, np.ndarray], *, source: str = "split") -> None:
    X = arrays["X"]
    y = arrays["y"]
    lengths = arrays["lengths"]
    mask = arrays["mask"]
    transaction_y = arrays["transaction_y"]
    row_id = arrays["row_id"]
    n, sequence_length, _ = X.shape
    if any(len(arrays[name]) != n for name in REQUIRED_ARRAYS if name != "X"):
        raise AssertionError(f"Dimensión inicial inconsistente en {source}")
    if mask.shape != (n, sequence_length):
        raise AssertionError(f"Máscara con shape inválido en {source}: {mask.shape}")
    if transaction_y.shape != mask.shape or row_id.shape != mask.shape:
        raise AssertionError(f"Trazabilidad con shape inválido en {source}")
    if not np.isfinite(X).all():
        raise AssertionError(f"NaN/Inf en {source}")
    if not set(np.unique(y)).issubset({0, 1}):
        raise AssertionError(f"Labels fuera de 0/1 en {source}")
    if len(np.unique(arrays["sender_id"])) != n:
        raise AssertionError(f"Remitentes duplicados dentro de {source}")
    if not np.array_equal(mask.sum(axis=1), lengths):
        raise AssertionError(f"Longitudes y máscara no coinciden en {source}")
    expected_mask = np.arange(sequence_length)[None, :] < lengths[:, None]
    if not np.array_equal(mask, expected_mask):
        raise AssertionError(f"La máscara no representa post-padding en {source}")
    if not np.allclose(X[~mask], 0.0):
        raise AssertionError(f"El padding de X no es cero en {source}")
    if not np.all(transaction_y[~mask] == -1) or not np.all(row_id[~mask] == -1):
        raise AssertionError(f"El padding de trazabilidad es inválido en {source}")
    if not set(np.unique(transaction_y[mask])).issubset({0, 1}):
        raise AssertionError(f"Etiquetas transaccionales inválidas en {source}")
    if not np.array_equal((transaction_y == 1).any(axis=1).astype(np.int8), y):
        raise AssertionError(f"Label de secuencia inconsistente en {source}")
