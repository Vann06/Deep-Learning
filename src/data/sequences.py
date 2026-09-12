"""Selección y empaquetado de ventanas temporales por remitente."""

from __future__ import annotations

import numpy as np
import pandas as pd


def choose_windows(
    transactions: pd.DataFrame,
    *,
    sequence_length: int,
    min_transactions: int,
) -> dict[str, np.ndarray]:
    """Elige una ventana por remitente y garantiza incluir una etiqueta positiva.

    Para un remitente positivo, la ventana termina en su última transacción
    etiquetada salvo que se necesiten observaciones posteriores para alcanzar la
    longitud mínima. Para un remitente normal se usa su historia más reciente.
    """
    windows: dict[str, np.ndarray] = {}
    labels = transactions["is_laundering"].to_numpy(dtype=np.int8)
    for sender_id, positions in transactions.groupby(
        "sender_id", sort=True, observed=True
    ).indices.items():
        positions = np.asarray(positions, dtype=np.int64)
        n_transactions = len(positions)
        if n_transactions < min_transactions:
            continue
        local_labels = labels[positions]
        if local_labels.any():
            anchor = int(np.flatnonzero(local_labels)[-1])
            if n_transactions <= sequence_length:
                chosen = positions
            else:
                end = min(n_transactions, max(sequence_length, anchor + 1))
                start = end - sequence_length
                chosen = positions[start:end]
            if not labels[chosen].any():
                raise AssertionError(f"La ventana positiva de {sender_id} perdió su etiqueta")
        else:
            chosen = positions[-sequence_length:]
        windows[str(sender_id)] = chosen
    return windows


def pack_split(
    transactions: pd.DataFrame,
    feature_matrix: np.ndarray,
    windows: dict[str, np.ndarray],
    sender_ids: np.ndarray,
    *,
    sequence_length: int,
) -> dict[str, np.ndarray]:
    """Empaqueta secuencias con post-padding, máscara y trazabilidad de filas."""
    sender_ids = np.asarray(sorted(map(str, sender_ids.tolist())))
    num_features = feature_matrix.shape[1]
    n = len(sender_ids)
    X = np.zeros((n, sequence_length, num_features), dtype=np.float32)
    y = np.zeros(n, dtype=np.int8)
    lengths = np.zeros(n, dtype=np.int16)
    mask = np.zeros((n, sequence_length), dtype=bool)
    transaction_y = np.full((n, sequence_length), -1, dtype=np.int8)
    row_ids = np.full((n, sequence_length), -1, dtype=np.int64)
    labels = transactions["is_laundering"].to_numpy(dtype=np.int8)
    source_rows = transactions["source_row_id"].to_numpy(dtype=np.int64)

    for sequence_id, sender_id in enumerate(sender_ids):
        positions = windows[sender_id]
        length = len(positions)
        X[sequence_id, :length] = feature_matrix[positions]
        lengths[sequence_id] = length
        mask[sequence_id, :length] = True
        transaction_y[sequence_id, :length] = labels[positions]
        row_ids[sequence_id, :length] = source_rows[positions]
        y[sequence_id] = int(labels[positions].max())

    return {
        "X": X,
        "y": y,
        "sender_id": sender_ids,
        "lengths": lengths,
        "mask": mask,
        "transaction_y": transaction_y,
        "row_id": row_ids,
    }
