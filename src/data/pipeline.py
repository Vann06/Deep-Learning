"""Pipeline reproducible de IBM AML a secuencias listas para PyTorch."""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .dataset import load_split, save_split
from .preprocessing import FeaturePreprocessor, engineer_transaction_features
from .sequences import choose_windows, pack_split
from .splits import SPLIT_NAMES, assert_disjoint_senders, split_and_sample_senders


IBM_COLUMNS = [
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account.1",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
]


@dataclass(frozen=True)
class PipelineConfig:
    sequence_length: int = 32
    min_transactions: int = 3
    normal_to_positive_ratio: int = 20
    random_state: int = 42
    train_size: float = 0.70
    val_size: float = 0.15
    chunksize: int = 350_000


def _sha256(path: Path, block_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_selected_transactions(
    ibm_csv: Path,
    selected_senders: set[str],
    *,
    chunksize: int,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    previous_timestamp: pd.Timestamp | None = None
    source_is_monotonic = True
    for chunk in pd.read_csv(
        ibm_csv,
        chunksize=chunksize,
    ):
        unexpected_columns = set(chunk.columns).difference(IBM_COLUMNS)
        missing_columns = set(IBM_COLUMNS).difference(chunk.columns)
        if unexpected_columns or missing_columns:
            raise ValueError(
                "Esquema IBM AML inesperado: "
                f"faltan={sorted(missing_columns)}, extra={sorted(unexpected_columns)}"
            )
        for column in ("Account", "Account.1", "Receiving Currency", "Payment Currency", "Payment Format"):
            chunk[column] = chunk[column].astype(pd.StringDtype())
        for column in ("From Bank", "To Bank"):
            chunk[column] = chunk[column].astype(np.int32)
        for column in ("Amount Received", "Amount Paid"):
            chunk[column] = chunk[column].astype(np.float32)
        chunk["Is Laundering"] = chunk["Is Laundering"].astype(np.int8)
        chunk["Timestamp"] = pd.to_datetime(
            chunk["Timestamp"], format="%Y/%m/%d %H:%M", errors="raise"
        )
        timestamps = chunk["Timestamp"]
        source_is_monotonic &= bool(timestamps.is_monotonic_increasing)
        if previous_timestamp is not None and len(chunk):
            source_is_monotonic &= bool(timestamps.iloc[0] >= previous_timestamp)
        if len(chunk):
            previous_timestamp = timestamps.iloc[-1]
        chunk["sender_id"] = chunk["From Bank"].astype("string") + ":" + chunk["Account"]
        keep = chunk["sender_id"].isin(selected_senders)
        if keep.any():
            selected = chunk.loc[keep].copy()
            selected["source_row_id"] = selected.index.to_numpy(dtype=np.int64)
            parts.append(selected)

    if not parts:
        raise RuntimeError("Ninguna transacción coincide con los remitentes seleccionados")
    transactions = pd.concat(parts, ignore_index=True)
    transactions = transactions.rename(
        columns={
            "Timestamp": "timestamp",
            "From Bank": "from_bank",
            "Account": "sender_account",
            "To Bank": "to_bank",
            "Account.1": "destination_account",
            "Amount Received": "amount_received",
            "Receiving Currency": "receiving_currency",
            "Amount Paid": "amount_paid",
            "Payment Currency": "payment_currency",
            "Payment Format": "payment_format",
            "Is Laundering": "is_laundering",
        }
    )
    transactions = transactions.sort_values(
        ["sender_id", "timestamp", "source_row_id"], kind="stable"
    ).reset_index(drop=True)
    transactions.attrs["source_is_monotonic"] = source_is_monotonic
    return transactions


def _window_positions_for_split(
    windows: dict[str, np.ndarray], sender_ids: np.ndarray
) -> np.ndarray:
    return np.concatenate([windows[str(sender_id)] for sender_id in sender_ids])


def _write_context(
    path: Path,
    transactions: pd.DataFrame,
    windows: dict[str, np.ndarray],
    sender_ids: np.ndarray,
) -> None:
    columns = [
        "source_row_id",
        "timestamp",
        "sender_id",
        "from_bank",
        "sender_account",
        "to_bank",
        "destination_account",
        "amount_received",
        "receiving_currency",
        "amount_paid",
        "payment_currency",
        "payment_format",
        "is_laundering",
    ]
    parts = []
    for sequence_id, sender_id in enumerate(sorted(map(str, sender_ids.tolist()))):
        context = transactions.iloc[windows[sender_id]][columns].copy()
        context.insert(0, "timestep", np.arange(len(context), dtype=np.int16))
        context.insert(0, "sequence_id", sequence_id)
        parts.append(context)
    pd.concat(parts, ignore_index=True).to_csv(path, index=False, compression="gzip")


def _plot_sequence_lengths(
    sender_summary: pd.DataFrame, output: Path, config: PipelineConfig
) -> None:
    eligible = sender_summary.loc[
        sender_summary["transaction_count"].ge(config.min_transactions), "transaction_count"
    ]
    clipped = eligible.clip(upper=150)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(clipped, bins=50, color="#2f7d32", alpha=0.85)
    ax.axvline(config.sequence_length, color="#c62828", linestyle="--", linewidth=2)
    ax.set_yscale("log")
    ax.set_xlabel("Transacciones por remitente (valores >150 recortados para visualizar)")
    ax.set_ylabel("Número de remitentes (escala log)")
    ax.set_title("IBM AML: distribución de longitudes elegibles")
    ax.text(
        config.sequence_length + 2,
        ax.get_ylim()[1] / 4,
        f"longitud elegida = {config.sequence_length}",
        color="#c62828",
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_class_balance(
    audit: dict, split_arrays: dict[str, dict[str, np.ndarray]], output: Path
) -> None:
    labels = ["Transacciones\noriginales", "Train", "Validation", "Test"]
    rates = [
        audit["ibm_aml"]["positive_transactions"] / audit["ibm_aml"]["rows"],
        *[float(split_arrays[name]["y"].mean()) for name in SPLIT_NAMES],
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, np.asarray(rates) * 100, color=["#757575", "#2f7d32", "#1976d2", "#7b1fa2"])
    ax.set_ylabel("Casos positivos (%)")
    ax.set_title("Desbalance original y después del submuestreo por remitente")
    for bar, value in zip(bars, rates, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3%}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_examples(
    transactions: pd.DataFrame,
    windows: dict[str, np.ndarray],
    test_senders: np.ndarray,
    output: Path,
) -> None:
    candidates: list[tuple[str, int, float]] = []
    for sender_id in map(str, test_senders.tolist()):
        rows = transactions.iloc[windows[sender_id]]
        candidates.append(
            (sender_id, int(rows["is_laundering"].max()), float(rows["amount_paid"].max()))
        )
    candidate_frame = pd.DataFrame(
        {
            "sender_id": [row[0] for row in candidates],
            "label": [row[1] for row in candidates],
            "max_amount": [row[2] for row in candidates],
        }
    )
    positives = candidate_frame.query("label == 1").nlargest(3, "max_amount")
    normals = candidate_frame.query("label == 0").nlargest(3, "max_amount")
    if len(positives) < 3 or len(normals) < 3:
        raise RuntimeError("No hay tres ejemplos de cada clase en test")

    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=False)
    for row_index in range(3):
        for column_index, selection in enumerate((normals, positives)):
            sender_id = selection.iloc[row_index]["sender_id"]
            rows = transactions.iloc[windows[sender_id]].reset_index(drop=True)
            ax = axes[row_index, column_index]
            ax.plot(rows.index, rows["amount_paid"], marker="o", color="#1565c0" if column_index == 0 else "#c62828")
            suspicious = rows["is_laundering"].eq(1)
            if suspicious.any():
                ax.scatter(rows.index[suspicious], rows.loc[suspicious, "amount_paid"], marker="X", s=100, color="#ffb300", edgecolor="black", label="Transacción etiquetada")
                ax.legend(loc="best", fontsize=8)
            ax.set_yscale("symlog")
            ax.set_ylabel("Monto pagado (symlog)")
            ax.set_xlabel("Paso temporal")
            kind = "normal" if column_index == 0 else "sospechosa"
            ax.set_title(f"Secuencia {kind} {row_index + 1} — {sender_id}")
    fig.suptitle("Tres secuencias normales versus tres sospechosas (antes del modelado)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _split_metadata(arrays: dict[str, np.ndarray]) -> dict:
    positives = int(arrays["y"].sum())
    negatives = int(len(arrays["y"]) - positives)
    return {
        "shape_X": list(arrays["X"].shape),
        "shape_y": list(arrays["y"].shape),
        "sequences": int(len(arrays["y"])),
        "positive": positives,
        "negative": negatives,
        "positive_rate": float(arrays["y"].mean()),
        "length_min": int(arrays["lengths"].min()),
        "length_median": float(np.median(arrays["lengths"])),
        "length_max": int(arrays["lengths"].max()),
    }


def run_pipeline(
    *,
    ibm_csv: str | Path,
    sender_summary_csv: str | Path,
    audit_json: str | Path,
    output_dir: str | Path = "artifacts",
    figures_dir: str | Path = "reports/figures",
    config: PipelineConfig = PipelineConfig(),
) -> dict:
    """Ejecuta el pipeline de ingeniería de datos y devuelve la metadata final."""
    ibm_csv = Path(ibm_csv)
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    sender_summary = pd.read_csv(
        sender_summary_csv, dtype={"sender_id": pd.StringDtype()}
    )
    audit = json.loads(Path(audit_json).read_text(encoding="utf-8"))
    selected, sampling_stats = split_and_sample_senders(
        sender_summary,
        min_transactions=config.min_transactions,
        normal_to_positive_ratio=config.normal_to_positive_ratio,
        random_state=config.random_state,
        train_size=config.train_size,
        val_size=config.val_size,
    )
    split_lookup = selected.set_index("sender_id")["split"].to_dict()
    transactions = _load_selected_transactions(
        ibm_csv, set(split_lookup), chunksize=config.chunksize
    )
    found_senders = set(transactions["sender_id"].astype(str).unique())
    missing = set(split_lookup).difference(found_senders)
    if missing:
        raise AssertionError(f"Faltan {len(missing)} remitentes seleccionados en el CSV")

    windows = choose_windows(
        transactions,
        sequence_length=config.sequence_length,
        min_transactions=config.min_transactions,
    )
    split_senders = {
        name: selected.loc[selected["split"].eq(name), "sender_id"].astype(str).to_numpy()
        for name in SPLIT_NAMES
    }
    assert_disjoint_senders(split_senders)

    engineered = engineer_transaction_features(transactions)
    train_positions = _window_positions_for_split(windows, split_senders["train"])
    preprocessor = FeaturePreprocessor().fit(engineered.iloc[train_positions])
    feature_matrix = preprocessor.transform(engineered)
    feature_names = preprocessor.get_feature_names()

    split_arrays: dict[str, dict[str, np.ndarray]] = {}
    for split_name in SPLIT_NAMES:
        arrays = pack_split(
            transactions,
            feature_matrix,
            windows,
            split_senders[split_name],
            sequence_length=config.sequence_length,
        )
        save_split(output_dir / f"{split_name}.npz", arrays)
        split_arrays[split_name] = arrays

    assert_disjoint_senders(
        {name: arrays["sender_id"] for name, arrays in split_arrays.items()}
    )
    with (output_dir / "scaler.pkl").open("wb") as stream:
        pickle.dump(preprocessor, stream, protocol=pickle.HIGHEST_PROTOCOL)
    (output_dir / "feature_names.json").write_text(
        json.dumps(feature_names, indent=2), encoding="utf-8"
    )
    selected[["sender_id", "split", "label", "transaction_count"]].to_csv(
        output_dir / "sender_splits.csv.gz", index=False, compression="gzip"
    )
    for split_name in ("val", "test"):
        _write_context(
            output_dir / f"{split_name}_context.csv.gz",
            transactions,
            windows,
            split_senders[split_name],
        )

    _plot_sequence_lengths(
        sender_summary, figures_dir / "sequence_length_distribution.png", config
    )
    _plot_class_balance(
        audit, split_arrays, figures_dir / "class_distribution.png"
    )
    _plot_examples(
        transactions,
        windows,
        split_senders["test"],
        figures_dir / "sequence_examples.png",
    )

    split_stats = {name: _split_metadata(split_arrays[name]) for name in SPLIT_NAMES}
    train_positive = split_stats["train"]["positive"]
    train_negative = split_stats["train"]["negative"]
    metadata = {
        "schema_version": 1,
        "dataset": {
            "selected": "IBM AML HI-Small",
            "kaggle_handle": "ealtman2019/ibm-transactions-for-anti-money-laundering-aml",
            "filename": ibm_csv.name,
            "sha256": _sha256(ibm_csv),
            "paysim_rejected_for_sequences": True,
            "audit_file": "audit/dataset_audit.json",
        },
        "config": asdict(config),
        "sequence_length": config.sequence_length,
        "min_transactions": config.min_transactions,
        "padding": "post-padding con ceros; mask=True únicamente en pasos reales",
        "num_features": len(feature_names),
        "feature_names": feature_names,
        "label_definition": "1 si la ventana contiene al menos una transacción Is Laundering=1",
        "sender_definition": "From Bank + ':' + Account",
        "window_policy": "una ventana por remitente; positiva anclada en la última transacción positiva, normal usa las más recientes",
        "normalization": "log1p + StandardScaler ajustado solo con timesteps TRAIN; cíclicas/binarias sin escalar; categóricas one-hot ajustadas solo con TRAIN",
        "split_criterion": "70/15/15 estratificado por etiqueta de remitente; ningún remitente compartido; submuestreo normal 20:1 dentro de cada split",
        "source_timestamp_monotonic": bool(transactions.attrs["source_is_monotonic"]),
        "sampling": sampling_stats,
        "splits": split_stats,
        "imbalance": {
            "original_transaction_positive_rate": audit["ibm_aml"]["positive_transactions"] / audit["ibm_aml"]["rows"],
            "eligible_sender_positive_rate": sampling_stats["eligible_positive_rate"],
            "train_pos_weight_recommended": train_negative / max(1, train_positive),
            "evaluation_note": "Validation/test conservan el mismo submuestreo 20:1, no la prevalencia operativa original; recalibrar antes de producción.",
        },
        "files": {
            name: f"{name}.npz" for name in SPLIT_NAMES
        }
        | {
            "preprocessor": "scaler.pkl",
            "feature_names": "feature_names.json",
            "sender_splits": "sender_splits.csv.gz",
            "validation_context": "val_context.csv.gz",
            "test_context": "test_context.csv.gz",
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Prueba inmediata de round-trip con archivos ya escritos.
    for split_name in SPLIT_NAMES:
        reloaded = load_split(output_dir / f"{split_name}.npz", validate=True)
        if not np.array_equal(reloaded["sender_id"], split_arrays[split_name]["sender_id"]):
            raise AssertionError(f"Round-trip de sender_id falló para {split_name}")
        if not np.array_equal(reloaded["y"], split_arrays[split_name]["y"]):
            raise AssertionError(f"Round-trip de y falló para {split_name}")
    return metadata
