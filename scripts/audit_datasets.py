"""Audita si PaySim e IBM AML permiten secuencias temporales por remitente.

El script procesa los CSV por bloques para no cargar los datasets completos en RAM.
Es una utilidad de EDA y no entrena modelos.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _merge_sender_summaries(parts: list[pd.DataFrame]) -> pd.DataFrame:
    summary = pd.concat(parts, ignore_index=True)
    return (
        summary.groupby("sender_id", sort=False, observed=True)
        .agg(
            transaction_count=("transaction_count", "sum"),
            positive_transactions=("positive_transactions", "sum"),
            first_timestamp=("first_timestamp", "min"),
            last_timestamp=("last_timestamp", "max"),
        )
        .reset_index()
    )


def audit_paysim(path: Path, chunksize: int = 500_000) -> tuple[dict, pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    rows = positives = 0
    for chunk in pd.read_csv(
        path,
        usecols=lambda column: column in {"nameOrig", "step", "isFraud"},
        chunksize=chunksize,
    ):
        chunk["nameOrig"] = chunk["nameOrig"].astype(pd.StringDtype())
        chunk["step"] = chunk["step"].astype(np.int16)
        chunk["isFraud"] = chunk["isFraud"].astype(np.int8)
        rows += len(chunk)
        positives += int(chunk["isFraud"].sum())
        grouped = chunk.groupby("nameOrig", sort=False, observed=True)
        part = grouped.agg(
            transaction_count=("isFraud", "size"),
            positive_transactions=("isFraud", "sum"),
            first_timestamp=("step", "min"),
            last_timestamp=("step", "max"),
        ).reset_index(names="sender_id")
        parts.append(part)

    senders = _merge_sender_summaries(parts)
    counts = senders["transaction_count"]
    positive_sender = senders["positive_transactions"].gt(0)
    result = {
        "dataset": "PaySim",
        "rows": rows,
        "positive_transactions": positives,
        "senders": int(len(senders)),
        "positive_senders": int(positive_sender.sum()),
        "transactions_per_sender": {
            "mean": float(counts.mean()),
            "median": float(counts.median()),
            "p90": float(counts.quantile(0.90)),
            "p95": float(counts.quantile(0.95)),
            "p99": float(counts.quantile(0.99)),
            "max": int(counts.max()),
        },
        "senders_ge_2": int(counts.ge(2).sum()),
        "senders_ge_3": int(counts.ge(3).sum()),
        "senders_ge_4": int(counts.ge(4).sum()),
        "positive_senders_ge_2": int((positive_sender & counts.ge(2)).sum()),
        "positive_senders_ge_3": int((positive_sender & counts.ge(3)).sum()),
        "suitable_for_sender_sequences": bool(
            counts.quantile(0.90) >= 4 and (positive_sender & counts.ge(4)).sum() >= 100
        ),
    }
    return result, senders


def audit_ibm(path: Path, chunksize: int = 400_000) -> tuple[dict, pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    rows = positives = 0
    for chunk in pd.read_csv(
        path,
        usecols=lambda column: column
        in {"Timestamp", "From Bank", "Account", "Is Laundering"},
        chunksize=chunksize,
    ):
        chunk["From Bank"] = chunk["From Bank"].astype(np.int32)
        chunk["Account"] = chunk["Account"].astype(pd.StringDtype())
        chunk["Is Laundering"] = chunk["Is Laundering"].astype(np.int8)
        chunk["Timestamp"] = pd.to_datetime(
            chunk["Timestamp"], format="%Y/%m/%d %H:%M", errors="raise"
        )
        rows += len(chunk)
        positives += int(chunk["Is Laundering"].sum())
        chunk["sender_id"] = (
            chunk["From Bank"].astype("string") + ":" + chunk["Account"]
        )
        grouped = chunk.groupby("sender_id", sort=False, observed=True)
        part = grouped.agg(
            transaction_count=("Is Laundering", "size"),
            positive_transactions=("Is Laundering", "sum"),
            first_timestamp=("Timestamp", "min"),
            last_timestamp=("Timestamp", "max"),
        ).reset_index()
        parts.append(part)

    senders = _merge_sender_summaries(parts)
    counts = senders["transaction_count"]
    positive_sender = senders["positive_transactions"].gt(0)
    eligible = counts.ge(4)
    result = {
        "dataset": "IBM AML HI-Small",
        "rows": rows,
        "positive_transactions": positives,
        "senders": int(len(senders)),
        "positive_senders": int(positive_sender.sum()),
        "transactions_per_sender": {
            "mean": float(counts.mean()),
            "median": float(counts.median()),
            "p75": float(counts.quantile(0.75)),
            "p90": float(counts.quantile(0.90)),
            "p95": float(counts.quantile(0.95)),
            "p99": float(counts.quantile(0.99)),
            "max": int(counts.max()),
        },
        "senders_ge_4": int(eligible.sum()),
        "positive_senders_ge_4": int((positive_sender & eligible).sum()),
        "positive_sender_coverage_ge_4": float(
            (positive_sender & eligible).sum() / max(1, positive_sender.sum())
        ),
        "suitable_for_sender_sequences": bool(
            (positive_sender & eligible).sum() >= 100
        ),
    }
    return result, senders


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paysim", type=Path, required=True)
    parser.add_argument("--ibm", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/audit"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paysim, paysim_senders = audit_paysim(args.paysim)
    ibm, ibm_senders = audit_ibm(args.ibm)

    (args.output_dir / "dataset_audit.json").write_text(
        json.dumps({"paysim": paysim, "ibm_aml": ibm}, indent=2),
        encoding="utf-8",
    )
    paysim_senders.to_csv(args.output_dir / "paysim_sender_summary.csv.gz", index=False)
    ibm_senders.to_csv(args.output_dir / "ibm_sender_summary.csv.gz", index=False)
    print(json.dumps({"paysim": paysim, "ibm_aml": ibm}, indent=2))


if __name__ == "__main__":
    main()
