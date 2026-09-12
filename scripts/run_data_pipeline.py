"""CLI reproducible del pipeline de ingeniería de datos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.pipeline import PipelineConfig, run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ibm", type=Path, required=True)
    parser.add_argument("--sender-summary", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--figures-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--min-transactions", type=int, default=3)
    parser.add_argument("--normal-ratio", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = PipelineConfig(
        sequence_length=args.sequence_length,
        min_transactions=args.min_transactions,
        normal_to_positive_ratio=args.normal_ratio,
        random_state=args.seed,
    )
    metadata = run_pipeline(
        ibm_csv=args.ibm,
        sender_summary_csv=args.sender_summary,
        audit_json=args.audit,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
        config=config,
    )
    print(json.dumps(metadata["splits"], indent=2))


if __name__ == "__main__":
    main()
