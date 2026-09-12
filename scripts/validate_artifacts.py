"""Validación independiente de los artefactos entregados a Personas 2 y 3."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import load_split
from src.data.splits import assert_disjoint_senders


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    splits = {
        name: load_split(args.artifacts / f"{name}.npz", validate=True)
        for name in ("train", "val", "test")
    }
    assert_disjoint_senders({name: data["sender_id"] for name, data in splits.items()})
    metadata = json.loads((args.artifacts / "metadata.json").read_text(encoding="utf-8"))
    feature_names = json.loads(
        (args.artifacts / "feature_names.json").read_text(encoding="utf-8")
    )
    with (args.artifacts / "scaler.pkl").open("rb") as stream:
        preprocessor = pickle.load(stream)
    if len(feature_names) != metadata["num_features"]:
        raise AssertionError("feature_names.json y metadata.json no coinciden")
    if preprocessor.get_feature_names() != feature_names:
        raise AssertionError("El preprocesador recargado no coincide con feature_names.json")
    for name, arrays in splits.items():
        if list(arrays["X"].shape) != metadata["splits"][name]["shape_X"]:
            raise AssertionError(f"Shape de {name} no coincide con metadata")
    print("OK: shapes, labels, NaN/Inf, padding, round-trip y leakage validados")
    print(json.dumps(metadata["splits"], indent=2))


if __name__ == "__main__":
    main()
