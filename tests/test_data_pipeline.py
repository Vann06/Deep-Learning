from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import load_split, save_split
from src.data.preprocessing import FeaturePreprocessor, engineer_transaction_features
from src.data.sequences import choose_windows, pack_split
from src.data.splits import assert_disjoint_senders, split_and_sample_senders
from scripts.audit_datasets import audit_ibm, audit_paysim


def synthetic_transactions() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=12, freq="2h")
    return pd.DataFrame(
        {
            "sender_id": ["1:A"] * 5 + ["2:B"] * 4 + ["3:C"] * 3,
            "timestamp": timestamps,
            "from_bank": [1] * 5 + [2] * 4 + [3] * 3,
            "to_bank": [9] * 12,
            "destination_account": ["X", "Y", "Y", "Z", "X"] + ["K"] * 4 + ["M", "N", "M"],
            "amount_received": np.arange(1, 13, dtype=float),
            "receiving_currency": ["USD"] * 12,
            "amount_paid": np.arange(1, 13, dtype=float),
            "payment_currency": ["USD"] * 12,
            "payment_format": ["Wire"] * 12,
            "is_laundering": [0, 0, 1, 0, 0] + [0] * 7,
            "source_row_id": np.arange(12),
        }
    )


class DataPipelineTests(unittest.TestCase):
    def test_chunked_audits_accept_expected_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            paysim_path = directory_path / "paysim.csv"
            ibm_path = directory_path / "ibm.csv"
            pd.DataFrame(
                {
                    "nameOrig": ["C1", "C1", "C2"],
                    "step": [1, 2, 2],
                    "isFraud": [0, 1, 0],
                }
            ).to_csv(paysim_path, index=False)
            pd.DataFrame(
                {
                    "Timestamp": ["2022/09/01 00:00", "2022/09/01 01:00", "2022/09/01 02:00"],
                    "From Bank": [1, 1, 2],
                    "Account": ["A", "A", "B"],
                    "Is Laundering": [0, 1, 0],
                }
            ).to_csv(ibm_path, index=False)

            paysim_result, _ = audit_paysim(paysim_path, chunksize=2)
            ibm_result, _ = audit_ibm(ibm_path, chunksize=2)

        self.assertEqual(paysim_result["rows"], 3)
        self.assertEqual(paysim_result["positive_transactions"], 1)
        self.assertEqual(ibm_result["rows"], 3)
        self.assertEqual(ibm_result["positive_transactions"], 1)

    def test_positive_window_keeps_positive_anchor(self) -> None:
        frame = synthetic_transactions()
        windows = choose_windows(frame, sequence_length=3, min_transactions=3)
        labels = frame["is_laundering"].to_numpy()
        self.assertEqual(labels[windows["1:A"]].max(), 1)
        self.assertEqual(len(windows["1:A"]), 3)

    def test_features_pack_and_round_trip(self) -> None:
        frame = synthetic_transactions()
        engineered = engineer_transaction_features(frame)
        preprocessor = FeaturePreprocessor().fit(engineered.iloc[:9])
        matrix = preprocessor.transform(engineered)
        windows = choose_windows(frame, sequence_length=4, min_transactions=3)
        arrays = pack_split(
            frame, matrix, windows, np.array(["1:A", "3:C"]), sequence_length=4
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.npz"
            save_split(path, arrays)
            loaded = load_split(path)
        self.assertTrue(np.array_equal(loaded["y"], np.array([1, 0], dtype=np.int8)))
        self.assertTrue(np.isfinite(loaded["X"]).all())
        self.assertTrue(np.all(loaded["X"][~loaded["mask"]] == 0))

    def test_group_split_has_no_leakage(self) -> None:
        rows = [
            {
                "sender_id": f"S{index:03d}",
                "transaction_count": 3 + index % 8,
                "positive_transactions": int(index % 10 == 0),
            }
            for index in range(120)
        ]
        selected, _ = split_and_sample_senders(
            pd.DataFrame(rows),
            min_transactions=3,
            normal_to_positive_ratio=5,
            random_state=7,
        )
        mapping = {
            name: selected.loc[selected["split"].eq(name), "sender_id"].to_numpy()
            for name in ("train", "val", "test")
        }
        assert_disjoint_senders(mapping)


if __name__ == "__main__":
    unittest.main()
