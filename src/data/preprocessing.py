"""Features transaccionales y normalización ajustada solo con TRAIN."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CONTINUOUS_FEATURES = [
    "log_amount_paid",
    "log_amount_received",
    "log_amount_abs_difference",
    "log_time_gap_minutes",
]

PASSTHROUGH_FEATURES = [
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "same_bank",
    "same_currency",
    "destination_changed",
    "new_destination",
]

CATEGORICAL_FEATURES = ["payment_format", "payment_currency", "receiving_currency"]


def engineer_transaction_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Crea features causales sin incluir identificadores ni la etiqueta."""
    required = {
        "sender_id",
        "timestamp",
        "from_bank",
        "to_bank",
        "destination_account",
        "amount_received",
        "receiving_currency",
        "amount_paid",
        "payment_currency",
        "payment_format",
    }
    missing = required.difference(transactions.columns)
    if missing:
        raise ValueError(f"Faltan columnas transaccionales: {sorted(missing)}")

    frame = transactions.copy()
    timestamp = pd.to_datetime(frame["timestamp"], errors="raise")
    amount_paid = frame["amount_paid"].astype("float64").clip(lower=0)
    amount_received = frame["amount_received"].astype("float64").clip(lower=0)
    destination_id = (
        frame["to_bank"].astype("string") + ":" + frame["destination_account"].astype("string")
    )
    grouped_time = timestamp.groupby(frame["sender_id"], sort=False)
    time_gap = grouped_time.diff().dt.total_seconds().div(60).fillna(0).clip(lower=0)
    previous_destination = destination_id.groupby(frame["sender_id"], sort=False).shift()
    first_destination = ~pd.DataFrame(
        {"sender_id": frame["sender_id"], "destination_id": destination_id}
    ).duplicated()

    engineered = pd.DataFrame(index=frame.index)
    engineered["log_amount_paid"] = np.log1p(amount_paid)
    engineered["log_amount_received"] = np.log1p(amount_received)
    engineered["log_amount_abs_difference"] = np.log1p(
        (amount_paid - amount_received).abs()
    )
    engineered["log_time_gap_minutes"] = np.log1p(time_gap)
    hour = timestamp.dt.hour + timestamp.dt.minute / 60.0
    weekday = timestamp.dt.dayofweek.astype("float64")
    engineered["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    engineered["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    engineered["weekday_sin"] = np.sin(2 * np.pi * weekday / 7.0)
    engineered["weekday_cos"] = np.cos(2 * np.pi * weekday / 7.0)
    engineered["same_bank"] = frame["from_bank"].eq(frame["to_bank"]).astype("float32")
    engineered["same_currency"] = (
        frame["payment_currency"].eq(frame["receiving_currency"]).astype("float32")
    )
    engineered["destination_changed"] = (
        previous_destination.notna() & destination_id.ne(previous_destination)
    ).astype("float32")
    engineered["new_destination"] = first_destination.astype("float32")
    for column in CATEGORICAL_FEATURES:
        engineered[column] = frame[column].fillna("UNKNOWN").astype("string")
    return engineered


@dataclass
class FeaturePreprocessor:
    """Transformador serializable con ajuste exclusivo sobre TRAIN."""

    scaler: StandardScaler = field(default_factory=StandardScaler)
    encoder: OneHotEncoder = field(
        default_factory=lambda: OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, dtype=np.float32
        )
    )
    fitted: bool = False

    def fit(self, features: pd.DataFrame) -> "FeaturePreprocessor":
        self.scaler.fit(features[CONTINUOUS_FEATURES])
        self.encoder.fit(features[CATEGORICAL_FEATURES])
        self.fitted = True
        return self

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("El preprocesador debe ajustarse antes de transformar")
        continuous = self.scaler.transform(features[CONTINUOUS_FEATURES]).astype(np.float32)
        passthrough = features[PASSTHROUGH_FEATURES].to_numpy(dtype=np.float32)
        categorical = self.encoder.transform(features[CATEGORICAL_FEATURES]).astype(np.float32)
        matrix = np.concatenate([continuous, passthrough, categorical], axis=1)
        if not np.isfinite(matrix).all():
            raise ValueError("Las features contienen NaN o Inf después de transformar")
        return matrix

    def get_feature_names(self) -> list[str]:
        if not self.fitted:
            raise RuntimeError("El preprocesador debe ajustarse antes de listar features")
        categorical_names = self.encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
        return CONTINUOUS_FEATURES + PASSTHROUGH_FEATURES + categorical_names
