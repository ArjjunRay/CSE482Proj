"""Chronological model training and artifact management."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nascar_performance.features import FEATURE_COLUMNS


def chronological_split(
    frame: pd.DataFrame, test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split entire races chronologically so a race never crosses partitions."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one")
    races = (
        frame[["RaceID", "DateTime"]]
        .drop_duplicates("RaceID")
        .sort_values(["DateTime", "RaceID"])
        .reset_index(drop=True)
    )
    if len(races) < 5:
        raise ValueError("At least five races are required for chronological evaluation")
    split_index = min(len(races) - 1, max(1, int(len(races) * (1 - test_fraction))))
    train_ids = set(races.iloc[:split_index]["RaceID"])
    test_ids = set(races.iloc[split_index:]["RaceID"])
    return frame[frame["RaceID"].isin(train_ids)].copy(), frame[
        frame["RaceID"].isin(test_ids)
    ].copy()


def _pairwise_accuracy(actual: pd.Series, predicted: np.ndarray, race_ids: pd.Series) -> float:
    correct = 0
    total = 0
    evaluation = pd.DataFrame(
        {"actual": actual.to_numpy(), "predicted": predicted, "race": race_ids.to_numpy()}
    )
    for _, group in evaluation.groupby("race"):
        actual_values = group["actual"].to_numpy()
        predicted_values = group["predicted"].to_numpy()
        actual_delta = actual_values[:, None] - actual_values[None, :]
        predicted_delta = predicted_values[:, None] - predicted_values[None, :]
        upper = np.triu(np.ones_like(actual_delta, dtype=bool), k=1)
        comparable = upper & (actual_delta != 0) & (predicted_delta != 0)
        correct += int(
            np.sum(np.sign(actual_delta[comparable]) == np.sign(predicted_delta[comparable]))
        )
        total += int(np.sum(comparable))
    return correct / total if total else 0.0


def _candidates() -> dict[str, Pipeline | TransformedTargetRegressor]:
    ridge = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]
    )
    forest = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=10,
                    min_samples_leaf=4,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )
    return {"ridge": ridge, "random_forest": forest}


def train_model(frame: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate candidates chronologically, then fit the winner on all rows."""
    if frame.empty:
        raise ValueError("No training rows were provided")
    train, test = chronological_split(frame)
    evaluations: dict[str, dict[str, float]] = {}
    fitted_candidates: dict[str, Any] = {}
    for name, candidate in _candidates().items():
        model = clone(candidate)
        model.fit(train[FEATURE_COLUMNS], train["FinishPosition"])
        predicted = model.predict(test[FEATURE_COLUMNS])
        evaluations[name] = {
            "mae": float(mean_absolute_error(test["FinishPosition"], predicted)),
            "rmse": float(mean_squared_error(test["FinishPosition"], predicted) ** 0.5),
            "pairwise_accuracy": float(
                _pairwise_accuracy(test["FinishPosition"], predicted, test["RaceID"])
            ),
        }
        fitted_candidates[name] = model

    winner = min(evaluations, key=lambda name: evaluations[name]["mae"])
    final_model = clone(_candidates()[winner])
    final_model.fit(frame[FEATURE_COLUMNS], frame["FinishPosition"])
    generated_at = datetime.now(UTC).isoformat()
    metadata = {
        "artifact_version": 1,
        "generated_at": generated_at,
        "selected_model": winner,
        "feature_columns": FEATURE_COLUMNS,
        "training_rows": len(frame),
        "training_races": int(frame["RaceID"].nunique()),
        "training_start": pd.Timestamp(frame["DateTime"].min()).isoformat(),
        "training_end": pd.Timestamp(frame["DateTime"].max()).isoformat(),
        "holdout_races": int(test["RaceID"].nunique()),
        "evaluation": evaluations,
        "limitations": [
            "Predictions are relative ranking estimates, not probabilities.",
            "Evaluation validity depends on the provenance and integrity of the input data.",
            "The model has not been validated for betting or financial decisions.",
        ],
    }
    return {"model": final_model, "metadata": metadata}, metadata


def save_artifact(
    artifact: dict[str, Any], metadata: dict[str, Any], model_path: Path, metrics_path: Path
) -> None:
    """Persist a model atomically and metadata as reviewable JSON."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = model_path.with_suffix(model_path.suffix + ".tmp")
    joblib.dump(artifact, temporary)
    temporary.replace(model_path)
    metrics_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def load_artifact(path: Path) -> dict[str, Any]:
    """Load and validate the minimum artifact contract."""
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or not {"model", "metadata"}.issubset(artifact):
        raise ValueError(f"Invalid model artifact: {path}")
    return artifact
