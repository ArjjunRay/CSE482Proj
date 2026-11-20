from __future__ import annotations

import numpy as np
import pandas as pd

from nascar_performance.features import FEATURE_COLUMNS
from nascar_performance.modeling import chronological_split, train_model


def make_training_frame() -> pd.DataFrame:
    rows = []
    for race_id in range(1, 11):
        for driver in range(1, 7):
            start = float(driver)
            rows.append(
                {
                    "RaceID": race_id,
                    "DateTime": pd.Timestamp("2025-01-01") + pd.to_timedelta(race_id, unit="D"),
                    "FinishPosition": start + ((race_id + driver) % 3 - 1) * 0.25,
                    "StartPosition": start,
                    "prior_races": race_id - 1,
                    "prior_avg_finish": start + 0.5,
                    "prior_avg_start": start,
                    "prior_top10_rate": 1.0,
                    "recent5_avg_finish": start + 0.25,
                    "track_prior_races": max(0, race_id // 3),
                    "track_avg_finish": start + 0.75 if race_id > 2 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def test_chronological_split_keeps_later_races_in_test() -> None:
    train, test = chronological_split(make_training_frame())
    assert train["RaceID"].max() < test["RaceID"].min()
    assert set(train["RaceID"]).isdisjoint(test["RaceID"])


def test_train_model_returns_versioned_artifact() -> None:
    artifact, metadata = train_model(make_training_frame())
    assert metadata["selected_model"] in {"ridge", "random_forest"}
    assert metadata["feature_columns"] == FEATURE_COLUMNS
    assert metadata["training_races"] == 10
    predictions = artifact["model"].predict(make_training_frame()[FEATURE_COLUMNS].head(2))
    assert len(predictions) == 2
