from __future__ import annotations

import pandas as pd

from nascar_performance.features import build_historical_features


def test_historical_features_do_not_include_current_finish() -> None:
    frame = pd.DataFrame(
        {
            "RaceID": [1, 2, 3],
            "DriverID": [10, 10, 10],
            "DateTime": pd.to_datetime(["2025-01-01", "2025-01-08", "2025-01-15"]),
            "Track": ["A", "B", "A"],
            "StartPosition": [3.0, 8.0, 2.0],
            "FinishPosition": [5.0, 20.0, 1.0],
        }
    )
    result = build_historical_features(frame, minimum_history=0)

    assert pd.isna(result.loc[0, "prior_avg_finish"])
    assert result.loc[1, "prior_avg_finish"] == 5.0
    assert result.loc[2, "prior_avg_finish"] == 12.5
    assert result.loc[2, "track_avg_finish"] == 5.0
    assert result.loc[2, "recent5_avg_finish"] == 12.5


def test_minimum_history_filters_cold_start_rows() -> None:
    frame = pd.DataFrame(
        {
            "RaceID": [1, 2],
            "DriverID": [10, 10],
            "DateTime": pd.to_datetime(["2025-01-01", "2025-01-08"]),
            "Track": ["A", "A"],
            "StartPosition": [3.0, 8.0],
            "FinishPosition": [5.0, 20.0],
        }
    )
    assert build_historical_features(frame, minimum_history=1)["RaceID"].tolist() == [2]
