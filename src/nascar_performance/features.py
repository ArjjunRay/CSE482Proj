"""Leakage-safe historical feature construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "StartPosition",
    "prior_races",
    "prior_avg_finish",
    "prior_avg_start",
    "prior_top10_rate",
    "recent5_avg_finish",
    "track_prior_races",
    "track_avg_finish",
]


def _ordinalize(group: pd.Series) -> pd.Series:
    """Convert order-preserving scrambled values to one-based ranks."""
    return group.rank(method="first", ascending=True).astype(float)


def load_modeling_data(data_dir: Path, normalize_scrambled: bool = False) -> pd.DataFrame:
    """Load the three project tables and create one row per driver-race."""
    results = pd.read_csv(data_dir / "race_results.csv")
    races = pd.read_csv(data_dir / "races.csv")
    drivers = pd.read_csv(data_dir / "drivers.csv")

    required_results = {"RaceID", "DriverID", "StartPosition", "FinishPosition"}
    if missing := required_results.difference(results.columns):
        raise ValueError(f"race_results.csv is missing columns: {sorted(missing)}")

    results["DateTime"] = pd.to_datetime(results.get("DateTime"), errors="coerce")
    races["DateTime"] = pd.to_datetime(races.get("DateTime"), errors="coerce")
    for column in ["RaceID", "DriverID", "StartPosition", "FinishPosition"]:
        results[column] = pd.to_numeric(results[column], errors="coerce")

    if normalize_scrambled:
        grouped = results.groupby("RaceID", sort=False)
        results["StartPosition"] = grouped["StartPosition"].transform(_ordinalize)
        results["FinishPosition"] = grouped["FinishPosition"].transform(_ordinalize)

    race_columns = [
        column for column in ["RaceID", "Name", "Track", "DateTime", "Season"] if column in races
    ]
    race_info = (
        races[race_columns]
        .drop_duplicates("RaceID")
        .rename(columns={"Name": "RaceName", "DateTime": "RaceDateTime", "Track": "RaceTrack"})
    )
    driver_columns = [
        column for column in ["DriverID", "FirstName", "LastName"] if column in drivers
    ]
    driver_info = drivers[driver_columns].drop_duplicates("DriverID")
    frame = results.merge(race_info, on="RaceID", how="left").merge(
        driver_info, on="DriverID", how="left"
    )
    frame["DateTime"] = frame.get("RaceDateTime").combine_first(frame.get("DateTime"))
    frame["Track"] = frame.get("RaceTrack").combine_first(frame.get("Track"))
    frame["DriverName"] = (
        frame.get("FirstName", pd.Series("", index=frame.index)).fillna("").astype(str)
        + " "
        + frame.get("LastName", pd.Series("", index=frame.index)).fillna("").astype(str)
    ).str.strip()
    frame["DriverName"] = frame["DriverName"].replace("", np.nan)
    frame = frame.dropna(
        subset=["RaceID", "DriverID", "DateTime", "StartPosition", "FinishPosition"]
    )
    return frame.drop_duplicates(["RaceID", "DriverID"]).reset_index(drop=True)


def build_historical_features(frame: pd.DataFrame, minimum_history: int = 5) -> pd.DataFrame:
    """Create features using only races that occurred before the target race."""
    required = {"RaceID", "DriverID", "DateTime", "Track", "StartPosition", "FinishPosition"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Input frame is missing columns: {sorted(missing)}")
    if minimum_history < 0:
        raise ValueError("minimum_history must be non-negative")

    featured = frame.copy().sort_values(["DateTime", "RaceID", "DriverID"]).reset_index(drop=True)
    driver_group = featured.groupby("DriverID", sort=False)
    featured["prior_races"] = driver_group.cumcount()
    featured["prior_avg_finish"] = driver_group["FinishPosition"].transform(
        lambda series: series.shift().expanding().mean()
    )
    featured["prior_avg_start"] = driver_group["StartPosition"].transform(
        lambda series: series.shift().expanding().mean()
    )
    featured["prior_top10_rate"] = driver_group["FinishPosition"].transform(
        lambda series: series.le(10).astype(float).shift().expanding().mean()
    )
    featured["recent5_avg_finish"] = driver_group["FinishPosition"].transform(
        lambda series: series.shift().rolling(window=5, min_periods=1).mean()
    )
    track_group = featured.groupby(["DriverID", "Track"], sort=False)
    featured["track_prior_races"] = track_group.cumcount()
    featured["track_avg_finish"] = track_group["FinishPosition"].transform(
        lambda series: series.shift().expanding().mean()
    )
    return featured[featured["prior_races"] >= minimum_history].reset_index(drop=True)
