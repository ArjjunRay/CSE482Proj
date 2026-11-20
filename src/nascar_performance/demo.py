"""Deterministic synthetic data for installation and deployment smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_demo_data(
    output_dir: Path, race_count: int = 30, driver_count: int = 18
) -> dict[str, int]:
    """Create realistic-shaped, explicitly synthetic CSV input tables."""
    if race_count < 10 or driver_count < 8:
        raise ValueError("Demo data needs at least 10 races and 8 drivers")
    rng = np.random.default_rng(42)
    tracks = ["Demo Speedway", "Example Raceway", "Synthetic Motor Park"]
    driver_ids = np.arange(1, driver_count + 1)
    skill = rng.normal(0, 3, driver_count)
    drivers = pd.DataFrame(
        {
            "DriverID": driver_ids,
            "FirstName": ["Demo"] * driver_count,
            "LastName": [f"Driver {index:02d}" for index in driver_ids],
            "Number": driver_ids,
            "Team": [f"Example Team {(index - 1) // 3 + 1}" for index in driver_ids],
            "Manufacturer": ["Synthetic"] * driver_count,
        }
    )

    races = []
    results = []
    start_date = pd.Timestamp("2025-01-05 14:00:00")
    for offset in range(race_count):
        race_id = 10_000 + offset
        race_date = start_date + pd.to_timedelta(7 * offset, unit="D")
        track = tracks[offset % len(tracks)]
        start_order = rng.permutation(driver_count) + 1
        noise = rng.normal(0, 4, driver_count)
        latent_finish = (0.38 * start_order) + skill + noise
        finish_order = pd.Series(latent_finish).rank(method="first").astype(int).to_numpy()
        races.append(
            {
                "RaceID": race_id,
                "SeriesID": 1,
                "SeriesName": "Synthetic Cup",
                "Season": race_date.year,
                "Name": f"Demonstration Race {offset + 1}",
                "Day": race_date.date().isoformat(),
                "DateTime": race_date.isoformat(),
                "Track": track,
                "IsInProgress": False,
                "IsOver": True,
            }
        )
        for index, driver_id in enumerate(driver_ids):
            finish = int(finish_order[index])
            results.append(
                {
                    "RaceID": race_id,
                    "Season": race_date.year,
                    "SeriesName": "Synthetic Cup",
                    "DateTime": race_date.isoformat(),
                    "Track": track,
                    "DriverID": driver_id,
                    "StartPosition": int(start_order[index]),
                    "FinishPosition": finish,
                    "Laps": 200,
                    "Points": max(0, driver_count + 1 - finish),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    drivers.to_csv(output_dir / "drivers.csv", index=False)
    pd.DataFrame(races).to_csv(output_dir / "races.csv", index=False)
    pd.DataFrame(results).to_csv(output_dir / "race_results.csv", index=False)
    return {"races": race_count, "drivers": driver_count, "results": len(results)}
