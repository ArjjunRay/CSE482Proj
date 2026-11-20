from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from nascar_performance.config import sportsdataio_key
from nascar_performance.data import SportsDataClient, raw_to_tables
from nascar_performance.features import load_modeling_data


def test_raw_to_tables_preserves_zero_values(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    (raw / "races_2025.json").write_text(
        json.dumps(
            [
                {
                    "RaceID": 1,
                    "SeriesID": 1,
                    "SeriesName": "Series",
                    "Season": 2025,
                    "Name": "Race",
                    "DateTime": "2025-01-01T12:00:00",
                    "Track": "Track",
                    "IsOver": True,
                }
            ]
        )
    )
    (raw / "drivers.json").write_text(
        json.dumps([{"DriverID": 10, "FirstName": "A", "LastName": "Driver"}])
    )
    (raw / "raceresultfinal_1.json").write_text(
        json.dumps(
            {
                "Race": {
                    "RaceID": 1,
                    "Season": 2025,
                    "SeriesName": "Series",
                    "DateTime": "2025-01-01T12:00:00",
                    "Track": "Track",
                },
                "DriverRaces": [
                    {
                        "DriverID": 10,
                        "StartPosition": 0,
                        "StartingPosition": 99,
                        "FinishPosition": 0,
                        "FinalPosition": 99,
                        "Laps": 0,
                        "LapsCompleted": 99,
                        "Points": 0,
                    }
                ],
            }
        )
    )

    counts = raw_to_tables(raw, processed)
    results = pd.read_csv(processed / "race_results.csv")

    assert counts == {"races": 1, "drivers": 1, "results": 1}
    assert results.loc[0, "StartPosition"] == 0
    assert results.loc[0, "FinishPosition"] == 0
    assert results.loc[0, "Laps"] == 0


def test_load_modeling_data_normalizes_scrambled_order(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "RaceID": [1, 1],
            "DriverID": [10, 20],
            "DateTime": ["2025-01-01", "2025-01-01"],
            "Track": ["Track", "Track"],
            "StartPosition": [8.2, 4.1],
            "FinishPosition": [2.0, 1.0],
        }
    ).to_csv(tmp_path / "race_results.csv", index=False)
    pd.DataFrame(
        {
            "RaceID": [1],
            "Name": ["Race"],
            "Track": ["Track"],
            "DateTime": ["2025-01-01"],
            "Season": [2025],
        }
    ).to_csv(tmp_path / "races.csv", index=False)
    pd.DataFrame(
        {
            "DriverID": [10, 20],
            "FirstName": ["A", "B"],
            "LastName": ["Driver", "Driver"],
        }
    ).to_csv(tmp_path / "drivers.csv", index=False)

    frame = load_modeling_data(tmp_path, normalize_scrambled=True)

    assert frame["StartPosition"].tolist() == [2.0, 1.0]
    assert frame["FinishPosition"].tolist() == [2.0, 1.0]
    assert frame["DriverName"].tolist() == ["A Driver", "B Driver"]


def test_missing_api_key_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPORTSDATAIO_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SPORTSDATAIO_KEY"):
        sportsdataio_key()


def test_client_rejects_empty_key() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        SportsDataClient(" ")
