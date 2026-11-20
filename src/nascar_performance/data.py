"""Data collection and table preparation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://api.sportsdata.io/nascar/v2/json"


class SportsDataClient:
    """Small, retrying client for the SportsDataIO NASCAR API."""

    def __init__(self, api_key: str, timeout: float = 30.0, delay: float = 0.25) -> None:
        if not api_key.strip():
            raise ValueError("api_key must not be empty")
        self.timeout = timeout
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({"Ocp-Apim-Subscription-Key": api_key})
        retries = Retry(
            total=4,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def get(self, endpoint: str) -> Any:
        response = self.session.get(f"{BASE_URL}/{endpoint.lstrip('/')}", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _write_json(value: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        temporary.replace(path)

    def collect_season(self, season: int, output_dir: Path) -> dict[str, int]:
        """Collect a season and return row/failure counts."""
        races = self.get(f"races/{season}")
        self._write_json(races, output_dir / f"races_{season}.json")
        drivers = self.get("Drivers")
        self._write_json(drivers, output_dir / "drivers.json")

        completed = [race for race in races if race.get("IsOver") and race.get("RaceID")]
        failures = 0
        for race in completed:
            race_id = int(race["RaceID"])
            try:
                result = self.get(f"raceresultfinal/{race_id}")
                self._write_json(result, output_dir / f"raceresultfinal_{race_id}.json")
            except requests.RequestException:
                failures += 1
            time.sleep(self.delay)
        return {"races": len(completed), "drivers": len(drivers), "failures": failures}


def _prefer(record: dict[str, Any], primary: str, fallback: str) -> Any:
    """Prefer a present value without treating numeric zero as missing."""
    value = record.get(primary)
    return record.get(fallback) if value is None else value


def raw_to_tables(raw_dir: Path, output_dir: Path) -> dict[str, int]:
    """Convert collected JSON files into stable CSV tables."""
    race_files = sorted(raw_dir.glob("races_*.json"))
    if not race_files:
        raise FileNotFoundError(f"No races_*.json files found in {raw_dir}")

    races: list[dict[str, Any]] = []
    for path in race_files:
        races.extend(json.loads(path.read_text(encoding="utf-8")))
    races_df = pd.DataFrame(races)
    race_columns = [
        "RaceID",
        "SeriesID",
        "SeriesName",
        "Season",
        "Name",
        "Day",
        "DateTime",
        "Track",
        "IsInProgress",
        "IsOver",
    ]
    races_df = races_df[[column for column in race_columns if column in races_df]]
    races_df = races_df.dropna(subset=["RaceID"]).drop_duplicates("RaceID")
    if "IsOver" in races_df:
        races_df = races_df[races_df["IsOver"].eq(True)]

    driver_path = raw_dir / "drivers.json"
    if not driver_path.exists():
        raise FileNotFoundError(f"Missing {driver_path}")
    drivers_df = pd.DataFrame(json.loads(driver_path.read_text(encoding="utf-8")))
    driver_columns = ["DriverID", "FirstName", "LastName", "Number", "Team", "Manufacturer"]
    drivers_df = drivers_df[[column for column in driver_columns if column in drivers_df]]
    drivers_df = drivers_df.dropna(subset=["DriverID"]).drop_duplicates("DriverID")

    rows: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("raceresultfinal_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        race = payload.get("Race", {})
        for result in payload.get("DriverRaces", []):
            rows.append(
                {
                    "RaceID": race.get("RaceID"),
                    "Season": race.get("Season"),
                    "SeriesName": race.get("SeriesName"),
                    "DateTime": race.get("DateTime"),
                    "Track": race.get("Track"),
                    "DriverID": result.get("DriverID"),
                    "StartPosition": _prefer(result, "StartPosition", "StartingPosition"),
                    "FinishPosition": _prefer(result, "FinishPosition", "FinalPosition"),
                    "Laps": _prefer(result, "Laps", "LapsCompleted"),
                    "Points": result.get("Points"),
                }
            )
    results_df = pd.DataFrame(rows)
    required = ["RaceID", "DriverID", "StartPosition", "FinishPosition"]
    if results_df.empty or any(column not in results_df for column in required):
        raise ValueError("No usable race results were found")
    for column in ["RaceID", "DriverID", "StartPosition", "FinishPosition", "Laps", "Points"]:
        results_df[column] = pd.to_numeric(results_df[column], errors="coerce")
    results_df = results_df.dropna(subset=required).drop_duplicates(["RaceID", "DriverID"])

    output_dir.mkdir(parents=True, exist_ok=True)
    races_df.to_csv(output_dir / "races.csv", index=False)
    drivers_df.to_csv(output_dir / "drivers.csv", index=False)
    results_df.to_csv(output_dir / "race_results.csv", index=False)
    return {"races": len(races_df), "drivers": len(drivers_df), "results": len(results_df)}
