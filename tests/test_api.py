from __future__ import annotations

from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyRegressor

from nascar_performance.features import FEATURE_COLUMNS


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model = DummyRegressor(strategy="mean").fit([[1.0] * len(FEATURE_COLUMNS)], [5.0])
    artifact = {
        "model": model,
        "metadata": {"generated_at": "2026-11-20T00:00:00+00:00"},
    }
    artifact_path = tmp_path / "model.joblib"
    joblib.dump(artifact, artifact_path)
    monkeypatch.setenv("MODEL_PATH", str(artifact_path))
    from nascar_performance.api import app

    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_loaded_model(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_rank_endpoint_orders_all_entrants(client: TestClient) -> None:
    entrant = {
        "driver_id": "1",
        "driver_name": "Example Driver",
        "start_position": 1,
        "prior_races": 10,
        "prior_avg_finish": 8,
        "prior_avg_start": 9,
        "prior_top10_rate": 0.6,
        "recent5_avg_finish": 7,
        "track_prior_races": 2,
        "track_avg_finish": 6,
    }
    response = client.post(
        "/v1/predictions/rank",
        json={"entrants": [entrant, {**entrant, "driver_id": "2", "driver_name": "Other Driver"}]},
    )
    assert response.status_code == 200
    positions = [item["projected_finish_position"] for item in response.json()["predictions"]]
    assert positions == [1, 2]
