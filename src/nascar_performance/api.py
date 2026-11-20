"""Versioned FastAPI inference service."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from nascar_performance import __version__
from nascar_performance.config import model_path
from nascar_performance.features import FEATURE_COLUMNS
from nascar_performance.modeling import load_artifact

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
LOGGER = logging.getLogger(__name__)


class Entrant(BaseModel):
    driver_id: str = Field(min_length=1, max_length=100)
    driver_name: str = Field(min_length=1, max_length=120)
    start_position: Annotated[float, Field(gt=0, le=100)]
    prior_races: Annotated[int, Field(ge=0)]
    prior_avg_finish: Annotated[float | None, Field(gt=0, le=100)] = None
    prior_avg_start: Annotated[float | None, Field(gt=0, le=100)] = None
    prior_top10_rate: Annotated[float | None, Field(ge=0, le=1)] = None
    recent5_avg_finish: Annotated[float | None, Field(gt=0, le=100)] = None
    track_prior_races: Annotated[int, Field(ge=0)] = 0
    track_avg_finish: Annotated[float | None, Field(gt=0, le=100)] = None


class PredictionRequest(BaseModel):
    race_name: str | None = Field(default=None, max_length=200)
    entrants: list[Entrant] = Field(min_length=2, max_length=100)


class Prediction(BaseModel):
    driver_id: str
    driver_name: str
    projected_finish_position: int
    model_score: float


class PredictionResponse(BaseModel):
    race_name: str | None
    generated_at: datetime
    model_version: str
    predictions: list[Prediction]
    disclaimer: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = model_path()
    try:
        app.state.artifact = load_artifact(path)
        LOGGER.info("Loaded model artifact from %s", path)
    except (FileNotFoundError, ValueError, OSError) as exc:
        app.state.artifact = None
        LOGGER.warning("API started without a model artifact: %s", exc)
    yield


app = FastAPI(
    title="NASCAR Performance Lab API",
    version=__version__,
    description="Experimental, leakage-aware NASCAR relative finish ranking.",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home() -> str:
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>NASCAR Performance Lab</title><style>
body{font-family:system-ui,sans-serif;max-width:760px;margin:10vh auto;padding:0 24px;
line-height:1.6;color:#171717}h1{line-height:1.1}a{color:#b21f2d}
code{background:#f2f2f2;padding:.15rem .35rem;border-radius:4px}
</style></head><body><h1>NASCAR Performance Lab</h1>
<p>A versioned API for experimental, leakage-aware race ranking.</p>
<p><a href="/docs">Interactive API documentation</a> · <a href="/healthz">Health check</a></p>
<p><strong>Scope:</strong> research and demonstration only.
Do not use these estimates for betting.</p>
</body></html>"""


@app.get("/healthz")
def health(request: Request) -> dict[str, str | bool]:
    return {
        "status": "ok",
        "version": __version__,
        "model_loaded": request.app.state.artifact is not None,
    }


@app.get("/v1/model")
def model_info(request: Request) -> dict:
    if request.app.state.artifact is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model unavailable"
        )
    return request.app.state.artifact["metadata"]


@app.post("/v1/predictions/rank", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    artifact = request.app.state.artifact
    if artifact is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model unavailable"
        )
    rows = []
    for entrant in payload.entrants:
        row = entrant.model_dump()
        rows.append(
            {
                "StartPosition": row["start_position"],
                "prior_races": row["prior_races"],
                "prior_avg_finish": row["prior_avg_finish"],
                "prior_avg_start": row["prior_avg_start"],
                "prior_top10_rate": row["prior_top10_rate"],
                "recent5_avg_finish": row["recent5_avg_finish"],
                "track_prior_races": row["track_prior_races"],
                "track_avg_finish": row["track_avg_finish"],
            }
        )
    scores = artifact["model"].predict(pd.DataFrame(rows)[FEATURE_COLUMNS])
    order = sorted(range(len(scores)), key=lambda index: (scores[index], index))
    predictions = [
        Prediction(
            driver_id=payload.entrants[index].driver_id,
            driver_name=payload.entrants[index].driver_name,
            projected_finish_position=rank,
            model_score=round(max(1.0, float(scores[index])), 3),
        )
        for rank, index in enumerate(order, start=1)
    ]
    return PredictionResponse(
        race_name=payload.race_name,
        generated_at=datetime.now(UTC),
        model_version=artifact["metadata"]["generated_at"],
        predictions=predictions,
        disclaimer=(
            "Experimental relative ranking; not validated for betting or financial decisions."
        ),
    )
