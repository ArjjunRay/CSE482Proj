"""Application configuration."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_MODEL_PATH = Path("models/nascar_rank_model.joblib")


def model_path() -> Path:
    """Return the configured model artifact path."""
    return Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))


def sportsdataio_key() -> str:
    """Return the API key without providing an insecure fallback."""
    key = os.getenv("SPORTSDATAIO_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "SPORTSDATAIO_KEY is not set. Copy .env.example and export the key before "
            "collecting fresh data."
        )
    return key
