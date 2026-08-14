"""FastAPI microservice for Budapest apartment price estimation.

Artifacts (``model.joblib``, ``columns.json``, ``metrics.json``) are loaded
once in the lifespan handler — never per request, never under ``__main__``.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACTS_DIR = _REPO_ROOT / "artifacts"
_STATIC_DIR = _REPO_ROOT / "static"


class PredictRequest(BaseModel):
    location: str
    sqm: float = Field(gt=0)
    rooms: float = Field(ge=0)


class PredictResponse(BaseModel):
    estimated_price_huf: int
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    columns = json.loads((_ARTIFACTS_DIR / "columns.json").read_text(encoding="utf-8"))
    metrics = json.loads((_ARTIFACTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    app.state.columns = columns
    app.state.model = joblib.load(_ARTIFACTS_DIR / "model.joblib")
    app.state.model_version = metrics["date"]
    yield


app = FastAPI(title="Ingatlan Budapest Price API", lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/locations")
def locations() -> list[str]:
    """Canonical district labels (roman numerals) known to the model."""
    return app.state.columns[2:]


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    columns = app.state.columns
    x = np.zeros(len(columns))
    x[0] = req.sqm
    x[1] = req.rooms
    location = req.location.strip()
    if location in columns[2:]:
        x[columns.index(location)] = 1.0
    else:
        # Unknown locations fall back to the "other" column (always trained).
        x[columns.index("other")] = 1.0
    price = float(app.state.model.predict(pd.DataFrame([x], columns=columns))[0])
    return PredictResponse(
        estimated_price_huf=int(round(price)),
        model_version=app.state.model_version,
    )


app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
