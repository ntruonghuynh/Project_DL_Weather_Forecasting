"""FastAPI application for an already-trained, versioned ModelBundle."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover - lets compileall work before optional install
    FastAPI = None  # type: ignore[assignment,misc]
    HTTPException = None  # type: ignore[assignment,misc]
    BaseModel = object  # type: ignore[assignment,misc]
    Field = None  # type: ignore[assignment,misc]

from src.serving.bundle import load_bundle
from src.serving.predictor import Predictor

if BaseModel is not object:

    class PredictRequest(BaseModel):
        """Exactly one 168-hour observation sequence."""

        timestamps: list[str]
        observations: list[dict[str, float]]


    class PredictionItem(BaseModel):
        timestamp: str
        y_true: float | None
        y_pred: float
        model_name: str
        model_version: str
        split: str
        run_id: str
        horizon: int
        unit: str


    class PredictResponse(BaseModel):
        predictions: list[PredictionItem]
        persistence_baseline: list[float]
        attention_weights: list[list[float]] | None
        latency_ms: float = Field(ge=0)
        model: dict[str, Any]


def create_app(predictor: Predictor | None = None, bundle_path: Path | None = None) -> object:
    """Create an API shell; optionally load one bundle exactly once at startup."""
    if FastAPI is None:
        raise RuntimeError("Install API dependencies before creating the FastAPI app")
    if predictor is not None and bundle_path is not None:
        raise ValueError("provide predictor or bundle_path, not both")
    initial_predictor = predictor

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.predictor = initial_predictor
        application.state.startup_error = None
        if application.state.predictor is None and bundle_path is not None:
            try:
                application.state.predictor = Predictor(load_bundle(bundle_path))
            except Exception as error:  # diagnostic health state; never train/fallback
                application.state.startup_error = str(error)
        yield

    application = FastAPI(
        title="Jena Weather Forecasting API",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.state.predictor = initial_predictor
    application.state.startup_error = None

    @application.get("/health")
    def health() -> dict[str, str]:
        if application.state.predictor is not None:
            return {"status": "ready"}
        return {
            "status": "not_ready",
            "detail": application.state.startup_error or "no model bundle configured",
        }

    @application.get("/model-info")
    def model_info() -> dict[str, object]:
        active = application.state.predictor
        if active is None:
            raise HTTPException(status_code=503, detail="model bundle is not ready")
        return active.model_info()

    @application.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        active = application.state.predictor
        if active is None:
            raise HTTPException(status_code=503, detail="model bundle is not ready")
        started = time.perf_counter()
        try:
            records = active.predict(request.observations, request.timestamps)
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        latency_ms = (time.perf_counter() - started) * 1000
        return PredictResponse(
            predictions=[PredictionItem(**asdict(record)) for record in records],
            persistence_baseline=active.persistence_baseline(request.observations),
            attention_weights=active.last_attention_weights(),
            latency_ms=latency_ms,
            model=active.model_info(),
        )

    return application


_configured_bundle = os.getenv("JENA_MODEL_BUNDLE")
app = (
    create_app(bundle_path=Path(_configured_bundle) if _configured_bundle else None)
    if FastAPI is not None
    else None
)
