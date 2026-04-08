"""FastAPI application for Kilter Board grade prediction."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.models.predict import DEFAULT_DB_PATH, DEFAULT_MODEL_PATH, load_model, predict_grade
from src.serving.schemas import HealthResponse, PredictRequest, PredictResponse


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load model and hold usability scores once at startup."""
    from src.data.ingest import load_climbs, load_placements
    from src.features.hold_usability import compute_hold_usability

    app.state.model = load_model(DEFAULT_MODEL_PATH)

    climbs = load_climbs(DEFAULT_DB_PATH, min_ascents=5)
    placements = load_placements(DEFAULT_DB_PATH)
    app.state.hold_scores = compute_hold_usability(climbs, placements)
    yield


def create_app(use_lifespan: bool = True) -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="Kilter Grade Predictor",
        version="0.1.0",
        lifespan=lifespan if use_lifespan else None,
    )

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        model_loaded = hasattr(application.state, "model") and application.state.model is not None
        return HealthResponse(status="ok", model_loaded=model_loaded)

    @application.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        try:
            result = predict_grade(
                holds=[h.model_dump() for h in req.holds],
                angle=req.angle,
                model=application.state.model,
                hold_scores=application.state.hold_scores,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        return PredictResponse(**result)

    return application


app = create_app()
