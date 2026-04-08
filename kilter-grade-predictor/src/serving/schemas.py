"""Pydantic request/response models for the prediction API."""

from pydantic import BaseModel, Field, field_validator


class Hold(BaseModel):
    """A single hold on the Kilter Board."""

    x: int = Field(..., ge=0, le=144, description="X coordinate on board grid")
    y: int = Field(..., ge=0, le=156, description="Y coordinate on board grid")
    role: str = Field(..., description="Hold role: start, middle, finish, foot")

    @field_validator("role")
    @classmethod
    def role_must_be_valid(cls, v: str) -> str:
        allowed = {"start", "middle", "finish", "foot"}
        if v not in allowed:
            msg = f"role must be one of {sorted(allowed)}"
            raise ValueError(msg)
        return v


class PredictRequest(BaseModel):
    """Request body for POST /predict."""

    holds: list[Hold] = Field(..., min_length=2, description="Route holds")
    angle: int = Field(..., ge=0, le=70, description="Board angle in degrees")


class PredictResponse(BaseModel):
    """Response body for POST /predict."""

    predicted_grade: float
    v_grade: str


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str
    model_loaded: bool
