from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to classify")


class PredictionResponse(BaseModel):
    prediction: str
    probabilities: dict[str, float]
