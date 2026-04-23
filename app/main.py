from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import Settings
from app.schemas import PredictionRequest, PredictionResponse
from app.services import DatasetLoader, SentimentModelService, SentimentModelTrainer

dataset_loader = DatasetLoader(Settings.DATASET_PATH)
trainer = SentimentModelTrainer(Settings.MODEL_PATH)
model_service = SentimentModelService(dataset_loader, trainer)


@asynccontextmanager
async def lifespan(_: FastAPI):
    model_service.load_or_train()
    yield


app = FastAPI(
    title="Sentiment Analysis API",
    description="Minimal FastAPI MVP using a scikit-learn Logistic Regression model.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
def read_root():
    return {
        "message": "Sentiment analysis API is running.",
        "model_artifact": str(Settings.MODEL_PATH.relative_to(Settings.BASE_DIR)),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    prediction, probabilities = model_service.predict(payload.text)
    return PredictionResponse(prediction=prediction, probabilities=probabilities)
