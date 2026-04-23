from pathlib import Path


class Settings:
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATASET_PATH = BASE_DIR / "data" / "data.csv"
    ARTIFACTS_DIR = BASE_DIR / "artifacts"
    MODEL_PATH = ARTIFACTS_DIR / "sentiment_pipeline.joblib"
    RANDOM_STATE = 42
