from __future__ import annotations

from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.config import Settings


@dataclass
class TrainingDataset:
    texts: list[str]
    labels: list[str]


class DatasetLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def load(self) -> TrainingDataset:
        dataframe = pd.read_csv(self.dataset_path)
        dataframe = dataframe[["Sentence", "Sentiment"]].dropna()
        dataframe["Sentence"] = dataframe["Sentence"].astype(str)
        dataframe["Sentiment"] = dataframe["Sentiment"].astype(str)
        return TrainingDataset(
            texts=dataframe["Sentence"].tolist(),
            labels=dataframe["Sentiment"].tolist(),
        )


class SentimentModelTrainer:
    def __init__(self, model_path):
        self.model_path = model_path

    def train(self, dataset: TrainingDataset) -> Pipeline:
        pipeline = Pipeline(
            steps=[
                ("vectorizer", TfidfVectorizer(stop_words="english", max_features=5000)),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=Settings.RANDOM_STATE,
                    ),
                ),
            ]
        )
        pipeline.fit(dataset.texts, dataset.labels)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, self.model_path)
        return pipeline


class SentimentModelService:
    def __init__(self, dataset_loader: DatasetLoader, trainer: SentimentModelTrainer):
        self.dataset_loader = dataset_loader
        self.trainer = trainer
        self.model: Pipeline | None = None

    def load_or_train(self) -> None:
        if self.trainer.model_path.exists():
            self.model = joblib.load(self.trainer.model_path)
            return

        dataset = self.dataset_loader.load()
        self.model = self.trainer.train(dataset)

    def predict(self, text: str) -> tuple[str, dict[str, float]]:
        if self.model is None:
            raise RuntimeError("Model has not been loaded.")

        prediction = self.model.predict([text])[0]
        probabilities = self.model.predict_proba([text])[0]
        class_names = self.model.classes_
        probability_map = {
            str(label): float(score)
            for label, score in zip(class_names, probabilities, strict=True)
        }
        return str(prediction), probability_map
