from __future__ import annotations

from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
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

    def _build_pipeline(self) -> Pipeline:
        return Pipeline(
            steps=[
                ("vectorizer", TfidfVectorizer(stop_words="english", max_features=5000)),
                (
                    "classifier",
                    LogisticRegression(
                        l1_ratio = Settings.L1_RATIO,
                        class_weight = Settings.CLASS_WEIGHT,
                        max_iter = Settings.MAX_ITER,
                        random_state = Settings.RANDOM_STATE,
                        solver = Settings.SOLVER
                    ),
                ),
            ]
        )

    @staticmethod
    def _print_classification_metrics(y_true: list[str], y_pred: list[str]) -> None:
        labels = sorted(set(y_true) | set(y_pred))
        acc = accuracy_score(y_true, y_pred)
        print(f"\n--- Hold-out evaluation (test split) ---\nAccuracy: {acc:.4f}\n")
        print("Classification report:\n")
        print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
        print("Confusion matrix (rows=true, cols=predicted):\n")
        print(
            confusion_matrix(y_true, y_pred, labels=labels),
            f"\nLabel order: {labels}\n",
        )

    def train(
        self,
        dataset: TrainingDataset,
        *,
        print_holdout_metrics: bool = False,
        holdout_size: float = 0.2,
    ) -> Pipeline:
        pipeline = self._build_pipeline()

        if print_holdout_metrics:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    dataset.texts,
                    dataset.labels,
                    test_size=holdout_size,
                    stratify=dataset.labels,
                    random_state=Settings.RANDOM_STATE,
                )
            except ValueError:
                X_train, X_test, y_train, y_test = train_test_split(
                    dataset.texts,
                    dataset.labels,
                    test_size=holdout_size,
                    random_state=Settings.RANDOM_STATE,
                )
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            self._print_classification_metrics(list(y_test), list(y_pred))
            pipeline.fit(dataset.texts, dataset.labels)
        else:
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
