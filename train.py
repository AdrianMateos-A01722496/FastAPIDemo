from app.config import Settings
from app.services import DatasetLoader, SentimentModelTrainer


def main() -> None:
    dataset_loader = DatasetLoader(Settings.DATASET_PATH)
    trainer = SentimentModelTrainer(Settings.MODEL_PATH)
    dataset = dataset_loader.load()
    trainer.train(dataset, print_holdout_metrics=True)
    print(f"Model saved to {Settings.MODEL_PATH}")


if __name__ == "__main__":
    main()
