import numpy as np
import pandas as pd
from pathlib import Path
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments

LABELS = ["Technical", "Performance", "UX", "Data/Security"]
MODEL_DIR = Path(__file__).parent.parent / "models" / "setfit"
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class CommentClassifier:
    def __init__(self):
        self.model: SetFitModel | None = None

    def _parse_csv(self, csv_path: str) -> tuple[list[str], list[list[int]]]:
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["End-User Comment"])
        df = df[df["End-User Comment"].str.strip() != ""]
        texts = df["End-User Comment"].tolist()
        labels = (df[LABELS].fillna("").eq("X")).astype(int).values.tolist()
        return texts, labels

    def train(self, csv_path: str, num_epochs: int = 1, num_iterations: int = 20):
        texts, labels = self._parse_csv(csv_path)
        dataset = Dataset.from_dict({"text": texts, "label": labels})

        model = SetFitModel.from_pretrained(
            BASE_MODEL,
            multi_target_strategy="one-vs-rest",
            labels=LABELS,
        )

        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                num_epochs=num_epochs,
                batch_size=16,
                num_iterations=num_iterations,
                output_dir=str(MODEL_DIR),
            ),
            train_dataset=dataset,
        )

        print(f"Fine-tuning SetFit on {len(texts)} comments...")
        trainer.train()

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(MODEL_DIR))
        self.model = model
        print(f"Model saved to {MODEL_DIR}")

    def load(self):
        if not self.is_trained():
            raise FileNotFoundError(
                f"No trained model at {MODEL_DIR}. Run: python main.py train"
            )
        self.model = SetFitModel.from_pretrained(str(MODEL_DIR))

    def predict(self, text: str) -> dict:
        if self.model is None:
            self.load()
        probas = self.model.predict_proba([text])[0].tolist()
        predictions = self.model.predict([text])[0].tolist()
        return {
            label: {"predicted": bool(pred), "confidence": round(float(prob), 3)}
            for label, pred, prob in zip(LABELS, predictions, probas)
        }

    def is_trained(self) -> bool:
        return (MODEL_DIR / "config.json").exists()
