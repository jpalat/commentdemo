import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

LABELS = ["Technical", "Performance", "UX", "Data/Security"]
MODEL_DIR = Path(__file__).parent.parent / "models"
BASE_MODEL = "all-MiniLM-L6-v2"


class CommentClassifier:
    def __init__(self):
        self.encoder = SentenceTransformer(BASE_MODEL)
        self.classifier: MultiOutputClassifier | None = None
        self._model_path = MODEL_DIR / "classifier.joblib"

    def train(self, csv_path: str):
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["End-User Comment"])
        df = df[df["End-User Comment"].str.strip() != ""]

        y = (df[LABELS].fillna("").eq("X")).astype(int).values
        X_text = df["End-User Comment"].tolist()

        print(f"Encoding {len(X_text)} comments...")
        embeddings = self.encoder.encode(X_text, show_progress_bar=True)

        self.classifier = MultiOutputClassifier(
            LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        )
        self.classifier.fit(embeddings, y)

        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(self.classifier, self._model_path)
        print(f"Model saved to {self._model_path}")

    def load(self):
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"No trained model at {self._model_path}. Run: python main.py train"
            )
        self.classifier = joblib.load(self._model_path)

    def predict(self, text: str) -> dict:
        if self.classifier is None:
            self.load()
        embedding = self.encoder.encode([text])
        predictions = self.classifier.predict(embedding)[0]
        probas = [
            est.predict_proba(embedding)[0][1]
            for est in self.classifier.estimators_
        ]
        return {
            label: {"predicted": bool(pred), "confidence": round(float(prob), 3)}
            for label, pred, prob in zip(LABELS, predictions, probas)
        }

    def is_trained(self) -> bool:
        return self._model_path.exists()
