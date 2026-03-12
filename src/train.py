import numpy as np
import pandas as pd
from pathlib import Path
from datasets import Dataset
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from setfit import SetFitModel, Trainer, TrainingArguments

from .classifier import LABELS, BASE_MODEL, MODEL_DIR, CommentClassifier

DATA_PATH = Path(__file__).parent.parent / "data" / "labeled_data.csv"


def train(csv_path: str = str(DATA_PATH)):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["End-User Comment"])
    df = df[df["End-User Comment"].str.strip() != ""]

    X = df["End-User Comment"].tolist()
    y = (df[LABELS].fillna("").eq("X")).astype(int).values.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    print(f"Fine-tuning SetFit on {len(X_train)} train / {len(X_test)} test comments...")

    eval_model = SetFitModel.from_pretrained(
        BASE_MODEL,
        multi_target_strategy="one-vs-rest",
        labels=LABELS,
    )

    eval_trainer = Trainer(
        model=eval_model,
        args=TrainingArguments(num_epochs=1, batch_size=16, num_iterations=20),
        train_dataset=Dataset.from_dict({"text": X_train, "label": y_train}),
    )
    eval_trainer.train()

    y_pred = eval_model.predict(X_test)
    y_test_arr = np.array(y_test)

    print(f"\n--- Evaluation on held-out {len(X_test)} examples ---")
    for i, label in enumerate(LABELS):
        print(f"\n{label}:")
        print(
            classification_report(
                y_test_arr[:, i], y_pred[:, i], target_names=["No", "Yes"], zero_division=0
            )
        )

    # Train final model on all data and save
    print("\nTraining final model on all data...")
    clf = CommentClassifier()
    clf.train(csv_path)
    return clf
