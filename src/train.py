import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier

from .classifier import LABELS, CommentClassifier

DATA_PATH = Path(__file__).parent.parent / "data" / "labeled_data.csv"


def train(csv_path: str = str(DATA_PATH)):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["End-User Comment"])
    df = df[df["End-User Comment"].str.strip() != ""]

    X = df["End-User Comment"].tolist()
    y = (df[LABELS].fillna("").eq("X")).astype(int).values

    clf = CommentClassifier()

    # Held-out evaluation split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    print(f"Encoding {len(X_train)} train / {len(X_test)} test comments...")
    train_emb = clf.encoder.encode(X_train, show_progress_bar=True)
    test_emb = clf.encoder.encode(X_test, show_progress_bar=False)

    eval_clf = MultiOutputClassifier(
        LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    )
    eval_clf.fit(train_emb, y_train)
    y_pred = eval_clf.predict(test_emb)

    print(f"\n--- Evaluation on held-out {len(X_test)} examples ---")
    for i, label in enumerate(LABELS):
        print(f"\n{label}:")
        print(
            classification_report(
                y_test[:, i], y_pred[:, i], target_names=["No", "Yes"], zero_division=0
            )
        )

    # Train final model on all data and save
    print("\nTraining final model on all data...")
    clf.train(csv_path)
    return clf
