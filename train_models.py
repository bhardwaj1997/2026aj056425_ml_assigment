"""
Train and persist five classification models for the Breast Cancer
Wisconsin (Diagnostic) dataset.

Each model is stored as a complete scikit-learn Pipeline (feature scaling +
estimator) so that the Streamlit app can load it and score raw test data
directly, without any retraining or manual preprocessing.

Run:
    python train_models.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_SEED = 2026
TEST_FRACTION = 0.20
MODEL_DIR = Path(__file__).resolve().parent / "model"
DATA_DIR = Path(__file__).resolve().parent


def load_diagnostic_frame():
    """Load the diagnostic dataset into a tidy feature frame and label series."""
    bundle = load_breast_cancer(as_frame=True)
    features = bundle.data.copy()
    # Clean column names so they survive a CSV round-trip without surprises.
    features.columns = [c.strip().replace(" ", "_") for c in features.columns]
    labels = bundle.target.rename("diagnosis")
    class_names = list(bundle.target_names)  # ['malignant', 'benign']
    return features, labels, class_names


def build_model_zoo():
    """Return the five classifiers, each wrapped in a scaling pipeline."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, C=1.0, random_state=RANDOM_SEED)),
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_SEED)),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance")),
        ]),
        "Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianNB()),
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=2.0, gamma="scale",
                        random_state=RANDOM_SEED)),
        ]),
    }


def positive_class_scores(pipeline, x_frame):
    """Return probability-like scores for the positive class for AUC."""
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(x_frame)[:, 1]
    return pipeline.decision_function(x_frame)


def evaluate(pipeline, x_frame, y_true):
    """Compute the six required metrics for a fitted pipeline."""
    y_pred = pipeline.predict(x_frame)
    y_score = positive_class_scores(pipeline, x_frame)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def slugify(name):
    return name.lower().replace(" ", "_").replace("-", "")


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    features, labels, class_names = load_diagnostic_frame()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=TEST_FRACTION,
        stratify=labels,
        random_state=RANDOM_SEED,
    )

    print(f"Features: {features.shape[1]} | Instances: {features.shape[0]}")
    print(f"Train rows: {len(x_train)} | Test rows: {len(x_test)}\n")

    model_zoo = build_model_zoo()
    metrics_table = {}
    file_map = {}

    for name, pipeline in model_zoo.items():
        pipeline.fit(x_train, y_train)
        scores = evaluate(pipeline, x_test, y_test)
        metrics_table[name] = scores

        out_file = MODEL_DIR / f"{slugify(name)}.joblib"
        joblib.dump(pipeline, out_file)
        file_map[name] = out_file.name

        pretty = "  ".join(f"{k}={v:.4f}" for k, v in scores.items())
        print(f"{name:<24} {pretty}")

    # Persist the held-out test set exactly as the app will consume it.
    test_frame = x_test.copy()
    test_frame["diagnosis"] = y_test.values
    test_frame.to_csv(DATA_DIR / "test_data.csv", index=False)

    metadata = {
        "feature_names": list(features.columns),
        "target_name": "diagnosis",
        "class_names": class_names,       # index 0 -> malignant, 1 -> benign
        "positive_label": 1,
        "model_files": file_map,
        "metrics": metrics_table,
        "random_seed": RANDOM_SEED,
        "test_fraction": TEST_FRACTION,
    }
    joblib.dump(metadata, MODEL_DIR / "metadata.joblib")

    # Emit a markdown comparison table for the README.
    print("\n| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |")
    print("|---|---|---|---|---|---|---|")
    for name, scores in metrics_table.items():
        print("| {} | {Accuracy:.4f} | {AUC:.4f} | {Precision:.4f} | "
              "{Recall:.4f} | {F1:.4f} | {MCC:.4f} |".format(name, **scores))

    print("\nSaved models + metadata to:", MODEL_DIR)
    print("Saved test data to:", DATA_DIR / "test_data.csv")


if __name__ == "__main__":
    main()
