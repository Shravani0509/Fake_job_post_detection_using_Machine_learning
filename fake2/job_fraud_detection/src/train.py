import os
from dataclasses import dataclass
from typing import Dict, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


DATA_PATH = os.environ.get("JOB_FRAUD_DATA_PATH", "dataset/job_postings.csv")
MODEL_OUT_PATH = os.environ.get(
    "JOB_FRAUD_MODEL_OUT_PATH", "trained_models/random_forest_job_fraud.joblib"
)
PLOTS_DIR = os.environ.get("JOB_FRAUD_PLOTS_DIR", "screenshots")


COMMON_INDIAN_LOCATIONS = [
    "Mumbai",
    "Delhi",
    "Bangalore",
    "Hyderabad",
    "Ahmedabad",
    "Chennai",
    "Kolkata",
    "Pune",
    "Jaipur",
    "Surat",
    "Lucknow",
    "Kanpur",
    "Nagpur",
    "Visakhapatnam",
    "Patna",
]


def _ensure_dirs():
    os.makedirs(os.path.dirname(MODEL_OUT_PATH) or ".", exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


def _plot_fraud_distribution(y_train: np.ndarray, y_test: np.ndarray) -> None:
    labels = ["Fraudulent", "Not Fraudulent"]
    train_counts = [int((y_train == 1).sum()), int((y_train == 0).sum())]
    test_counts = [int((y_test == 1).sum()), int((y_test == 0).sum())]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width / 2, train_counts, width, label="Train", color=["red", "green"])
    ax.bar(x + width / 2, test_counts, width, label="Test", color=["red", "green"], alpha=0.6)

    ax.set_ylabel("Count")
    ax.set_title("Fraud vs Legitimate Distribution")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    out_path = os.path.join(PLOTS_DIR, "fraud_distribution.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_top_terms(text_transformer: TfidfVectorizer, rf_model: RandomForestClassifier) -> None:
    """Plots top TF-IDF terms by RandomForest feature importance."""
    if not hasattr(rf_model, "feature_importances_"):
        return

    feature_importances = rf_model.feature_importances_
    vocab = text_transformer.get_feature_names_out()

    # rf_model gets combined features (tfidf + encoded location as extra column)
    # We will only visualize TF-IDF part.
    tfidf_feature_count = len(vocab)
    importances_tfidf = feature_importances[:tfidf_feature_count]

    top_n = 20
    top_idx = np.argsort(importances_tfidf)[-top_n:][::-1]
    top_terms = [str(vocab[i]) for i in top_idx]
    top_scores = importances_tfidf[top_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top_terms[::-1], top_scores[::-1])
    ax.set_title("Top TF-IDF Features (approx.)")
    ax.set_xlabel("Feature Importance")

    out_path = os.path.join(PLOTS_DIR, "top_tfidf_features.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Not Fraudulent", "Fraudulent"])
    ax.set_yticklabels(["Not Fraudulent", "Fraudulent"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    ax.set_title("Confusion Matrix")
    out_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def train_and_save() -> Dict[str, float]:
    _ensure_dirs()

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset CSV not found at: {DATA_PATH}. Put your dataset into dataset/job_postings.csv"
        )

    data = pd.read_csv(DATA_PATH)

    required_cols = {"job_description", "location", "fraudulent"}
    missing = required_cols - set(data.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(list(missing))}")

    data = data.dropna(subset=["job_description", "location", "fraudulent"])

    X = data[["job_description", "location"]]
    y = data["fraudulent"].astype(int)

    text_transformer = TfidfVectorizer(max_features=1500, stop_words="english")
    location_encoder = LabelEncoder()
    location_encoder.fit(COMMON_INDIAN_LOCATIONS + list(data["location"].unique()))

    X_text = text_transformer.fit_transform(X["job_description"])
    X_loc = location_encoder.transform(X["location"]).reshape(-1, 1)

    X_combined = hstack([X_text, X_loc])

    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42, stratify=y
    )

    rf_model = RandomForestClassifier(
        n_estimators=250,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        max_depth=None,
    )

    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    y_proba = rf_model.predict_proba(X_test)

    # Determine index for fraudulent class=1
    fraudulent_index = 1 if len(rf_model.classes_) > 1 else 0
    # If classes_ are [0,1], index for class 1 is where classes_==1
    if 1 in list(rf_model.classes_):
        fraudulent_index = list(rf_model.classes_).index(1)

    roc_auc = roc_auc_score(y_test, y_proba[:, fraudulent_index]) if len(np.unique(y_test)) > 1 else 0.0

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc),
    }

    _plot_fraud_distribution(y_train.to_numpy(), y_test.to_numpy())
    _plot_top_terms(text_transformer, rf_model)
    _plot_confusion_matrix(y_test.to_numpy(), y_pred)

    joblib.dump(
        {
            "model": rf_model,
            "text_transformer": text_transformer,
            "location_encoder": location_encoder,
            "metrics": metrics,
            "feature_info": {
                "tfidf_max_features": 1500,
                "location_encoded_type": "LabelEncoder",
            },
        },
        MODEL_OUT_PATH,
    )

    metrics_out = {k: round(v, 6) for k, v in metrics.items()}
    return metrics_out


if __name__ == "__main__":
    results = train_and_save()
    print("Training complete. Metrics:")
    for k, v in results.items():
        print(f"- {k}: {v}")

