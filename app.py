"""
Streamlit front-end for the Breast Cancer (Diagnostic) classification suite.

The app loads five pre-trained scikit-learn pipelines and scores an uploaded
test CSV *without any retraining*. It reports the six evaluation metrics, a
confusion matrix, and a full classification report for the selected model, plus
a side-by-side comparison of every model.

Run locally:
    streamlit run app.py
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

MODEL_DIR = Path(__file__).resolve().parent / "model"

st.set_page_config(
    page_title="Breast Cancer Diagnosis Classifier",
    page_icon="🩺",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_assets():
    """Load metadata and every persisted pipeline exactly once."""
    metadata = joblib.load(MODEL_DIR / "metadata.joblib")
    pipelines = {
        name: joblib.load(MODEL_DIR / file_name)
        for name, file_name in metadata["model_files"].items()
    }
    return metadata, pipelines


def positive_class_scores(pipeline, x_frame):
    """Return positive-class scores, preferring probabilities for AUC."""
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(x_frame)[:, 1]
    return pipeline.decision_function(x_frame)


def compute_metrics(pipeline, x_frame, y_true):
    y_pred = pipeline.predict(x_frame)
    y_score = positive_class_scores(pipeline, x_frame)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }, y_pred


def draw_confusion(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="PuBu",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    return fig


def main():
    try:
        metadata, pipelines = load_assets()
    except FileNotFoundError:
        st.error(
            "Trained models were not found. Run `python train_models.py` first "
            "so the `model/` folder is populated."
        )
        st.stop()

    feature_names = metadata["feature_names"]
    target_name = metadata["target_name"]
    class_names = metadata["class_names"]

    st.title("🩺 Breast Cancer Diagnosis — Multi-Model Classifier")
    st.caption(
        "Binary classification on the Breast Cancer Wisconsin (Diagnostic) "
        "dataset — 30 features, 569 instances. Upload the held-out test CSV to "
        "score five pre-trained models."
    )

    with st.sidebar:
        st.header("⚙️ Controls")
        uploaded = st.file_uploader(
            "Upload test data (CSV)",
            type=["csv"],
            help="CSV must contain all 30 feature columns and the "
            f"'{target_name}' target column.",
        )
        chosen_model = st.selectbox("Select a model", list(pipelines.keys()))
        st.markdown("---")
        st.markdown(
            "**Classes**\n\n"
            f"- `0` → {class_names[0]}\n"
            f"- `1` → {class_names[1]}"
        )

    if uploaded is None:
        st.info(
            "⬅️ Upload the `test_data.csv` file from the repository to begin. "
            "The models are only used for inference — they are never retrained "
            "on your uploaded data."
        )
        st.subheader("Reference metrics (held-out test set)")
        st.dataframe(
            pd.DataFrame(metadata["metrics"]).T.round(4),
            use_container_width=True,
        )
        return

    try:
        data = pd.read_csv(uploaded)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read the CSV file: {exc}")
        return

    missing_features = [c for c in feature_names if c not in data.columns]
    if missing_features:
        st.error(
            "The uploaded CSV is missing required feature columns: "
            + ", ".join(missing_features)
        )
        return
    if target_name not in data.columns:
        st.error(
            f"The uploaded CSV must include the target column '{target_name}'."
        )
        return

    x_test = data[feature_names]
    y_test = data[target_name]

    st.success(
        f"Loaded {len(data)} rows. Scoring with **{chosen_model}** "
        "(no retraining)."
    )

    pipeline = pipelines[chosen_model]
    metrics, y_pred = compute_metrics(pipeline, x_test, y_test)

    st.subheader(f"📊 Evaluation metrics — {chosen_model}")
    metric_cols = st.columns(6)
    for col, (name, value) in zip(metric_cols, metrics.items()):
        col.metric(name, f"{value:.4f}")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Confusion matrix")
        st.pyplot(draw_confusion(y_test, y_pred, class_names))
    with right:
        st.subheader("Classification report")
        report = classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        st.dataframe(
            pd.DataFrame(report).T.style.format("{:.4f}"),
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("🏁 All-model comparison on the uploaded test data")
    comparison = {}
    for name, pipe in pipelines.items():
        scores, _ = compute_metrics(pipe, x_test, y_test)
        comparison[name] = scores
    comparison_df = pd.DataFrame(comparison).T
    st.dataframe(
        comparison_df.style.format("{:.4f}").highlight_max(axis=0, color="#c6f6d5"),
        use_container_width=True,
    )
    winner = comparison_df["F1"].idxmax()
    st.success(f"🥇 Best F1 on this test data: **{winner}**")

    with st.expander("Preview uploaded data (first 10 rows)"):
        st.dataframe(data.head(10), use_container_width=True)


if __name__ == "__main__":
    main()
