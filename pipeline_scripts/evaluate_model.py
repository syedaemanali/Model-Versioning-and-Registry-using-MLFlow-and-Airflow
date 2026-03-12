import os
import json
import pickle
import mlflow
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)

GRAPH_DIR = "/opt/airflow/data/processed/graphs"
HISTORY_FILE = "/opt/airflow/data/processed/graphs/history.json"
PALETTE = ["#FFB3BA", "#BAFFC9", "#BAE1FF", "#D4BAFF", "#FFDFBA"]
BG = "#FAFAFA"

def setup_plot():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.facecolor": BG, "figure.facecolor": BG,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3, "grid.color": "#CCCCCC",
    })

def evaluate_model(**context):
    ti = context['ti']
    model_path   = ti.xcom_pull(task_ids='train_model', key='model_path')
    encoded_path = ti.xcom_pull(task_ids='train_model', key='encoded_path')
    run_id       = ti.xcom_pull(task_ids='train_model', key='run_id')
    model_type   = ti.xcom_pull(task_ids='train_model', key='model_type')

    df = pd.read_csv(encoded_path)
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)

    print(f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    # push to XCom first before anything can fail
    ti.xcom_push(key='accuracy', value=accuracy)
    ti.xcom_push(key='run_id',   value=run_id)

    # log to mlflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("accuracy",  accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall",    recall)
        mlflow.log_metric("f1_score",  f1)

    # create graph dirs
    run_dir = os.path.join(GRAPH_DIR, run_id[:8])
    os.makedirs(run_dir, exist_ok=True)
    run_label = f"{model_type} | acc={accuracy:.4f}"
    setup_plot()

    # confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", ax=ax,
                cmap=sns.diverging_palette(350, 145, s=50, l=85, as_cmap=True),
                linewidths=2, linecolor="white",
                xticklabels=["Did Not Survive", "Survived"],
                yticklabels=["Did Not Survive", "Survived"],
                annot_kws={"size": 13, "fontweight": "bold"})
    ax.set_title(f"Confusion Matrix\n{run_label}", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{run_dir}/confusion_matrix.png", dpi=150)
    plt.close()

    # ROC curve
    proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#BAE1FF", linewidth=2.5, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4)
    ax.set_title(f"ROC Curve\n{run_label}", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{run_dir}/roc_curve.png", dpi=150)
    plt.close()

    # metrics bar chart for this run
    fig, ax = plt.subplots(figsize=(8, 5))
    metric_vals = [accuracy, precision, recall, f1]
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    bars = ax.bar(metric_names, metric_vals, color=PALETTE[:4], edgecolor="white", linewidth=1.5, width=0.5)
    for bar, val in zip(bars, metric_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", fontsize=11, color="#555")
    ax.axhline(0.80, color="#E07070", linestyle="--", linewidth=1.5, alpha=0.7, label="0.80 threshold")
    ax.set_title(f"Model Metrics\n{run_label}", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{run_dir}/metrics_bar.png", dpi=150)
    plt.close()

    # load history, append this run, save back
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)

    # avoid duplicate entries if DAG is rerun with same run_id
    history = [h for h in history if h["run_id"] != run_id]
    history.append({
        "run_id": run_id, "label": run_label,
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1, "auc": roc_auc,
    })
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

    # comparison graphs only if more than one run exists
    if len(history) >= 2:
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(history))]
        labels = [h["label"] for h in history]

        # accuracy comparison
        fig, ax = plt.subplots(figsize=(9, 5))
        accs = [h["accuracy"] for h in history]
        bars = ax.bar(labels, accs, color=colors, width=0.45, edgecolor="white", linewidth=1.5)
        ax.axhline(0.80, color="#E07070", linestyle="--", linewidth=1.5, label="0.80 threshold")
        for bar, val in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.4f}", ha="center", fontsize=10, color="#555")
        ax.set_title("Accuracy Comparison Across Experiments", fontsize=14, fontweight="bold", pad=15)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_ylim(0.5, 1.05)
        ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
        ax.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{GRAPH_DIR}/comparison_accuracy.png", dpi=150)
        plt.close()

        # all metrics comparison
        fig, ax = plt.subplots(figsize=(11, 6))
        display_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
        metric_keys   = ["accuracy", "precision", "recall", "f1"]
        x = np.arange(len(display_names))
        width = 0.8 / len(history)
        for i, entry in enumerate(history):
            vals = [entry[m] for m in metric_keys]
            bars = ax.bar(x + i * width, vals, width, label=entry["label"],
                          color=PALETTE[i % len(PALETTE)], edgecolor="white", linewidth=1.5)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.2f}", ha="center", fontsize=8, color="#555")
        ax.set_title("All Metrics Comparison Across Experiments", fontsize=14, fontweight="bold", pad=15)
        ax.set_xticks(x + width * (len(history) - 1) / 2)
        ax.set_xticklabels(display_names, fontsize=11)
        ax.set_ylim(0, 1.2)
        ax.set_ylabel("Score", fontsize=11)
        ax.axhline(0.80, color="#E07070", linestyle="--", linewidth=1.2, alpha=0.6)
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{GRAPH_DIR}/comparison_all_metrics.png", dpi=150)
        plt.close()

        print(f"Comparison graphs updated with {len(history)} runs")

    print(f"Per-run graphs saved to {run_dir}")