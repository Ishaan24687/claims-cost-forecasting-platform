"""
Model evaluation with publication-quality plots.

Generates ROC curve, PR curve, calibration plot, and confusion matrix.
These get logged as MLflow artifacts and also saved locally for the dashboard.
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray,
                   model_name: str = "XGBoost") -> Path:
    """Plot ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#2563eb", lw=2,
            label=f"{model_name} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Claims Denial Prediction", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path = ARTIFACTS_DIR / "roc_curve.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_precision_recall_curve(y_true: np.ndarray, y_proba: np.ndarray,
                                 model_name: str = "XGBoost") -> Path:
    """
    PR curve is more informative than ROC when classes are imbalanced.
    With ~25% positive class, a high AUC can mask poor recall.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="#dc2626", lw=2,
            label=f"{model_name} (AP = {avg_precision:.3f})")
    ax.axhline(y=y_true.mean(), color="gray", linestyle="--",
               label=f"Baseline (prevalence = {y_true.mean():.2f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve — Claims Denial Prediction", fontsize=14)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path = ARTIFACTS_DIR / "pr_curve.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_calibration(y_true: np.ndarray, y_proba: np.ndarray,
                     model_name: str = "XGBoost", n_bins: int = 10) -> Path:
    """
    Calibration plot: does the model's 30% prediction actually mean 30% denial?

    Well-calibrated models are important here because we use the probability
    to triage claims for manual review, not just the binary prediction.
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_proba, n_bins=n_bins
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(mean_predicted_value, fraction_of_positives, "o-",
            color="#059669", lw=2, label=model_name)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Calibration Plot — Claims Denial Prediction", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path = ARTIFACTS_DIR / "calibration_plot.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          model_name: str = "XGBoost") -> Path:
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Approved", "Denied"],
    )
    disp.plot(ax=ax, cmap="Blues", values_format=",d")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14)
    fig.tight_layout()

    output_path = ARTIFACTS_DIR / "confusion_matrix.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "XGBoost",
) -> dict:
    """
    Full evaluation suite. Returns metrics dict and saves all plots.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    y_true_np = y_test.to_numpy()

    roc_path = plot_roc_curve(y_true_np, y_proba, model_name)
    pr_path = plot_precision_recall_curve(y_true_np, y_proba, model_name)
    cal_path = plot_calibration(y_true_np, y_proba, model_name)
    cm_path = plot_confusion_matrix(y_true_np, y_pred, model_name)

    report = classification_report(y_test, y_pred, output_dict=True)
    auc_score = roc_auc_score(y_test, y_proba)

    metrics = {
        "auc": auc_score,
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"],
        "accuracy": report["accuracy"],
        "average_precision": average_precision_score(y_test, y_proba),
    }

    print(f"\nEvaluation Results for {model_name}:")
    print(f"  AUC:               {metrics['auc']:.4f}")
    print(f"  Precision:         {metrics['precision']:.4f}")
    print(f"  Recall:            {metrics['recall']:.4f}")
    print(f"  F1:                {metrics['f1']:.4f}")
    print(f"  Avg Precision:     {metrics['average_precision']:.4f}")
    print(f"\nPlots saved to {ARTIFACTS_DIR}/")

    return {
        "metrics": metrics,
        "artifacts": {
            "roc_curve": str(roc_path),
            "pr_curve": str(pr_path),
            "calibration_plot": str(cal_path),
            "confusion_matrix": str(cm_path),
        },
        "classification_report": report,
    }
