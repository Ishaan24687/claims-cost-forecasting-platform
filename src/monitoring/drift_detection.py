"""
Feature drift detection using Population Stability Index (PSI).

PSI compares the distribution of features in the current scoring batch against
the training distribution. If a feature's distribution shifts significantly,
the model's predictions may become unreliable.

Rule of thumb:
  PSI < 0.1  → no significant shift
  PSI 0.1-0.2 → moderate shift, investigate
  PSI > 0.2  → significant shift, consider retraining

I chose PSI over KS-test or chi-squared because it's the standard in insurance/
financial model monitoring and our compliance team already understands it.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.engineering import FEATURE_COLUMNS


PSI_THRESHOLD_WARNING = 0.10
PSI_THRESHOLD_CRITICAL = 0.20
N_BINS = 10


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = N_BINS,
) -> float:
    """
    Calculate Population Stability Index between two distributions.

    Uses equal-width binning on the expected (training) distribution,
    then applies the same bin edges to the actual (scoring) distribution.
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())

    if min_val == max_val:
        return 0.0

    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    expected_counts = np.histogram(expected, bins=bin_edges)[0]
    actual_counts = np.histogram(actual, bins=bin_edges)[0]

    # Avoid division by zero with small epsilon
    expected_pct = (expected_counts + 1e-6) / (expected_counts.sum() + n_bins * 1e-6)
    actual_pct = (actual_counts + 1e-6) / (actual_counts.sum() + n_bins * 1e-6)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def detect_drift(
    training_df: pd.DataFrame,
    scoring_df: pd.DataFrame,
    feature_columns: list[str] = FEATURE_COLUMNS,
) -> dict:
    """
    Compare feature distributions between training and scoring data.

    Returns a report with PSI per feature and overall drift status.
    """
    results = {}
    features_with_drift = []
    features_critical = []

    for col in feature_columns:
        if col not in training_df.columns or col not in scoring_df.columns:
            continue

        psi = calculate_psi(
            training_df[col].values.astype(float),
            scoring_df[col].values.astype(float),
        )

        if psi >= PSI_THRESHOLD_CRITICAL:
            status = "critical"
            features_critical.append(col)
        elif psi >= PSI_THRESHOLD_WARNING:
            status = "warning"
            features_with_drift.append(col)
        else:
            status = "stable"

        results[col] = {
            "psi": round(psi, 6),
            "status": status,
            "training_mean": round(float(training_df[col].mean()), 4),
            "training_std": round(float(training_df[col].std()), 4),
            "scoring_mean": round(float(scoring_df[col].mean()), 4),
            "scoring_std": round(float(scoring_df[col].std()), 4),
        }

    if features_critical:
        overall_status = "critical"
    elif features_with_drift:
        overall_status = "warning"
    else:
        overall_status = "stable"

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "overall_status": overall_status,
        "n_features_monitored": len(results),
        "n_features_warning": len(features_with_drift),
        "n_features_critical": len(features_critical),
        "features_with_drift": features_with_drift,
        "features_critical": features_critical,
        "feature_details": results,
        "recommendation": _get_recommendation(overall_status, features_critical),
    }

    return report


def _get_recommendation(status: str, critical_features: list[str]) -> str:
    if status == "critical":
        feature_list = ", ".join(critical_features[:5])
        return (
            f"RETRAIN RECOMMENDED: Significant distribution shift detected in "
            f"{feature_list}. Model predictions may be unreliable."
        )
    elif status == "warning":
        return (
            "MONITOR: Moderate distribution shift detected in some features. "
            "Schedule a review but retraining may not be necessary yet."
        )
    return "No action needed. Feature distributions are stable."


def save_drift_report(report: dict, output_dir: str = "artifacts") -> Path:
    """Save drift report as JSON artifact."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_path = output_path / f"drift_report_{timestamp}.json"

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report_path


def monitor_prediction_distribution(
    predictions: np.ndarray,
    historical_mean: float = 0.25,
    historical_std: float = 0.05,
) -> dict:
    """
    Quick check: has the overall denial rate shifted?

    If the model suddenly starts predicting 50% denial rate when the
    historical average is 25%, something is wrong — either the data
    changed or the model is degrading.
    """
    current_mean = float(np.mean(predictions))
    current_std = float(np.std(predictions))

    z_score = abs(current_mean - historical_mean) / max(historical_std, 1e-6)

    if z_score > 3:
        status = "critical"
    elif z_score > 2:
        status = "warning"
    else:
        status = "stable"

    return {
        "current_denial_rate": round(current_mean, 4),
        "historical_denial_rate": historical_mean,
        "z_score": round(z_score, 2),
        "status": status,
    }
