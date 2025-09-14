"""
SHAP-based model explanations.

Provides per-claim explanations that the clinical ops team can use to
understand why the model flagged (or didn't flag) a claim. The SHAP values
tell us exactly how much each feature contributed to the denial probability.
"""

import numpy as np
import pandas as pd
import shap

from src.features.engineering import FEATURE_COLUMNS, engineer_single_claim
from src.models.predict import load_model

_explainer = None


def _get_explainer():
    """Initialize SHAP TreeExplainer. Cached after first call."""
    global _explainer
    if _explainer is None:
        model = load_model()
        _explainer = shap.TreeExplainer(model)
    return _explainer


def explain_prediction(claim_dict: dict) -> dict:
    """
    Generate SHAP explanation for a single claim prediction.

    Returns feature contributions sorted by absolute impact, plus
    the base value and final prediction for waterfall chart rendering.
    """
    model = load_model()
    explainer = _get_explainer()

    features = engineer_single_claim(claim_dict)
    feature_array = np.array([[features[col] for col in FEATURE_COLUMNS]])

    shap_values = explainer.shap_values(feature_array)

    # For binary classification, shap_values might be a list of two arrays
    if isinstance(shap_values, list):
        shap_vals = shap_values[1][0]  # positive class
    else:
        shap_vals = shap_values[0]

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1] if len(base_value) > 1 else base_value[0]

    denial_prob = float(model.predict_proba(feature_array)[0, 1])

    feature_contributions = {}
    for i, col in enumerate(FEATURE_COLUMNS):
        feature_contributions[col] = round(float(shap_vals[i]), 6)

    sorted_features = sorted(
        feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True
    )

    top_risk_factors = [
        name for name, val in sorted_features if val > 0.01
    ][:5]

    top_protective_factors = [
        name for name, val in sorted_features if val < -0.01
    ][:5]

    return {
        "claim_id": claim_dict.get("claim_id", "unknown"),
        "base_value": round(float(base_value), 6),
        "prediction": round(denial_prob, 4),
        "feature_contributions": feature_contributions,
        "top_risk_factors": top_risk_factors,
        "top_protective_factors": top_protective_factors,
        "feature_values": {col: features[col] for col in FEATURE_COLUMNS},
    }


def explain_batch(claim_df: pd.DataFrame) -> list[dict]:
    """
    Generate SHAP explanations for a batch of claims.

    More efficient than calling explain_prediction in a loop because
    SHAP can vectorize the tree traversal.
    """
    from src.features.engineering import get_model_input

    model = load_model()
    explainer = _get_explainer()

    X = get_model_input(claim_df)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_matrix = shap_values[1]
    else:
        shap_matrix = shap_values

    probas = model.predict_proba(X)[:, 1]

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(base_value[1]) if len(base_value) > 1 else float(base_value[0])

    explanations = []
    for i in range(len(claim_df)):
        contributions = {
            col: round(float(shap_matrix[i, j]), 6)
            for j, col in enumerate(FEATURE_COLUMNS)
        }

        sorted_contribs = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)

        explanations.append({
            "claim_id": claim_df.iloc[i].get("claim_id", f"claim_{i}"),
            "base_value": round(base_value, 6),
            "prediction": round(float(probas[i]), 4),
            "feature_contributions": contributions,
            "top_risk_factors": [n for n, v in sorted_contribs if v > 0.01][:5],
            "top_protective_factors": [n for n, v in sorted_contribs if v < -0.01][:5],
        })

    return explanations


def get_global_feature_importance(X: pd.DataFrame, max_display: int = 15) -> dict:
    """
    Compute mean absolute SHAP values across the dataset.

    This gives a global view of feature importance — more reliable than
    single-tree feature importance because it accounts for feature interactions.
    """
    explainer = _get_explainer()
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_matrix = shap_values[1]
    else:
        shap_matrix = shap_values

    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)

    importance = {
        col: round(float(mean_abs_shap[i]), 6)
        for i, col in enumerate(FEATURE_COLUMNS)
    }

    sorted_importance = dict(
        sorted(importance.items(), key=lambda x: x[1], reverse=True)[:max_display]
    )

    return sorted_importance
