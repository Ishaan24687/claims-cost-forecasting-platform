"""
Inference module: load model and make predictions.

Handles both single claim and batch prediction. The denial_reasons logic
maps model output back to human-readable explanations that the clinical ops
team actually cares about.
"""

import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src.features.engineering import (
    FEATURE_COLUMNS,
    engineer_single_claim,
    get_model_input,
)

MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pkl")
PREDICTION_THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))
MODEL_VERSION = "1.2.0"

_model = None


def load_model(model_path: str = MODEL_PATH):
    """Load the trained model from disk. Cached after first load."""
    global _model
    if _model is None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found at {path}. Run 'make train' first."
            )
        _model = joblib.load(path)
        print(f"Loaded model from {path}")
    return _model


def get_model():
    """Get the cached model instance."""
    global _model
    return _model


def _determine_denial_reasons(features: dict, denial_prob: float) -> list[str]:
    """
    Map feature values to human-readable denial reasons.

    These roughly correspond to the CARC/RARC codes we see in real 835 remittance
    files, but simplified for the dashboard.
    """
    reasons = []

    if features.get("late_submission", 0) == 1:
        reasons.append("Late submission (>90 days from date of service)")

    if features.get("submission_lag_days", 0) > 60:
        reasons.append("Submission significantly delayed")

    if features.get("has_high_cost_procedure", 0) == 1 and denial_prob > 0.5:
        reasons.append("High-cost procedure requires prior authorization")

    if features.get("log_billed_amount", 0) > np.log1p(10000):
        reasons.append("Billed amount exceeds usual and customary rates")

    if features.get("billed_to_allowed_ratio", 1) > 3.0:
        reasons.append("Billed amount significantly exceeds allowed amount")

    if features.get("provider_historical_denial_rate", 0) > 0.4:
        reasons.append("Provider has elevated denial history — possible documentation issues")

    if features.get("is_inpatient", 0) == 1 and denial_prob > 0.5:
        reasons.append("Inpatient stay may require medical necessity review")

    if not reasons and denial_prob > 0.5:
        reasons.append("Multiple risk factors combined")

    return reasons


def _calculate_reimbursement(
    billed_amount: float,
    allowed_amount: Optional[float],
    denial_prob: float,
    is_denied: bool,
) -> float:
    """
    Estimate expected reimbursement.

    If denied, reimbursement is $0. If approved, it's the allowed amount
    (or ~65% of billed if allowed isn't provided).
    """
    if is_denied:
        return 0.0

    if allowed_amount and allowed_amount > 0:
        return round(allowed_amount, 2)

    return round(billed_amount * 0.65, 2)


def predict_single(claim_dict: dict) -> dict:
    """
    Predict denial probability for a single claim.

    Returns a dict matching the ClaimPrediction schema.
    """
    model = load_model()

    features = engineer_single_claim(claim_dict)
    feature_array = np.array([[features[col] for col in FEATURE_COLUMNS]])

    denial_prob = float(model.predict_proba(feature_array)[0, 1])
    is_denied = denial_prob >= PREDICTION_THRESHOLD

    denial_reasons = _determine_denial_reasons(features, denial_prob)

    reimbursement = _calculate_reimbursement(
        billed_amount=claim_dict.get("billed_amount", 0),
        allowed_amount=claim_dict.get("allowed_amount"),
        denial_prob=denial_prob,
        is_denied=is_denied,
    )

    if denial_prob > 0.8:
        confidence = "high"
    elif denial_prob > 0.6 or denial_prob < 0.2:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "claim_id": claim_dict.get("claim_id", "unknown"),
        "denial_probability": round(denial_prob, 4),
        "predicted_denied": is_denied,
        "denial_reasons": denial_reasons,
        "estimated_reimbursement": reimbursement,
        "confidence": confidence,
        "model_version": MODEL_VERSION,
    }


def predict_batch(claim_df: pd.DataFrame) -> pd.DataFrame:
    """
    Batch prediction on a DataFrame of claims.

    More efficient than calling predict_single in a loop because we
    vectorize the feature engineering.
    """
    model = load_model()

    X = get_model_input(claim_df)

    proba = model.predict_proba(X)[:, 1]
    predictions = (proba >= PREDICTION_THRESHOLD).astype(int)

    result_df = claim_df[["claim_id"]].copy()
    result_df["denial_probability"] = np.round(proba, 4)
    result_df["predicted_denied"] = predictions.astype(bool)

    result_df["confidence"] = pd.cut(
        proba,
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["high_approve", "medium", "low", "medium", "high_deny"],
        ordered=False,
    )

    result_df["estimated_reimbursement"] = np.where(
        result_df["predicted_denied"],
        0.0,
        np.where(
            claim_df["allowed_amount"] > 0,
            claim_df["allowed_amount"],
            claim_df["billed_amount"] * 0.65,
        ),
    )

    return result_df
