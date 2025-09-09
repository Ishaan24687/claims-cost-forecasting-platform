"""
Feature engineering pipeline for claims denial prediction.

These features are based on patterns I've observed in real PBM claims data.
The healthcare domain knowledge matters here — things like submission lag,
chronic condition flags, and provider denial history are much stronger
predictors than raw amounts alone.
"""

from datetime import datetime

import numpy as np
import pandas as pd

from src.features.cpt_categories import (
    HIGH_COST_PROCEDURES,
    get_cpt_category,
    get_procedure_complexity,
)

CHRONIC_CONDITION_PREFIXES = ["E11", "I10", "J44", "N18"]

EMERGENCY_POS_CODES = {"23"}
INPATIENT_POS_CODES = {"21", "31"}

FEATURE_COLUMNS = [
    "submission_lag_days",
    "log_billed_amount",
    "billed_to_allowed_ratio",
    "has_chronic_condition",
    "has_high_cost_procedure",
    "is_emergency",
    "is_inpatient",
    "provider_historical_denial_rate",
    "n_diagnosis_codes",
    "n_procedure_codes",
    "cost_per_procedure",
    "is_weekend_service",
    "service_month",
    "late_submission",
    "is_pharmacy_claim",
    "is_dental_claim",
    "member_age",
    "is_male",
    "has_surgery",
    "has_radiology",
    "has_pathology",
    "has_eval_mgmt",
    "days_supply_normalized",
]


def compute_submission_lag(claim_df: pd.DataFrame) -> pd.Series:
    service = pd.to_datetime(claim_df["service_date"])
    submission = pd.to_datetime(claim_df["submission_date"])
    lag = (submission - service).dt.days
    return lag.clip(lower=0)


def detect_chronic_conditions(diagnosis_codes_str: str) -> bool:
    """Check if any diagnosis codes indicate a chronic condition."""
    if not diagnosis_codes_str or pd.isna(diagnosis_codes_str):
        return False
    codes = str(diagnosis_codes_str).split("|")
    return any(
        code.startswith(prefix)
        for code in codes
        for prefix in CHRONIC_CONDITION_PREFIXES
    )


def detect_high_cost_procedure(procedure_codes_str: str) -> bool:
    if not procedure_codes_str or pd.isna(procedure_codes_str):
        return False
    codes = str(procedure_codes_str).split("|")
    return any(code in HIGH_COST_PROCEDURES for code in codes)


def compute_provider_denial_rate(claim_df: pd.DataFrame) -> pd.Series:
    """
    Rolling historical denial rate per provider.

    In production this would be a 90-day rolling window against a database.
    Here I compute a leave-one-out mean to avoid data leakage.
    """
    provider_stats = claim_df.groupby("provider_npi")["is_denied"].agg(["sum", "count"])
    provider_stats.columns = ["total_denied", "total_claims"]

    claim_df = claim_df.merge(provider_stats, left_on="provider_npi", right_index=True, how="left")

    # Leave-one-out to prevent target leakage
    loo_rate = (claim_df["total_denied"] - claim_df["is_denied"]) / (claim_df["total_claims"] - 1)
    loo_rate = loo_rate.fillna(0.25).clip(0, 1)

    claim_df.drop(columns=["total_denied", "total_claims"], inplace=True)
    return loo_rate


def get_procedure_category_flags(procedure_codes_str: str) -> dict[str, int]:
    """One-hot encode procedure categories for a claim."""
    if not procedure_codes_str or pd.isna(procedure_codes_str):
        return {"has_surgery": 0, "has_radiology": 0, "has_pathology": 0, "has_eval_mgmt": 0}

    codes = str(procedure_codes_str).split("|")
    categories = get_procedure_complexity(codes)

    return {
        "has_surgery": int("surgery" in categories),
        "has_radiology": int("radiology" in categories),
        "has_pathology": int("pathology" in categories),
        "has_eval_mgmt": int("evaluation_management" in categories),
    }


def engineer_features(claim_df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline. Takes raw claims, returns model-ready features.

    I keep the original dataframe intact and add new columns — makes debugging
    easier when you can see both raw and engineered features side by side.
    """
    df = claim_df.copy()

    df["submission_lag_days"] = compute_submission_lag(df)
    df["log_billed_amount"] = np.log1p(df["billed_amount"])

    df["billed_to_allowed_ratio"] = np.where(
        df["allowed_amount"] > 0,
        df["billed_amount"] / df["allowed_amount"],
        1.0,
    )
    df["billed_to_allowed_ratio"] = df["billed_to_allowed_ratio"].clip(upper=10.0)

    df["has_chronic_condition"] = df["diagnosis_codes"].apply(detect_chronic_conditions).astype(int)
    df["has_high_cost_procedure"] = df["procedure_codes"].apply(detect_high_cost_procedure).astype(int)

    df["is_emergency"] = df["place_of_service"].isin(EMERGENCY_POS_CODES).astype(int)
    df["is_inpatient"] = df["place_of_service"].isin(INPATIENT_POS_CODES).astype(int)

    df["provider_historical_denial_rate"] = compute_provider_denial_rate(df)

    df["n_diagnosis_codes"] = df["diagnosis_codes"].apply(
        lambda x: len(str(x).split("|")) if x and not pd.isna(x) else 0
    )
    df["n_procedure_codes"] = df["procedure_codes"].apply(
        lambda x: len(str(x).split("|")) if x and str(x).strip() and not pd.isna(x) else 0
    )

    df["cost_per_procedure"] = np.where(
        df["n_procedure_codes"] > 0,
        df["billed_amount"] / df["n_procedure_codes"],
        df["billed_amount"],
    )

    service_dt = pd.to_datetime(df["service_date"])
    df["is_weekend_service"] = service_dt.dt.dayofweek.isin([5, 6]).astype(int)
    df["service_month"] = service_dt.dt.month

    df["late_submission"] = (df["submission_lag_days"] > 90).astype(int)

    df["is_pharmacy_claim"] = (df["claim_type"] == "pharmacy").astype(int)
    df["is_dental_claim"] = (df["claim_type"] == "dental").astype(int)
    df["is_male"] = (df["member_gender"] == "M").astype(int)

    category_flags = df["procedure_codes"].apply(get_procedure_category_flags)
    flags_df = pd.DataFrame(category_flags.tolist(), index=df.index)
    df = pd.concat([df, flags_df], axis=1)

    # Normalize days_supply: 0 for non-pharmacy, scaled for pharmacy
    df["days_supply_normalized"] = np.where(
        df["is_pharmacy_claim"] == 1,
        df["days_supply"] / 90.0,
        0.0,
    )

    # TODO: add provider specialty as a feature — it was the #3 predictor at work
    # but I can't share that data. Would need to encode ~20 specialties.

    return df


def get_model_input(claim_df: pd.DataFrame) -> pd.DataFrame:
    """Extract just the feature columns needed for model training/inference."""
    df = engineer_features(claim_df)
    missing_cols = set(FEATURE_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing features after engineering: {missing_cols}")
    return df[FEATURE_COLUMNS].fillna(0)


def engineer_single_claim(claim_dict: dict) -> dict:
    """
    Engineer features for a single claim (API inference path).

    This duplicates some logic from engineer_features() but works on a dict
    instead of a DataFrame. In production I'd refactor to share code, but for
    single-claim latency this is faster than creating a 1-row DataFrame.
    """
    service_date = claim_dict.get("service_date")
    submission_date = claim_dict.get("submission_date")

    if isinstance(service_date, str):
        service_date = datetime.strptime(service_date, "%Y-%m-%d")
    if isinstance(submission_date, str):
        submission_date = datetime.strptime(submission_date, "%Y-%m-%d")

    if submission_date and service_date:
        submission_lag = (submission_date - service_date).days
    else:
        submission_lag = 0

    billed = claim_dict.get("billed_amount", 0)
    allowed = claim_dict.get("allowed_amount") or billed * 0.65

    dx_codes = claim_dict.get("diagnosis_codes", [])
    proc_codes = claim_dict.get("procedure_codes", [])

    has_chronic = any(
        code.startswith(prefix)
        for code in dx_codes
        for prefix in CHRONIC_CONDITION_PREFIXES
    )

    has_high_cost = any(c in HIGH_COST_PROCEDURES for c in proc_codes)

    categories = get_procedure_complexity(proc_codes)
    pos = str(claim_dict.get("place_of_service", "11"))

    n_procs = len(proc_codes) if proc_codes else 1

    features = {
        "submission_lag_days": max(submission_lag, 0),
        "log_billed_amount": np.log1p(billed),
        "billed_to_allowed_ratio": min(billed / allowed, 10.0) if allowed > 0 else 1.0,
        "has_chronic_condition": int(has_chronic),
        "has_high_cost_procedure": int(has_high_cost),
        "is_emergency": int(pos in EMERGENCY_POS_CODES),
        "is_inpatient": int(pos in INPATIENT_POS_CODES),
        "provider_historical_denial_rate": 0.25,  # default for unknown providers
        "n_diagnosis_codes": len(dx_codes),
        "n_procedure_codes": len(proc_codes),
        "cost_per_procedure": billed / n_procs,
        "is_weekend_service": int(service_date.weekday() in (5, 6)) if service_date else 0,
        "service_month": service_date.month if service_date else 1,
        "late_submission": int(submission_lag > 90),
        "is_pharmacy_claim": int(claim_dict.get("claim_type") == "pharmacy"),
        "is_dental_claim": int(claim_dict.get("claim_type") == "dental"),
        "member_age": claim_dict.get("member_age", 50),
        "is_male": int(claim_dict.get("member_gender") == "M"),
        "has_surgery": int("surgery" in categories),
        "has_radiology": int("radiology" in categories),
        "has_pathology": int("pathology" in categories),
        "has_eval_mgmt": int("evaluation_management" in categories),
        "days_supply_normalized": (claim_dict.get("days_supply") or 0) / 90.0
                                  if claim_dict.get("claim_type") == "pharmacy" else 0.0,
    }

    return features
