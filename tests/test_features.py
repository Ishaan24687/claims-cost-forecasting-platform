"""
Tests for feature engineering pipeline.

Covers the core transformations: submission lag calculation, chronic condition
detection, ICD-10 prefix matching, CPT category mapping, and edge cases.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    FEATURE_COLUMNS,
    compute_submission_lag,
    detect_chronic_conditions,
    detect_high_cost_procedure,
    engineer_features,
    engineer_single_claim,
    get_model_input,
    get_procedure_category_flags,
)
from src.features.cpt_categories import (
    get_cpt_category,
    is_high_cost_procedure,
    get_procedure_complexity,
)


def _make_claim_row(**overrides) -> dict:
    """Helper to create a single claim dict with sensible defaults."""
    defaults = {
        "claim_id": "CLM-TEST-001",
        "member_id": "MBR-000001",
        "provider_npi": "1234567890",
        "claim_type": "medical",
        "place_of_service": "11",
        "diagnosis_codes": "E11.9|I10",
        "procedure_codes": "99214",
        "billed_amount": 285.00,
        "allowed_amount": 195.00,
        "service_date": "2024-06-01",
        "submission_date": "2024-06-10",
        "drug_ndc": "",
        "days_supply": 0,
        "member_age": 55,
        "member_gender": "M",
        "provider_specialty": "internal_medicine",
        "is_denied": 0,
    }
    defaults.update(overrides)
    return defaults


def _make_claim_df(**overrides) -> pd.DataFrame:
    return pd.DataFrame([_make_claim_row(**overrides)])


class TestSubmissionLag:
    def test_basic_lag(self):
        df = _make_claim_df(service_date="2024-01-01", submission_date="2024-01-15")
        lag = compute_submission_lag(df)
        assert lag.iloc[0] == 14

    def test_same_day_submission(self):
        df = _make_claim_df(service_date="2024-03-01", submission_date="2024-03-01")
        lag = compute_submission_lag(df)
        assert lag.iloc[0] == 0

    def test_late_submission_over_90_days(self):
        df = _make_claim_df(service_date="2024-01-01", submission_date="2024-05-01")
        lag = compute_submission_lag(df)
        assert lag.iloc[0] == 121

    def test_negative_lag_clipped_to_zero(self):
        """If submission_date is before service_date (bad data), clip to 0."""
        df = _make_claim_df(service_date="2024-06-01", submission_date="2024-05-01")
        lag = compute_submission_lag(df)
        assert lag.iloc[0] == 0


class TestChronicConditionDetection:
    def test_diabetes_detected(self):
        assert detect_chronic_conditions("E11.9") is True

    def test_hypertension_detected(self):
        assert detect_chronic_conditions("I10") is True

    def test_copd_detected(self):
        assert detect_chronic_conditions("J44.1") is True

    def test_ckd_detected(self):
        assert detect_chronic_conditions("N18.3") is True

    def test_multiple_codes_with_chronic(self):
        assert detect_chronic_conditions("R10.9|E11.65|Z23") is True

    def test_no_chronic_condition(self):
        assert detect_chronic_conditions("R10.9|Z23|M54.5") is False

    def test_empty_string(self):
        assert detect_chronic_conditions("") is False

    def test_nan_value(self):
        assert detect_chronic_conditions(float("nan")) is False


class TestHighCostProcedure:
    def test_knee_replacement(self):
        assert detect_high_cost_procedure("27447") is True

    def test_hip_replacement(self):
        assert detect_high_cost_procedure("27130") is True

    def test_regular_office_visit(self):
        assert detect_high_cost_procedure("99213") is False

    def test_multiple_with_high_cost(self):
        assert detect_high_cost_procedure("99214|27447") is True

    def test_empty_string(self):
        assert detect_high_cost_procedure("") is False


class TestCPTCategories:
    def test_office_visit(self):
        assert get_cpt_category("99213") == "evaluation_management"

    def test_surgery(self):
        assert get_cpt_category("27447") == "surgery"

    def test_radiology(self):
        assert get_cpt_category("71046") == "radiology"

    def test_pathology(self):
        assert get_cpt_category("85025") == "pathology"

    def test_medicine(self):
        assert get_cpt_category("90834") == "medicine"

    def test_unknown_code(self):
        assert get_cpt_category("00000") == "other"

    def test_procedure_complexity(self):
        codes = ["99213", "99214", "71046", "27447"]
        result = get_procedure_complexity(codes)
        assert result["evaluation_management"] == 2
        assert result["radiology"] == 1
        assert result["surgery"] == 1


class TestProcedureCategoryFlags:
    def test_surgery_flag(self):
        flags = get_procedure_category_flags("27447")
        assert flags["has_surgery"] == 1
        assert flags["has_radiology"] == 0

    def test_mixed_procedures(self):
        flags = get_procedure_category_flags("99213|71046")
        assert flags["has_eval_mgmt"] == 1
        assert flags["has_radiology"] == 1
        assert flags["has_surgery"] == 0

    def test_empty_procedures(self):
        flags = get_procedure_category_flags("")
        assert all(v == 0 for v in flags.values())


class TestFeatureEngineering:
    def test_engineer_features_returns_all_columns(self):
        df = _make_claim_df()
        result = engineer_features(df)
        for col in FEATURE_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_log_billed_amount(self):
        df = _make_claim_df(billed_amount=285.00)
        result = engineer_features(df)
        expected = np.log1p(285.00)
        assert abs(result["log_billed_amount"].iloc[0] - expected) < 0.001

    def test_late_submission_flag(self):
        df = _make_claim_df(service_date="2024-01-01", submission_date="2024-05-01")
        result = engineer_features(df)
        assert result["late_submission"].iloc[0] == 1

    def test_not_late_submission(self):
        df = _make_claim_df(service_date="2024-01-01", submission_date="2024-01-15")
        result = engineer_features(df)
        assert result["late_submission"].iloc[0] == 0

    def test_pharmacy_claim_flag(self):
        df = _make_claim_df(claim_type="pharmacy")
        result = engineer_features(df)
        assert result["is_pharmacy_claim"].iloc[0] == 1
        assert result["is_dental_claim"].iloc[0] == 0

    def test_emergency_flag(self):
        df = _make_claim_df(place_of_service="23")
        result = engineer_features(df)
        assert result["is_emergency"].iloc[0] == 1

    def test_inpatient_flag(self):
        df = _make_claim_df(place_of_service="21")
        result = engineer_features(df)
        assert result["is_inpatient"].iloc[0] == 1

    def test_get_model_input_shape(self):
        df = _make_claim_df()
        X = get_model_input(df)
        assert X.shape[1] == len(FEATURE_COLUMNS)
        assert list(X.columns) == FEATURE_COLUMNS


class TestSingleClaimEngineering:
    def test_single_claim_features(self):
        claim = {
            "claim_id": "CLM-TEST-001",
            "claim_type": "medical",
            "place_of_service": "23",
            "diagnosis_codes": ["E11.9", "I10"],
            "procedure_codes": ["99214"],
            "billed_amount": 285.00,
            "allowed_amount": 195.00,
            "service_date": "2024-06-01",
            "submission_date": "2024-06-10",
            "member_age": 55,
            "member_gender": "M",
            "days_supply": None,
        }
        features = engineer_single_claim(claim)

        assert features["is_emergency"] == 1
        assert features["has_chronic_condition"] == 1
        assert features["submission_lag_days"] == 9
        assert features["member_age"] == 55
        assert features["is_male"] == 1

    def test_single_claim_has_all_features(self):
        claim = {
            "claim_type": "medical",
            "diagnosis_codes": ["M54.5"],
            "procedure_codes": ["99213"],
            "billed_amount": 150.00,
            "service_date": "2024-06-01",
            "submission_date": "2024-06-05",
        }
        features = engineer_single_claim(claim)
        for col in FEATURE_COLUMNS:
            assert col in features, f"Missing feature: {col}"
