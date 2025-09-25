"""
Tests for the FastAPI endpoints.

Uses TestClient so we don't need a running server. The model loading
is mocked for tests that need predictions — the actual model tests
live in test_features.py and the training pipeline tests.
"""

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


VALID_CLAIM = {
    "claim_id": "CLM-2025-00001",
    "member_id": "MBR-100234",
    "provider_npi": "1234567890",
    "claim_type": "medical",
    "place_of_service": "11",
    "diagnosis_codes": ["E11.9", "I10"],
    "procedure_codes": ["99214"],
    "billed_amount": 285.00,
    "allowed_amount": 195.00,
    "service_date": "2025-01-10",
    "submission_date": "2025-01-12",
}


class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "timestamp" in data
        assert data["status"] == "healthy"


class TestModelInfoEndpoint:
    def test_model_info_returns_200(self):
        response = client.get("/model/info")
        assert response.status_code == 200

    def test_model_info_has_features(self):
        response = client.get("/model/info")
        data = response.json()
        assert "feature_columns" in data
        assert "n_features" in data
        assert data["n_features"] > 0

    def test_model_info_has_supported_types(self):
        response = client.get("/model/info")
        data = response.json()
        assert "medical" in data["supported_claim_types"]
        assert "pharmacy" in data["supported_claim_types"]


class TestPredictEndpoint:
    @patch("src.api.main.get_model")
    @patch("src.api.main.predict_single")
    def test_predict_valid_claim(self, mock_predict, mock_get_model):
        mock_get_model.return_value = MagicMock()
        mock_predict.return_value = {
            "claim_id": "CLM-2025-00001",
            "denial_probability": 0.23,
            "predicted_denied": False,
            "denial_reasons": [],
            "estimated_reimbursement": 195.00,
            "confidence": "high",
            "model_version": "1.2.0",
        }

        response = client.post("/predict", json=VALID_CLAIM)
        assert response.status_code == 200
        data = response.json()
        assert data["claim_id"] == "CLM-2025-00001"
        assert 0 <= data["denial_probability"] <= 1
        assert isinstance(data["predicted_denied"], bool)

    def test_predict_returns_503_without_model(self):
        with patch("src.api.main.get_model", return_value=None):
            response = client.post("/predict", json=VALID_CLAIM)
            assert response.status_code == 503


class TestValidation:
    def test_invalid_npi_too_short(self):
        bad_claim = VALID_CLAIM.copy()
        bad_claim["provider_npi"] = "12345"
        response = client.post("/predict", json=bad_claim)
        assert response.status_code == 422

    def test_invalid_npi_with_letters(self):
        bad_claim = VALID_CLAIM.copy()
        bad_claim["provider_npi"] = "123456789A"
        response = client.post("/predict", json=bad_claim)
        assert response.status_code == 422

    def test_invalid_icd10_code(self):
        bad_claim = VALID_CLAIM.copy()
        bad_claim["diagnosis_codes"] = ["INVALID"]
        response = client.post("/predict", json=bad_claim)
        assert response.status_code == 422

    def test_negative_billed_amount(self):
        bad_claim = VALID_CLAIM.copy()
        bad_claim["billed_amount"] = -100.00
        response = client.post("/predict", json=bad_claim)
        assert response.status_code == 422

    def test_zero_billed_amount(self):
        bad_claim = VALID_CLAIM.copy()
        bad_claim["billed_amount"] = 0
        response = client.post("/predict", json=bad_claim)
        assert response.status_code == 422

    def test_empty_diagnosis_codes(self):
        bad_claim = VALID_CLAIM.copy()
        bad_claim["diagnosis_codes"] = []
        response = client.post("/predict", json=bad_claim)
        assert response.status_code == 422

    def test_invalid_claim_type(self):
        bad_claim = VALID_CLAIM.copy()
        bad_claim["claim_type"] = "vision"
        response = client.post("/predict", json=bad_claim)
        assert response.status_code == 422

    def test_valid_pharmacy_claim(self):
        pharmacy_claim = VALID_CLAIM.copy()
        pharmacy_claim["claim_type"] = "pharmacy"
        pharmacy_claim["drug_ndc"] = "00002323301"
        pharmacy_claim["days_supply"] = 30
        pharmacy_claim["procedure_codes"] = []
        with patch("src.api.main.get_model", return_value=MagicMock()), \
             patch("src.api.main.predict_single", return_value={
                 "claim_id": "CLM-2025-00001",
                 "denial_probability": 0.15,
                 "predicted_denied": False,
                 "denial_reasons": [],
                 "estimated_reimbursement": 195.00,
                 "confidence": "high",
                 "model_version": "1.2.0",
             }):
            response = client.post("/predict", json=pharmacy_claim)
            assert response.status_code == 200


class TestBatchEndpoint:
    @patch("src.api.main.get_model")
    @patch("src.api.main.predict_single")
    def test_batch_predict(self, mock_predict, mock_get_model):
        mock_get_model.return_value = MagicMock()
        mock_predict.return_value = {
            "claim_id": "CLM-2025-00001",
            "denial_probability": 0.23,
            "predicted_denied": False,
            "denial_reasons": [],
            "estimated_reimbursement": 195.00,
            "confidence": "high",
            "model_version": "1.2.0",
        }

        batch = {"claims": [VALID_CLAIM, VALID_CLAIM]}
        response = client.post("/predict/batch", json=batch)
        assert response.status_code == 200
        data = response.json()
        assert data["total_claims"] == 2
        assert len(data["predictions"]) == 2

    def test_empty_batch_rejected(self):
        response = client.post("/predict/batch", json={"claims": []})
        assert response.status_code == 422
