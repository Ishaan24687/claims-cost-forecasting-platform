"""
Pydantic models for API request/response validation.

Validators enforce healthcare-specific constraints — NPI format, ICD-10 pattern,
positive amounts. These catch a lot of garbage before it ever hits the model.
"""

import re
from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ClaimInput(BaseModel):
    claim_id: str = Field(..., examples=["CLM-2025-00001"])
    member_id: str = Field(..., examples=["MBR-100234"])
    provider_npi: str = Field(..., min_length=10, max_length=10, examples=["1234567890"])
    claim_type: str = Field(..., examples=["medical"])
    place_of_service: str = Field(default="11", examples=["11", "21", "23"])
    diagnosis_codes: list[str] = Field(..., min_length=1, examples=[["E11.9", "I10"]])
    procedure_codes: list[str] = Field(default_factory=list, examples=[["99214"]])
    billed_amount: float = Field(..., gt=0, examples=[285.00])
    allowed_amount: Optional[float] = Field(None, ge=0, examples=[195.00])
    service_date: date = Field(..., examples=["2025-01-10"])
    submission_date: Optional[date] = Field(None, examples=["2025-01-12"])
    drug_ndc: Optional[str] = Field(None, examples=["00002323301"])
    days_supply: Optional[int] = Field(None, ge=0, examples=[30])
    member_age: Optional[int] = Field(None, ge=0, le=120, examples=[55])
    member_gender: Optional[str] = Field(None, examples=["M"])
    provider_specialty: Optional[str] = Field(None, examples=["internal_medicine"])

    @field_validator("provider_npi")
    @classmethod
    def validate_npi(cls, v: str) -> str:
        if not v.isdigit() or len(v) != 10:
            raise ValueError("NPI must be exactly 10 digits")
        return v

    @field_validator("diagnosis_codes")
    @classmethod
    def validate_icd10(cls, codes: list[str]) -> list[str]:
        icd10_pattern = re.compile(r"^[A-Z]\d{2}(\.\d{1,4})?$")
        for code in codes:
            if not icd10_pattern.match(code):
                raise ValueError(
                    f"Invalid ICD-10 code: {code}. "
                    f"Expected format like E11.9 or I10"
                )
        return codes

    @field_validator("procedure_codes")
    @classmethod
    def validate_cpt(cls, codes: list[str]) -> list[str]:
        cpt_pattern = re.compile(r"^\d{5}$|^[A-Z]\d{4}$|^D\d{4}$")
        for code in codes:
            if not cpt_pattern.match(code):
                raise ValueError(
                    f"Invalid procedure code: {code}. "
                    f"Expected 5-digit CPT or HCPCS format"
                )
        return codes

    @field_validator("claim_type")
    @classmethod
    def validate_claim_type(cls, v: str) -> str:
        allowed = {"medical", "pharmacy", "dental"}
        if v.lower() not in allowed:
            raise ValueError(f"claim_type must be one of {allowed}")
        return v.lower()

    @field_validator("submission_date")
    @classmethod
    def validate_submission_after_service(cls, v, info):
        if v and "service_date" in info.data and info.data["service_date"]:
            if v < info.data["service_date"]:
                raise ValueError("submission_date cannot be before service_date")
        return v


class ClaimPrediction(BaseModel):
    claim_id: str
    denial_probability: float = Field(..., ge=0, le=1)
    predicted_denied: bool
    denial_reasons: list[str] = Field(default_factory=list)
    estimated_reimbursement: float
    confidence: str = Field(..., examples=["high", "medium", "low"])
    model_version: str


class BatchRequest(BaseModel):
    claims: list[ClaimInput] = Field(..., min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: list[ClaimPrediction]
    total_claims: int
    total_predicted_denied: int
    batch_denial_rate: float
    processing_time_ms: float


class HealthCheck(BaseModel):
    status: str = "healthy"
    model_loaded: bool = False
    model_version: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
