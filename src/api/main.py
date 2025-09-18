"""
FastAPI application for claims denial prediction.

The API is designed for two use cases:
1. Real-time single claim scoring (latency matters, ~50ms target)
2. Batch scoring for daily pipeline runs (throughput matters, 1000 claims/request)
"""

import time
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.data.schemas import (
    BatchRequest,
    BatchResponse,
    ClaimInput,
    ClaimPrediction,
    HealthCheck,
)
from src.models.predict import (
    MODEL_VERSION,
    get_model,
    load_model,
    predict_single,
)
from src.models.explain import explain_prediction


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup instead of on every request."""
    try:
        load_model()
        print(f"Model loaded successfully (version {MODEL_VERSION})")
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        print("API will start but predictions will fail until model is trained.")
    yield


app = FastAPI(
    title="Claims Cost Forecasting API",
    description=(
        "ML-powered API for predicting healthcare claim denials. "
        "Serves an XGBoost model trained on 50K+ claims with CPT, ICD-10, "
        "and NDC code features."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthCheck)
async def health_check():
    model = get_model()
    return HealthCheck(
        status="healthy",
        model_loaded=model is not None,
        model_version=MODEL_VERSION if model else None,
        timestamp=datetime.utcnow(),
    )


@app.post("/predict", response_model=ClaimPrediction)
async def predict_claim(claim: ClaimInput):
    """Predict denial probability for a single healthcare claim."""
    model = get_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model first with 'make train'.",
        )

    claim_dict = claim.model_dump()
    # Convert date objects to strings for the feature engineering pipeline
    claim_dict["service_date"] = str(claim_dict["service_date"])
    if claim_dict.get("submission_date"):
        claim_dict["submission_date"] = str(claim_dict["submission_date"])

    result = predict_single(claim_dict)
    return ClaimPrediction(**result)


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(request: BatchRequest):
    """Score a batch of claims (up to 1000)."""
    model = get_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model first with 'make train'.",
        )

    start_time = time.time()

    predictions = []
    for claim in request.claims:
        claim_dict = claim.model_dump()
        claim_dict["service_date"] = str(claim_dict["service_date"])
        if claim_dict.get("submission_date"):
            claim_dict["submission_date"] = str(claim_dict["submission_date"])

        result = predict_single(claim_dict)
        predictions.append(ClaimPrediction(**result))

    processing_time_ms = (time.time() - start_time) * 1000
    total_denied = sum(1 for p in predictions if p.predicted_denied)

    return BatchResponse(
        predictions=predictions,
        total_claims=len(predictions),
        total_predicted_denied=total_denied,
        batch_denial_rate=total_denied / len(predictions) if predictions else 0.0,
        processing_time_ms=round(processing_time_ms, 2),
    )


@app.post("/explain")
async def explain_claim(claim: ClaimInput):
    """Get SHAP explanation for a claim prediction."""
    model = get_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model first with 'make train'.",
        )

    claim_dict = claim.model_dump()
    claim_dict["service_date"] = str(claim_dict["service_date"])
    if claim_dict.get("submission_date"):
        claim_dict["submission_date"] = str(claim_dict["submission_date"])

    try:
        explanation = explain_prediction(claim_dict)
        return explanation
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"SHAP explanation failed: {str(e)}",
        )


@app.get("/model/info")
async def model_info():
    """Return current model metadata and feature configuration."""
    from src.features.engineering import FEATURE_COLUMNS

    model = get_model()
    model_type = type(model).__name__ if model else None

    return {
        "model_version": MODEL_VERSION,
        "model_type": model_type,
        "model_loaded": model is not None,
        "feature_columns": FEATURE_COLUMNS,
        "n_features": len(FEATURE_COLUMNS),
        "prediction_threshold": 0.5,
        "supported_claim_types": ["medical", "pharmacy", "dental"],
    }
