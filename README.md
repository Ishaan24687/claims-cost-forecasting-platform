# Claims Cost Forecasting Platform

End-to-end ML pipeline for predicting healthcare claim denials and forecasting costs across medical, pharmacy, and dental claims.

## Why I Built This

At Lantern Care I process 500K+ healthcare claims monthly through our PBM adjudication pipeline. The biggest pain point I kept running into was claim denials — they cost us time, money, and create friction with providers. About 25% of claims get denied on first submission, and most of those denials are predictable if you look at the right signals.

I built this platform to catch likely denials *before* submission. The system ingests raw claims data (CPT codes, ICD-10 diagnoses, NDC drug codes, provider info), engineers features that capture the patterns I've seen in real adjudication — things like submission lag, chronic condition flags, provider historical denial rates — and runs them through an XGBoost model that hits 87% AUC on our test set. That accuracy maps to real savings: on $50M+ annual claims volume, even a 5% reduction in preventable denials is meaningful.

The whole thing is containerized with Docker, orchestrated through Airflow for daily scoring and weekly retraining, tracked with MLflow for reproducibility, and served through FastAPI with a Streamlit dashboard for the clinical ops team.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLAIMS DATA SOURCES                          │
│   Medical Claims │ Pharmacy Claims │ Dental Claims │ Provider Data  │
└────────┬─────────┴────────┬────────┴───────┬───────┴───────┬────────┘
         │                  │                │               │
         ▼                  ▼                ▼               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    AIRFLOW ORCHESTRATION                             │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ Daily Scoring │  │Weekly Retrain│  │  Monitoring   │              │
│  │   DAG (2AM)  │  │ DAG (Sun 3AM)│  │   Alerts     │              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
└─────────┼─────────────────┼─────────────────┼───────────────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     ML PIPELINE                                     │
│                                                                     │
│  ┌────────────┐  ┌───────────────┐  ┌────────────┐  ┌──────────┐  │
│  │   Data      │  │   Feature     │  │   Model    │  │  Model   │  │
│  │ Generation  │──│ Engineering   │──│  Training  │──│ Registry │  │
│  │ (50K claims)│  │ (CPT, ICD-10) │  │ (XGBoost)  │  │ (MLflow) │  │
│  └────────────┘  └───────────────┘  └────────────┘  └────┬─────┘  │
│                                                           │        │
│  ┌────────────┐  ┌───────────────┐  ┌────────────┐       │        │
│  │   SHAP     │  │  Evaluation   │  │   Drift    │       │        │
│  │ Explainer  │  │  (ROC, PR)    │  │ Detection  │       │        │
│  └────────────┘  └───────────────┘  └────────────┘       │        │
└──────────────────────────────────────────────────────────┼────────┘
                                                           │
                    ┌──────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     SERVING LAYER                                   │
│                                                                     │
│  ┌──────────────────────┐      ┌────────────────────────────────┐  │
│  │      FastAPI          │      │     Streamlit Dashboard        │  │
│  │  /predict             │      │  ┌──────────┐ ┌────────────┐  │  │
│  │  /predict/batch       │◄────►│  │ KPI      │ │ Claim      │  │  │
│  │  /explain             │      │  │ Overview │ │ Lookup     │  │  │
│  │  /model/info          │      │  └──────────┘ └────────────┘  │  │
│  │  /health              │      │  ┌──────────────────────────┐  │  │
│  └──────────────────────┘      │  │ Model Performance Trends │  │  │
│                                 │  └──────────────────────────┘  │  │
│                                 └────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Model Comparison

I trained four models and tracked everything in MLflow. XGBoost won, but LightGBM was close and trained 3x faster — in production I'd probably A/B test them.

| Model | AUC | Precision | Recall | F1 | Training Time |
|---|---|---|---|---|---|
| Logistic Regression | 0.72 | 0.68 | 0.61 | 0.64 | 2s |
| Random Forest | 0.81 | 0.76 | 0.74 | 0.75 | 15s |
| **XGBoost** | **0.87** | **0.82** | **0.79** | **0.80** | 45s |
| LightGBM | 0.86 | 0.81 | 0.78 | 0.79 | 12s |

## SHAP Feature Importance

The top predictors from SHAP analysis make clinical sense:

1. **submission_lag_days** — Claims submitted more than 90 days after service get denied at 3x the rate. This was the single strongest signal.
2. **log_billed_amount** — High-cost claims face more scrutiny. The relationship is nonlinear, which is why tree models beat logistic regression here.
3. **has_chronic_condition** — Patients with diabetes (E11.x), hypertension (I10), COPD (J44.x) have more complex claims that require better documentation.
4. **provider_historical_denial_rate** — Some providers consistently submit incomplete claims. A rolling 90-day denial rate per provider was a strong feature.
5. **is_emergency** — Emergency claims have different adjudication rules and lower denial rates.
6. **claim_type** — Pharmacy claims have different denial patterns than medical claims. NDC-based claims are more formulary-dependent.

The SHAP dependence plots showed interesting interactions — for example, high billed amounts combined with late submission are almost always denied. That interaction is hard to capture with linear models.

## Quick Start

### With Docker (recommended)

```bash
# Clone and start all services
git clone https://github.com/ishaan-gupta/claims-cost-forecasting-platform.git
cd claims-cost-forecasting-platform
cp .env.example .env
docker-compose up -d

# Generate training data and train the model
make generate-data
make train

# The dashboard is at http://localhost:8501
# The API is at http://localhost:8000/docs
# MLflow UI is at http://localhost:5001
```

### Local Development

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Generate synthetic claims data
python -m src.data.generate_claims

# Train models
python -m src.models.train

# Start the API
uvicorn src.api.main:app --reload --port 8000

# Start the dashboard (separate terminal)
streamlit run dashboard/streamlit_app.py
```

## API Endpoints

### `GET /health`
Health check with model status.

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.2.0",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### `POST /predict`
Predict denial probability for a single claim.

```json
// Request
{
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
  "submission_date": "2025-01-12"
}

// Response
{
  "claim_id": "CLM-2025-00001",
  "denial_probability": 0.23,
  "predicted_denied": false,
  "denial_reasons": [],
  "estimated_reimbursement": 195.00,
  "confidence": "high",
  "model_version": "1.2.0"
}
```

### `POST /predict/batch`
Score up to 1,000 claims at once.

### `POST /explain`
Get SHAP explanation for a prediction.

```json
// Response
{
  "claim_id": "CLM-2025-00001",
  "base_value": 0.25,
  "prediction": 0.23,
  "feature_contributions": {
    "submission_lag_days": -0.08,
    "log_billed_amount": 0.04,
    "has_chronic_condition": 0.03,
    "provider_historical_denial_rate": -0.02
  },
  "top_risk_factors": ["log_billed_amount", "has_chronic_condition"],
  "top_protective_factors": ["submission_lag_days", "provider_historical_denial_rate"]
}
```

### `GET /model/info`
Current model metadata and feature list.

## Dashboard

The Streamlit dashboard has three pages:

1. **Overview** — KPIs (total claims processed, current denial rate, average billed amount, model AUC), with trend charts
2. **Claim Lookup** — Search by claim ID, see the prediction, SHAP waterfall chart showing which features drove the decision
3. **Model Performance** — AUC/F1 over time, feature importance bar chart, confusion matrix

## Project Structure

```
claims-cost-forecasting-platform/
├── src/
│   ├── data/
│   │   ├── generate_claims.py     # Synthetic data generation (50K claims)
│   │   └── schemas.py             # Pydantic models with healthcare validators
│   ├── features/
│   │   ├── engineering.py         # Feature engineering pipeline
│   │   └── cpt_categories.py     # CPT code → clinical category mapping
│   ├── models/
│   │   ├── train.py              # MLflow training pipeline (4 models)
│   │   ├── evaluate.py           # ROC, PR curves, calibration plots
│   │   ├── predict.py            # Inference with denial reason logic
│   │   └── explain.py            # SHAP explanations
│   ├── api/
│   │   └── main.py               # FastAPI serving layer
│   └── monitoring/
│       └── drift_detection.py    # PSI-based feature drift detection
├── dags/
│   ├── daily_scoring_dag.py      # Airflow: daily batch scoring
│   └── weekly_retrain_dag.py     # Airflow: weekly model retraining
├── dashboard/
│   └── streamlit_app.py          # Multi-page Streamlit dashboard
├── tests/
│   ├── test_features.py          # Feature engineering tests
│   └── test_api.py               # API endpoint tests
├── docker-compose.yml            # 4-service stack
├── Dockerfile
├── Makefile
├── requirements.txt
└── README.md
```

## Known Issues / What I'd Improve

- **Provider specialty encoding**: At work this was the #3 predictor, but I can't share that proprietary encoding. The synthetic data approximates it, but the real relationship is stronger.
- **Real-time streaming**: Currently batch-oriented. I'd add Kafka for real-time claim scoring as they hit the adjudication system.
- **GBM hyperparameter tuning**: I used Optuna at work but kept it simple here with reasonable defaults. The gap between tuned and untuned XGBoost was about 2% AUC on our production data.
- **Multi-target modeling**: This predicts denial yes/no, but in production I also built a separate regression model for expected reimbursement amount. Would be good to add that here.
- **DRG grouper integration**: Inpatient claims use DRG codes for reimbursement, and I'd like to add the MS-DRG grouper logic. That's a separate project on its own.
- **CI/CD**: Would add GitHub Actions for automated testing and model validation on PR. The MLflow registry handles model versioning but the deployment step is manual.

## License

MIT
