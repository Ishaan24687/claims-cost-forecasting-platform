"""
Airflow DAG: Weekly model retraining pipeline.

Runs every Sunday at 3 AM:
1. Load latest training data
2. Train a new model candidate
3. Validate: only deploy if AUC improved over current production model
4. Deploy or skip based on validation
5. Notify on completion

The validation step is critical — we had an incident at work where an
auto-retrained model degraded because a data feed was corrupted. Now
the pipeline always compares against the production baseline.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator

default_args = {
    "owner": "ishaan",
    "depends_on_past": False,
    "email": ["ishaan@example.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}


def load_training_data(**context):
    """Load and prepare training data."""
    import pandas as pd
    from src.features.engineering import get_model_input

    claim_df = pd.read_csv("data/synthetic_claims.csv")
    X = get_model_input(claim_df)
    y = claim_df["is_denied"]

    X.to_csv("data/retrain_features.csv", index=False)
    y.to_csv("data/retrain_labels.csv", index=False)

    print(f"Loaded {len(X)} samples with {X.shape[1]} features")
    print(f"Denial rate: {y.mean():.1%}")

    context["ti"].xcom_push(key="n_samples", value=len(X))


def train_model(**context):
    """Train a new model candidate."""
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import cross_val_score
    from xgboost import XGBClassifier
    import joblib

    X = pd.read_csv("data/retrain_features.csv")
    y = pd.read_csv("data/retrain_labels.csv").iloc[:, 0]

    pos_weight = (y == 0).sum() / (y == 1).sum()

    model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    cv_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
    candidate_auc = float(np.mean(cv_scores))
    print(f"Candidate model AUC: {candidate_auc:.4f} (+/- {np.std(cv_scores):.4f})")

    model.fit(X, y)
    joblib.dump(model, "models/candidate_model.pkl")

    context["ti"].xcom_push(key="candidate_auc", value=candidate_auc)


def validate_model(**context):
    """
    Compare candidate model against production model.

    Returns the task_id to branch to: deploy_model if improved, skip_deploy otherwise.
    """
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.metrics import roc_auc_score

    candidate_auc = context["ti"].xcom_pull(task_ids="train_model", key="candidate_auc")

    X = pd.read_csv("data/retrain_features.csv")
    y = pd.read_csv("data/retrain_labels.csv").iloc[:, 0]

    try:
        prod_model = joblib.load("models/best_model.pkl")
        prod_proba = prod_model.predict_proba(X)[:, 1]
        prod_auc = roc_auc_score(y, prod_proba)
    except FileNotFoundError:
        prod_auc = 0.0
        print("No production model found — candidate will be deployed")

    improvement = candidate_auc - prod_auc
    print(f"Production AUC: {prod_auc:.4f}")
    print(f"Candidate AUC:  {candidate_auc:.4f}")
    print(f"Improvement:    {improvement:+.4f}")

    # Only deploy if there's meaningful improvement (0.5% AUC threshold)
    if improvement > 0.005:
        print("DEPLOYING: Candidate model is better")
        return "deploy_model"
    else:
        print("SKIPPING: Candidate model is not significantly better")
        return "skip_deploy"


def deploy_model(**context):
    """Promote candidate model to production."""
    import shutil
    from pathlib import Path

    candidate_path = Path("models/candidate_model.pkl")
    production_path = Path("models/best_model.pkl")

    # Keep a backup of the current production model
    if production_path.exists():
        backup_path = Path("models") / f"backup_{datetime.now().strftime('%Y%m%d')}.pkl"
        shutil.copy2(production_path, backup_path)
        print(f"Backed up current model to {backup_path}")

    shutil.copy2(candidate_path, production_path)
    print(f"Deployed candidate model to {production_path}")

    # TODO: in production, also update the MLflow model registry stage
    # and trigger a rolling restart of the API containers


def skip_deploy(**context):
    """Log that deployment was skipped."""
    print("Deployment skipped — current production model retained")


def notify(**context):
    """Send notification about retrain results."""
    candidate_auc = context["ti"].xcom_pull(task_ids="train_model", key="candidate_auc")

    # In production this would send to Slack or email
    # Here we just print the summary
    print(f"\n{'='*50}")
    print("Weekly Retrain Complete")
    print(f"{'='*50}")
    print(f"Candidate AUC: {candidate_auc:.4f}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"{'='*50}")


with DAG(
    dag_id="weekly_model_retrain",
    default_args=default_args,
    description="Weekly model retraining with validation gate",
    schedule_interval="0 3 * * 0",  # 3 AM every Sunday
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["claims", "ml", "retrain"],
) as dag:

    t_load = PythonOperator(
        task_id="load_training_data",
        python_callable=load_training_data,
    )

    t_train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    t_validate = BranchPythonOperator(
        task_id="validate_model",
        python_callable=validate_model,
    )

    t_deploy = PythonOperator(
        task_id="deploy_model",
        python_callable=deploy_model,
    )

    t_skip = PythonOperator(
        task_id="skip_deploy",
        python_callable=skip_deploy,
    )

    t_notify = PythonOperator(
        task_id="notify",
        python_callable=notify,
        trigger_rule="none_failed_min_one_success",
    )

    t_load >> t_train >> t_validate >> [t_deploy, t_skip] >> t_notify
