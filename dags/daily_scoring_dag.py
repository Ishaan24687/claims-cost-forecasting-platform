"""
Airflow DAG: Daily batch scoring pipeline.

Runs at 2 AM every day:
1. Ingest new claims from the data source
2. Run feature engineering
3. Score claims with the production model
4. Store prediction results
5. Run drift monitoring

In production this would pull from an EDI 837 feed or a claims data warehouse.
Here it scores the synthetic data as a demo.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "ishaan",
    "depends_on_past": False,
    "email": ["ishaan@example.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


def ingest_claims(**context):
    """Pull new claims from the data source."""
    import pandas as pd

    claim_df = pd.read_csv("data/synthetic_claims.csv")

    # In production: filter to claims submitted since last run
    # Here we just take a random sample to simulate a daily batch
    batch_size = min(2000, len(claim_df))
    daily_batch = claim_df.sample(n=batch_size, random_state=datetime.now().day)

    daily_batch.to_csv("data/daily_batch.csv", index=False)
    print(f"Ingested {len(daily_batch)} claims for scoring")

    context["ti"].xcom_push(key="batch_size", value=len(daily_batch))


def run_feature_engineering(**context):
    """Engineer features for the daily batch."""
    import pandas as pd
    from src.features.engineering import get_model_input

    claim_df = pd.read_csv("data/daily_batch.csv")
    X = get_model_input(claim_df)

    X.to_csv("data/daily_features.csv", index=False)
    print(f"Engineered {X.shape[1]} features for {len(X)} claims")


def batch_scoring(**context):
    """Score the daily batch with the production model."""
    import pandas as pd
    from src.models.predict import predict_batch

    claim_df = pd.read_csv("data/daily_batch.csv")
    results = predict_batch(claim_df)

    results.to_csv("data/daily_predictions.csv", index=False)

    n_denied = results["predicted_denied"].sum()
    denial_rate = n_denied / len(results)
    print(f"Scored {len(results)} claims: {n_denied} predicted denied ({denial_rate:.1%})")

    context["ti"].xcom_push(key="n_denied", value=int(n_denied))
    context["ti"].xcom_push(key="denial_rate", value=float(denial_rate))


def store_results(**context):
    """Store predictions. In production this writes to a database."""
    import pandas as pd

    predictions = pd.read_csv("data/daily_predictions.csv")
    print(f"Stored {len(predictions)} predictions")

    # TODO: write to PostgreSQL or Snowflake in production
    # For now, just append to a running history file
    history_path = "data/prediction_history.csv"
    try:
        history = pd.read_csv(history_path)
        predictions["scored_date"] = datetime.now().strftime("%Y-%m-%d")
        history = pd.concat([history, predictions], ignore_index=True)
    except FileNotFoundError:
        predictions["scored_date"] = datetime.now().strftime("%Y-%m-%d")
        history = predictions

    history.to_csv(history_path, index=False)


def run_monitoring(**context):
    """Check for feature drift against training distribution."""
    import pandas as pd
    from src.features.engineering import get_model_input
    from src.monitoring.drift_detection import detect_drift, save_drift_report

    training_df = pd.read_csv("data/synthetic_claims.csv")
    scoring_df = pd.read_csv("data/daily_batch.csv")

    X_train = get_model_input(training_df)
    X_score = get_model_input(scoring_df)

    report = detect_drift(X_train, X_score)
    report_path = save_drift_report(report)

    print(f"Drift status: {report['overall_status']}")
    print(f"Features with drift: {report['features_with_drift']}")
    print(f"Report saved to {report_path}")

    if report["overall_status"] == "critical":
        # In production this would trigger a PagerDuty alert or Slack notification
        print("ALERT: Critical drift detected — consider triggering retrain")


with DAG(
    dag_id="daily_claims_scoring",
    default_args=default_args,
    description="Daily batch scoring pipeline for healthcare claims",
    schedule_interval="0 2 * * *",  # 2 AM daily
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["claims", "ml", "production"],
) as dag:

    t_ingest = PythonOperator(
        task_id="ingest_claims",
        python_callable=ingest_claims,
    )

    t_features = PythonOperator(
        task_id="feature_engineering",
        python_callable=run_feature_engineering,
    )

    t_score = PythonOperator(
        task_id="batch_scoring",
        python_callable=batch_scoring,
    )

    t_store = PythonOperator(
        task_id="store_results",
        python_callable=store_results,
    )

    t_monitor = PythonOperator(
        task_id="run_monitoring",
        python_callable=run_monitoring,
    )

    t_ingest >> t_features >> t_score >> t_store >> t_monitor
