"""
Training pipeline for claims denial prediction models.

Trains 4 models, logs everything to MLflow, and registers the best one.
XGBoost slightly beats LightGBM on this data — I think the categorical
encoding matters more than speed here. But LightGBM trains 3x faster,
which matters for the weekly retrain pipeline.
"""

import json
import os
import time
import warnings
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.features.engineering import FEATURE_COLUMNS, get_model_input

warnings.filterwarnings("ignore", category=UserWarning)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
MODEL_REGISTRY_NAME = os.getenv("MODEL_REGISTRY_NAME", "claims_denial_model")
DATA_PATH = os.getenv("CLAIMS_DATA_PATH", "data/synthetic_claims.csv")


def load_data(data_path: str = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    claim_df = pd.read_csv(data_path)
    print(f"Loaded {len(claim_df):,} claims from {data_path}")
    print(f"Denial rate: {claim_df['is_denied'].mean():.1%}")

    X = get_model_input(claim_df)
    y = claim_df["is_denied"]
    return X, y


def get_models(pos_weight: float) -> dict:
    """
    Initialize all candidate models.

    scale_pos_weight handles the class imbalance (~25% denial rate).
    I've found that class weights work better than oversampling for this
    problem — SMOTE introduced too many near-boundary synthetic samples
    that confused the tree splits.
    """
    return {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            C=1.0,
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
    }


def cross_validate_model(model, X: pd.DataFrame, y: pd.Series, n_folds: int = 5) -> dict:
    """
    5-fold stratified CV. Returns mean and std of all metrics.

    Stratified because the 75/25 split needs to be preserved in each fold,
    otherwise the model sees different class distributions during training.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    metrics = {"auc": [], "precision": [], "recall": [], "f1": [], "accuracy": []}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        metrics["auc"].append(roc_auc_score(y_val, y_proba))
        metrics["precision"].append(precision_score(y_val, y_pred))
        metrics["recall"].append(recall_score(y_val, y_pred))
        metrics["f1"].append(f1_score(y_val, y_pred))
        metrics["accuracy"].append(accuracy_score(y_val, y_pred))

    return {k: {"mean": np.mean(v), "std": np.std(v)} for k, v in metrics.items()}


def log_feature_importance(model, feature_names: list[str], model_name: str):
    """Extract and log feature importance as a CSV artifact."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    artifact_path = Path("artifacts") / f"{model_name}_feature_importance.csv"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(artifact_path, index=False)
    mlflow.log_artifact(str(artifact_path))

    return importance_df


def train_all_models(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Train all models, log to MLflow, return results.

    Each model gets its own MLflow run nested under an experiment.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("claims_denial_prediction")

    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    pos_weight = neg_count / pos_count
    print(f"Class balance: {neg_count:,} negative / {pos_count:,} positive "
          f"(weight={pos_weight:.2f})")

    models = get_models(pos_weight)
    results = {}
    best_auc = 0
    best_model_name = None

    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name}...")
        print(f"{'='*60}")

        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(model.get_params())
            mlflow.log_param("n_features", X.shape[1])
            mlflow.log_param("n_samples", X.shape[0])
            mlflow.log_param("pos_class_ratio", pos_count / len(y))

            start_time = time.time()
            cv_results = cross_validate_model(model, X, y)
            training_time = time.time() - start_time

            # Final fit on full data for the model artifact
            model.fit(X, y)

            for metric_name, values in cv_results.items():
                mlflow.log_metric(f"cv_{metric_name}_mean", values["mean"])
                mlflow.log_metric(f"cv_{metric_name}_std", values["std"])
            mlflow.log_metric("training_time_seconds", training_time)

            log_feature_importance(model, FEATURE_COLUMNS, model_name)

            report = classification_report(y, model.predict(X), output_dict=True)
            report_path = Path("artifacts") / f"{model_name}_classification_report.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            mlflow.log_artifact(str(report_path))

            if model_name in ("XGBoost",):
                mlflow.xgboost.log_model(model, model_name)
            else:
                mlflow.sklearn.log_model(model, model_name)

            mean_auc = cv_results["auc"]["mean"]
            print(f"  AUC: {mean_auc:.4f} (+/- {cv_results['auc']['std']:.4f})")
            print(f"  Precision: {cv_results['precision']['mean']:.4f}")
            print(f"  Recall: {cv_results['recall']['mean']:.4f}")
            print(f"  F1: {cv_results['f1']['mean']:.4f}")
            print(f"  Time: {training_time:.1f}s")

            if mean_auc > best_auc:
                best_auc = mean_auc
                best_model_name = model_name

            results[model_name] = {
                "model": model,
                "cv_results": cv_results,
                "training_time": training_time,
            }

    print(f"\n{'='*60}")
    print(f"Best model: {best_model_name} (AUC={best_auc:.4f})")
    print(f"{'='*60}")

    # Register the best model
    best_model = results[best_model_name]["model"]
    with mlflow.start_run(run_name=f"best_model_{best_model_name}"):
        mlflow.log_param("model_type", best_model_name)
        for metric_name, values in results[best_model_name]["cv_results"].items():
            mlflow.log_metric(f"cv_{metric_name}_mean", values["mean"])

        if best_model_name == "XGBoost":
            mlflow.xgboost.log_model(
                best_model, "model",
                registered_model_name=MODEL_REGISTRY_NAME,
            )
        else:
            mlflow.sklearn.log_model(
                best_model, "model",
                registered_model_name=MODEL_REGISTRY_NAME,
            )

    # Also save locally for the API to pick up without MLflow
    import joblib
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_dir / "best_model.pkl")
    print(f"Saved best model to {model_dir / 'best_model.pkl'}")

    return results


def main():
    X, y = load_data()
    results = train_all_models(X, y)

    print("\n\nFinal Summary:")
    print("-" * 70)
    print(f"{'Model':<25} {'AUC':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Time':>8}")
    print("-" * 70)
    for name, res in results.items():
        cv = res["cv_results"]
        print(f"{name:<25} {cv['auc']['mean']:>8.4f} {cv['precision']['mean']:>8.4f} "
              f"{cv['recall']['mean']:>8.4f} {cv['f1']['mean']:>8.4f} "
              f"{res['training_time']:>7.1f}s")


if __name__ == "__main__":
    main()
