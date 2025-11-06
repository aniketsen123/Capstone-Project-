import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import mlflow
import mlflow.sklearn
import dagshub
import os
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from src.logger import logging


# -------------------- DAGSHUB / MLFLOW SETUP --------------------
mlflow.set_tracking_uri('https://dagshub.com/aniketsen123/Capstone-Project-.mlflow')
dagshub.init(repo_owner='aniketsen123', repo_name='Capstone-Project-', mlflow=True)

# ---------------------------------------------------------------
def ensure_experiment(experiment_name: str) -> None:
    """
    Ensures that the MLflow experiment exists.
    If deleted, restores it; if missing, creates it.
    """
    client = MlflowClient()

    # 1️⃣ Check if it already exists
    existing = client.get_experiment_by_name(experiment_name)
    if existing:
        # If it's active, just set it
        if existing.lifecycle_stage == "active":
            mlflow.set_experiment(experiment_name)
            logging.info(f"Using existing active experiment: {experiment_name}")
            return
        # If deleted, restore it
        else:
            client.restore_experiment(existing.experiment_id)
            mlflow.set_experiment(experiment_name)
            logging.info(f"Restored deleted experiment: {experiment_name}")
            return

    # 2️⃣ Check deleted-only experiments (edge case)
    deleted_experiments = client.list_experiments(view_type=ViewType.DELETED_ONLY)
    for exp in deleted_experiments:
        if exp.name == experiment_name:
            client.restore_experiment(exp.experiment_id)
            mlflow.set_experiment(experiment_name)
            logging.info(f"Restored deleted experiment: {experiment_name}")
            return

    # 3️⃣ Create a new one if nothing found
    client.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    logging.info(f"Created new experiment: {experiment_name}")


# -------------------- HELPER FUNCTIONS --------------------
def load_model(file_path: str):
    """Load the trained model from a file."""
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    logging.info(f'Model loaded from {file_path}')
    return model


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    df = pd.read_csv(file_path)
    logging.info(f'Data loaded from {file_path}')
    return df


def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    metrics_dict = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba)
    }
    logging.info('Model evaluation metrics calculated')
    return metrics_dict


def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    with open(file_path, 'w') as file:
        json.dump(metrics, file, indent=4)
    logging.info(f'Metrics saved to {file_path}')


def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    model_info = {'run_id': run_id, 'model_path': model_path}
    with open(file_path, 'w') as file:
        json.dump(model_info, file, indent=4)
    logging.debug(f'Model info saved to {file_path}')


# -------------------- MAIN FUNCTION --------------------
def main():
    ensure_experiment("my-dvc-pipeline")

    with mlflow.start_run() as run:
        clf = load_model('./models/model.pkl')
        test_data = load_data('./data/processed/test_bow.csv')

        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics = evaluate_model(clf, X_test, y_test)
        save_metrics(metrics, 'reports/metrics.json')

        # ✅ Log each metric individually
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # ✅ Log model parameters
        if hasattr(clf, 'get_params'):
            params = clf.get_params()
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)

        # ✅ Log model and files
        mlflow.sklearn.log_model(clf, "model")
        mlflow.log_artifact('reports/metrics.json')
        save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')

        logging.info("Model evaluation logged successfully.")


if __name__ == '__main__':
    main()
