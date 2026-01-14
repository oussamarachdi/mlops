import os
import mlflow
import logging
from .config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_mlflow():
    # Check for credentials
    if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
         logger.warning("AWS credentials for MinIO (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) are missing! Artifact logging may fail.")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger.info(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")
    logger.info(f"MLflow experiment set to: {MLFLOW_EXPERIMENT_NAME}")
