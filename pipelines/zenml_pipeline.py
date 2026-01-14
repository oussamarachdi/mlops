from zenml import pipeline, step
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import mlflow

from src.config import RANDOM_STATE, TEST_SIZE, MODEL_PATH, MLFLOW_EXPERIMENT_NAME

@step(enable_cache=False)
def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv("data/raw/breast_cancer.csv")
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y

@step(experiment_tracker="mlflow_tracker", enable_cache=False)
def train_model(X: pd.DataFrame, y: pd.Series, C: float = 1.0) -> object:
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, solver="liblinear", max_iter=2000, random_state=RANDOM_STATE))
    ])
    model.fit(X_train, y_train)
    
    # Log parameters - ZenML handles the run lifecycle
    mlflow.log_param("C", C)
    mlflow.log_param("solver", "liblinear")
    mlflow.log_param("max_iter", 2000)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("test_size", TEST_SIZE)
        
    return model

@step(experiment_tracker="mlflow_tracker", enable_cache=False)
def evaluate_model(model: object, X: pd.DataFrame, y: pd.Series) -> float:
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    preds = model.predict(X_test)
    score = float(f1_score(y_test, preds))
    
    # Log metric to active MLflow run managed by ZenML
    mlflow.log_metric("f1_score", score)
    
    return score

@step(enable_cache=False)
def export_model(model: object) -> str:
    try:
        # Get the project root (where the script is running from)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        abs_model_path = os.path.join(project_root, MODEL_PATH)
        
        print(f"Attempting to export model to: {abs_model_path}")
        os.makedirs(os.path.dirname(abs_model_path), exist_ok=True)
        joblib.dump(model, abs_model_path)
        print(f"Model successfully exported to {abs_model_path}")
        return abs_model_path
    except Exception as e:
        print(f"Error exporting model: {e}")
        raise e

@pipeline
def bc_pipeline():
    X, y = load_data()
    model = train_model(X, y)
    score = evaluate_model(model, X, y)
    path = export_model(model)
    return score

if __name__ == "__main__":
    result = bc_pipeline()
    print(f"Pipeline completed! Result: {result}")
