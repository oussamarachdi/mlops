from zenml import pipeline, step
import pandas as pd
import joblib
import os
import optuna
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src.config import RANDOM_STATE, TEST_SIZE, MODEL_PATH, MLFLOW_EXPERIMENT_NAME

@step(enable_cache=False)
def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv("data/raw/breast_cancer.csv")
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y

@step(experiment_tracker="mlflow_tracker", enable_cache=False)
def hp_tuning_step(X: pd.DataFrame, y: pd.Series) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    def objective(trial):
        params = {
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
            "solver": trial.suggest_categorical("solver", ["liblinear", "lbfgs"])
        }
        
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=params["C"], 
                solver=params["solver"], 
                max_iter=2000, 
                random_state=RANDOM_STATE
            ))
        ])
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        f1 = float(f1_score(y_test, preds))
        
        return f1

    print("Starting Optuna optimization...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    
    print("Best params:", study.best_params)
    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_f1", study.best_value)
    
    return study.best_params

@step(experiment_tracker="mlflow_tracker", enable_cache=False)
def train_best_model(X: pd.DataFrame, y: pd.Series, best_params: dict) -> object:
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=best_params["C"], 
            solver=best_params["solver"], 
            max_iter=2000, 
            random_state=RANDOM_STATE
        ))
    ])
    model.fit(X_train, y_train)
    
    # Log final parameters
    mlflow.log_params(best_params)
    mlflow.log_param("model_type", "tuned_logistic_regression")
        
    return model

@step(experiment_tracker="mlflow_tracker", enable_cache=False)
def evaluate_model(model: object, X: pd.DataFrame, y: pd.Series) -> float:
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    preds = model.predict(X_test)
    score = float(f1_score(y_test, preds))
    
    # Log metric to active MLflow run managed by ZenML
    mlflow.log_metric("final_f1_score", score)
    
    return score

@step(enable_cache=False)
def export_model(model: object) -> str:
    try:
        # Get the project root
        project_root = os.getcwd()
        abs_model_path = os.path.join(project_root, MODEL_PATH)
        
        print(f"Attempting to export tuned model to: {abs_model_path}")
        os.makedirs(os.path.dirname(abs_model_path), exist_ok=True)
        joblib.dump(model, abs_model_path)
        print(f"Model successfully exported to {abs_model_path}")
        return abs_model_path
    except Exception as e:
        print(f"Error exporting model: {e}")
        raise e

@pipeline
def tuning_pipeline():
    X, y = load_data()
    best_params = hp_tuning_step(X, y)
    model = train_best_model(X, y, best_params)
    score = evaluate_model(model, X, y)
    _ = export_model(model)
    return score

if __name__ == "__main__":
    result = tuning_pipeline()
    print(f"Tuning pipeline completed! Final F1 Score: {result}")
