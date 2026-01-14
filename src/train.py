import os
import argparse
import joblib
import pandas as pd
import optuna
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from .config import RANDOM_STATE, TEST_SIZE, MODEL_PATH
from .mlflow_utils import setup_mlflow

def load_data():
    df = pd.read_csv("data/raw/breast_cancer.csv")
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y

def build_model(params):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=params.get("C", 1.0),
            solver=params.get("solver", "liblinear"),
            max_iter=2000,
            random_state=RANDOM_STATE
        ))
    ])

def log_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    
    filename = "confusion_matrix.png"
    plt.savefig(filename)
    plt.close()
    
    mlflow.log_artifact(filename)
    if os.path.exists(filename):
        os.remove(filename)

def train_and_evaluate(params, run_name, version_tag):
    setup_mlflow()
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_param("version", version_tag)

        model = build_model(params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_metrics({"f1": f1, "accuracy": acc})
        
        # Log confusion matrix
        log_confusion_matrix(y_test, y_pred)
        
        # Log model to MLflow (MinIO)
        mlflow.sklearn.log_model(model, "model")

        # Save local artifacts
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        # Versioned model
        v_path = f"models/exported/model_{version_tag}.pkl"
        joblib.dump(model, v_path)
        print(f"Saved {version_tag} model to {v_path}")
        
        # Active model (symlink simulation or copy) for API
        joblib.dump(model, MODEL_PATH)
        print(f"Updated active model at {MODEL_PATH}")
        
        mlflow.log_artifact(v_path, artifact_path="exported_models")

        return f1

def objective(trial):
    params = {
        "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
        "solver": trial.suggest_categorical("solver", ["liblinear", "lbfgs"])
    }
    
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        mlflow.log_params(params)
        
        X, y = load_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        model = build_model(params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        
        mlflow.log_metric("f1", f1)
        
        return f1

def run_optuna():
    setup_mlflow()
    print("Starting Optuna optimization...")
    study = optuna.create_study(direction="maximize")
    with mlflow.start_run(run_name="optuna_optimization"):
        # Log study info in a parent run
        study.optimize(objective, n_trials=10)
    
    print("Best params:", study.best_params)
    return study.best_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["v1", "v2"], required=True, help="v1 for baseline, v2 for optuna tuned")
    args = parser.parse_args()

    # Ensure data directory exists or is accessible
    if not os.path.exists("data/raw/breast_cancer.csv"):
        print("Error: data/raw/breast_cancer.csv not found!")
        exit(1)

    if args.mode == "v1":
        print("Training Baseline (v1)...")
        train_and_evaluate(params={"C": 1.0, "solver": "liblinear"}, run_name="baseline_v1", version_tag="v1")
    
    elif args.mode == "v2":
        print("Tuning with Optuna (v2)...")
        best_params = run_optuna()
        print("Training Best Model (v2)...")
        train_and_evaluate(params=best_params, run_name="optuna_best_v2", version_tag="v2")
