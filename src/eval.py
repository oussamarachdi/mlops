import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

import mlflow
from .mlflow_utils import setup_mlflow
from .config import MODEL_PATH

def load_test_split():
    df = pd.read_csv("data/raw/breast_cancer.csv")
    X = df.drop(columns=["target"])
    y = df["target"]
    # simple deterministic split for evaluation demo (same as train.py)
    from sklearn.model_selection import train_test_split
    from .config import TEST_SIZE, RANDOM_STATE
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_test, y_test

def evaluate(run_name: str = "eval"):
    setup_mlflow()
    model = joblib.load(MODEL_PATH)
    X_test, y_test = load_test_split()

    y_pred = model.predict(X_test)

    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    os.makedirs("artifacts", exist_ok=True)
    fig_path = "artifacts/confusion_matrix.png"
    plt.figure()
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()

    with mlflow.start_run(run_name=run_name):
        mlflow.log_metric("f1", float(f1))
        mlflow.log_metric("accuracy", float(acc))
        mlflow.log_artifact(fig_path, artifact_path="figures")

    print({"f1": f1, "accuracy": acc})

if __name__ == "__main__":
    evaluate(run_name="eval_exported_model")
