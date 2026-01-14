from zenml import pipeline, step
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src.config import RANDOM_STATE, TEST_SIZE, MODEL_PATH

@step
def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv("data/raw/breast_cancer.csv")
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y

@step
def train_model(X: pd.DataFrame, y: pd.Series, C: float = 1.0) -> object:
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, solver="liblinear", max_iter=2000, random_state=RANDOM_STATE))
    ])
    model.fit(X_train, y_train)
    return model

@step
def evaluate_model(model: object, X: pd.DataFrame, y: pd.Series) -> float:
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    preds = model.predict(X_test)
    return float(f1_score(y_test, preds))

@step
def export_model(model: object) -> str:
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return MODEL_PATH

@pipeline
def bc_pipeline():
    X, y = load_data()
    model = train_model(X, y)
    score = evaluate_model(model, X, y)
    _ = export_model(model)
    return score

if __name__ == "__main__":
    bc_pipeline()
