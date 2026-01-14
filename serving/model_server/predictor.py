import os
import joblib
import numpy as np

MODEL_PATH = os.getenv("MODEL_PATH", "models/exported/model.pkl")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

_model = None

def load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model

def predict(features: list[float]) -> dict:
    model = load_model()

    x = np.array(features, dtype=float).reshape(1, -1)

    proba = float(model.predict_proba(x)[0, 1])
    prediction = int(proba >= 0.5)

    return {
        "prediction": prediction,
        "proba": proba,
        "model_version": MODEL_VERSION
    }
