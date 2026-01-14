import joblib
import numpy as np
from .config import MODEL_PATH, MODEL_VERSION

def load_model():
    return joblib.load(MODEL_PATH)

def predict(features: list[float]):
    model = load_model()
    x = np.array(features, dtype=float).reshape(1, -1)

    proba = float(model.predict_proba(x)[0, 1])
    pred = int(proba >= 0.5)

    return {
        "prediction": pred,
        "proba": proba,
        "model_version": MODEL_VERSION
    }

if __name__ == "__main__":
    # Example test
    sample = [14.0,20.0,90.0,600.0,0.10,0.20,0.15,0.10,0.18,0.06,
              0.40,1.2,3.0,40.0,0.01,0.02,0.03,0.01,0.02,0.003,
              16.0,25.0,110.0,900.0,0.12,0.30,0.25,0.15,0.28,0.08]

    print(predict(sample))
