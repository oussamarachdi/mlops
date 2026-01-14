import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .config import MODEL_PATH, MODEL_VERSION

app = FastAPI(title="Breast Cancer Prediction API")

class PredictionRequest(BaseModel):
    features: list[float]
    # Check dataset for feature names if detailed input schema is needed.
    # For now, assuming list of floats matching model input.

class PredictionResponse(BaseModel):
    prediction: int
    probability: float

model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"Loaded model from {MODEL_PATH}")
        else:
            print(f"Model not found at {MODEL_PATH}. Waiting for training.")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.get("/version")
def get_version():
    return {"version": MODEL_VERSION, "model_path": MODEL_PATH}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert request features to 2D array-like structure expected by sklearn
        # If the model was trained with a DataFrame and expects feature names, we might need adjustments.
        # But standard Pipelines work fine with numpy arrays (list of lists).
        data = [request.features]
        
        # Get prediction and probability
        prediction = model.predict(data)[0]
        # Check if predict_proba is available
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(data)[0][1] # Probability of class 1
        else:
            probability = 0.0 # Fallback if not available

        return {"prediction": int(prediction), "probability": float(probability)}
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health():
    if model:
        return {"status": "healthy"}
    return {"status": "unhealthy", "detail": "Model not loaded"}

@app.post("/reload")
def reload_model():
    load_model()
    if model:
        return {"status": "reloaded", "model_path": MODEL_PATH}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model")
