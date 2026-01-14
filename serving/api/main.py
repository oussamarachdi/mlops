from fastapi import FastAPI
from .schemas import PredictRequest, PredictResponse
from serving.model_server.predictor import predict

app = FastAPI(title="Breast Cancer MLOps API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def run_prediction(req: PredictRequest):
    result = predict(req.features)
    return PredictResponse(**result)
