from pydantic import BaseModel
from typing import List

class PredictRequest(BaseModel):
    features: List[float]

class PredictResponse(BaseModel):
    prediction: int
    proba: float
    model_version: str
