# Breast Cancer Classification MLOps

This project implements an MLOps pipeline for Breast Cancer Classification using scikit-learn, MLflow, Optuna, DVC, MinIO, and FastAPI.

## Prerequisites

- Python 3.9+
- Docker & Docker Compose
- Git

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repo_url>
    cd mlops-mini-project
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv .venv
    # Windows
    .\.venv\Scripts\activate
    # Linux/Mac
    source .venv/bin/activate

    pip install -r requirements.txt
    ```

3.  **Start Infrastructure:**
    Start MinIO (S3 artifact store) and MLflow Tracking Server.
    ```bash
    docker compose up -d --build
    ```
    - **MinIO Console**: http://localhost:9001 (User: `minio`, Pass: `minio12345`)
    - **MLflow UI**: http://localhost:5000

## Training

The training pipeline is managed by DVC.

To run the full training pipeline (including Optuna hyperparameter tuning):
```bash
dvc repro
```

Alternatively, you can run the script manually:
```bash
# Baseline model (v1)
python src/train.py --mode v1

# Tuned model (v2)
python src/train.py --mode v2
```

Training metrics and artifacts are logged to MLflow.

## Serving

The FastAPI service serves the latest trained model.

- **API URL**: http://localhost:8000
- **Docs**: http://localhost:8000/docs

### API Endpoints

- `GET /health`: Check service health.
- `GET /version`: Get loaded model version.
- `POST /predict`: Make a prediction.
- `POST /reload`: Reload the model from disk (useful after retraining).

### Example Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]}'
```
