import os
import mlflow
import joblib
from dotenv import load_dotenv
from src.config import MLFLOW_TRACKING_URI, MODEL_PATH

def export_best_model():
    load_dotenv()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Search for the best run in the tuning_pipeline experiment
    # Experiment name was 'tuning_pipeline' as seen in the logs
    experiment_name = "tuning_pipeline"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if not experiment:
        print(f"Experiment '{experiment_name}' not found!")
        return

    print(f"Searching for the best run in experiment: {experiment_name}")
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.final_f1_score DESC"],
        max_results=1
    )

    if runs.empty:
        print("No runs found in the experiment.")
        return

    best_run = runs.iloc[0]
    run_id = best_run.run_id
    f1_score = best_run["metrics.final_f1_score"]
    
    print(f"Best Run ID: {run_id}")
    print(f"Best F1 Score: {f1_score}")

    # Download the model artifact
    # ZenML logs the model as 'model' artifact in the train_best_model step
    # However, we also have the local export. 
    # Let's try to load it from MLflow to be sure.
    
    model_uri = f"runs:/{run_id}/model"
    print(f"Loading model from: {model_uri}")
    
    try:
        model = mlflow.sklearn.load_model(model_uri)
        
        # Save to the serving path
        project_root = os.getcwd()
        abs_model_path = os.path.join(project_root, MODEL_PATH)
        
        os.makedirs(os.path.dirname(abs_model_path), exist_ok=True)
        joblib.dump(model, abs_model_path)
        print(f"Successfully exported best model to {abs_model_path}")
    except Exception as e:
        print(f"Error loading/saving model from MLflow: {e}")
        print("Falling back to local version if available...")

if __name__ == "__main__":
    export_best_model()
