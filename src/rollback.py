import os
import shutil
import argparse
from src.config import MODEL_PATH

def rollback(version):
    project_root = os.getcwd()
    abs_model_path = os.path.join(project_root, MODEL_PATH)
    versioned_model_path = os.path.join(project_root, f"models/exported/model_{version}.pkl")

    if not os.path.exists(versioned_model_path):
        print(f"Error: Versioned model '{versioned_model_path}' not found!")
        return

    print(f"Rolling back to version: {version}")
    shutil.copy(versioned_model_path, abs_model_path)
    print(f"Successfully updated active model at {abs_model_path} with version {version}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["v1", "v2"], required=True, help="Version to rollback to (v1 or v2)")
    args = parser.parse_args()

    rollback(args.version)
