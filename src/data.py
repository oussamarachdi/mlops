import os
import pandas as pd
from sklearn.datasets import load_breast_cancer

def ensure_dirs():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

def download_raw():
    ensure_dirs()
    ds = load_breast_cancer(as_frame=True)
    df = ds.frame
    raw_path = "data/raw/breast_cancer.csv"
    df.to_csv(raw_path, index=False)
    return raw_path

if __name__ == "__main__":
    path = download_raw()
    print(f"Saved raw dataset to: {path}")
