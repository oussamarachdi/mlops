import boto3
from botocore.client import Config
import time
import os

def create_buckets():
    s3 = boto3.client(
        "s3",
        endpoint_url="http://localhost:9000",
        aws_access_key_id="minio",
        aws_secret_access_key="minio12345",
        config=Config(signature_version="s3v4"),
        region_name="us-east-1"
    )

    buckets = ["dvc", "mlflow"]
    
    # Retry logic for MinIO startup
    for _ in range(10):
        try:
            existing = [b["Name"] for b in s3.list_buckets()["Buckets"]]
            break
        except Exception as e:
            print(f"Waiting for MinIO... {e}")
            time.sleep(2)
    else:
        print("Could not connect to MinIO")
        return

    for bucket in buckets:
        if bucket not in existing:
            s3.create_bucket(Bucket=bucket)
            print(f"Created bucket: {bucket}")
        else:
            print(f"Bucket {bucket} already exists")

if __name__ == "__main__":
    create_buckets()
