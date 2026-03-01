"""
AWS S3 Data Lake utilities for the Rage Quit Predictor MLOps pipeline.

Bucket: rage-quit-mlops-yashraj
Structure:
    raw/                - OpenDota API JSON responses
    streaming-features/ - Spark output Parquet by date partition
    models/             - PyTorch .pt checkpoints by version
    mlflow-artifacts/   - MLflow run artifacts
"""

import boto3
import json
from pathlib import Path
from datetime import datetime

BUCKET = "rage-quit-mlops-yashraj"
s3 = boto3.client("s3", region_name="us-east-1")


def upload_model(local_path: str, version: str = "v1") -> str:
    key = f"models/best_model_{version}.pt"
    s3.upload_file(local_path, BUCKET, key)
    s3_uri = f"s3://{BUCKET}/{key}"
    print(f"✅ Uploaded model to {s3_uri}")
    return s3_uri


def upload_metrics(local_path: str) -> str:
    filename = Path(local_path).name
    key = f"mlflow-artifacts/metrics/{filename}"
    s3.upload_file(local_path, BUCKET, key)
    s3_uri = f"s3://{BUCKET}/{key}"
    print(f"✅ Uploaded metrics to {s3_uri}")
    return s3_uri


def list_bucket() -> None:
    print(f"\n--- S3 Bucket: {BUCKET} ---")
    response = s3.list_objects_v2(Bucket=BUCKET)
    for obj in response.get("Contents", []):
        size_kb = obj["Size"] / 1024
        print(f"  {obj['Key']:<60} {size_kb:.1f} KB")


def verify_setup() -> bool:
    try:
        s3.head_bucket(Bucket=BUCKET)
        print(f"✅ Bucket {BUCKET} exists and is accessible")
        list_bucket()
        return True
    except Exception as e:
        print(f"❌ Bucket check failed: {e}")
        return False


if __name__ == "__main__":
    verify_setup()