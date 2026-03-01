"""
MLflow tracking wrapper for the RageQuitTransformer.

Logs all hyperparameters, per-epoch metrics, final test results,
model artifacts, and registers the model in the MLflow Model Registry.

Usage:
    python mlops/mlflow_tracking.py
"""

import json
import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch

# Add project root to path so we can import from model/
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.vocab import NUM_CONTINUOUS_FEATURES, VOCAB_SIZE
from model.transformer import RageQuitTransformer

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "rage-quit-predictor"

# Ground-truth metrics from results/metrics/test_metrics.json
METRICS_PATH = Path("results/metrics/test_metrics.json")
HISTORY_PATH = Path("results/metrics/training_history.json")
WEIGHTS_PATH = Path("results/weights/best_model.pt")

# Hyperparameters matching configs/default.yaml
HPARAMS = {
    "vocab_size": 22,
    "embed_dim": 128,
    "num_heads": 4,
    "num_layers": 4,
    "num_continuous_features": 6,
    "ff_dim": 512,
    "max_seq_len": 256,
    "max_minutes": 90,
    "dropout": 0.1,
    "batch_size": 64,
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "epochs": 30,
    "warmup_fraction": 0.05,
    "max_grad_norm": 1.0,
    "patience": 5,
    "pos_weight": 163.31,
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingWarmRestarts",
    "loss": "BCEWithLogitsLoss",
    "positional_encoding": "game_time_minute",
    "train_size": 139990,
    "val_size": 30010,
    "test_size": 30020,
    "num_positives": 183,
    "positive_rate": 0.0061,
    "dataset": "OpenDota_20k_matches",
    "label_definition": "leaver_status>=2 AND duration<median",
}


def load_test_metrics() -> dict:
    with open(METRICS_PATH) as f:
        return json.load(f)


def load_training_history() -> list:
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH) as f:
            return json.load(f)
    return []


def load_model() -> RageQuitTransformer:
    checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})

    model = RageQuitTransformer(
        vocab_size=config.get("vocab_size", VOCAB_SIZE),
        embed_dim=config.get("embed_dim", 128),
        num_heads=config.get("num_heads", 4),
        num_layers=config.get("num_layers", 4),
        num_continuous_features=config.get("num_continuous_features", NUM_CONTINUOUS_FEATURES),
        ff_dim=config.get("ff_dim", 512),
        max_seq_len=config.get("max_seq_len", 256),
        dropout=config.get("dropout", 0.1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def log_run():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    test_metrics = load_test_metrics()
    history = load_training_history()

    with mlflow.start_run(run_name="transformer-baseline") as run:
        # --- Log hyperparameters ---
        mlflow.log_params(HPARAMS)
        mlflow.set_tags({
            "model_type": "transformer",
            "dataset": "opendota",
            "task": "rage_quit_prediction",
            "framework": "pytorch",
        })

        # --- Log per-epoch metrics from training history ---
        for entry in history:
            step = entry["epoch"]
            mlflow.log_metrics({
                "train_loss": entry["train_loss"],
                "val_loss": entry["val_loss"],
                "val_auc_roc": entry["val_auc_roc"],
                "val_auc_pr": entry["val_auc_pr"],
                "val_f1": entry["val_f1"],
            }, step=step)

        # --- Log final test metrics ---
        mlflow.log_metrics({
            "test_auc_pr":    test_metrics["auc_pr"],
            "test_auc_roc":   test_metrics["auc_roc"],
            "test_f1":        test_metrics["f1"],
            "test_precision": test_metrics["precision"],
            "test_recall":    test_metrics["recall"],
            "test_accuracy":  test_metrics["accuracy"],
            "test_tp":        test_metrics["tp"],
            "test_fp":        test_metrics["fp"],
            "test_fn":        test_metrics["fn"],
            "test_tn":        test_metrics["tn"],
        })

        # --- Log model artifact ---
        model = load_model()
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name="RageQuitTransformer",
        )

        # --- Log existing result figures as artifacts ---
        figures_dir = Path("results/figures")
        if figures_dir.exists():
            for fig_path in figures_dir.glob("*.png"):
                mlflow.log_artifact(str(fig_path), artifact_path="figures")

        # --- Log metrics JSON files ---
        for json_path in Path("results/metrics").glob("*.json"):
            mlflow.log_artifact(str(json_path), artifact_path="metrics")

        run_id = run.info.run_id
        print(f"\n✅ MLflow run logged successfully!")
        print(f"   Run ID:     {run_id}")
        print(f"   Experiment: {EXPERIMENT_NAME}")
        print(f"   Test AUC-PR: {test_metrics['auc_pr']:.4f}")
        print(f"   View at:    {MLFLOW_TRACKING_URI}")
        return run_id


if __name__ == "__main__":
    log_run()