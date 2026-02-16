"""
Training pipeline for the RageQuitTransformer.

Features:
    - Weighted BCE loss for class imbalance
    - Cosine annealing LR with linear warmup
    - Gradient clipping
    - Early stopping on validation AUC-PR
    - Checkpointing best model
    - TensorBoard-compatible logging

Usage:
    python -m model.train --config configs/default.yaml
    python -m model.train --data_path data/processed/sequences.pkl --epochs 30
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from data.dataset import create_dataloaders
from data.vocab import NUM_CONTINUOUS_FEATURES, VOCAB_SIZE
from model.evaluate import evaluate_model
from model.transformer import RageQuitTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_cosine_warmup_scheduler(
    optimizer, num_warmup_steps: int, num_total_steps: int
) -> LambdaLR:
    """Cosine annealing with linear warmup."""

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_total_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    criterion,
    scheduler,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> dict:
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        event_ids = batch["event_ids"].to(device)
        continuous = batch["continuous_features"].to(device)
        minutes = batch["minutes"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(event_ids, continuous, minutes, mask).squeeze(-1)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        with torch.no_grad():
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / max(num_batches, 1)
    accuracy = np.mean(
        [(1 if p > 0.5 else 0) == int(l) for p, l in zip(all_preds, all_labels)]
    )

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "lr": optimizer.param_groups[0]["lr"],
    }


@torch.no_grad()
def validate(
    model: nn.Module, dataloader, criterion, device: torch.device
) -> dict:
    """Run validation and return loss + metrics."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        event_ids = batch["event_ids"].to(device)
        continuous = batch["continuous_features"].to(device)
        minutes = batch["minutes"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(event_ids, continuous, minutes, mask).squeeze(-1)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        num_batches += 1

        preds = torch.sigmoid(logits).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / max(num_batches, 1)
    metrics = evaluate_model(np.array(all_labels), np.array(all_preds))
    metrics["loss"] = avg_loss

    return metrics


def train(
    data_path: str = "data/processed/sequences.pkl",
    output_dir: str = "results",
    # Model hyperparameters
    embed_dim: int = 128,
    num_heads: int = 4,
    num_layers: int = 4,
    ff_dim: int = 512,
    dropout: float = 0.1,
    max_seq_len: int = 256,
    # Training hyperparameters
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    epochs: int = 30,
    warmup_fraction: float = 0.05,
    max_grad_norm: float = 1.0,
    patience: int = 5,
    # Data
    num_workers: int = 4,
    seed: int = 42,
):
    """Full training pipeline."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "weights").mkdir(exist_ok=True)

    # --- Data ---
    logger.info("Loading data...")
    train_loader, val_loader, test_loader, metadata = create_dataloaders(
        data_path, batch_size=batch_size, max_seq_len=max_seq_len, num_workers=num_workers
    )
    logger.info(f"Train: {metadata['train_size']}, Val: {metadata['val_size']}, Test: {metadata['test_size']}")
    logger.info(f"Rage quit rate: {metadata['rage_quit_rate']:.2%}")
    logger.info(f"Pos weight: {metadata['pos_weight']:.2f}")

    # --- Model ---
    model = RageQuitTransformer(
        vocab_size=VOCAB_SIZE,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        num_continuous_features=NUM_CONTINUOUS_FEATURES,
        ff_dim=ff_dim,
        max_seq_len=max_seq_len,
        dropout=dropout,
    ).to(device)

    logger.info(f"Model parameters: {model.num_trainable_parameters:,}")

    # --- Loss ---
    pos_weight = torch.tensor([metadata["pos_weight"]], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- Optimizer + Scheduler ---
    optimizer = AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_fraction)
    scheduler = get_cosine_warmup_scheduler(optimizer, warmup_steps, total_steps)

    # --- Training loop ---
    best_val_auc_pr = 0.0
    patience_counter = 0
    history = []

    logger.info("Starting training...")
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, scheduler, device, max_grad_norm
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start

        # Log
        logger.info(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val AUC-ROC: {val_metrics['auc_roc']:.4f} | "
            f"Val AUC-PR: {val_metrics['auc_pr']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"LR: {train_metrics['lr']:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_auc_roc": val_metrics["auc_roc"],
            "val_auc_pr": val_metrics["auc_pr"],
            "val_f1": val_metrics["f1"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "lr": train_metrics["lr"],
        })

        # Check for improvement (using AUC-PR as primary metric for imbalanced data)
        if val_metrics["auc_pr"] > best_val_auc_pr:
            best_val_auc_pr = val_metrics["auc_pr"]
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc_pr": best_val_auc_pr,
                    "val_auc_roc": val_metrics["auc_roc"],
                    "config": {
                        "vocab_size": VOCAB_SIZE,
                        "embed_dim": embed_dim,
                        "num_heads": num_heads,
                        "num_layers": num_layers,
                        "num_continuous_features": NUM_CONTINUOUS_FEATURES,
                        "ff_dim": ff_dim,
                        "max_seq_len": max_seq_len,
                        "dropout": dropout,
                    },
                },
                output_path / "weights" / "best_model.pt",
            )
            logger.info(f"  ↑ New best model saved (AUC-PR: {best_val_auc_pr:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break

    # --- Final evaluation on test set ---
    logger.info("Loading best model for test evaluation...")
    checkpoint = torch.load(output_path / "weights" / "best_model.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = validate(model, test_loader, criterion, device)
    logger.info("=" * 60)
    logger.info("TEST SET RESULTS")
    logger.info(f"  AUC-ROC:   {test_metrics['auc_roc']:.4f}")
    logger.info(f"  AUC-PR:    {test_metrics['auc_pr']:.4f}")
    logger.info(f"  F1:        {test_metrics['f1']:.4f}")
    logger.info(f"  Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Recall:    {test_metrics['recall']:.4f}")
    logger.info("=" * 60)

    # Save history and final metrics
    with open(output_path / "metrics" / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(output_path / "metrics" / "test_results.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    logger.info(f"Results saved to {output_path}")
    return model, history, test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train the RageQuitTransformer")
    parser.add_argument("--data_path", type=str, default="data/processed/sequences.pkl")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(**vars(args))


if __name__ == "__main__":
    main()
