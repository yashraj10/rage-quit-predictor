"""
Evaluation metrics for the rage quit predictor.

Primary metrics (for imbalanced binary classification):
    - AUC-ROC: Overall ranking quality
    - AUC-PR (Average Precision): Better than AUC-ROC for imbalanced data
    - Precision@K: "Of the top K predicted rage quits, how many actually quit?"
    - F1 at optimal threshold
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def find_optimal_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    """Find the threshold that maximizes F1 score."""
    precisions, recalls, thresholds = precision_recall_curve(labels, probs)

    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)

    return float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5


def precision_at_k(labels: np.ndarray, probs: np.ndarray, k: int = 100) -> float:
    """Compute precision among the top-K highest predicted probabilities."""
    if len(labels) < k:
        k = len(labels)

    top_k_indices = np.argsort(probs)[-k:]
    top_k_labels = labels[top_k_indices]

    return float(np.mean(top_k_labels))


def evaluate_model(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float | None = None,
) -> dict:
    """
    Compute all evaluation metrics.

    Args:
        labels: Binary ground truth labels (0 or 1)
        probs: Predicted probabilities [0, 1]
        threshold: Classification threshold (if None, uses optimal)

    Returns:
        Dictionary of all metrics
    """
    # Handle edge cases
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return {
            "auc_roc": 0.0,
            "auc_pr": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "accuracy": 0.0,
            "optimal_threshold": 0.5,
            "precision_at_100": 0.0,
            "precision_at_50": 0.0,
            "num_samples": len(labels),
            "num_positive": int(np.sum(labels)),
        }

    # Core metrics
    auc_roc = roc_auc_score(labels, probs)
    auc_pr = average_precision_score(labels, probs)

    # Find optimal threshold
    optimal_thresh = find_optimal_threshold(labels, probs)
    use_threshold = threshold if threshold is not None else optimal_thresh

    # Threshold-dependent metrics
    preds = (probs >= use_threshold).astype(int)
    f1 = f1_score(labels, preds, zero_division=0)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    accuracy = accuracy_score(labels, preds)

    # Precision@K
    p_at_100 = precision_at_k(labels, probs, k=100)
    p_at_50 = precision_at_k(labels, probs, k=50)

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        "auc_roc": float(auc_roc),
        "auc_pr": float(auc_pr),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "optimal_threshold": float(optimal_thresh),
        "threshold_used": float(use_threshold),
        "precision_at_100": float(p_at_100),
        "precision_at_50": float(p_at_50),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "num_samples": len(labels),
        "num_positive": int(np.sum(labels)),
    }


def get_roc_curve_data(labels: np.ndarray, probs: np.ndarray) -> dict:
    """Get ROC curve data for plotting."""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()}


def get_pr_curve_data(labels: np.ndarray, probs: np.ndarray) -> dict:
    """Get Precision-Recall curve data for plotting."""
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
    }


def print_evaluation_report(labels: np.ndarray, probs: np.ndarray, model_name: str = "Model"):
    """Print a formatted evaluation report."""
    metrics = evaluate_model(labels, probs)

    print(f"\n{'=' * 50}")
    print(f" {model_name} — Evaluation Report")
    print(f"{'=' * 50}")
    print(f" Samples: {metrics['num_samples']:,} (pos: {metrics['num_positive']:,})")
    print(f" AUC-ROC:         {metrics['auc_roc']:.4f}")
    print(f" AUC-PR:          {metrics['auc_pr']:.4f}")
    print(f" F1:              {metrics['f1']:.4f}")
    print(f" Precision:       {metrics['precision']:.4f}")
    print(f" Recall:          {metrics['recall']:.4f}")
    print(f" Accuracy:        {metrics['accuracy']:.4f}")
    print(f" Threshold:       {metrics['threshold_used']:.4f}")
    print(f" Precision@100:   {metrics['precision_at_100']:.4f}")
    print(f" Precision@50:    {metrics['precision_at_50']:.4f}")
    print(f"{'=' * 50}")
    print(f" Confusion Matrix:")
    print(f"   TP: {metrics['true_positives']:6d}  FP: {metrics['false_positives']:6d}")
    print(f"   FN: {metrics['false_negatives']:6d}  TN: {metrics['true_negatives']:6d}")
    print(f"{'=' * 50}\n")

    return metrics
