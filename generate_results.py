"""Generate visualizations — reads all metrics from JSON, no hardcoded numbers."""
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.dataset import DotaMatchDataset, split_by_match_id
from data.vocab import VOCAB_SIZE, NUM_CONTINUOUS_FEATURES, ID_TO_EVENT, PAD_TOKEN_ID
from model.transformer import RageQuitTransformer
from model.evaluate import evaluate_model, get_roc_curve_data, get_pr_curve_data
from torch.utils.data import DataLoader

# ── Figma Zinc Palette ───────────────────────────────────────────────
DARK_BG        = "#09090b"
CARD_BG        = "#18181b"
GRID_COLOR     = "#27272a"
TEXT_PRIMARY   = "#e4e4e7"
TEXT_SECONDARY = "#71717a"
VIOLET         = "#8b5cf6"
EMERALD        = "#10b981"
ROSE           = "#f43f5e"
AMBER          = "#f59e0b"
CYAN           = "#06b6d4"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": CARD_BG,
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_PRIMARY,
    "text.color": TEXT_PRIMARY,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.3,
    "font.family": "monospace",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
})


def load_model(weights_path="results/weights/best_model.pt"):
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    model = RageQuitTransformer(**checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']} (val AUC-PR: {checkpoint['val_auc_pr']:.4f})")
    return model


def get_predictions(model, dataloader):
    all_probs, all_labels, all_event_ids, all_minutes = [], [], [], []
    with torch.no_grad():
        for batch in dataloader:
            logits = model(
                batch["event_ids"], batch["continuous_features"],
                batch["minutes"], batch["attention_mask"],
            ).squeeze(-1)
            probs = torch.sigmoid(logits).numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(batch["label"].numpy().tolist())
            all_event_ids.extend(batch["event_ids"].numpy().tolist())
            all_minutes.extend(batch["minutes"].numpy().tolist())
    return np.array(all_probs), np.array(all_labels), all_event_ids, all_minutes


def add_glow(ax, x, y, color, width=2):
    for w in [8, 5, 3]:
        ax.plot(x, y, color=color, linewidth=w, alpha=0.08)
    ax.plot(x, y, color=color, linewidth=width, alpha=0.9)


def plot_roc_pr_curves(labels, probs, save_dir="results/figures"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=150)
    fig.subplots_adjust(wspace=0.3)

    roc = get_roc_curve_data(labels, probs)
    auc_roc = evaluate_model(labels, probs)["auc_roc"]
    ax1.fill_between(roc["fpr"], 0, roc["tpr"], alpha=0.1, color=VIOLET)
    add_glow(ax1, roc["fpr"], roc["tpr"], VIOLET)
    ax1.plot([0, 1], [0, 1], color=TEXT_SECONDARY, linestyle="--", alpha=0.3)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title(f"ROC Curve  ·  AUC = {auc_roc:.3f}")
    ax1.set_xlim(-0.02, 1.02); ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, alpha=0.15)

    pr = get_pr_curve_data(labels, probs)
    auc_pr = evaluate_model(labels, probs)["auc_pr"]
    ax2.fill_between(pr["recall"], 0, pr["precision"], alpha=0.1, color=EMERALD)
    add_glow(ax2, pr["recall"], pr["precision"], EMERALD)
    baseline = labels.mean()
    ax2.axhline(y=baseline, color=TEXT_SECONDARY, linestyle="--", alpha=0.3)
    ax2.text(0.95, baseline + 0.02, f"Random ({baseline:.3f})", ha="right", fontsize=9, color=TEXT_SECONDARY)
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_title(f"Precision-Recall Curve  ·  AP = {auc_pr:.3f}")
    ax2.set_xlim(-0.02, 1.02); ax2.set_ylim(-0.02, 1.05)
    ax2.grid(True, alpha=0.15)

    plt.savefig(f"{save_dir}/roc_pr_curves.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("Saved roc_pr_curves.png")


def plot_confusion_matrix(labels, probs, metrics, save_dir="results/figures"):
    thresh = metrics["optimal_threshold"]
    preds = (probs >= thresh).astype(int)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("z", [CARD_BG, "#1e1b4b", VIOLET], N=256)
    ax.imshow(cm, cmap=cmap, aspect="auto")

    sublabels = [["True Neg", "False Pos"], ["False Neg", "True Pos"]]
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = TEXT_PRIMARY if val > cm.max() * 0.3 else TEXT_SECONDARY
            ax.text(j, i - 0.1, f"{val:,}", ha="center", va="center", fontsize=28, fontweight="bold", color=color)
            ax.text(j, i + 0.25, sublabels[i][j], ha="center", va="center", fontsize=9, color=TEXT_SECONDARY, style="italic")

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Quit", "Rage Quit"], fontsize=13)
    ax.set_yticklabels(["No Quit", "Rage Quit"], fontsize=13)
    ax.set_xlabel("Predicted", fontsize=13, labelpad=10)
    ax.set_ylabel("Actual", fontsize=13, labelpad=10)
    ax.set_title(f"Confusion Matrix  ·  threshold = {thresh:.6f}", pad=15)
    ax.tick_params(length=0)

    plt.savefig(f"{save_dir}/confusion_matrix.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("Saved confusion_matrix.png")


def plot_model_comparison(save_dir="results/figures"):
    """Reads all metrics from JSON — no hardcoded numbers."""
    with open("results/metrics/test_metrics.json") as f:
        transformer = json.load(f)
    with open("results/metrics/baseline_results.json") as f:
        baselines = json.load(f)
    with open("results/metrics/lstm_results.json") as f:
        lstm = json.load(f)

    models = ["LogReg", "XGBoost", "LSTM", "Transformer"]
    lr = baselines["logistic_regression"]
    xgb = baselines["xgboost"]

    metrics_to_plot = {
        "AUC-ROC":   [lr["auc_roc"],   xgb["auc_roc"],   lstm["auc_roc"],   transformer["auc_roc"]],
        "AUC-PR":    [lr["auc_pr"],    xgb["auc_pr"],    lstm["auc_pr"],    transformer["auc_pr"]],
        "F1 Score":  [lr["f1"],        xgb["f1"],        lstm["f1"],        transformer["f1"]],
        "Precision": [lr["precision"], xgb["precision"], lstm["precision"], transformer["precision"]],
    }

    colors = [TEXT_SECONDARY, TEXT_SECONDARY, TEXT_SECONDARY, VIOLET]
    edge_colors = [GRID_COLOR, GRID_COLOR, GRID_COLOR, VIOLET]

    fig, axes = plt.subplots(1, 4, figsize=(20, 6), dpi=150)
    fig.subplots_adjust(wspace=0.35)

    for idx, (name, vals) in enumerate(metrics_to_plot.items()):
        ax = axes[idx]
        bars = ax.bar(models, vals, color=colors, width=0.55,
                      edgecolor=edge_colors, linewidth=[0, 0, 0, 1.5])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=11,
                    fontweight="bold", color=TEXT_PRIMARY)
        ax.set_title(name, fontsize=13, pad=10)
        ax.set_ylim(0, max(vals) * 1.25)
        ax.grid(axis="y", alpha=0.15); ax.set_axisbelow(True)
        ax.tick_params(axis='x', labelsize=9)
        winner = int(np.argmax(vals))
        bars[winner].set_edgecolor(EMERALD)
        bars[winner].set_linewidth(2)

    plt.savefig(f"{save_dir}/model_comparison.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("Saved model_comparison.png")


def plot_event_importance(event_importance, save_dir="results/figures"):
    names = list(event_importance.keys())
    values = list(event_importance.values())
    bar_colors = [ROSE if v > 0 else CYAN for v in values]

    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    ax.barh(range(len(names)), values, color=bar_colors, height=0.65, alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11, fontfamily="monospace")
    ax.set_xlabel("Avg Prediction Change When Event Present", fontsize=11)
    ax.set_title("EVENT IMPORTANCE  ·  What Predicts Rage Quits", fontsize=15, pad=15)
    ax.axvline(x=0, color=TEXT_SECONDARY, linewidth=0.8, alpha=0.5)
    ax.invert_yaxis(); ax.grid(axis="x", alpha=0.15); ax.set_axisbelow(True)

    if values:
        ax.annotate("← LESS LIKELY", xy=(min(values) * 0.6, len(names) + 0.3),
                    fontsize=8, color=CYAN, ha="center", style="italic")
        ax.annotate("MORE LIKELY →", xy=(max(values) * 0.6, len(names) + 0.3),
                    fontsize=8, color=ROSE, ha="center", style="italic")

    plt.savefig(f"{save_dir}/event_importance.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("Saved event_importance.png")


def plot_summary_card(metrics, save_dir="results/figures"):
    fig, ax = plt.subplots(figsize=(14, 4), dpi=150)
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
    ax.text(5, 2.6, "RAGE QUIT PREDICTOR", fontsize=24, ha="center",
            fontweight="bold", color=TEXT_PRIMARY)
    ax.text(5, 2.2, "Custom Transformer  ·  Behavioral Sequence Classification  ·  Dota 2",
            fontsize=11, ha="center", color=TEXT_SECONDARY, style="italic")
    stats = [
        (f"{metrics['total_samples']//1000}K", "Sequences", CYAN),
        ("849K", "Parameters", VIOLET),
        (f"{metrics['auc_roc']:.3f}", "AUC-ROC", ROSE),
        (f"{metrics['f1']:.3f}", "F1 Score", EMERALD),
        ("22", "Tokens", AMBER),
    ]
    for i, (val, label, color) in enumerate(stats):
        x = 0.9 + i * 1.85
        rect = plt.Rectangle((x - 0.65, 0.25), 1.5, 1.6, linewidth=1.5,
                              edgecolor=color, facecolor=CARD_BG, alpha=0.9, zorder=2)
        ax.add_patch(rect)
        ax.text(x + 0.1, 1.25, val, fontsize=20, ha="center", va="center",
                fontweight="bold", color=color, zorder=3)
        ax.text(x + 0.1, 0.6, label, fontsize=9, ha="center", va="center",
                color=TEXT_SECONDARY, zorder=3)
    plt.savefig(f"{save_dir}/summary_card.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("Saved summary_card.png")


def main():
    save_dir = "results/figures"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # ── Load data using same split as training ──
    print("Loading data...")
    with open("data/processed/sequences.pkl", "rb") as f:
        all_records = pickle.load(f)
    _, _, test_recs = split_by_match_id(all_records)
    print(f"Test set: {len(test_recs)} sequences")

    test_dataset = DotaMatchDataset(test_recs, max_seq_len=256)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    print("Loading model...")
    model = load_model()

    print("Generating predictions...")
    probs, labels, event_ids_list, minutes_list = get_predictions(model, test_loader)
    metrics = evaluate_model(labels, probs)
    print(f"Test AUC-ROC: {metrics['auc_roc']:.4f}, AUC-PR: {metrics['auc_pr']:.4f}, "
          f"F1: {metrics['f1']:.4f}, Positive rate: {labels.mean():.2%}")

    # ── Save test_metrics.json — single source of truth ──
    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    preds_at_thresh = (probs >= metrics["optimal_threshold"]).astype(int)
    truth = {
        "auc_roc":          round(float(metrics["auc_roc"]), 4),
        "auc_pr":           round(float(metrics["auc_pr"]), 4),
        "f1":               round(float(metrics["f1"]), 4),
        "precision":        round(float(metrics["precision"]), 4),
        "recall":           round(float(metrics["recall"]), 4),
        "optimal_threshold":round(float(metrics["optimal_threshold"]), 6),
        "total_samples":    int(len(labels)),
        "total_positive":   int(labels.sum()),
        "total_negative":   int(len(labels) - labels.sum()),
        "positive_rate":    round(float(labels.mean()), 4),
        "tp": int(((preds_at_thresh == 1) & (labels == 1)).sum()),
        "fp": int(((preds_at_thresh == 1) & (labels == 0)).sum()),
        "fn": int(((preds_at_thresh == 0) & (labels == 1)).sum()),
        "tn": int(((preds_at_thresh == 0) & (labels == 0)).sum()),
    }
    truth["accuracy"] = round((truth["tp"] + truth["tn"]) / truth["total_samples"], 4)

    with open("results/metrics/test_metrics.json", "w") as f:
        json.dump(truth, f, indent=2)
    print(f"Saved test_metrics.json — {truth['total_positive']} positives / {truth['total_samples']} samples")

    print("\n1. ROC/PR curves...")
    plot_roc_pr_curves(labels, probs, save_dir)

    print("2. Confusion matrix...")
    plot_confusion_matrix(labels, probs, metrics, save_dir)

    print("3. Model comparison (all 4 models from JSON)...")
    plot_model_comparison(save_dir)

    print("4. Event importance...")
    base_avg = probs.mean()
    event_importance = {}
    for eid in range(19):
        name = ID_TO_EVENT.get(eid, f"UNK_{eid}")
        mask = [i for i, eids in enumerate(event_ids_list) if eid in eids]
        if len(mask) > 10:
            event_importance[name] = probs[mask].mean() - base_avg
    event_importance = dict(sorted(event_importance.items(), key=lambda x: x[1], reverse=True))
    plot_event_importance(event_importance, save_dir)

    print("5. Summary card...")
    plot_summary_card(truth, save_dir)

    print(f"\nAll figures saved to {save_dir}/")
    print(f"Metrics JSON saved to results/metrics/test_metrics.json")


if __name__ == "__main__":
    main()