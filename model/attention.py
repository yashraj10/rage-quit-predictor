"""
Attention extraction and visualization for model interpretability.

This code produces visual evidence of what the model learned. The key finding to aim for:
"The model consistently attends to death streaks followed by action droughts,
which matches the intuitive rage quit pattern."
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from data.vocab import ID_TO_EVENT, PAD_TOKEN_ID

logger = logging.getLogger(__name__)


def extract_cls_attention(
    model,
    batch: dict,
    device: torch.device,
    layer_idx: int = -1,
) -> np.ndarray:
    """
    Extract the attention weights from [CLS] token to all other tokens.

    Args:
        model: RageQuitTransformer
        batch: Single batch from DataLoader
        device: torch device
        layer_idx: Which transformer layer (-1 = last)

    Returns:
        (batch_size, seq_len) attention weights from [CLS] to each position
    """
    event_ids = batch["event_ids"].to(device)
    continuous = batch["continuous_features"].to(device)
    minutes = batch["minutes"].to(device)
    mask = batch["attention_mask"].to(device)

    attention_maps = model.get_attention_weights(event_ids, continuous, minutes, mask)

    if not attention_maps:
        logger.warning("No attention maps captured")
        return np.zeros((event_ids.shape[0], event_ids.shape[1]))

    # Get specified layer's attention
    attn = attention_maps[layer_idx]  # (B, num_heads, seq_len, seq_len)

    # Average across heads
    attn_avg = attn.mean(dim=1)  # (B, seq_len, seq_len)

    # Get [CLS] token's attention to all positions (row 0)
    cls_attn = attn_avg[:, 0, :]  # (B, seq_len)

    return cls_attn.cpu().numpy()


def plot_attention_heatmap(
    event_ids: list[int],
    attention_weights: np.ndarray,
    title: str = "Attention from [CLS] Token",
    save_path: str | None = None,
    max_tokens: int = 60,
):
    """
    Plot attention heatmap showing which events the model focuses on.

    Args:
        event_ids: List of event token IDs for one sequence
        attention_weights: (seq_len,) attention from [CLS]
        title: Plot title
        save_path: Optional path to save figure
        max_tokens: Maximum tokens to display
    """
    # Filter out padding
    valid_mask = [eid != PAD_TOKEN_ID for eid in event_ids]
    valid_events = [eid for eid, v in zip(event_ids, valid_mask) if v]
    valid_attn = attention_weights[: len(valid_events)]

    # Truncate for readability
    valid_events = valid_events[:max_tokens]
    valid_attn = valid_attn[:max_tokens]

    event_labels = [ID_TO_EVENT.get(eid, f"?{eid}") for eid in valid_events]

    fig, ax = plt.subplots(figsize=(max(12, len(event_labels) * 0.3), 3))

    # Heatmap as a single row
    attn_2d = valid_attn.reshape(1, -1)
    sns.heatmap(
        attn_2d,
        ax=ax,
        cmap="YlOrRd",
        xticklabels=event_labels,
        yticklabels=["[CLS] →"],
        cbar_kws={"label": "Attention Weight"},
    )
    ax.set_xticklabels(event_labels, rotation=45, ha="right", fontsize=7)
    ax.set_title(title, fontsize=12, pad=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved attention heatmap to {save_path}")
    plt.close()


def plot_attention_comparison(
    rage_quit_events: list[int],
    rage_quit_attn: np.ndarray,
    normal_events: list[int],
    normal_attn: np.ndarray,
    save_path: str | None = None,
    max_tokens: int = 50,
):
    """
    Side-by-side comparison of attention patterns for rage quit vs normal game.
    This is the money shot for interviews.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 6))

    for ax, events, attn, title, cmap in [
        (ax1, rage_quit_events, rage_quit_attn, "Rage Quit — Predicted Correctly", "Reds"),
        (ax2, normal_events, normal_attn, "Normal Game — No Quit", "Blues"),
    ]:
        valid = [eid for eid in events if eid != PAD_TOKEN_ID][:max_tokens]
        a = attn[: len(valid)]
        labels = [ID_TO_EVENT.get(eid, "?") for eid in valid]

        attn_2d = a.reshape(1, -1)
        sns.heatmap(
            attn_2d,
            ax=ax,
            cmap=cmap,
            xticklabels=labels,
            yticklabels=["[CLS] →"],
            cbar_kws={"label": "Attention"},
        )
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
        ax.set_title(title, fontsize=11)

    plt.suptitle("Attention Comparison: Rage Quit vs Normal", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved comparison to {save_path}")
    plt.close()


def compute_event_importance(
    model,
    dataloader,
    device: torch.device,
    num_batches: int = 50,
) -> dict[str, float]:
    """
    Compute average attention weight per event type across the dataset.
    Shows which events the model considers most important for its predictions.

    Returns:
        Dict mapping event name → average attention weight
    """
    model.eval()
    event_attention_sum = {}
    event_count = {}

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        cls_attn = extract_cls_attention(model, batch, device)  # (B, seq_len)
        event_ids = batch["event_ids"].numpy()

        for b in range(cls_attn.shape[0]):
            for pos in range(cls_attn.shape[1]):
                eid = event_ids[b, pos]
                if eid == PAD_TOKEN_ID:
                    continue
                event_name = ID_TO_EVENT.get(eid, f"UNK_{eid}")
                event_attention_sum[event_name] = (
                    event_attention_sum.get(event_name, 0.0) + cls_attn[b, pos]
                )
                event_count[event_name] = event_count.get(event_name, 0) + 1

    # Average
    event_importance = {}
    for name in event_attention_sum:
        if event_count[name] > 0:
            event_importance[name] = event_attention_sum[name] / event_count[name]

    # Sort by importance
    event_importance = dict(
        sorted(event_importance.items(), key=lambda x: x[1], reverse=True)
    )

    return event_importance


def plot_event_importance(
    event_importance: dict[str, float],
    save_path: str | None = None,
):
    """Bar chart of average attention per event type."""
    # Filter out meta tokens
    filtered = {k: v for k, v in event_importance.items() if k not in ["[CLS]", "[SEP]", "[PAD]"]}

    names = list(filtered.keys())
    values = list(filtered.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(names)))

    bars = ax.barh(range(len(names)), values, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Average Attention Weight from [CLS]", fontsize=10)
    ax.set_title("Event Importance — What the Model Focuses On", fontsize=12)
    ax.invert_yaxis()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved event importance plot to {save_path}")
    plt.close()
