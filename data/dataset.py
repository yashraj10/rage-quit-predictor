"""
PyTorch Dataset and DataLoader utilities for behavioral sequences.

Handles padding, truncation, attention masks, stratified splitting by match_id,
and weighted sampling for class imbalance.
"""

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from data.vocab import NUM_CONTINUOUS_FEATURES, PAD_TOKEN_ID


class DotaMatchDataset(Dataset):
    """PyTorch Dataset for Dota 2 behavioral event sequences."""

    def __init__(self, records: list[dict], max_seq_len: int = 256):
        self.records = records
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        event_ids = rec["event_ids"][: self.max_seq_len]
        continuous = rec["continuous"][: self.max_seq_len]
        minutes = rec["minutes"][: self.max_seq_len]
        label = rec["label"]

        actual_len = len(event_ids)
        attention_mask = [1] * actual_len

        # Pad to max_seq_len
        pad_len = self.max_seq_len - actual_len
        if pad_len > 0:
            event_ids = event_ids + [PAD_TOKEN_ID] * pad_len
            continuous = continuous + [[0.0] * NUM_CONTINUOUS_FEATURES] * pad_len
            minutes = minutes + [0] * pad_len
            attention_mask = attention_mask + [0] * pad_len

        return {
            "event_ids": torch.tensor(event_ids, dtype=torch.long),
            "continuous_features": torch.tensor(continuous, dtype=torch.float32),
            "minutes": torch.tensor(minutes, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float32),
        }


def split_by_match_id(
    records: list[dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split records into train/val/test by match_id to prevent data leakage.
    Players from the same match always go in the same split.
    Stratified to maintain label distribution across splits.
    """
    rng = np.random.RandomState(seed)

    # Group records by match_id
    match_groups = defaultdict(list)
    for rec in records:
        match_groups[rec["match_id"]].append(rec)

    # Classify matches as having a rage quit or not
    positive_matches = []
    negative_matches = []
    for mid, recs in match_groups.items():
        has_quit = any(r["label"] == 1 for r in recs)
        if has_quit:
            positive_matches.append(mid)
        else:
            negative_matches.append(mid)

    rng.shuffle(positive_matches)
    rng.shuffle(negative_matches)

    def split_list(items, train_r, val_r):
        n = len(items)
        train_end = int(n * train_r)
        val_end = train_end + int(n * val_r)
        return items[:train_end], items[train_end:val_end], items[val_end:]

    pos_train, pos_val, pos_test = split_list(positive_matches, train_ratio, val_ratio)
    neg_train, neg_val, neg_test = split_list(negative_matches, train_ratio, val_ratio)

    train_ids = set(pos_train + neg_train)
    val_ids = set(pos_val + neg_val)
    test_ids = set(pos_test + neg_test)

    train_records = [r for r in records if r["match_id"] in train_ids]
    val_records = [r for r in records if r["match_id"] in val_ids]
    test_records = [r for r in records if r["match_id"] in test_ids]

    return train_records, val_records, test_records


def make_weighted_sampler(records: list[dict]) -> WeightedRandomSampler:
    """Create a weighted sampler to handle class imbalance in training."""
    labels = [r["label"] for r in records]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[l] for l in labels]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def create_dataloaders(
    data_path: str,
    batch_size: int = 64,
    max_seq_len: int = 256,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Load processed data and create train/val/test DataLoaders.

    Returns:
        (train_loader, val_loader, test_loader, metadata)
    """
    with open(data_path, "rb") as f:
        records = pickle.load(f)

    train_recs, val_recs, test_recs = split_by_match_id(records)

    train_dataset = DotaMatchDataset(train_recs, max_seq_len)
    val_dataset = DotaMatchDataset(val_recs, max_seq_len)
    test_dataset = DotaMatchDataset(test_recs, max_seq_len)

    # Weighted sampler for training
    sampler = make_weighted_sampler(train_recs) if use_weighted_sampler else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Compute metadata for loss weighting
    train_labels = [r["label"] for r in train_recs]
    num_pos = sum(train_labels)
    num_neg = len(train_labels) - num_pos

    metadata = {
        "train_size": len(train_recs),
        "val_size": len(val_recs),
        "test_size": len(test_recs),
        "num_positive": num_pos,
        "num_negative": num_neg,
        "pos_weight": num_neg / max(num_pos, 1),
        "rage_quit_rate": num_pos / len(train_recs) if train_recs else 0,
    }

    return train_loader, val_loader, test_loader, metadata
