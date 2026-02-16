"""
Baseline models for comparison against the transformer.

1. Logistic Regression on aggregated features
2. XGBoost on aggregated features
3. LSTM on behavioral sequences

Having baselines is critical for interviews — it shows you understand that
transformers aren't always the answer and you can justify your architecture choice
with empirical evidence.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data.vocab import EVENT_VOCAB, NUM_CONTINUOUS_FEATURES, PAD_TOKEN_ID, VOCAB_SIZE
from model.evaluate import evaluate_model, print_evaluation_report

logger = logging.getLogger(__name__)


# =============================================================================
# Feature aggregation (for non-sequential baselines)
# =============================================================================

def aggregate_sequence_features(record: dict) -> np.ndarray:
    """
    Convert a behavioral event sequence into a fixed-length feature vector.
    Used by Logistic Regression and XGBoost baselines.
    """
    event_ids = record["event_ids"]
    continuous = record["continuous"]
    minutes = record["minutes"]

    # Event count features (one per event type, excluding meta tokens)
    event_counts = np.zeros(19)  # vocab size minus meta tokens
    for eid in event_ids:
        if eid < 19:
            event_counts[eid] += 1

    # Sequence length
    seq_len = sum(1 for eid in event_ids if eid != PAD_TOKEN_ID)

    # Continuous feature statistics (mean, std, min, max, last)
    cont_array = np.array([c for c, eid in zip(continuous, event_ids) if eid != PAD_TOKEN_ID])
    if len(cont_array) == 0:
        cont_stats = np.zeros(NUM_CONTINUOUS_FEATURES * 5)
    else:
        cont_mean = np.mean(cont_array, axis=0)
        cont_std = np.std(cont_array, axis=0)
        cont_min = np.min(cont_array, axis=0)
        cont_max = np.max(cont_array, axis=0)
        cont_last = cont_array[-1]
        cont_stats = np.concatenate([cont_mean, cont_std, cont_min, cont_max, cont_last])

    # Temporal features
    max_minute = max(minutes) if minutes else 0

    # Event pattern features
    deaths = event_counts[EVENT_VOCAB["DEATH"]]
    kills = event_counts[EVENT_VOCAB["KILL"]]
    death_streaks = event_counts[EVENT_VOCAB["DEATH_STREAK"]]
    action_droughts = event_counts[EVENT_VOCAB["ACTION_DROUGHT"]]
    team_fight_losses = event_counts[EVENT_VOCAB["TEAM_FIGHT_LOSS"]]
    gold_drops = event_counts[EVENT_VOCAB["GOLD_SPIKE_DOWN"]]

    # Derived features
    kd_ratio = kills / max(deaths, 1)
    frustration_score = death_streaks * 2 + action_droughts + team_fight_losses + gold_drops

    extra_features = np.array([
        seq_len,
        max_minute,
        kd_ratio,
        frustration_score,
    ])

    return np.concatenate([event_counts, cont_stats, extra_features])


# =============================================================================
# Baseline 1: Logistic Regression
# =============================================================================

class LogRegBaseline:
    """Logistic Regression on aggregated sequence features."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
        )

    def fit(self, train_records: list[dict]):
        X = np.array([aggregate_sequence_features(r) for r in train_records])
        y = np.array([r["label"] for r in train_records])

        X = self.scaler.fit_transform(X)
        self.model.fit(X, y)
        logger.info(f"LogReg trained on {len(train_records)} samples")

    def predict_proba(self, records: list[dict]) -> np.ndarray:
        X = np.array([aggregate_sequence_features(r) for r in records])
        X = self.scaler.transform(X)
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, records: list[dict], name: str = "Logistic Regression") -> dict:
        probs = self.predict_proba(records)
        labels = np.array([r["label"] for r in records])
        return print_evaluation_report(labels, probs, name)


# =============================================================================
# Baseline 2: XGBoost
# =============================================================================

class XGBoostBaseline:
    """XGBoost on aggregated sequence features."""

    def __init__(self):
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError("pip install xgboost")

        self.model = None  # Initialized in fit()

    def fit(self, train_records: list[dict]):
        from xgboost import XGBClassifier

        X = np.array([aggregate_sequence_features(r) for r in train_records])
        y = np.array([r["label"] for r in train_records])

        # Compute scale_pos_weight for imbalance
        num_neg = np.sum(y == 0)
        num_pos = np.sum(y == 1)
        scale_pos_weight = num_neg / max(num_pos, 1)

        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="aucpr",
            early_stopping_rounds=20,
            random_state=42,
        )

        # Use 10% of training data for early stopping
        split = int(len(X) * 0.9)
        self.model.fit(
            X[:split], y[:split],
            eval_set=[(X[split:], y[split:])],
            verbose=False,
        )
        logger.info(f"XGBoost trained on {len(train_records)} samples")

    def predict_proba(self, records: list[dict]) -> np.ndarray:
        X = np.array([aggregate_sequence_features(r) for r in records])
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, records: list[dict], name: str = "XGBoost") -> dict:
        probs = self.predict_proba(records)
        labels = np.array([r["label"] for r in records])
        return print_evaluation_report(labels, probs, name)


# =============================================================================
# Baseline 3: LSTM on sequences
# =============================================================================

class LSTMClassifier(nn.Module):
    """Bidirectional LSTM baseline for sequential behavioral data."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_continuous_features: int = NUM_CONTINUOUS_FEATURES,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.continuous_proj = nn.Linear(num_continuous_features, embed_dim)
        self.combine = nn.Linear(embed_dim * 2, embed_dim)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, event_ids, continuous_features, minutes=None, attention_mask=None):
        event_emb = self.embedding(event_ids)
        cont_emb = self.continuous_proj(continuous_features)
        x = self.combine(torch.cat([event_emb, cont_emb], dim=-1))

        # Pack padded sequences for efficiency
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            output, (hidden, _) = self.lstm(packed)
        else:
            output, (hidden, _) = self.lstm(x)

        # Concatenate final forward and backward hidden states
        hidden_fwd = hidden[-2, :, :]
        hidden_bwd = hidden[-1, :, :]
        combined = torch.cat([hidden_fwd, hidden_bwd], dim=1)

        return self.classifier(combined)


# =============================================================================
# Run all baselines
# =============================================================================

def run_baselines(
    data_path: str = "data/processed/sequences.pkl",
    output_dir: str = "results/metrics",
):
    """Train and evaluate all baselines."""
    import json

    from data.dataset import split_by_match_id

    with open(data_path, "rb") as f:
        records = pickle.load(f)

    train_recs, val_recs, test_recs = split_by_match_id(records)
    logger.info(f"Train: {len(train_recs)}, Val: {len(val_recs)}, Test: {len(test_recs)}")

    results = {}

    # --- Logistic Regression ---
    logger.info("Training Logistic Regression baseline...")
    logreg = LogRegBaseline()
    logreg.fit(train_recs)
    results["logistic_regression"] = logreg.evaluate(test_recs)

    # --- XGBoost ---
    try:
        logger.info("Training XGBoost baseline...")
        xgb = XGBoostBaseline()
        xgb.fit(train_recs)
        results["xgboost"] = xgb.evaluate(test_recs)
    except ImportError:
        logger.warning("XGBoost not installed. Skipping.")

    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Baseline results saved to {output_dir}/baseline_results.json")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_baselines()
