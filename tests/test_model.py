"""
Tests for the RageQuitTransformer model and data pipeline.

Run: pytest tests/ -v
"""

import numpy as np
import pytest
import torch

from data.vocab import (
    CLS_TOKEN_ID,
    EVENT_VOCAB,
    NUM_CONTINUOUS_FEATURES,
    PAD_TOKEN_ID,
    SEP_TOKEN_ID,
    VOCAB_SIZE,
)
from model.evaluate import evaluate_model, find_optimal_threshold, precision_at_k
from model.transformer import (
    BehavioralTokenEmbedding,
    GameTimePositionalEncoding,
    RageQuitTransformer,
)


# =============================================================================
# Vocabulary tests
# =============================================================================

class TestVocab:
    def test_vocab_size(self):
        assert VOCAB_SIZE == 22

    def test_special_tokens(self):
        assert PAD_TOKEN_ID == 19
        assert CLS_TOKEN_ID == 20
        assert SEP_TOKEN_ID == 21

    def test_no_duplicate_ids(self):
        ids = list(EVENT_VOCAB.values())
        assert len(ids) == len(set(ids))


# =============================================================================
# Model architecture tests
# =============================================================================

class TestRageQuitTransformer:
    @pytest.fixture
    def model(self):
        return RageQuitTransformer(
            vocab_size=VOCAB_SIZE,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            num_continuous_features=NUM_CONTINUOUS_FEATURES,
            ff_dim=128,
            max_seq_len=32,
            dropout=0.1,
        )

    @pytest.fixture
    def sample_batch(self):
        batch_size = 4
        seq_len = 32
        return {
            "event_ids": torch.randint(0, VOCAB_SIZE, (batch_size, seq_len)),
            "continuous_features": torch.randn(batch_size, seq_len, NUM_CONTINUOUS_FEATURES),
            "minutes": torch.randint(0, 60, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        }

    def test_forward_shape(self, model, sample_batch):
        logits = model(
            sample_batch["event_ids"],
            sample_batch["continuous_features"],
            sample_batch["minutes"],
            sample_batch["attention_mask"],
        )
        assert logits.shape == (4, 1)

    def test_forward_with_padding(self, model, sample_batch):
        # Set last 10 positions to padding
        sample_batch["attention_mask"][:, -10:] = 0
        sample_batch["event_ids"][:, -10:] = PAD_TOKEN_ID

        logits = model(
            sample_batch["event_ids"],
            sample_batch["continuous_features"],
            sample_batch["minutes"],
            sample_batch["attention_mask"],
        )
        assert logits.shape == (4, 1)
        assert not torch.isnan(logits).any()

    def test_output_is_finite(self, model, sample_batch):
        logits = model(
            sample_batch["event_ids"],
            sample_batch["continuous_features"],
            sample_batch["minutes"],
            sample_batch["attention_mask"],
        )
        assert torch.isfinite(logits).all()

    def test_parameter_count(self, model):
        assert model.num_parameters > 0
        assert model.num_trainable_parameters == model.num_parameters

    def test_gradient_flow(self, model, sample_batch):
        logits = model(
            sample_batch["event_ids"],
            sample_batch["continuous_features"],
            sample_batch["minutes"],
            sample_batch["attention_mask"],
        )
        loss = logits.sum()
        loss.backward()

        # Check that gradients exist and are finite
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"


class TestBehavioralTokenEmbedding:
    def test_output_shape(self):
        emb = BehavioralTokenEmbedding(VOCAB_SIZE, 64, NUM_CONTINUOUS_FEATURES)
        event_ids = torch.randint(0, VOCAB_SIZE, (2, 16))
        continuous = torch.randn(2, 16, NUM_CONTINUOUS_FEATURES)
        output = emb(event_ids, continuous)
        assert output.shape == (2, 16, 64)


class TestGameTimePositionalEncoding:
    def test_output_shape(self):
        pe = GameTimePositionalEncoding(64, max_minutes=90)
        x = torch.randn(2, 16, 64)
        minutes = torch.randint(0, 60, (2, 16))
        output = pe(x, minutes)
        assert output.shape == (2, 16, 64)


# =============================================================================
# Evaluation tests
# =============================================================================

class TestEvaluation:
    def test_evaluate_perfect_predictions(self):
        labels = np.array([0, 0, 0, 1, 1])
        probs = np.array([0.1, 0.2, 0.1, 0.9, 0.8])
        metrics = evaluate_model(labels, probs)
        assert metrics["auc_roc"] > 0.9
        assert metrics["auc_pr"] > 0.8

    def test_evaluate_random_predictions(self):
        np.random.seed(42)
        labels = np.random.randint(0, 2, 1000)
        probs = np.random.rand(1000)
        metrics = evaluate_model(labels, probs)
        # Random should be near 0.5 AUC-ROC
        assert 0.4 < metrics["auc_roc"] < 0.6

    def test_precision_at_k(self):
        labels = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        probs = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        p_at_3 = precision_at_k(labels, probs, k=3)
        assert p_at_3 == pytest.approx(2 / 3, abs=0.01)

    def test_find_optimal_threshold(self):
        labels = np.array([0, 0, 0, 1, 1])
        probs = np.array([0.1, 0.2, 0.3, 0.7, 0.9])
        thresh = find_optimal_threshold(labels, probs)
        assert 0.3 < thresh < 0.8

    def test_evaluate_handles_all_same_label(self):
        labels = np.array([0, 0, 0, 0])
        probs = np.array([0.1, 0.2, 0.3, 0.4])
        metrics = evaluate_model(labels, probs)
        assert metrics["auc_roc"] == 0.0  # edge case


# =============================================================================
# Data processing tests
# =============================================================================

class TestDataProcessing:
    def test_aggregate_sequence_features(self):
        from model.baselines import aggregate_sequence_features

        record = {
            "event_ids": [CLS_TOKEN_ID, 0, 1, 1, 4, 13, SEP_TOKEN_ID, 8, 16],
            "continuous": [[0.0] * NUM_CONTINUOUS_FEATURES] * 9,
            "minutes": [0, 1, 1, 2, 2, 2, 2, 3, 3],
            "label": 1,
        }
        features = aggregate_sequence_features(record)
        assert len(features) > 0
        assert np.isfinite(features).all()
