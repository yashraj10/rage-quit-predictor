"""
RageQuitTransformer: Custom transformer encoder for behavioral sequence classification.

Architecture:
    Input → Token Embedding + Continuous Feature Projection + Game-Time Positional Encoding
        → Transformer Encoder (N layers)
        → [CLS] token representation
        → Classification Head
        → P(rage_quit)

Key design decisions:
    1. Game-time positional encoding (not sequence position) — events are unevenly
       distributed across time, so encoding by actual game minute preserves temporal structure.
    2. Dual embedding — discrete event tokens + continuous features are projected
       separately then fused, giving the model both symbolic and numerical signal.
    3. [CLS] token pooling — standard approach for sequence-level classification.
"""

import math

import torch
import torch.nn as nn


class BehavioralTokenEmbedding(nn.Module):
    """Combines discrete event embedding with projected continuous features."""

    def __init__(self, vocab_size: int, embed_dim: int, num_continuous_features: int):
        super().__init__()
        self.event_embedding = nn.Embedding(vocab_size, embed_dim)
        self.continuous_proj = nn.Linear(num_continuous_features, embed_dim)
        self.combine = nn.Linear(embed_dim * 2, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, event_ids: torch.Tensor, continuous_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            event_ids: (B, seq_len) — discrete event token IDs
            continuous_features: (B, seq_len, num_features) — continuous features per token
        Returns:
            (B, seq_len, embed_dim)
        """
        event_emb = self.event_embedding(event_ids)
        cont_emb = self.continuous_proj(continuous_features)
        combined = torch.cat([event_emb, cont_emb], dim=-1)
        combined = self.combine(combined)
        return self.layer_norm(combined)


class GameTimePositionalEncoding(nn.Module):
    """
    Positional encoding based on actual game time (minutes), not sequence position.

    This is critical because events are unevenly distributed across game time.
    A death at minute 5 and a death at minute 25 should have different positional
    encodings even if they're adjacent in the sequence.
    """

    def __init__(self, embed_dim: int, max_minutes: int = 90, dropout: float = 0.1):
        super().__init__()
        self.minute_embedding = nn.Embedding(max_minutes, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, minutes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, embed_dim) — embedded tokens
            minutes: (B, seq_len) — game minute for each token
        Returns:
            (B, seq_len, embed_dim)
        """
        pos_emb = self.minute_embedding(minutes)
        return self.dropout(x + pos_emb)


class RageQuitTransformer(nn.Module):
    """Custom transformer for predicting rage quits from behavioral sequences."""

    def __init__(
        self,
        vocab_size: int = 22,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        num_continuous_features: int = 6,
        ff_dim: int = 512,
        max_seq_len: int = 256,
        max_minutes: int = 90,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Embedding layers
        self.token_embedding = BehavioralTokenEmbedding(
            vocab_size, embed_dim, num_continuous_features
        )
        self.positional_encoding = GameTimePositionalEncoding(
            embed_dim, max_minutes, dropout
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize with Xavier uniform for linear layers, normal for embeddings."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        event_ids: torch.Tensor,
        continuous_features: torch.Tensor,
        minutes: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            event_ids: (B, seq_len) — discrete event token IDs
            continuous_features: (B, seq_len, num_features) — continuous features
            minutes: (B, seq_len) — game minute for each token
            attention_mask: (B, seq_len) — 1 for real tokens, 0 for padding

        Returns:
            logits: (B, 1) — raw logits for rage quit probability
        """
        # Embed tokens (discrete + continuous fusion)
        x = self.token_embedding(event_ids, continuous_features)

        # Add game-time positional encoding
        x = self.positional_encoding(x, minutes)

        # Padding mask for transformer (True = positions to IGNORE)
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0

        # Transformer encoding
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # [CLS] token representation (position 0)
        cls_repr = x[:, 0, :]

        # Classification
        logits = self.classifier(cls_repr)
        return logits

    def get_attention_weights(
        self,
        event_ids: torch.Tensor,
        continuous_features: torch.Tensor,
        minutes: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """
        Extract attention weights from all layers for interpretability.

        Returns:
            List of (B, num_heads, seq_len, seq_len) tensors, one per layer.
        """
        self.eval()
        attention_maps = []

        # Temporarily enable need_weights on self-attention layers
        hooks = []
        for layer in self.transformer_encoder.layers:
            original_fn = layer.self_attn.forward

            def make_hook(orig_fn):
                def hooked_forward(*args, **kwargs):
                    kwargs["need_weights"] = True
                    kwargs["average_attn_weights"] = False
                    output, weights = orig_fn(*args, **kwargs)
                    attention_maps.append(weights.detach())
                    return output, weights
                return hooked_forward

            layer.self_attn.forward = make_hook(original_fn)
            hooks.append((layer.self_attn, original_fn))

        with torch.no_grad():
            _ = self.forward(event_ids, continuous_features, minutes, attention_mask)

        # Restore original forward methods
        for attn_module, orig_fn in hooks:
            attn_module.forward = orig_fn

        return attention_maps

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
