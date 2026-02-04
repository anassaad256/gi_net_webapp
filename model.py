"""
ABMIL (Attention-Based Multiple Instance Learning) model for Ki-67 grade prediction.

This module implements the attention-based MIL classifier that takes tile features
from H-optimus-0 and predicts Ki-67 proliferation grade (G1 vs G2+G3).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ABMIL(nn.Module):
    """
    Attention-Based Multiple Instance Learning for Ki-67 grade prediction.

    Input: (batch, n_tiles, 1536) - tile features from H-optimus-0
    Output: (batch, 2) - logits for [G1, G2+G3]

    Args:
        feature_dim: Dimension of input features (1536 for H-optimus-0)
        hidden_dim: Hidden dimension for encoder and classifier
        n_classes: Number of output classes (2 for binary G1 vs G2+G3)
        dropout: Dropout rate for regularization
    """

    def __init__(
        self,
        feature_dim: int = 1536,
        hidden_dim: int = 256,
        n_classes: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()

        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the ABMIL model.

        Args:
            x: Input tensor of shape (batch, n_tiles, feature_dim)
            mask: Optional mask tensor of shape (batch, n_tiles) where 1 indicates
                  valid tiles and 0 indicates padding

        Returns:
            logits: Classification logits of shape (batch, n_classes)
            attn_weights: Attention weights of shape (batch, n_tiles)
        """
        # Encode tile features
        h = self.encoder(x)  # (batch, n_tiles, hidden_dim)

        # Compute attention scores
        attn = self.attention(h).squeeze(-1)  # (batch, n_tiles)

        # Apply mask if provided (for padded sequences)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        # Normalize attention weights
        attn_weights = F.softmax(attn, dim=1)

        # Weighted aggregation of tile features
        out = torch.bmm(attn_weights.unsqueeze(1), h).squeeze(1)  # (batch, hidden_dim)

        # Classification
        logits = self.classifier(out)

        return logits, attn_weights
