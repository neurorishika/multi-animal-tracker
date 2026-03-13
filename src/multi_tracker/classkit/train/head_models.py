"""
Lightweight classifier heads for embeddings.

These are trained on top of frozen embeddings for fast iteration.
"""

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

from typing import Optional


class LinearHead(nn.Module):
    """Simple linear classifier on embeddings."""

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(input_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)


class MLPHead(nn.Module):
    """Small MLP classifier on embeddings."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Build hidden layers
        layers = []
        current_dim = input_dim

        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))

            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            else:
                raise ValueError(f"Unknown activation: {activation}")

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            current_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(current_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)


def create_head(
    model_type: str,
    input_dim: int,
    num_classes: int,
    hidden_dim: Optional[int] = None,
    dropout: float = 0.1,
    **kwargs,
) -> nn.Module:
    """
    Factory for creating classifier heads.

    Args:
        model_type: 'linear' or 'mlp'
        input_dim: Embedding dimension
        num_classes: Number of classes
        hidden_dim: Hidden dimension for MLP
        dropout: Dropout rate
        **kwargs: Additional arguments

    Returns:
        PyTorch model
    """
    if torch is None:
        raise ImportError("torch required for head models")

    if model_type == "linear":
        return LinearHead(input_dim, num_classes, dropout=dropout)

    elif model_type == "mlp":
        if hidden_dim is None:
            hidden_dim = 512
        return MLPHead(
            input_dim,
            num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
            **kwargs,
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")
