"""
Temperature scaling for probability calibration.

Calibrates classifier confidence to match true accuracy.
Reference: "On Calibration of Modern Neural Networks" (Guo et al., 2017)
https://arxiv.org/abs/1706.04599
"""

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import LBFGS
except ImportError:
    torch = None
    nn = None
    F = None
    LBFGS = None
    np = None


class TemperatureScaling:
    """
    Post-hoc calibration via temperature scaling.

    Learns a single temperature parameter T such that
    calibrated_probs = softmax(logits / T)
    """

    def __init__(self):
        self.temperature = 1.0

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        max_iter: int = 50,
        device: str = "cpu",
    ):
        """
        Fit temperature parameter on validation set.

        Args:
            logits: (N, num_classes) raw logits
            labels: (N,) ground-truth labels
            max_iter: Maximum optimization iterations
            device: 'cpu', 'cuda', or 'mps'
        """
        if torch is None:
            raise ImportError("torch required for calibration")

        # Convert to tensors
        logits_tensor = torch.FloatTensor(logits).to(device)
        labels_tensor = torch.LongTensor(labels).to(device)

        # Initialize temperature parameter
        temperature = nn.Parameter(torch.ones(1).to(device))

        # Define NLL loss
        def eval_loss():
            scaled_logits = logits_tensor / temperature
            loss = F.cross_entropy(scaled_logits, labels_tensor)
            return loss

        # Optimize using LBFGS (recommended for temperature scaling)
        optimizer = LBFGS([temperature], lr=0.01, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = eval_loss()
            loss.backward()
            return loss

        optimizer.step(closure)

        # Store learned temperature
        self.temperature = temperature.item()

        # Clamp to reasonable range
        self.temperature = max(0.1, min(self.temperature, 10.0))

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to logits.

        Args:
            logits: (N, num_classes) raw logits

        Returns:
            calibrated_probs: (N, num_classes) calibrated probabilities
        """
        if torch is None:
            raise ImportError("torch required for calibration")

        logits_tensor = torch.FloatTensor(logits)
        scaled_logits = logits_tensor / self.temperature
        probs = F.softmax(scaled_logits, dim=1)

        return probs.numpy()
