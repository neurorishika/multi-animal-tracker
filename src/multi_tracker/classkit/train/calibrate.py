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


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Measures how well predicted confidence matches accuracy.

    Args:
        probs: (N, num_classes) probabilities
        labels: (N,) ground-truth labels
        n_bins: Number of confidence bins

    Returns:
        ece: Expected Calibration Error (0 = perfect calibration)
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    # Bin by confidence
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Compute ECE
    ece = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() == 0:
            continue

        bin_confidence = confidences[mask].mean()
        bin_accuracy = accuracies[mask].mean()
        bin_weight = mask.sum() / len(labels)

        ece += bin_weight * abs(bin_confidence - bin_accuracy)

    return ece


def get_calibration_curve(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> tuple:
    """
    Compute calibration curve for plotting.

    Args:
        probs: (N, num_classes) probabilities
        labels: (N,) ground-truth labels
        n_bins: Number of bins

    Returns:
        bin_confidences: (n_bins,) mean confidence per bin
        bin_accuracies: (n_bins,) mean accuracy per bin
        bin_counts: (n_bins,) number of samples per bin
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    # Bin by confidence
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_confidences = []
    bin_accuracies = []
    bin_counts = []

    for b in range(n_bins):
        mask = bin_indices == b
        count = mask.sum()

        if count > 0:
            bin_confidences.append(confidences[mask].mean())
            bin_accuracies.append(accuracies[mask].mean())
        else:
            bin_confidences.append(0.0)
            bin_accuracies.append(0.0)

        bin_counts.append(count)

    return (
        np.array(bin_confidences),
        np.array(bin_accuracies),
        np.array(bin_counts),
    )
