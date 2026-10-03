"""Temperature tuning and Expected Calibration Error (ECE) for O-Laya decision heads.

Implements post-hoc temperature scaling (Guo et al. 2017) to ensure predicted
confidence scores match empirical accuracy:
1. Expected Calibration Error (ECE) with adaptive or uniform binning
2. Maximum Calibration Error (MCE)
3. Platt / Temperature Scaling via Negative Log-Likelihood minimization
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class CalibrationMetrics:
    """Summary metrics of model calibration."""

    ece: float
    mce: float
    avg_confidence: float
    accuracy: float
    nll: float
    bin_accuracies: list[float]
    bin_confidences: list[float]
    bin_counts: list[int]

    def summary(self) -> str:
        return (
            f"Acc: {self.accuracy:.2%} | Conf: {self.avg_confidence:.2%} | "
            f"ECE: {self.ece:.4f} ({self.ece:.2%}) | MCE: {self.mce:.4f} | NLL: {self.nll:.4f}"
        )


def compute_ece(
    probs: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    num_bins: int = 10,
) -> CalibrationMetrics:
    """Compute Expected Calibration Error (ECE) and binning statistics.

    Args:
        probs: Predicted class probabilities [N, K] or binary confidences [N].
        targets: Ground-truth class indices [N].
        num_bins: Number of confidence bins in [0, 1].

    Returns:
        CalibrationMetrics dataclass with ECE, MCE, and per-bin stats.
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    if probs.ndim == 2:
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
    else:
        confidences = probs
        predictions = (confidences >= 0.5).astype(int)

    correct = (predictions == targets).astype(float)
    n = len(targets)
    if n == 0:
        return CalibrationMetrics(0.0, 0.0, 0.0, 0.0, 0.0, [], [], [])

    # Calculate NLL
    eps = 1e-12
    if probs.ndim == 2:
        target_probs = probs[np.arange(n), targets.astype(int)]
        nll = float(-np.mean(np.log(np.clip(target_probs, eps, 1.0))))
    else:
        p = np.clip(probs, eps, 1.0 - eps)
        nll = float(-np.mean(targets * np.log(p) + (1 - targets) * np.log(1 - p)))

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_accuracies: list[float] = []
    bin_confidences: list[float] = []
    bin_counts: list[int] = []

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Samples within this bin
        if i == num_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        count = int(np.sum(in_bin))
        bin_counts.append(count)

        if count > 0:
            bin_acc = float(np.mean(correct[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            bin_accuracies.append(bin_acc)
            bin_confidences.append(bin_conf)

            diff = abs(bin_acc - bin_conf)
            ece += (count / n) * diff
            mce = max(mce, diff)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append((bin_lower + bin_upper) / 2.0)

    accuracy = float(np.mean(correct))
    avg_confidence = float(np.mean(confidences))

    return CalibrationMetrics(
        ece=float(ece),
        mce=float(mce),
        avg_confidence=avg_confidence,
        accuracy=accuracy,
        nll=nll,
        bin_accuracies=bin_accuracies,
        bin_confidences=bin_confidences,
        bin_counts=bin_counts,
    )


class TemperatureScaler(nn.Module):
    """Post-hoc temperature scaling layer for multi-class and binary logits."""

    def __init__(self, init_temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temp = self.temperature.clamp(min=1e-3)
        return logits / temp


def fit_temperature(
    logits: torch.Tensor,
    targets: torch.Tensor,
    init_temp: float = 1.0,
    max_iter: int = 50,
    candidate_mask: torch.Tensor | None = None,
) -> tuple[float, CalibrationMetrics, CalibrationMetrics]:
    """Fit temperature parameter via L-BFGS to minimize Negative Log-Likelihood.

    Args:
        logits: Uncalibrated logits [N, K].
        targets: Ground truth class indices [N].
        init_temp: Initial temperature guess.
        max_iter: Maximum optimization iterations.

    Returns:
        (optimal_temperature, uncalibrated_metrics, calibrated_metrics)
    """
    scaler = TemperatureScaler(init_temp)
    criterion = nn.CrossEntropyLoss()

    logits = logits.detach()
    targets = targets.detach()

    def scaled_logits() -> torch.Tensor:
        scaled = scaler(logits)
        return scaled.masked_fill(~candidate_mask, -torch.inf) if candidate_mask is not None else scaled

    # Mask after scaling so invalid slots never acquire mass at large temperatures.
    initial = logits.masked_fill(~candidate_mask, -torch.inf) if candidate_mask is not None else logits
    init_probs = torch.softmax(initial, dim=-1)
    uncalibrated_metrics = compute_ece(init_probs, targets)

    optimizer = torch.optim.LBFGS([scaler.temperature], lr=0.01, max_iter=max_iter)

    def eval_loss():
        optimizer.zero_grad()
        loss = criterion(scaled_logits(), targets)
        loss.backward()
        return loss

    optimizer.step(eval_loss)

    optimal_t = float(scaler.temperature.item())

    # Calculate post-calibration metrics
    with torch.no_grad():
        calibrated_logits = scaled_logits()
        calibrated_probs = torch.softmax(calibrated_logits, dim=-1)
        calibrated_metrics = compute_ece(calibrated_probs, targets)

    return optimal_t, uncalibrated_metrics, calibrated_metrics


def pad_and_cat_logits(logits_list: list[torch.Tensor], pad_value: float = -1e4) -> torch.Tensor:
    """Concatenate a list of candidate logits batches that may have variable candidate counts.

    Pads candidate dimension to the maximum observed across batches with `pad_value`.
    """
    if not logits_list:
        return torch.empty(0)
    max_c = max(t.shape[1] for t in logits_list)
    padded = []
    for t in logits_list:
        if t.shape[1] < max_c:
            pad_size = max_c - t.shape[1]
            pad_tensor = torch.full((t.shape[0], pad_size), pad_value, dtype=t.dtype, device=t.device)
            t = torch.cat([t, pad_tensor], dim=1)
        padded.append(t)
    return torch.cat(padded, dim=0)
