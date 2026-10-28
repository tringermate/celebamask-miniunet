"""F-measures: per image or batch, and pooled over a whole dataset.

The per-image (or per-batch) F-measure averages F over the classes present. Face
parsing papers usually pool true/false positives and negatives over the whole test
set first and then average the per-class F1 (``confusion_matrix`` + ``f1_per_class``).
"""

from __future__ import annotations

import numpy as np
import torch


def compute_multiclass_fscore(mask_gt: np.ndarray, mask_pred: np.ndarray, beta: float = 1) -> float:
    """Mean F-beta score over the classes that occur in ``mask_gt``.

    Classes that appear only in the prediction are not averaged over (their false
    positives still lower the precision of the present classes). If ``mask_gt`` is
    a batch, the classes are pooled over the whole batch.
    """
    f_scores = []

    for class_id in np.unique(mask_gt):
        tp = np.sum((mask_gt == class_id) & (mask_pred == class_id))
        fp = np.sum((mask_gt != class_id) & (mask_pred == class_id))
        fn = np.sum((mask_gt == class_id) & (mask_pred != class_id))

        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f_score = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-7)

        f_scores.append(f_score)

    if len(f_scores) == 0:
        return 0.0
    return float(np.mean(f_scores))


def compute_f_measure(
    preds: torch.Tensor | np.ndarray, targets: torch.Tensor | np.ndarray, beta: float = 1
) -> float:
    """``compute_multiclass_fscore`` for tensors or arrays (prediction first, as in training)."""
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    return compute_multiclass_fscore(targets, preds, beta=beta)


def compute_f_measure_per_image(
    preds: torch.Tensor | np.ndarray, targets: torch.Tensor | np.ndarray, beta: float = 1
) -> list[float]:
    """F-measure of each image in a batch separately."""
    return [compute_f_measure(p, t, beta) for p, t in zip(preds, targets)]


def confusion_matrix(
    preds: torch.Tensor | np.ndarray, targets: torch.Tensor | np.ndarray, n_classes: int
) -> np.ndarray:
    """``(n_classes, n_classes)`` pixel counts, rows = ground truth, columns = prediction."""
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    idx = targets.astype(np.int64).ravel() * n_classes + preds.astype(np.int64).ravel()
    return np.bincount(idx, minlength=n_classes**2).reshape(n_classes, n_classes)


def f1_per_class(cm: np.ndarray) -> np.ndarray:
    """F1 of each class from a confusion matrix; NaN for a class absent from both
    ground truth and prediction."""
    tp = np.diag(cm).astype(np.float64)
    denom = 2 * tp + (cm.sum(axis=0) - tp) + (cm.sum(axis=1) - tp)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, 2 * tp / denom, np.nan)
