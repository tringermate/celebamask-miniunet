"""Weighted cross-entropy + weighted Dice loss."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class MulticlassDiceLoss(nn.Module):
    """Soft Dice loss over softmax probabilities, optionally weighted per class.

    With weights ``w``: ``1 - sum_{n,c} w_c * dice_{n,c} / (sum_c w_c * N)``.
    Without weights: ``1 - mean_{n,c} dice_{n,c}``.

    Args:
        num_classes: Number of classes.
        weight: Optional tensor of shape ``(num_classes,)``.
    """

    def __init__(self, num_classes: int = 19, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = (
            F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        )

        intersection = torch.sum(probs * targets_one_hot, dim=(2, 3))
        cardinality = torch.sum(probs + targets_one_hot, dim=(2, 3))

        dice = (2.0 * intersection + 1e-6) / (cardinality + 1e-6)

        if self.weight is not None:
            weighted_dice = (dice * self.weight).sum() / (self.weight.sum() * logits.size(0))
            return 1.0 - weighted_dice

        return 1.0 - dice.mean()


def build_class_weights(
    bg_weight: float,
    normal_weight: float,
    tiny_weight: float,
    tiny_classes: list[int] | tuple[int, ...] = (15, 16),
    num_classes: int = 19,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Class weights: ``bg_weight`` for class 0, ``tiny_weight`` for ``tiny_classes``
    (15 earrings, 16 necklace), ``normal_weight`` for every other class."""
    weights = torch.ones(num_classes, device=device)
    weights[0] = bg_weight
    weights[1:] = normal_weight
    for c in tiny_classes:
        weights[c] = tiny_weight
    return weights


def build_loss(
    loss_cfg: dict, num_classes: int = 19, device: torch.device | str = "cpu"
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build ``CE(weight=w) + Dice(weight=w)`` from the ``loss`` section of the config.

    ``use_class_weights: false`` drops ``w`` from both terms; ``use_dice_loss: false``
    drops the Dice term. Both are ablation switches and default to True.
    """
    weights = None
    if loss_cfg.get("use_class_weights", True):
        weights = build_class_weights(
            loss_cfg["bg_weight"],
            loss_cfg["normal_weight"],
            loss_cfg["tiny_weight"],
            loss_cfg.get("tiny_classes", (15, 16)),
            num_classes,
            device,
        )

    ce_loss_fn = nn.CrossEntropyLoss(weight=weights)
    dice_loss_fn = (
        MulticlassDiceLoss(num_classes, weight=weights)
        if loss_cfg.get("use_dice_loss", True)
        else None
    )

    def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = ce_loss_fn(logits, targets)
        if dice_loss_fn is not None:
            loss = loss + dice_loss_fn(logits, targets)
        return loss

    return loss_fn
