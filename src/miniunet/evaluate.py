"""Evaluate a checkpoint, with and without flip TTA.

Usage:
    # validation split of the config
    python -m miniunet.evaluate --config configs/default.yaml --checkpoint outputs/default/best.pth

    # any folder of images and masks, e.g. the official test split
    python -m miniunet.evaluate --checkpoint outputs/default/best.pth \
        --images dataset_official/test/images --masks dataset_official/test/masks --per-class

Four numbers are reported for each setting:
- batch-averaged: F-measure per batch of ``batch_size`` images (classes pooled over
  the batch), averaged over batches. This is the training-time validation metric.
- per-image: F-measure of each image, averaged over images.
- dataset F1: per-class F1 with pixels pooled over the whole set, averaged over all
  19 classes, and over the 18 classes without background. This is how face parsing
  papers usually report CelebAMask-HQ results.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from miniunet.celebamask import CLASS_NAMES
from miniunet.data import FaceParsingDataset, list_pairs
from miniunet.metrics import (
    compute_f_measure,
    compute_f_measure_per_image,
    confusion_matrix,
    f1_per_class,
)
from miniunet.predict import flipped_logits, load_model
from miniunet.train import build_loaders
from miniunet.utils import get_device, load_config


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    tta: bool = False,
    amp: bool = True,
) -> dict:
    """Return the F-measures over ``loader``.

    Keys: ``batch_averaged``, ``per_image``, ``dataset_f1`` (all classes),
    ``dataset_f1_no_bg`` and ``f1_per_class`` (array, NaN for absent classes).
    """
    return evaluate_with_and_without_tta(model, loader, device, amp, settings=(tta,))[tta]


@torch.no_grad()
def evaluate_with_and_without_tta(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    amp: bool = True,
    settings: tuple[bool, ...] = (False, True),
) -> dict[bool, dict]:
    """``evaluate`` for each TTA setting in one pass; the un-flipped forward is shared."""
    model.eval()
    acc = {tta: {"batch": [], "image": [], "cm": 0} for tta in settings}
    for img, mask in loader:
        img = img.to(device)
        with torch.amp.autocast(device.type, enabled=amp and device.type == "cuda"):
            logits = {False: model(img)}
            if True in settings:
                logits[True] = (logits[False] + flipped_logits(model, img)) / 2.0
        for tta in settings:
            preds = torch.argmax(logits[tta], dim=1)
            acc[tta]["batch"].append(compute_f_measure(preds, mask))
            acc[tta]["image"].extend(compute_f_measure_per_image(preds, mask))
            acc[tta]["cm"] = acc[tta]["cm"] + confusion_matrix(preds, mask, logits[tta].size(1))

    results = {}
    for tta, a in acc.items():
        per_class = f1_per_class(a["cm"])
        results[tta] = {
            "batch_averaged": float(np.mean(a["batch"])),
            "per_image": float(np.mean(a["image"])),
            "dataset_f1": float(np.nanmean(per_class)),
            "dataset_f1_no_bg": float(np.nanmean(per_class[1:])),
            "f1_per_class": per_class,
        }
    return results


def folder_loader(cfg: dict, img_dir: str, mask_dir: str, device: torch.device) -> DataLoader:
    """Loader over every image in ``img_dir`` with its mask in ``mask_dir``."""
    imgs, masks = list_pairs(img_dir, mask_dir)
    ds = FaceParsingDataset(imgs, masks, size=tuple(cfg["data"].get("image_size", (512, 512))))
    return DataLoader(
        ds,
        batch_size=cfg["data"].get("batch_size", 4),
        num_workers=cfg["data"].get("num_workers", 0),
        pin_memory=device.type == "cuda",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="F-measure of a checkpoint.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images", help="evaluate this image folder instead of the val split")
    parser.add_argument("--masks", help="mask folder for --images")
    parser.add_argument("--per-class", action="store_true", help="print the F1 of each class")
    parser.add_argument("--device", help="e.g. cpu or cuda:0")
    args = parser.parse_args()
    if bool(args.images) != bool(args.masks):
        parser.error("--images and --masks go together")

    cfg = load_config(args.config)
    device = get_device(args.device or cfg.get("device"))
    if args.images:
        loader = folder_loader(cfg, args.images, args.masks, device)
    else:
        _, loader = build_loaders(cfg, device)
    model = load_model(cfg, args.checkpoint, device)
    amp = bool(cfg["train"].get("amp", True))

    print(f"Images: {len(loader.dataset)}")
    results = evaluate_with_and_without_tta(model, loader, device, amp)
    for tta, scores in results.items():
        print(
            f"TTA {'on ' if tta else 'off'} | batch-averaged F {scores['batch_averaged']:.4f}"
            f" | per-image F {scores['per_image']:.4f}"
            f" | dataset F1 {scores['dataset_f1']:.4f}"
            f" | dataset F1 w/o background {scores['dataset_f1_no_bg']:.4f}"
        )

    if args.per_class:
        print(f"\n{'class':12s} {'TTA off':>8s} {'TTA on':>8s}")
        for c, (off, on) in enumerate(
            zip(results[False]["f1_per_class"], results[True]["f1_per_class"])
        ):
            name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c)
            print(f"{name:12s} {off:8.4f} {on:8.4f}")


if __name__ == "__main__":
    main()
