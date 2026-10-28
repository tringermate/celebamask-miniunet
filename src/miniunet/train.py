"""Train the Mini-U-Net.

Usage:
    python -m miniunet.train --config configs/default.yaml [--epochs N] [--output-dir DIR]

Writes ``best_model_Ep{epoch}_F{score}.pth`` every time the validation F-measure
improves, a copy of the latest best as ``best.pth``, the resolved config as
``config.yaml`` and the per-epoch history as ``history.csv``.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from miniunet.data import FaceParsingDataset, list_pairs, split_pairs, subset_pairs
from miniunet.losses import build_loss
from miniunet.metrics import compute_f_measure
from miniunet.model import build_model, count_parameters
from miniunet.utils import get_device, load_config, seed_everything


def _seed_worker(worker_id: int) -> None:
    """Seed NumPy in each DataLoader worker (PyTorch already seeds ``random`` and torch)."""
    np.random.seed(torch.initial_seed() % 2**32)


def build_loaders(cfg: dict, device: torch.device) -> tuple[DataLoader, DataLoader]:
    """Build the training (augmented, shuffled) and validation loaders from the config.

    ``data.train_limit`` keeps a fixed random subset of that many images of the
    training folder (drawn with ``split_seed``). With ``data.val_images`` set,
    validation uses that folder (and ``data.val_masks``) and the whole subset is
    trained on; otherwise ``val_fraction`` of the subset is split off for validation.
    """
    data_cfg = cfg["data"]
    seed = data_cfg.get("split_seed", 42)
    pool = subset_pairs(
        *list_pairs(data_cfg["train_images"], data_cfg["train_masks"]),
        data_cfg.get("train_limit"),
        seed,
    )
    if data_cfg.get("val_images"):
        tr_imgs, tr_masks = pool
        va_imgs, va_masks = list_pairs(data_cfg["val_images"], data_cfg["val_masks"])
    else:
        (tr_imgs, tr_masks), (va_imgs, va_masks) = split_pairs(
            *pool, data_cfg.get("val_fraction", 0.1), seed
        )
    size = tuple(data_cfg.get("image_size", (512, 512)))
    train_ds = FaceParsingDataset(
        tr_imgs, tr_masks, augment=True, size=size, aug_cfg=cfg.get("augmentation")
    )
    val_ds = FaceParsingDataset(va_imgs, va_masks, augment=False, size=size)

    loader_kwargs = dict(
        batch_size=data_cfg.get("batch_size", 4),
        num_workers=data_cfg.get("num_workers", 0),
        pin_memory=device.type == "cuda",
        worker_init_fn=_seed_worker,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> float:
    """Validation F-measure used for model selection.

    The F-measure is computed per batch (classes pooled over the batch) and then
    averaged over batches.
    """
    model.eval()
    total = 0.0
    for img, mask in loader:
        with torch.amp.autocast(device.type, enabled=amp):
            preds = torch.argmax(model(img.to(device)), dim=1)
        total += compute_f_measure(preds, mask)
    return total / len(loader)


def train(cfg: dict) -> dict:
    """Run a full training according to ``cfg``.

    Returns:
        A dict with ``best_f1``, ``best_epoch``, ``best_checkpoint``, ``n_params`` and
        ``output_dir``.
    """
    seed_everything(cfg.get("seed", 42))
    device = get_device(cfg.get("device"))
    out_dir = cfg["train"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    train_loader, val_loader = build_loaders(cfg, device)

    n_classes = cfg["model"].get("n_classes", 19)
    model = build_model(cfg["model"]).to(device)
    n_params = count_parameters(model)
    print(f"Device: {device} | trainable parameters: {n_params:,}")
    print(f"Train images: {len(train_loader.dataset)} | val images: {len(val_loader.dataset)}")

    loss_fn = build_loss(cfg["loss"], n_classes, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optimizer"]["lr"],
        weight_decay=cfg["optimizer"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg["scheduler"]["T_0"], T_mult=cfg["scheduler"]["T_mult"]
    )
    amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    epochs = cfg["train"]["epochs"]
    best_f1, best_epoch, best_path = 0.0, 0, None
    history_path = os.path.join(out_dir, "history.csv")
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_f_measure", "lr"])

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for img, mask in tqdm(train_loader, desc=f"Ep {epoch}/{epochs}", leave=False):
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device.type, enabled=amp):
                logits = model(img)
                loss = loss_fn(logits, mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        val_f1 = validate(model, val_loader, device, amp)
        lr_used = optimizer.param_groups[0]["lr"]
        scheduler.step()

        with open(history_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.6f}", f"{val_f1:.6f}", f"{lr_used:.8f}"])

        msg = f"Ep {epoch}/{epochs} | train loss {train_loss:.4f} | val F {val_f1:.4f}"
        msg += f" | lr {lr_used:.6f}"
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            best_path = os.path.join(out_dir, f"best_model_Ep{epoch}_F{best_f1:.4f}.pth")
            torch.save(model.state_dict(), best_path)
            shutil.copyfile(best_path, os.path.join(out_dir, "best.pth"))
            msg += " | new best, saved"
        print(msg)

    print(f"Best validation F-measure {best_f1:.4f} at epoch {best_epoch}")
    return {
        "best_f1": best_f1,
        "best_epoch": best_epoch,
        "best_checkpoint": best_path,
        "n_params": n_params,
        "output_dir": out_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Mini-U-Net face parser.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, help="override train.epochs")
    parser.add_argument("--output-dir", help="override train.output_dir")
    parser.add_argument("--seed", type=int, help="override seed")
    parser.add_argument("--device", help="override device, e.g. cpu or cuda:0")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.output_dir:
        cfg["train"]["output_dir"] = args.output_dir
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.device:
        cfg["device"] = args.device
    train(cfg)


if __name__ == "__main__":
    main()
