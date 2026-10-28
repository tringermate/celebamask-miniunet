"""Optuna search over the three loss-weight groups (background, normal, tiny classes).

Usage:
    python -m miniunet.tune_weights --config configs/default.yaml [--n-trials 10]

Each trial trains a fresh model for ``tuning.epochs_per_trial`` epochs with AdamW
(no scheduler) and is scored by the batch-averaged
validation F-measure. The best weights are printed and written to
``<train.output_dir>/best_loss_weights.yaml``.

Note: the weights and the final checkpoint are both selected on the same
validation split, so the resulting validation score is optimistic.
"""

from __future__ import annotations

import argparse
import copy
import os

import optuna
import torch
import yaml

from miniunet.losses import build_loss
from miniunet.model import build_model
from miniunet.train import build_loaders, validate
from miniunet.utils import get_device, load_config, seed_everything


def run_trial(cfg: dict, train_loader, val_loader, device: torch.device, epochs: int) -> float:
    """Train for ``epochs`` with the loss weights in ``cfg['loss']``; return validation F."""
    model = build_model(cfg["model"]).to(device)
    loss_fn = build_loss(cfg["loss"], cfg["model"].get("n_classes", 19), device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optimizer"]["lr"],
        weight_decay=cfg["optimizer"]["weight_decay"],
    )
    amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    for _ in range(epochs):
        model.train()
        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device.type, enabled=amp):
                loss = loss_fn(model(img), mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    return validate(model, val_loader, device, amp)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna search over the loss weights.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n-trials", type=int, help="override tuning.n_trials")
    parser.add_argument("--epochs", type=int, help="override tuning.epochs_per_trial")
    parser.add_argument("--device", help="e.g. cpu or cuda:0")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["tuning"]
    n_trials = args.n_trials or tcfg["n_trials"]
    epochs = args.epochs or tcfg["epochs_per_trial"]
    seed_everything(cfg.get("seed", 42))
    device = get_device(args.device or cfg.get("device"))
    train_loader, val_loader = build_loaders(cfg, device)

    def objective(trial: optuna.Trial) -> float:
        trial_cfg = copy.deepcopy(cfg)
        trial_cfg["loss"]["use_class_weights"] = True
        trial_cfg["loss"]["bg_weight"] = trial.suggest_float("bg_weight", *tcfg["bg_weight"])
        trial_cfg["loss"]["normal_weight"] = trial.suggest_float(
            "normal_weight", *tcfg["normal_weight"]
        )
        trial_cfg["loss"]["tiny_weight"] = trial.suggest_float("tiny_weight", *tcfg["tiny_weight"])
        return run_trial(trial_cfg, train_loader, val_loader, device, epochs)

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.get("seed", 42))
    )
    study.optimize(objective, n_trials=n_trials)

    print(f"Best validation F-measure: {study.best_value:.4f}")
    print(f"Best loss weights: {study.best_params}")
    out_dir = cfg["train"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "best_loss_weights.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(study.best_params, f)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
