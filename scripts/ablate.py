"""Ablation: retrain with one component switched off at a time.

Usage:
    python scripts/ablate.py --config configs/default.yaml [--epochs 105] [--only use_cbam aug_flip]
                             [--skip-full]

For every switch below, a copy of the config with that switch set to false is
trained into ``outputs/ablation/<switch>``. The full model is trained too (unless
``--skip-full``) and its best checkpoint is evaluated with and without flip TTA,
which gives the ``tta`` row. Results go to ``outputs/ablation/summary.csv``.

Removing a module (CBAM, attention gates, dilated context) changes the parameter
count; the count of each run is recorded in the summary.
"""

from __future__ import annotations

import argparse
import copy
import csv
import os

from miniunet.evaluate import evaluate
from miniunet.predict import load_model
from miniunet.train import build_loaders, train
from miniunet.utils import get_device, load_config

# switch name -> config section that holds it
SWITCHES: dict[str, str] = {
    "use_cbam": "model",
    "use_attention_gates": "model",
    "use_dilated_context": "model",
    "use_dice_loss": "loss",
    "use_class_weights": "loss",
    "aug_flip": "augmentation",
    "aug_rotate": "augmentation",
    "aug_elastic": "augmentation",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one model per switched-off component.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, help="override train.epochs for every run")
    parser.add_argument("--output-root", default="outputs/ablation")
    parser.add_argument("--only", nargs="+", choices=list(SWITCHES), help="run only these")
    parser.add_argument("--skip-full", action="store_true", help="do not train the full model")
    args = parser.parse_args()

    base = load_config(args.config)
    if args.epochs is not None:
        base["train"]["epochs"] = args.epochs
    os.makedirs(args.output_root, exist_ok=True)

    rows = []
    runs = [] if args.skip_full else [("full", None)]
    runs += [(name, SWITCHES[name]) for name in (args.only or SWITCHES)]

    for name, section in runs:
        cfg = copy.deepcopy(base)
        if section is not None:
            cfg[section][name] = False
        cfg["train"]["output_dir"] = os.path.join(args.output_root, name)
        print(f"=== Run: {name if section is None else name + '=false'} ===")
        result = train(cfg)
        rows.append(
            {
                "run": name if section is None else f"{name}=false",
                "n_params": result["n_params"],
                "best_val_f_measure": f"{result['best_f1']:.4f}",
                "best_epoch": result["best_epoch"],
            }
        )

        if name == "full" and result["best_checkpoint"]:
            device = get_device(cfg.get("device"))
            _, val_loader = build_loaders(cfg, device)
            model = load_model(cfg, result["best_checkpoint"], device)
            amp = bool(cfg["train"].get("amp", True))
            for tta in (False, True):
                scores = evaluate(model, val_loader, device, tta=tta, amp=amp)
                rows.append(
                    {
                        "run": f"full, tta={'on' if tta else 'off'}",
                        "n_params": result["n_params"],
                        "best_val_f_measure": f"{scores['batch_averaged']:.4f}",
                        "best_epoch": result["best_epoch"],
                    }
                )

        summary_path = os.path.join(args.output_root, "summary.csv")
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    print(f"Summary written to {os.path.join(args.output_root, 'summary.csv')}")


if __name__ == "__main__":
    main()
