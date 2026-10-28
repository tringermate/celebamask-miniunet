"""Plot training curves from a ``history.csv`` written by ``miniunet.train``.

Usage:
    python scripts/plot_history.py outputs/default/history.csv [--out docs/training_curves.png]

Requires matplotlib (not in requirements.txt: ``pip install matplotlib``).
"""

from __future__ import annotations

import argparse
import csv


def plot_history(history_csv: str, out_path: str) -> None:
    """Save a two-panel figure: validation F-measure and training loss, and learning rate."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(history_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    epoch = [int(r["epoch"]) for r in rows]
    loss = [float(r["train_loss"]) for r in rows]
    val_f = [float(r["val_f_measure"]) for r in rows]
    lr = [float(r["lr"]) for r in rows]

    fig, (ax1, ax3) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax1.plot(epoch, val_f, color="tab:green", label="validation F-measure")
    ax1.set_ylabel("validation F-measure")
    ax2 = ax1.twinx()
    ax2.plot(epoch, loss, color="tab:red", linestyle="--", label="training loss")
    ax2.set_ylabel("training loss")
    best = max(range(len(val_f)), key=val_f.__getitem__)
    ax1.set_title(f"Best validation F-measure {val_f[best]:.4f} at epoch {epoch[best]}")
    ax1.legend(loc="lower left")
    ax2.legend(loc="lower right")

    ax3.plot(epoch, lr, color="tab:brown")
    ax3.set_ylabel("learning rate")
    ax3.set_xlabel("epoch")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training curves from history.csv.")
    parser.add_argument("history_csv")
    parser.add_argument("--out", default="training_curves.png")
    args = parser.parse_args()
    plot_history(args.history_csv, args.out)


if __name__ == "__main__":
    main()
