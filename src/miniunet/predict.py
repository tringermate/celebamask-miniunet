"""Predict label masks for a folder of images.

Usage:
    python -m miniunet.predict --checkpoint outputs/default/best.pth \
        --input-dir dataset_official/test/images --output-dir predictions --tta

Each output is a single-channel PNG named after the input image (``.jpg`` -> ``.png``),
resized back to the original image size, with pixel values 0..18.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from miniunet.data import FLIP_PAIRS, FaceParsingDataset, list_images, mask_name
from miniunet.model import build_model
from miniunet.utils import get_device, load_config


def swap_flip_channels(logits: torch.Tensor) -> torch.Tensor:
    """Swap the left/right class channels (4<->5, 6<->7, 8<->9) of ``(N, C, H, W)`` logits."""
    perm = list(range(logits.size(1)))
    for a, b in FLIP_PAIRS:
        perm[a], perm[b] = b, a
    return logits[:, perm]


@torch.no_grad()
def flipped_logits(model: torch.nn.Module, img: torch.Tensor) -> torch.Tensor:
    """Logits of the horizontally flipped batch, flipped back and channel-swapped."""
    return swap_flip_channels(torch.flip(model(torch.flip(img, dims=[3])), dims=[3]))


@torch.no_grad()
def predict_logits(model: torch.nn.Module, img: torch.Tensor, tta: bool = False) -> torch.Tensor:
    """Logits for a batch; with ``tta``, the average with the un-flipped, channel-swapped
    logits of the horizontally flipped batch."""
    logits = model(img)
    if not tta:
        return logits
    return (logits + flipped_logits(model, img)) / 2.0


def load_model(cfg: dict, checkpoint: str, device: torch.device) -> torch.nn.Module:
    """Build the model from ``cfg['model']``, load a state dict and switch to eval mode."""
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    return model


def predict_folder(
    model: torch.nn.Module,
    input_dir: str,
    output_dir: str,
    device: torch.device,
    tta: bool = False,
    size: tuple[int, int] = (512, 512),
    batch_size: int = 4,
) -> list[str]:
    """Write one PNG mask per image in ``input_dir`` to ``output_dir``; return their paths."""
    os.makedirs(output_dir, exist_ok=True)
    files = list_images(input_dir)
    ds = FaceParsingDataset([os.path.join(input_dir, f) for f in files], size=size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    written = []
    for img, names in tqdm(loader, desc="Predict"):
        preds = torch.argmax(predict_logits(model, img.to(device), tta), dim=1)
        preds = preds.cpu().numpy().astype(np.uint8)
        for pred, name in zip(preds, names):
            with Image.open(os.path.join(input_dir, name)) as orig:
                w, h = orig.size
            out_path = os.path.join(output_dir, mask_name(name))
            Image.fromarray(pred).resize((w, h), Image.NEAREST).save(out_path)
            written.append(out_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict face-parsing masks for a folder.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default="predictions")
    parser.add_argument(
        "--tta",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="horizontal-flip TTA (default: inference.tta from the config)",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", help="e.g. cpu or cuda:0")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(args.device or cfg.get("device"))
    tta = cfg.get("inference", {}).get("tta", True) if args.tta is None else args.tta
    model = load_model(cfg, args.checkpoint, device)
    written = predict_folder(
        model,
        args.input_dir,
        args.output_dir,
        device,
        tta=tta,
        size=tuple(cfg["data"].get("image_size", (512, 512))),
        batch_size=args.batch_size,
    )
    print(f"Wrote {len(written)} masks to {args.output_dir} (TTA {'on' if tta else 'off'})")


if __name__ == "__main__":
    main()
