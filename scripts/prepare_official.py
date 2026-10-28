"""Build the official CelebAMask-HQ train/val/test split in this repository's layout.

Usage:
    python scripts/prepare_official.py --root path/to/CelebAMask-HQ --out dataset_official
        [--splits test] [--image-size 512] [--partition list_eval_partition.txt] [--workers 8]

Writes ``<out>/<split>/images/<idx>.jpg`` and ``<out>/<split>/masks/<idx>.png``, where
each mask is a single-channel label map (pixel value = class id 0..18) merged from
the per-part annotations. Without ``--partition`` the split is derived from the
thresholds of the release's g_partition.py, which papers use; the split sizes
are printed so they can be checked against 24,183 / 2,993 / 2,824.

Images are copied unchanged (1024 x 1024) unless ``--image-size`` is given; the
model resizes to 512 x 512 anyway, so ``--image-size 512`` saves disk space.
"""

from __future__ import annotations

import argparse
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from PIL import Image
from tqdm import tqdm

from miniunet.celebamask import (
    EXPECTED_SPLIT_SIZES,
    SPLITS,
    image_path,
    merge_part_masks,
    official_split,
)


def prepare_one(idx: int, root: str, img_dir: str, mask_dir: str, image_size: int | None) -> None:
    """Write the image and merged mask of CelebA-HQ image ``idx``."""
    src = image_path(root, idx)
    dst = os.path.join(img_dir, f"{idx}.jpg")
    if image_size:
        with Image.open(src) as im:
            im.convert("RGB").resize((image_size, image_size), Image.BILINEAR).save(dst, quality=95)
    else:
        shutil.copyfile(src, dst)
    Image.fromarray(merge_part_masks(root, idx)).save(os.path.join(mask_dir, f"{idx}.png"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the official CelebAMask-HQ split.")
    parser.add_argument("--root", required=True, help="the extracted CelebAMask-HQ folder")
    parser.add_argument("--out", default="dataset_official")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--image-size", type=int, help="resize images to this square size")
    parser.add_argument("--partition", help="CelebA list_eval_partition.txt (optional)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()

    splits = official_split(args.root, args.partition)
    for name in SPLITS:
        n, expected = len(splits[name]), EXPECTED_SPLIT_SIZES[name]
        note = "" if n == expected else f"  (expected {expected:,})"
        print(f"{name:5s}: {n:,} images{note}")

    for name in args.splits:
        img_dir = os.path.join(args.out, name, "images")
        mask_dir = os.path.join(args.out, name, "masks")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        job = partial(
            prepare_one, root=args.root, img_dir=img_dir, mask_dir=mask_dir,
            image_size=args.image_size,
        )  # fmt: skip
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(
                tqdm(pool.map(job, splits[name], chunksize=32), total=len(splits[name]), desc=name)
            )
        print(f"Wrote {len(splits[name]):,} image/mask pairs to {os.path.join(args.out, name)}")


if __name__ == "__main__":
    main()
