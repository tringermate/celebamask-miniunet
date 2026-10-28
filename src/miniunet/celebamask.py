"""Readers for the original CelebAMask-HQ release: label order, official split, mask merging.

Expected layout of the download (``root``)::

    CelebA-HQ-img/                      0.jpg ... 29999.jpg (1024 x 1024)
    CelebAMask-HQ-mask-anno/0..14/      00000_skin.png, 00000_hair.png, ... (512 x 512)
    CelebA-HQ-to-CelebA-mapping.txt     idx, orig_idx, orig_file

The official split is the one made by the release's ``face_parsing/Data_preprocessing/
g_partition.py``: 24,183 train / 2,993 val / 2,824 test, from thresholds on the
``orig_idx`` column of the mapping file.
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image

# Part names in the order of the official g_mask.py; class id = position + 1 (0 = background).
# A later part overwrites an earlier one where they overlap.
PART_NAMES: tuple[str, ...] = (
    "skin", "nose", "eye_g", "l_eye", "r_eye", "l_brow", "r_brow", "l_ear", "r_ear",
    "mouth", "u_lip", "l_lip", "hair", "hat", "ear_r", "neck_l", "neck", "cloth",
)  # fmt: skip
CLASS_NAMES: tuple[str, ...] = ("background", *PART_NAMES)

SPLITS: tuple[str, ...] = ("train", "val", "test")
EXPECTED_SPLIT_SIZES: dict[str, int] = {"train": 24183, "val": 2993, "test": 2824}

# Thresholds of g_partition.py, applied to the 0-based orig_idx. (CelebA's own
# partition starts one image earlier, at file numbers 162771 and 182638.)
VAL_START, TEST_START = 162771, 182638

IMAGE_DIR = "CelebA-HQ-img"
ANNO_DIR = "CelebAMask-HQ-mask-anno"
MAPPING_FILE = "CelebA-HQ-to-CelebA-mapping.txt"
ANNO_PER_FOLDER = 2000


def _stem_number(name: str) -> int:
    return int(os.path.splitext(os.path.basename(name))[0])


def read_partition(path: str) -> dict[int, str]:
    """Read CelebA's ``list_eval_partition`` (txt or csv): original number -> split name."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            fields = line.replace(",", " ").split()
            if len(fields) == 2 and fields[1].isdigit():
                out[_stem_number(fields[0])] = SPLITS[int(fields[1])]
    return out


def threshold_split(orig_idx: int) -> str:
    """Split of a CelebA-HQ image as in g_partition.py, from its ``orig_idx``."""
    if orig_idx >= TEST_START:
        return "test"
    if orig_idx >= VAL_START:
        return "val"
    return "train"


def official_split(root: str, partition_file: str | None = None) -> dict[str, list[int]]:
    """CelebA-HQ indices of each split, sorted, from the mapping file in ``root``.

    By default the split of g_partition.py (the one papers report); with
    ``partition_file``, CelebA's own partition of each ``orig_file``.
    """
    partition = read_partition(partition_file) if partition_file else None
    splits: dict[str, list[int]] = {s: [] for s in SPLITS}
    with open(os.path.join(root, MAPPING_FILE), encoding="utf-8") as f:
        for line in f:
            fields = line.split()
            if len(fields) != 3 or not fields[0].isdigit():
                continue  # header
            idx, orig_idx, orig_file = int(fields[0]), int(fields[1]), fields[2]
            split = (
                threshold_split(orig_idx)
                if partition is None
                else partition[_stem_number(orig_file)]
            )
            splits[split].append(idx)
    return {s: sorted(idx) for s, idx in splits.items()}


def image_path(root: str, idx: int) -> str:
    return os.path.join(root, IMAGE_DIR, f"{idx}.jpg")


def merge_part_masks(root: str, idx: int, size: tuple[int, int] = (512, 512)) -> np.ndarray:
    """Combine the per-part annotation PNGs of image ``idx`` into one label map (uint8).

    Parts without a file are absent from the image. ``size`` is the (width, height)
    of the annotations, used only if the image has no part file at all.
    """
    folder = os.path.join(root, ANNO_DIR, str(idx // ANNO_PER_FOLDER))
    label = None
    for class_id, part in enumerate(PART_NAMES, start=1):
        path = os.path.join(folder, f"{idx:05d}_{part}.png")
        if not os.path.exists(path):
            continue
        part_mask = np.array(Image.open(path))
        if part_mask.ndim == 3:
            part_mask = part_mask[..., 0]
        if label is None:
            label = np.zeros(part_mask.shape, dtype=np.uint8)
        label[part_mask != 0] = class_id
    if label is None:
        label = np.zeros((size[1], size[0]), dtype=np.uint8)
    return label
