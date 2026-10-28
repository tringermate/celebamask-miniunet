"""Official-split tooling on a tiny fake CelebAMask-HQ download."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from miniunet.celebamask import (
    ANNO_DIR,
    IMAGE_DIR,
    MAPPING_FILE,
    PART_NAMES,
    merge_part_masks,
    official_split,
)
from miniunet.metrics import confusion_matrix, f1_per_class

REPO = Path(__file__).resolve().parents[1]
# CelebA-HQ idx -> orig_idx (0-based; orig_file is orig_idx + 1), at the g_partition.py
# thresholds: 2 train, 1 val, 2 test images.
ORIG = {0: 999, 1: 162770, 2: 162771, 3: 182638, 4: 202598}
MASK_SIZE = 32


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    result = subprocess.run(
        [sys.executable, *args], cwd=cwd, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def _expected_label(idx: int) -> np.ndarray:
    """Skin everywhere, hair on the top half, nose (overwritten by hair) in the top-left
    corner, and for odd idx a necklace in the bottom-right corner."""
    label = np.full((MASK_SIZE, MASK_SIZE), 1, dtype=np.uint8)
    label[: MASK_SIZE // 2] = 13
    if idx % 2:
        label[-4:, -4:] = 16
    return label


@pytest.fixture(scope="module")
def fake_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("CelebAMask-HQ")
    (root / IMAGE_DIR).mkdir()
    (root / ANNO_DIR / "0").mkdir(parents=True)
    rng = np.random.default_rng(0)
    lines = ["idx         orig_idx    orig_file"]
    for idx, orig in ORIG.items():
        lines.append(f"{idx}   {orig}   {orig + 1:06d}.jpg")
        coarse = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)  # smooth, like a photo
        img = Image.fromarray(coarse).resize((64, 64), Image.BICUBIC)
        img.save(root / IMAGE_DIR / f"{idx}.jpg", quality=98)

        parts = {
            name: np.zeros((MASK_SIZE, MASK_SIZE), np.uint8) for name in ("skin", "nose", "hair")
        }
        parts["skin"][:] = 255
        parts["nose"][:4, :4] = 255
        parts["hair"][: MASK_SIZE // 2] = 255
        if idx % 2:
            parts["neck_l"] = np.zeros((MASK_SIZE, MASK_SIZE), np.uint8)
            parts["neck_l"][-4:, -4:] = 255
        for name, m in parts.items():
            rgb = np.repeat(m[..., None], 3, axis=2)  # the release stores some parts as RGB
            Image.fromarray(rgb if name == "hair" else m).save(
                root / ANNO_DIR / "0" / f"{idx:05d}_{name}.png"
            )
    (root / MAPPING_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def test_part_order_matches_flip_pairs_and_tiny_classes() -> None:
    assert PART_NAMES.index("l_eye") + 1 == 4 and PART_NAMES.index("r_ear") + 1 == 9
    assert PART_NAMES.index("ear_r") + 1 == 15 and PART_NAMES.index("neck_l") + 1 == 16


def test_official_split(fake_root: Path) -> None:
    assert official_split(str(fake_root)) == {"train": [0, 1], "val": [2], "test": [3, 4]}


def test_official_split_from_partition_file(fake_root: Path, tmp_path: Path) -> None:
    part = tmp_path / "list_eval_partition.txt"
    part.write_text("".join(f"{o + 1:06d}.jpg {2 if i == 0 else 0}\n" for i, o in ORIG.items()))
    assert official_split(str(fake_root), str(part)) == {
        "train": [1, 2, 3, 4],
        "val": [],
        "test": [0],
    }


def test_merge_part_masks(fake_root: Path) -> None:
    for idx in ORIG:
        np.testing.assert_array_equal(merge_part_masks(str(fake_root), idx), _expected_label(idx))


def test_prepare_official(fake_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "dataset_official"
    _run(
        [
            str(REPO / "scripts" / "prepare_official.py"),
            "--root", str(fake_root), "--out", str(out), "--image-size", "48", "--workers", "1",
        ],
        cwd=tmp_path,
    )  # fmt: skip
    assert sorted(os.listdir(out / "test" / "images")) == ["3.jpg", "4.jpg"]
    assert Image.open(out / "test" / "images" / "3.jpg").size == (48, 48)
    np.testing.assert_array_equal(
        np.array(Image.open(out / "val" / "masks" / "2.png")), _expected_label(2)
    )


def test_folder_loader(fake_root: Path, tmp_path: Path) -> None:
    from miniunet.evaluate import folder_loader

    out = tmp_path / "ds"
    _run(
        [
            str(REPO / "scripts" / "prepare_official.py"), "--root", str(fake_root),
            "--out", str(out), "--splits", "test", "--workers", "1",
        ],
        cwd=tmp_path,
    )  # fmt: skip
    cfg = {"data": {"image_size": [32, 32]}}
    loader = folder_loader(
        cfg, str(out / "test" / "images"), str(out / "test" / "masks"), torch.device("cpu")
    )
    assert [os.path.basename(p) for p in loader.dataset.img_paths] == ["3.jpg", "4.jpg"]


def test_dataset_f1() -> None:
    gt = torch.tensor([[0, 0, 1, 1], [2, 2, 2, 2]])
    pred = torch.tensor([[0, 1, 1, 1], [2, 2, 2, 0]])
    cm = confusion_matrix(pred, gt, n_classes=4)
    assert cm.sum() == 8
    f1 = f1_per_class(cm)
    # class 0: tp 1, fp 1, fn 1; class 1: tp 2, fp 1; class 2: tp 3, fn 1; class 3 absent
    np.testing.assert_allclose(f1[:3], [2 / 4, 4 / 5, 6 / 7])
    assert np.isnan(f1[3])
