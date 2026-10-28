"""End-to-end smoke test: train one epoch and predict with TTA on a random dummy dataset."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
TRAIN_SIZE = (96, 80)  # original (width, height); the dataset resizes to 512x512
TEST_SIZES = [(100, 70), (64, 128), (120, 120)]


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    result = subprocess.run(
        [sys.executable, *args], cwd=cwd, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture(scope="module")
def dummy_root(tmp_path_factory) -> Path:
    """Random images and masks in the layout that configs/default.yaml expects."""
    root = tmp_path_factory.mktemp("dummy")
    rng = np.random.default_rng(0)
    img_dir = root / "dataset_official" / "train" / "images"
    mask_dir = root / "dataset_official" / "train" / "masks"
    test_dir = root / "dataset_official" / "test" / "images"
    for d in (img_dir, mask_dir, test_dir):
        d.mkdir(parents=True)

    w, h = TRAIN_SIZE
    for i in range(10):
        img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        mask = rng.integers(0, 19, size=(h, w), dtype=np.uint8)
        Image.fromarray(img).save(img_dir / f"{i}.jpg")
        Image.fromarray(mask).save(mask_dir / f"{i}.png")
    for i, (w, h) in enumerate(TEST_SIZES):
        img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        Image.fromarray(img).save(test_dir / f"test_{i}.jpg")
    return root


@pytest.fixture(scope="module")
def trained(dummy_root: Path) -> Path:
    _run(
        [
            "-m",
            "miniunet.train",
            "--config",
            str(REPO / "configs" / "default.yaml"),
            "--epochs",
            "1",
            "--device",
            "cpu",
        ],
        cwd=dummy_root,
    )
    return dummy_root / "outputs" / "default"


def test_train_one_epoch(trained: Path) -> None:
    with open(trained / "history.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert set(rows[0]) == {"epoch", "train_loss", "val_f_measure", "lr"}
    assert np.isfinite(float(rows[0]["train_loss"]))
    assert (trained / "best.pth").exists()
    assert (trained / "config.yaml").exists()


def test_predict_with_tta(dummy_root: Path, trained: Path) -> None:
    out_dir = dummy_root / "pred"
    _run(
        [
            "-m",
            "miniunet.predict",
            "--config",
            str(REPO / "configs" / "default.yaml"),
            "--checkpoint",
            str(trained / "best.pth"),
            "--input-dir",
            str(dummy_root / "dataset_official" / "test" / "images"),
            "--output-dir",
            str(out_dir),
            "--tta",
            "--device",
            "cpu",
        ],
        cwd=dummy_root,
    )
    for i, size in enumerate(TEST_SIZES):
        mask = Image.open(out_dir / f"test_{i}.png")
        assert mask.size == size
        arr = np.array(mask)
        assert arr.ndim == 2
        assert arr.min() >= 0 and arr.max() <= 18


def test_evaluate_runs(dummy_root: Path, trained: Path) -> None:
    result = _run(
        [
            "-m",
            "miniunet.evaluate",
            "--config",
            str(REPO / "configs" / "default.yaml"),
            "--checkpoint",
            str(trained / "best.pth"),
            "--device",
            "cpu",
        ],
        cwd=dummy_root,
    )
    assert "TTA on" in result.stdout and "TTA off" in result.stdout


def test_evaluate_folder(dummy_root: Path, trained: Path) -> None:
    train_dir = dummy_root / "dataset_official" / "train"
    result = _run(
        [
            "-m", "miniunet.evaluate", "--config", str(REPO / "configs" / "default.yaml"),
            "--checkpoint", str(trained / "best.pth"), "--device", "cpu", "--per-class",
            "--images", str(train_dir / "images"), "--masks", str(train_dir / "masks"),
        ],
        cwd=dummy_root,
    )  # fmt: skip
    assert "Images: 10" in result.stdout
    assert "dataset F1 w/o background" in result.stdout and "neck_l" in result.stdout


def test_train_with_val_folder_and_limit(dummy_root: Path, tmp_path: Path) -> None:
    import yaml

    cfg = yaml.safe_load((REPO / "configs" / "official.yaml").read_text(encoding="utf-8"))
    train_dir = dummy_root / "dataset_official" / "train"
    cfg["data"].update(
        train_images=str(train_dir / "images"), train_masks=str(train_dir / "masks"),
        val_images=str(train_dir / "images"), val_masks=str(train_dir / "masks"),
        train_limit=3, num_workers=0,
    )  # fmt: skip
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    result = _run(
        [
            "-m", "miniunet.train", "--config", str(cfg_path), "--epochs", "1",
            "--device", "cpu", "--output-dir", str(tmp_path / "out"),
        ],
        cwd=tmp_path,
    )  # fmt: skip
    assert "Train images: 3 | val images: 10" in result.stdout


def test_limit_is_applied_before_the_val_split(dummy_root: Path) -> None:
    import torch

    from miniunet.train import build_loaders
    from miniunet.utils import load_config

    cfg = load_config(str(REPO / "configs" / "default.yaml"))
    train_dir = dummy_root / "dataset_official" / "train"
    cfg["data"].update(
        train_images=str(train_dir / "images"), train_masks=str(train_dir / "masks"),
        train_limit=6, val_fraction=0.5, num_workers=0,
    )  # fmt: skip
    tr, va = build_loaders(cfg, torch.device("cpu"))
    tr_paths, va_paths = set(tr.dataset.img_paths), set(va.dataset.img_paths)
    assert len(tr_paths) == 3 and len(va_paths) == 3 and not tr_paths & va_paths
