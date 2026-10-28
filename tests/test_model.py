"""Unit tests for the model, flip handling, loss and data split."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from miniunet.data import FLIP_PAIRS, hflip_mask, split_train_val, swap_flip_labels
from miniunet.losses import MulticlassDiceLoss, build_class_weights
from miniunet.model import MiniUNet, count_parameters
from miniunet.predict import predict_logits, swap_flip_channels

EXPECTED_PARAMS = 1_818_398
PARAM_LIMIT = 1_821_085


def test_parameter_count_matches_report() -> None:
    n = count_parameters(MiniUNet())
    assert n == EXPECTED_PARAMS
    assert n <= PARAM_LIMIT


def test_forward_shape_512() -> None:
    model = MiniUNet().eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 512, 512))
    assert out.shape == (1, 19, 512, 512)


@pytest.mark.parametrize("switch", ["use_cbam", "use_attention_gates", "use_dilated_context"])
def test_ablation_switches_run_and_change_param_count(switch: str) -> None:
    model = MiniUNet(**{switch: False}).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 64, 64))
    assert out.shape == (1, 19, 64, 64)
    assert count_parameters(model) < EXPECTED_PARAMS


def test_flip_label_swap_round_trip() -> None:
    rng = np.random.default_rng(0)
    mask = rng.integers(0, 19, size=(32, 48), dtype=np.uint8)
    assert np.array_equal(hflip_mask(hflip_mask(mask)), mask)
    assert np.array_equal(swap_flip_labels(swap_flip_labels(mask)), mask)

    flipped = hflip_mask(mask)
    for a, b in FLIP_PAIRS:
        assert np.array_equal(flipped == a, (mask == b)[:, ::-1])


def test_tta_channel_swap_matches_label_swap() -> None:
    rng = np.random.default_rng(1)
    mask = rng.integers(0, 19, size=(16, 16))
    one_hot = F.one_hot(torch.from_numpy(mask), 19).permute(2, 0, 1)[None].float()
    swapped = swap_flip_channels(one_hot).argmax(dim=1)[0].numpy()
    assert np.array_equal(swapped, swap_flip_labels(mask))
    assert torch.equal(swap_flip_channels(swap_flip_channels(one_hot)), one_hot)


def test_tta_is_flip_equivariant() -> None:
    """TTA logits of a flipped image equal the flipped, channel-swapped TTA logits."""
    torch.manual_seed(0)
    model = MiniUNet().eval()
    img = torch.rand(1, 3, 64, 64)
    a = predict_logits(model, torch.flip(img, dims=[3]), tta=True)
    b = swap_flip_channels(torch.flip(predict_logits(model, img, tta=True), dims=[3]))
    assert torch.allclose(a, b, atol=1e-4)


@pytest.mark.parametrize("weighted", [False, True])
def test_dice_loss_zero_for_perfect_prediction(weighted: bool) -> None:
    targets = torch.randint(0, 19, (2, 32, 32))
    logits = F.one_hot(targets, 19).permute(0, 3, 1, 2).float() * 100.0
    weight = build_class_weights(0.248, 1.564, 11.877) if weighted else None
    loss = MulticlassDiceLoss(19, weight=weight)(logits, targets)
    assert loss.item() == pytest.approx(0.0, abs=1e-4)


def test_split_is_seeded_shuffle(tmp_path) -> None:
    """The split helper is a random.seed(42) shuffle of the sorted file list."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    names = [f"{i:05d}.jpg" for i in range(1000)]
    for n in names:
        (img_dir / n).touch()

    (tr_imgs, tr_masks), (va_imgs, _) = split_train_val(str(img_dir), "masks", 0.1, 42)

    indices = list(range(1000))
    random.seed(42)
    random.shuffle(indices)
    assert [p.split("/")[-1].split("\\")[-1] for p in tr_imgs] == [names[i] for i in indices[:900]]
    assert [p.split("/")[-1].split("\\")[-1] for p in va_imgs] == [names[i] for i in indices[900:]]
    assert tr_masks[0].endswith(names[indices[0]].replace(".jpg", ".png"))
