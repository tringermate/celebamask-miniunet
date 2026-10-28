"""Dataset, augmentation and train/validation split for CelebAMask-HQ face parsing."""

from __future__ import annotations

import os
import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
from torch.utils.data import Dataset

# Left/right label pairs that swap under a horizontal flip:
# (4, 5) left/right eye, (6, 7) left/right eyebrow, (8, 9) left/right ear.
FLIP_PAIRS: tuple[tuple[int, int], ...] = ((4, 5), (6, 7), (8, 9))

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def swap_flip_labels(mask: np.ndarray) -> np.ndarray:
    """Swap left/right class ids in a label map (no spatial flip)."""
    out = mask.copy()
    for a, b in FLIP_PAIRS:
        out[mask == a] = b
        out[mask == b] = a
    return out


def hflip_mask(mask: np.ndarray) -> np.ndarray:
    """Mirror a label map horizontally and swap the left/right class ids."""
    return swap_flip_labels(mask[:, ::-1])


def apply_elastic_transform(
    image_pil: Image.Image, mask_pil: Image.Image, alpha: float = 150, sigma: float = 15
) -> tuple[Image.Image, Image.Image]:
    """Apply the same random elastic deformation to an RGB image and its mask.

    The displacement field is uniform noise in [-1, 1] smoothed by a Gaussian of
    width ``sigma`` and scaled by ``alpha``. The image is resampled bilinearly,
    the mask with nearest neighbour.
    """
    img_np = np.array(image_pil)
    mask_np = np.array(mask_pil)
    shape = img_np.shape[:2]

    # Global NumPy RNG, so the `seed` setting also controls this augmentation.
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    distorted_img = np.zeros_like(img_np)
    for i in range(3):
        distorted_img[:, :, i] = map_coordinates(
            img_np[:, :, i], indices, order=1, mode="reflect"
        ).reshape(shape)

    distorted_mask = map_coordinates(mask_np, indices, order=0, mode="reflect").reshape(shape)

    return Image.fromarray(distorted_img), Image.fromarray(distorted_mask.astype(np.uint8))


class FaceParsingDataset(Dataset):
    """Image/mask pairs resized to a fixed size, with optional training augmentation.

    Without ``mask_paths`` (inference), ``__getitem__`` returns the image tensor and
    the image file name instead of a mask.

    Args:
        img_paths: Paths to RGB images.
        mask_paths: Paths to label PNGs (pixel value = class id), or None.
        augment: Enable training augmentation.
        size: Output (width, height).
        aug_cfg: The ``augmentation`` section of the config. Defaults match configs/.
    """

    def __init__(
        self,
        img_paths: list[str],
        mask_paths: list[str] | None = None,
        augment: bool = False,
        size: tuple[int, int] = (512, 512),
        aug_cfg: dict | None = None,
    ) -> None:
        self.img_paths, self.mask_paths = img_paths, mask_paths
        self.augment, self.size = augment, tuple(size)
        cfg = aug_cfg or {}
        self.aug_flip = cfg.get("aug_flip", True)
        self.flip_p = cfg.get("flip_p", 0.5)
        self.aug_rotate = cfg.get("aug_rotate", True)
        self.rotate_p = cfg.get("rotate_p", 0.5)
        self.rotate_max_deg = cfg.get("rotate_max_deg", 10.0)
        self.aug_elastic = cfg.get("aug_elastic", True)
        self.elastic_p = cfg.get("elastic_p", 0.3)
        self.elastic_alpha = cfg.get("elastic_alpha", 150.0)
        self.elastic_sigma = cfg.get("elastic_sigma", 15.0)

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor | str]:
        image = Image.open(self.img_paths[idx]).convert("RGB").resize(self.size, Image.BILINEAR)
        mask = (
            Image.open(self.mask_paths[idx]).resize(self.size, Image.NEAREST)
            if self.mask_paths
            else None
        )

        if self.augment and mask is not None:
            if self.aug_flip and random.random() < self.flip_p:
                image = TF.hflip(image)
                mask = Image.fromarray(hflip_mask(np.array(mask)))

            if self.aug_rotate and random.random() < self.rotate_p:
                angle = random.uniform(-self.rotate_max_deg, self.rotate_max_deg)
                image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

            if self.aug_elastic and random.random() < self.elastic_p:
                image, mask = apply_elastic_transform(
                    image, mask, alpha=self.elastic_alpha, sigma=self.elastic_sigma
                )

        if mask is None:
            return TF.to_tensor(image), os.path.basename(self.img_paths[idx])
        return TF.to_tensor(image), torch.from_numpy(np.array(mask, dtype=np.int64))


def list_images(img_dir: str) -> list[str]:
    """Return the sorted image file names in ``img_dir``."""
    return sorted(f for f in os.listdir(img_dir) if f.lower().endswith(IMAGE_EXTENSIONS))


def mask_name(img_name: str) -> str:
    """Mask file name for an image: the ``.jpg`` extension replaced by ``.png``."""
    return img_name.replace(".jpg", ".png")


def list_pairs(img_dir: str, mask_dir: str) -> tuple[list[str], list[str]]:
    """Full paths of every image in ``img_dir`` and its mask in ``mask_dir``, sorted."""
    imgs = list_images(img_dir)
    return (
        [os.path.join(img_dir, f) for f in imgs],
        [os.path.join(mask_dir, mask_name(f)) for f in imgs],
    )


def subset_pairs(
    imgs: list[str], masks: list[str], limit: int | None, seed: int = 42
) -> tuple[list[str], list[str]]:
    """A fixed random subset of ``limit`` image/mask pairs, in their original order.

    Returns the lists unchanged when ``limit`` is None or not smaller than the list.
    """
    if not limit or limit >= len(imgs):
        return imgs, masks
    keep = sorted(random.Random(seed).sample(range(len(imgs)), limit))
    return [imgs[i] for i in keep], [masks[i] for i in keep]


def split_pairs(
    imgs: list[str], masks: list[str], val_fraction: float = 0.1, split_seed: int = 42
) -> tuple[tuple[list[str], list[str]], tuple[list[str], list[str]]]:
    """Shuffle the pairs with a fixed seed and split them into train and validation.

    The last ``val_fraction`` of the shuffled order is the validation split. A private
    ``random.Random`` is used, so the global RNG is not touched.

    Returns:
        ``((train_imgs, train_masks), (val_imgs, val_masks))``.
    """
    indices = list(range(len(imgs)))
    random.Random(split_seed).shuffle(indices)
    n_val = max(1, round(len(imgs) * val_fraction))
    tr_idx, va_idx = indices[: len(indices) - n_val], indices[len(indices) - n_val :]

    def pick(idx: list[int]) -> tuple[list[str], list[str]]:
        return [imgs[i] for i in idx], [masks[i] for i in idx]

    return pick(tr_idx), pick(va_idx)


def split_train_val(
    img_dir: str, mask_dir: str, val_fraction: float = 0.1, split_seed: int = 42
) -> tuple[tuple[list[str], list[str]], tuple[list[str], list[str]]]:
    """``split_pairs`` over every image in ``img_dir`` and its mask in ``mask_dir``."""
    return split_pairs(*list_pairs(img_dir, mask_dir), val_fraction, split_seed)
