"""Shared image preprocessing for posture training and inference.

Every place that turns an OAK-D jpg into a model input goes through this
module, so training, batch inference and the Colab notebook cannot drift
apart (crop box, resize, color order, normalization).

Checkpoints record how their inputs were prepared:
    ckpt["normalize"]  -> None (legacy, plain 0..1) or "imagenet"
    ckpt["crop"]       -> True if the stall crop was applied in memory
Models trained before v1.1 have neither key; they were trained on 0..1
inputs without ImageNet normalization, so the defaults below keep them
producing exactly the same predictions as before.
"""

import os
import sys
import pathlib

import cv2
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import CROP_TOF, IMG_SIZE

TOF_W, TOF_H = 640, 480
MIN_JPG_BYTES = 1000

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def crop_box_for_size(w, h, box=CROP_TOF):
    """Scale the ToF crop box (640x480 coords) to an image of size w x h."""
    x1, y1, x2, y2 = box
    sx, sy = w / TOF_W, h / TOF_H
    return int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)


def read_image(path, min_bytes=MIN_JPG_BYTES):
    """Read a jpg as BGR uint8. Returns None for missing, truncated or corrupt files."""
    if not path or not os.path.exists(path):
        return None
    try:
        if os.path.getsize(path) < min_bytes:
            return None
    except OSError:
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    if h < 10 or w < 10:
        return None
    return img


def is_precropped(path):
    """Legacy preprocessed files carry a _cropped suffix and must not be cropped twice."""
    return "_cropped" in os.path.basename(path or "")


def crop_stall(img, path=None):
    """Crop to the stall region unless the file was already cropped on disk."""
    if path is not None and is_precropped(path):
        return img
    h, w = img.shape[:2]
    x1, y1, x2, y2 = crop_box_for_size(w, h)
    return img[y1:y2, x1:x2]


def resize_rgb_uint8(img_bgr, size=IMG_SIZE):
    """Resize to size x size and convert BGR -> RGB, keeping uint8.

    Bilinear (cv2 default) on purpose: it is what every existing model was
    trained with, so legacy checkpoints keep identical predictions.
    """
    img = cv2.resize(img_bgr, (size, size))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def uint8_to_tensor(img_rgb_uint8, normalize=None):
    """HWC uint8 RGB -> CHW float32 tensor, optionally ImageNet-normalized."""
    x = img_rgb_uint8.astype(np.float32) / 255.0
    if normalize == "imagenet":
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
    elif normalize not in (None, "none"):
        raise ValueError(f"Unknown normalize mode: {normalize}")
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def normalize_batch(x, normalize=None):
    """Apply normalization to an already-batched float tensor in 0..1 (N,C,H,W)."""
    if normalize in (None, "none"):
        return x
    if normalize != "imagenet":
        raise ValueError(f"Unknown normalize mode: {normalize}")
    c = x.shape[1]
    mean = torch.tensor(np.resize(IMAGENET_MEAN, c), dtype=x.dtype, device=x.device).view(1, c, 1, 1)
    std = torch.tensor(np.resize(IMAGENET_STD, c), dtype=x.dtype, device=x.device).view(1, c, 1, 1)
    return (x - mean) / std


def load_model_input(path, size=IMG_SIZE, crop=True, normalize=None):
    """Read -> (crop) -> resize -> tensor. Returns None if the file is unusable."""
    img = read_image(path)
    if img is None:
        return None
    if crop:
        img = crop_stall(img, path)
    return uint8_to_tensor(resize_rgb_uint8(img, size), normalize)


def ckpt_preprocessing(ckpt):
    """Return (normalize, crop) recorded in a checkpoint, with legacy defaults."""
    normalize = ckpt.get("normalize", None)
    crop = ckpt.get("crop", True)
    return normalize, crop


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
