"""Vulva segmentation model and the image handling shared by training and inference.

Model: U-Net with a ResNet-18 encoder (ImageNet weights), one output channel
(vulva vs not). Xu et al. (2024) also used a U-Net for this step. The decoder
keeps skip connections down to half resolution, because on the OAK-D IR image
the vulva is only about 15 x 25 pixels.

Input: the IR frame cropped to the stall (config.CROP_TOF), resized to
SEG_W x SEG_H, ImageNet-normalised. Two input sets, stored in the checkpoint:
  "ir"        grey IR replicated to 3 channels (models up to r1)
  "ir+depth"  IR, height above the sow's own skin and 3D shading as the three
              channels (depth_views.py). The vulva sticks out of the skin, while
              feces and dark marks are flat, so depth separates the two.
A predicted mask is mapped back to the full 640 x 480 frame so it lines up with
the depth image from the same sensor.
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CROP_TOF  # noqa: E402

SEG_W, SEG_H = 320, 384          # crop 380 x 450 -> about 0.84 scale
FULL_W, FULL_H = 640, 480
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def crop_box(w=FULL_W, h=FULL_H):
    x1, y1, x2, y2 = CROP_TOF
    sx, sy = w / FULL_W, h / FULL_H
    return int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)


def read_ir(path):
    """IR jpg -> grey uint8 at 640 x 480 (resized if the file is another size), or None."""
    if not path or not os.path.exists(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if img.shape != (FULL_H, FULL_W):
        img = cv2.resize(img, (FULL_W, FULL_H), interpolation=cv2.INTER_LINEAR)
    return img


def read_mask(path):
    if not path or not os.path.exists(path):
        return None
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if m.shape != (FULL_H, FULL_W):
        m = cv2.resize(m, (FULL_W, FULL_H), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


INPUTS = ("ir", "ir+depth")
HEIGHT_SPAN_MM = 8.0


def default_depth_path(ir_path):
    """pigN_ir_vis_<ts>.jpg -> pigN_depth_<ts>.raw in the same folder (OAK naming)."""
    d, name = os.path.split(ir_path)
    return os.path.join(d, name.replace("_ir_vis_", "_depth_").rsplit(".", 1)[0] + ".raw")


def make_input(ir_gray, depth_mm=None, inputs="ir"):
    """Full-frame model input: the grey IR for "ir"; for "ir+depth" an H x W x 3 uint8
    stack of IR, height (+/-8 mm -> 0..255, 128 = level with the skin) and shading.
    Missing depth gives flat height and zero shading, which the model sees as 'no 3D cue'."""
    if inputs == "ir":
        return ir_gray
    if inputs != "ir+depth":
        raise ValueError(f"unknown inputs {inputs!r}; expected one of {INPUTS}")
    h = np.full(ir_gray.shape, 128, np.uint8)
    sh = np.zeros(ir_gray.shape, np.uint8)
    if depth_mm is not None:
        from depth_views import relief
        shade, height, _ = relief(depth_mm)
        y = np.clip((np.nan_to_num(height, nan=0.0) + HEIGHT_SPAN_MM) / (2 * HEIGHT_SPAN_MM), 0, 1)
        h = (y * 255).astype(np.uint8)
        sh = (np.clip((np.nan_to_num(shade, nan=0.6) - 0.6) / 0.4, 0, 1) * 255).astype(np.uint8)
    return np.dstack([ir_gray, h, sh])


def read_input(ir_path, depth_path=None, inputs="ir"):
    """IR (+ depth) files -> full-frame model input, or None if the IR is unreadable."""
    ir = read_ir(ir_path)
    if ir is None or inputs == "ir":
        return ir
    from measure3d import load_depth_raw
    dp = depth_path or default_depth_path(ir_path)
    return make_input(ir, load_depth_raw(dp), inputs)


def ir_channel(img):
    return img if img.ndim == 2 else np.ascontiguousarray(img[..., 0])


def to_model_space(img, mask=None):
    """Full-frame input (grey or H x W x 3) and mask -> stall crop resized to SEG_W x SEG_H."""
    x1, y1, x2, y2 = crop_box()
    im = cv2.resize(img[y1:y2, x1:x2], (SEG_W, SEG_H), interpolation=cv2.INTER_LINEAR)
    if mask is None:
        return im, None
    mk = cv2.resize(mask[y1:y2, x1:x2], (SEG_W, SEG_H), interpolation=cv2.INTER_NEAREST)
    return im, mk


def to_full_frame(mask_small):
    """Model-space mask -> full 640 x 480 frame (zeros outside the stall crop)."""
    x1, y1, x2, y2 = crop_box()
    full = np.zeros((FULL_H, FULL_W), np.uint8)
    full[y1:y2, x1:x2] = cv2.resize(mask_small.astype(np.uint8), (x2 - x1, y2 - y1),
                                    interpolation=cv2.INTER_NEAREST)
    return full


def to_tensor(img):
    """Grey (replicated) or H x W x 3 uint8 -> normalised 3 x H x W tensor."""
    x = img.astype(np.float32) / 255.0
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=2)
    x = (x - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class _Up(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin + cskip, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


class VulvaUNet(nn.Module):
    """ResNet-18 encoder + 4-stage decoder -> 1 logit per pixel at input resolution."""

    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18
        enc = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)      # 64, /2
        self.pool = enc.maxpool                                      # /4
        self.l1, self.l2, self.l3, self.l4 = enc.layer1, enc.layer2, enc.layer3, enc.layer4
        self.u4 = _Up(512, 256, 256)
        self.u3 = _Up(256, 128, 128)
        self.u2 = _Up(128, 64, 64)
        self.u1 = _Up(64, 64, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        size = x.shape[-2:]
        s0 = self.stem(x)
        s1 = self.l1(self.pool(s0))
        s2 = self.l2(s1)
        s3 = self.l3(s2)
        s4 = self.l4(s3)
        y = self.u4(s4, s3)
        y = self.u3(y, s2)
        y = self.u2(y, s1)
        y = self.u1(y, s0)
        return F.interpolate(self.head(y), size=size, mode="bilinear", align_corners=False)


def save_checkpoint(model, path, meta):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "arch": "unet_resnet18",
                "input": [SEG_W, SEG_H], "crop": list(CROP_TOF), **meta}, path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = VulvaUNet(pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


def blob_confidence(prob, mask):
    """Mean probability inside the predicted blob (0 when there is none): how sure the
    model is that the blob is a vulva. train_seg.py tunes a cut on it from held-out sows."""
    return float(prob[mask > 0].astype(np.float32).mean()) if mask.any() else 0.0


@torch.no_grad()
def predict(model, grays, device="cpu", threshold=0.5, min_area=None, batch_size=16, presence=0.0):
    """Full-frame inputs (read_input / make_input; plain grey for "ir" models)
    -> list of (full-frame mask uint8, max prob, area in model pixels, blob confidence).
    presence: blobs with a confidence below this are dropped (checkpoint "presence_threshold")."""
    out = []
    for i in range(0, len(grays), batch_size):
        chunk = grays[i:i + batch_size]
        x = torch.stack([to_tensor(to_model_space(g)[0]) for g in chunk]).to(device)
        probs = torch.sigmoid(model(x))[:, 0].cpu().numpy()
        for p in probs:
            m = (p >= threshold).astype(np.uint8)
            m = largest_component(m)
            area = int(m.sum())
            if min_area is not None and area < min_area:
                m[:] = 0
            conf = blob_confidence(p, m)
            if conf < presence:
                m[:] = 0
            out.append((to_full_frame(m), float(p.max()), area, conf))
    return out


def largest_component(m):
    """Keep the biggest blob: there is one vulva per frame."""
    if m.sum() == 0:
        return m
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 2:
        return m
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == k).astype(np.uint8)
