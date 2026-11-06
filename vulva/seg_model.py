"""Vulva segmentation model and the image handling shared by training and inference.

Model: U-Net with a ResNet-18 encoder (ImageNet weights), one output channel
(vulva vs not), or three (background / vulva / tail; checkpoint "classes": 3) so the model
learns what a tail looks like instead of only "vulva vs everything else". Xu et al. (2024) also used a U-Net for this step. The decoder
keeps skip connections down to half resolution, because on the OAK-D IR image
the vulva is only about 15 x 25 pixels.

Input: the IR frame cropped to the stall (config.CROP_TOF), resized to
SEG_W x SEG_H, ImageNet-normalised. Two input sets, stored in the checkpoint:
  "ir"        grey IR replicated to 3 channels (models up to r1)
  "ir+depth"  IR, height above the sow's own skin and 3D shading as the three
              channels (depth_views.py). The vulva sticks out of the skin, while
              feces and dark marks are flat, so depth separates the two. (Old way: it
              feeds depth into filters trained on colour photos; lost to IR, r2d / r8.)
  "depth"     depth only: height above the local skin (+/-20 mm), shape index (dome = +1,
              ridge such as a tail = +0.5, flat = 0) and curvedness (depth_features)
  "fusion"    IR replicated to 3 channels + height + shape index = 5 channels; the first
              layer keeps its ImageNet weights for the IR channels and starts at zero for
              the two depth channels, so training starts from the IR model and learns how
              much to use depth (standard way to add a channel to a pretrained network)
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
DEPTH_MEAN, DEPTH_STD = 0.5, 0.25   # depth feature channels are stored as 0..255 with 128 = neutral


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


INPUTS = ("ir", "ir+depth", "depth", "fusion")
HEIGHT_SPAN_MM = 8.0
FEAT_HEIGHT_MM = 20.0     # depth features: height clipped to +/- this (the vulva rises ~8-11 mm)
FEAT_SKIN_MM = 25.0       # the "skin" a bump is measured against: depth smoothed over this scale
FEAT_CURV_MM = 10.0       # smoothing before curvature (ToF noise; 6 mm left speckle everywhere)
FEAT_CURV_REF = 0.015     # 1/mm: curvature below this counts as flat (shape index faded out)


def in_channels_for(inputs):
    return 5 if inputs == "fusion" else 3


def depth_features(depth_mm, fx=471.583):
    """ToF depth (480 x 640, mm, 0 = no reading) -> three uint8 maps on the same grid:
    height above the local skin (+/-FEAT_HEIGHT_MM -> 0..255, 128 = level), shape index
    (-1 cup .. -0.5 valley .. 0 saddle/flat .. +0.5 ridge .. +1 dome -> 0..255) faded to 0
    where the surface is nearly flat, and curvedness (0 .. 4 x FEAT_CURV_REF -> 0..255).
    Holes are filled from their surroundings first; scales are in mm on the sow, converted to
    pixels with the median distance, so near and far sows give the same maps."""
    d = depth_mm.astype(np.float32)
    valid = ((d > 300) & (d < 3000)).astype(np.float32)
    if valid.sum() < 100:
        h = np.full(d.shape, 128, np.uint8)
        return h, h.copy(), np.zeros(d.shape, np.uint8)
    x1, y1, x2, y2 = crop_box(d.shape[1], d.shape[0])
    vz = d[y1:y2, x1:x2][valid[y1:y2, x1:x2] > 0]
    z = float(np.median(vz)) if vz.size else float(np.median(d[valid > 0]))
    px_per_mm = fx / max(z, 1.0)

    def blur(a, sigma_px):
        return cv2.GaussianBlur(a, (0, 0), max(sigma_px, 0.5))

    w3 = blur(valid, 3.0)
    filled = np.where(valid > 0, d, blur(d * valid, 3.0) / np.maximum(w3, 1e-3))
    seen = (valid > 0) | (w3 > 0.1)
    fine = blur(filled, 1.5)
    skin = blur(filled, FEAT_SKIN_MM * px_per_mm)
    height = np.clip(skin - fine, -FEAT_HEIGHT_MM, FEAT_HEIGHT_MM)        # + = towards the camera

    f = -blur(filled, FEAT_CURV_MM * px_per_mm).astype(np.float64)        # surface height field
    fy, fx_ = np.gradient(f)
    fxy, fxx = np.gradient(fx_)
    fyy, _ = np.gradient(fy)
    fxx, fyy, fxy = (a * px_per_mm ** 2 for a in (fxx, fyy, fxy))          # 1/mm
    tr, disc = fxx + fyy, np.sqrt(((fxx - fyy) / 2) ** 2 + fxy ** 2)
    k1, k2 = tr / 2 + disc, tr / 2 - disc                                 # k1 >= k2
    si = -(2 / np.pi) * np.arctan((k1 + k2) / (k1 - k2 + 1e-9))
    curv = np.sqrt((k1 ** 2 + k2 ** 2) / 2)
    si_eff = si * np.clip(curv / FEAT_CURV_REF, 0, 1)

    h8 = np.clip((height + FEAT_HEIGHT_MM) / (2 * FEAT_HEIGHT_MM) * 255, 0, 255).astype(np.uint8)
    s8 = np.clip((si_eff + 1) / 2 * 255, 0, 255).astype(np.uint8)
    c8 = np.clip(curv / (4 * FEAT_CURV_REF) * 255, 0, 255).astype(np.uint8)
    h8[~seen], s8[~seen], c8[~seen] = 128, 128, 0
    return h8, s8, c8


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
    if inputs in ("depth", "fusion"):
        if depth_mm is None:
            return None                       # no depth: these models cannot see the frame
        h8, s8, c8 = depth_features(depth_mm)
        return np.dstack([h8, s8, c8]) if inputs == "depth" else np.dstack([ir_gray, h8, s8])
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
    """IR (+ depth) files -> full-frame model input, or None if the IR is unreadable (or, for
    "depth" / "fusion", the depth file is missing)."""
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


def to_tensor(img, inputs=None):
    """Model-space input -> normalised C x H x W tensor.
    grey ("ir")        -> IR replicated to 3 channels, ImageNet mean/std
    3 channels         -> "ir+depth" (ImageNet mean/std) or "depth" (DEPTH_MEAN/STD)
    "fusion" [ir, h, s] -> 5 channels: IR x 3 (ImageNet) + height + shape (DEPTH_MEAN/STD)"""
    x = img.astype(np.float32) / 255.0
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=2)
    if inputs == "fusion":
        ir = (np.repeat(x[..., :1], 3, axis=2) - MEAN) / STD
        x = np.concatenate([ir, (x[..., 1:3] - DEPTH_MEAN) / DEPTH_STD], axis=2)
    elif inputs == "depth":
        x = (x - DEPTH_MEAN) / DEPTH_STD
    else:
        x = (x - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).astype(np.float32).copy())


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
    """ResNet-18 encoder + 4-stage decoder -> logits per pixel at input resolution:
    1 channel (vulva vs not) or, with n_out=3, background / vulva / tail."""

    def __init__(self, pretrained=True, n_out=1, in_ch=3):
        super().__init__()
        self.n_out, self.in_ch, self.inputs = n_out, in_ch, None
        from torchvision.models import ResNet18_Weights, resnet18
        enc = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        if in_ch != 3:                   # extra input channels: keep the RGB filters, new ones start at zero
            old = enc.conv1
            enc.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
            with torch.no_grad():
                enc.conv1.weight.zero_()
                enc.conv1.weight[:, :3] = old.weight
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)      # 64, /2
        self.pool = enc.maxpool                                      # /4
        self.l1, self.l2, self.l3, self.l4 = enc.layer1, enc.layer2, enc.layer3, enc.layer4
        self.u4 = _Up(512, 256, 256)
        self.u3 = _Up(256, 128, 128)
        self.u2 = _Up(128, 64, 64)
        self.u1 = _Up(64, 64, 32)
        self.head = nn.Conv2d(32, n_out, 1)

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


def n_out_for(ckpt):
    return 3 if int(ckpt.get("classes", 2)) == 3 else 1


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    inputs = ckpt.get("inputs", "ir")
    model = VulvaUNet(pretrained=False, n_out=n_out_for(ckpt), in_ch=in_channels_for(inputs))
    model.load_state_dict(ckpt["state_dict"])
    model.inputs = inputs
    model.to(device).eval()
    return model, ckpt


def blob_confidence(prob, mask):
    """Mean probability inside the predicted blob (0 when there is none): how sure the
    model is that the blob is a vulva. train_seg.py tunes a cut on it from held-out sows."""
    return float(prob[mask > 0].astype(np.float32).mean()) if mask.any() else 0.0


MIN_TAIL_AREA = 40      # model pixels; smaller tail specks are ignored
TAIL_COVER = 0.10       # a vulva is "covered" when the tail fills this share of the ellipse fitted to it


def parts_from_probs(p, threshold=0.5, min_area=None):
    """One frame's model output -> (vulva mask, vulva prob map, tail mask or None, blob area before
    the minimum-area check), model space.
    p: H x W vulva probability (1-channel models) or 3 x H x W softmax (bg, vulva, tail)."""
    if p.ndim == 3:
        # vulva only where it has the majority (p >= threshold, 0.5), stricter than "wins among three"
        lab = p.argmax(0)
        vul, pv = (p[1] >= threshold).astype(np.uint8), p[1]
        tail = (lab == 2).astype(np.uint8)
        n, cc, st, _ = cv2.connectedComponentsWithStats(tail, connectivity=8)
        keep = [k for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= MIN_TAIL_AREA]
        tail = np.isin(cc, keep).astype(np.uint8) if keep else np.zeros_like(tail)
    else:
        vul, pv, tail = (p >= threshold).astype(np.uint8), p, None
    vul = largest_component(vul)
    area = int(vul.sum())
    if min_area is not None and area < min_area:
        vul[:] = 0
    return vul, pv, tail, area


def covered_by_tail(vulva, tail, share=TAIL_COVER):
    """True when the tail cuts into the vulva: tail pixels fill at least `share` of the ellipse
    fitted to the vulva blob. A tail merely hanging next to it (the tail root sits right above the
    vulva, so that is normal) does not count."""
    if tail is None or not vulva.any() or not tail.any():
        return False
    cnts, _ = cv2.findContours(vulva.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    ell = np.zeros_like(vulva, np.uint8)
    if len(c) >= 5:
        cv2.ellipse(ell, cv2.fitEllipse(c), 1, -1)
    else:
        cv2.drawContours(ell, [cv2.convexHull(c)], -1, 1, -1)
    n = int(ell.sum())
    return bool(n and (ell & (tail > 0)).sum() >= share * n)


@torch.no_grad()
def predict_parts(model, grays, device="cpu", threshold=0.5, min_area=None, batch_size=16, presence=0.0):
    """Full-frame inputs -> one dict per frame: vulva (full-frame mask, emptied below the
    confidence cut), tail (full-frame mask, 3-class models only, else None), covered_by_tail
    (the tail cuts into the vulva: partly covered, so its size is not trusted), pmax, area,
    conf (mean vulva probability in the blob)."""
    out = []
    n_out, inputs = getattr(model, "n_out", 1), getattr(model, "inputs", None)
    for i in range(0, len(grays), batch_size):
        chunk = grays[i:i + batch_size]
        live = [g for g in chunk if g is not None]          # None: unreadable, or no depth for a depth model
        probs = []
        if live:
            x = torch.stack([to_tensor(to_model_space(g)[0], inputs) for g in live]).to(device)
            logits = model(x)
            probs = list((torch.softmax(logits, 1) if n_out == 3 else torch.sigmoid(logits)[:, 0]).cpu().numpy())
        probs.reverse()
        for g in chunk:
            if g is None:
                out.append({"vulva": np.zeros((FULL_H, FULL_W), np.uint8), "tail": None, "covered_by_tail": False,
                            "pmax": 0.0, "area": 0, "conf": 0.0, "no_input": True})
                continue
            p = probs.pop()
            vul, pv, tail, area = parts_from_probs(p, threshold, min_area)
            conf = blob_confidence(pv, vul)
            if conf < presence:
                vul[:] = 0
            out.append({"vulva": to_full_frame(vul), "tail": to_full_frame(tail) if tail is not None else None,
                        "covered_by_tail": covered_by_tail(vul, tail), "pmax": float(pv.max()), "area": area, "conf": conf})
    return out


def predict(model, grays, device="cpu", threshold=0.5, min_area=None, batch_size=16, presence=0.0):
    """Full-frame inputs (read_input / make_input; plain grey for "ir" models)
    -> list of (full-frame mask uint8, max prob, area in model pixels, blob confidence).
    presence: blobs with a confidence below this are dropped (checkpoint "presence_threshold").
    3-class models: the vulva class; use predict_parts for the tail."""
    return [(d["vulva"], d["pmax"], d["area"], d["conf"])
            for d in predict_parts(model, grays, device, threshold, min_area, batch_size, presence)]


def largest_component(m):
    """Keep the biggest blob: there is one vulva per frame."""
    if m.sum() == 0:
        return m
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 2:
        return m
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == k).astype(np.uint8)
