#!/usr/bin/env python3
"""Depth pictures that make the vulva's 3D shape visible next to the IR image.

The ToF depth is pixel-aligned with the IR image, so both views can be clicked
with the same coordinates. Two views, both limited to the sow's body (the
connected surface around the image centre; stall bars and floor are dropped):

  shade   shaded relief: the smoothed depth surface lit from the upper left,
          like a terrain map. The vulva shows as a lit oval lump with a shadow
          below and to the right; flat feces and dark skin marks disappear.
  height  millimetres above the sow's own body surface smoothed over about
          25 mm: + = sticks out toward the camera, - = groove or hole. At this
          scale the vulva is a red oval in the groove below the anus; wider
          blurs mostly show the curve of the hams.

Sizes are set in millimetres and converted with the distance to the sow, so a
close and a far sow get the same amount of smoothing.

Usage (writes a side-by-side picture for one frame):
    python vulva/depth_views.py <ir.jpg> [depth.raw] --out view.jpg
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure3d import DEFAULT_INTRINSICS, load_depth_raw  # noqa: E402

FX = DEFAULT_INTRINSICS["fx"]
VALID = (150, 2500)            # mm; ToF zeros and far background are dropped
LIGHT = np.array([-0.3, -0.4, 0.87]) / np.linalg.norm([-0.3, -0.4, 0.87])   # mostly frontal, from upper left


def _nblur(d, ok, sigma):
    """Gaussian blur that ignores invalid pixels (normalised convolution)."""
    w = cv2.GaussianBlur(ok.astype(np.float32), (0, 0), sigma)
    s = cv2.GaussianBlur(np.where(ok, d, 0).astype(np.float32), (0, 0), sigma)
    return s / np.maximum(w, 1e-3)


def body_mask(depth_mm):
    """Pixels on the sow: valid depth near the median depth of the image centre,
    then the connected part that holds the centre. Returns (mask, distance_mm)."""
    d = depth_mm.astype(np.float32)
    ok = (d > VALID[0]) & (d < VALID[1])
    c = d[100:380, 220:420]
    c = c[(c > VALID[0]) & (c < VALID[1])]
    if c.size < 100:
        return ok, 800.0
    zc = float(np.median(c))
    m = ok & (np.abs(d - zc) < max(250.0, 0.45 * zc))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n > 1:
        k = lab[240, 320] if lab[240, 320] > 0 else 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        m = lab == k
    return m, zc


def relief(depth_mm, smooth_mm=5.0, trend_mm=25.0):
    """depth (mm) -> (shade 0..1, height_mm, distance_mm); NaN off the body."""
    d = depth_mm.astype(np.float32)
    body, zc = body_mask(d)
    px = zc / FX                                   # mm per pixel at the sow
    ds = _nblur(d, body, max(1.0, smooth_mm / px))
    gy, gx = np.gradient(ds)
    nx, ny = -gx / px, -gy / px
    shade = (nx * LIGHT[0] + ny * LIGHT[1] + LIGHT[2]) / np.sqrt(nx ** 2 + ny ** 2 + 1)
    height = _nblur(d, body, trend_mm / px) - ds   # + = closer than the smooth body
    shade[~body] = np.nan
    height[~body] = np.nan
    return shade, height, zc


def shade_image(shade, lo=0.6, hi=1.0):
    y = np.clip((np.nan_to_num(shade, nan=lo) - lo) / (hi - lo), 0, 1)
    g = (y * 255).astype(np.uint8)
    out = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
    out[np.isnan(shade)] = (20, 20, 28)
    return out


def height_image(height, span_mm=8.0):
    y = np.clip((np.nan_to_num(height, nan=-span_mm) + span_mm) / (2 * span_mm), 0, 1)
    c = cv2.applyColorMap((y * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    c = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
    c[np.isnan(height)] = (20, 20, 28)
    return c


def views_for(depth_path, span_mm=8.0):
    """Depth file -> (shade RGB, height RGB, distance_mm), or None if no usable depth."""
    dep = load_depth_raw(depth_path) if depth_path else None
    if dep is None or not ((dep > VALID[0]) & (dep < VALID[1])).any():
        return None
    s, h, zc = relief(dep)
    return shade_image(s), height_image(h, span_mm), zc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ir")
    ap.add_argument("depth", nargs="?", default="")
    ap.add_argument("--out", default="depth_view.jpg")
    args = ap.parse_args()
    depth = args.depth or os.path.join(os.path.dirname(args.ir),
                                       os.path.basename(args.ir).replace("_ir_vis_", "_depth_").rsplit(".", 1)[0] + ".raw")
    ir = cv2.imread(args.ir, cv2.IMREAD_GRAYSCALE)
    if ir is None:
        sys.exit(f"cannot read {args.ir}")
    ir = cv2.resize(ir, (640, 480)) if ir.shape != (480, 640) else ir
    v = views_for(depth)
    if v is None:
        sys.exit(f"no usable depth in {depth}")
    a = cv2.cvtColor(cv2.createCLAHE(3.0, (8, 8)).apply(ir), cv2.COLOR_GRAY2RGB)
    img = np.hstack([a, v[0], v[1]])[30:480]
    cv2.imwrite(args.out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"sow at {v[2]:.0f} mm -> {args.out}")


if __name__ == "__main__":
    main()
