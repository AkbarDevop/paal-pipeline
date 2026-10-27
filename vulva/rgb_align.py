"""Put the OAK-D colour image on the ToF / IR pixel grid, so outlines can be drawn on it.

The colour camera (1280 x 800) sits next to the ToF sensor and has no calibration file,
so there is no exact pixel map. Two steps:

1. An approximate map fitted by hand on the stall bars and anus (rgb = 1.69 * ir + (30, -12)
   in 1280 x 800 pixels; label_ellipse.py --rgb-map uses the same numbers).
2. A residual shift for each frame, found by matching image edges (gradient magnitude) of the
   IR and the mapped colour image around the vulva: about +18 IR px to the right on the Feb
   test frames, varying 3-37 px between frames with the sow's position, so it is measured
   per frame rather than fixed. If the match is weak, a wider window over the rump is used,
   then the typical shift.

register(ir, rgb, center) returns the colour image resampled onto the 640 x 480 IR grid
(BGR uint8) plus the shift and match score. Outlines in IR pixels can then be drawn on it
directly. Expect a few pixels of error; check the score.
"""

from __future__ import annotations

import cv2
import numpy as np

MAP = (1.69, 30.0, -12.0)          # scale, dx, dy: IR pixel -> 1280 x 800 colour pixel
DEFAULT_SHIFT = (18, 0)            # typical residual, IR px (8 hand-labelled Feb 12-15 frames)
REAR = (310, 255)                  # middle of the sow's rear in the IR frame


def _edges(gray):
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 2)
    m = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return (m - m.mean()) / (m.std() + 1e-6)


def to_ir_grid(rgb, shift=(0, 0), map_=MAP, size=(640, 480)):
    """Colour image (any resolution) -> IR pixel grid; shift = where the colour content sits
    relative to the IR, in IR pixels."""
    s, dx, dy = map_
    h, w = rgb.shape[:2]
    kx, ky = w / 1280.0, h / 800.0
    A = np.array([[s * kx, 0, (s * shift[0] + dx) * kx], [0, s * ky, (s * shift[1] + dy) * ky]], np.float32)
    return cv2.warpAffine(rgb, A, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)


def find_shift(ir, rgb_grid, center, half=70, rx=(-5, 40), ry=(-15, 15)):
    """Best (dx, dy) and its normalised correlation, matching edges in a window around center."""
    H, W = ir.shape[:2]
    cx, cy = int(round(center[0])), int(round(center[1]))
    if not (half - ry[0] <= cy < H - half - ry[1] and half - rx[0] <= cx < W - half - rx[1]):
        return None, 0.0
    t = _edges(cv2.equalizeHist(ir))[cy - half:cy + half, cx - half:cx + half]
    g = _edges(cv2.cvtColor(rgb_grid, cv2.COLOR_BGR2GRAY))
    big = g[cy - half + ry[0]:cy + half + ry[1], cx - half + rx[0]:cx + half + rx[1]]
    res = cv2.matchTemplate(big, t, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    return (loc[0] + rx[0], loc[1] + ry[0]), float(score)


def register(ir, rgb, center=None, min_score=0.3):
    """-> (colour image on the IR grid, (dx, dy), score, how). center: the vulva in IR pixels if
    known; how is 'vulva', 'rear' or 'default' (which window the shift came from)."""
    rough = to_ir_grid(rgb)
    if center is not None:
        shift, score = find_shift(ir, rough, center)
        if shift is not None and score >= min_score:
            return to_ir_grid(rgb, shift), shift, score, "vulva"
    shift, score = find_shift(ir, rough, REAR, half=150)
    if shift is not None and score >= min_score:
        return to_ir_grid(rgb, shift), shift, score, "rear"
    return to_ir_grid(rgb, DEFAULT_SHIFT), DEFAULT_SHIFT, score, "default"
