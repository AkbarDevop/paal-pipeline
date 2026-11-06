#!/usr/bin/env python3
r"""Vulva outline from the depth bump (Xu et al. 2023, section 2.4), seeded by a 2D outline.

Xu et al. cut a region around the vulva by hand, filled the vulva area from the surrounding
skin ("No Vulva Surface"), subtracted that from the original surface ("Vulva Only Surface") and
fitted an ellipse to the part that rises above the skin. Here the hand cut is replaced by a
seed outline (the U-Net's, or a hand outline) that says where to look; the rest follows the
paper on the ToF depth of the same frame, at the camera's own resolution:

  1. original surface: height of every depth point in a window around the seed above a plane
     fitted to the skin around it (so a tilted rump does not count as height)
  2. no-vulva surface: the seed grown by 35 % (the paper's dilation) is cut out and filled
     with a smooth quadratic surface fitted to a ring of skin around it (tail pixels left out)
  3. vulva-only surface = original - no-vulva
  4. bump: inside the grown seed, the pixels rising at least max(MIN_RISE_MM, REL_RISE x peak),
     the connected piece that overlaps the seed most; the ellipse fitted to it is the outline

Every outline comes with quality numbers (peak height, roundness, overlap with the seed,
share of the bump's edge cut off by the grown seed) so a flat, ragged or clipped bump can be
flagged instead of trusted.

First look on a labelled set (hand outlines as seeds), sheets on the colour image:

    python vulva/depth_outline.py ~/Downloads/paal_exchange/vulva_test_feb12/labels.csv --n 20 --out outputs/depth_outline_look
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure3d import (DEFAULT_INTRINSICS, MAX_DEPTH_MM, MIN_DEPTH_MM, backproject,  # noqa: E402
                       fit_plane)

GROW = 0.35          # seed grown by this share of its equivalent radius (paper: 35 %)
RING_MM = 25.0       # width of the skin ring around the grown seed used to fill it
MIN_RISE_MM = 2.0    # a bump pixel rises at least this much above the filled skin ...
REL_RISE = 0.2       # ... and at least this share of the peak
MIN_PEAK_MM = 3.0    # a bump lower than this is "flat"
FAR_MM = 200.0       # depth this far from the seed's own depth is background (bars, floor)


def _grow(mask, share):
    r_eq = np.sqrt(mask.sum() / np.pi)
    k = max(int(round(share * r_eq)), 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    return cv2.dilate(mask.astype(np.uint8), ker).astype(bool)


def _ring(mask, px):
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.uint8), ker).astype(bool) & ~mask


def fit_quadratic(u, v, h, iters=3):
    """Robust quadratic surface h = a + bu + cv + du^2 + euv + fv^2 -> (coef, rmse of the kept points)."""
    A = np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], axis=1)
    keep = np.ones(len(h), bool)
    coef = np.zeros(6)
    for _ in range(iters):
        if keep.sum() < 12:
            break
        coef = np.linalg.lstsq(A[keep], h[keep], rcond=None)[0]
        res = h - A @ coef
        mad = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep]))) + 1e-6
        new = np.abs(res) < 3 * mad
        if new.sum() < 0.6 * len(h):
            break
        keep = new
    res = h - A @ coef
    return coef, float(np.sqrt(np.mean(res[keep] ** 2)))


def _quad_eval(coef, u, v):
    return coef[0] + coef[1] * u + coef[2] * v + coef[3] * u * u + coef[4] * u * v + coef[5] * v * v


def depth_outline(depth_mm, seed, tail=None, K=None, grow=GROW, ring_mm=RING_MM,
                  min_rise=MIN_RISE_MM, rel_rise=REL_RISE):
    """depth_mm (480 x 640 uint16 mm), seed (bool mask, same size), tail (optional bool mask).
    Returns a dict: status, mask (full-frame bool, the ellipse), ellipse, stats, and the window
    surfaces for plotting (os, nvs, vos, grown, bump, box)."""
    K = K or DEFAULT_INTRINSICS
    out = {"status": "", "mask": None, "ellipse": None, "peak_mm": np.nan, "thr_mm": np.nan,
           "roundness": np.nan, "iou_seed": np.nan, "seed_cover": np.nan, "clipped": np.nan,
           "skin_rmse_mm": np.nan, "n_valid": 0}
    seed = seed.astype(bool)
    if depth_mm is None:
        out["status"] = "no_depth"
        return out
    if seed.sum() < 10:
        out["status"] = "no_seed"
        return out
    H, W = depth_mm.shape
    tail = np.zeros_like(seed) if tail is None else tail.astype(bool)
    d = depth_mm.astype(np.float64)
    valid = (d > MIN_DEPTH_MM) & (d < MAX_DEPTH_MM)
    if (seed & valid).sum() < 10:
        out["status"] = "no_depth"
        return out
    z0 = float(np.median(d[seed & valid]))
    valid &= np.abs(d - z0) < FAR_MM
    ring_px = max(int(round(ring_mm * K["fx"] / z0)), 3)

    grown = _grow(seed, grow) & ~tail
    skin = _ring(grown | tail, ring_px) & valid & ~tail
    if skin.sum() < 30 or (grown & valid).sum() < 20:
        out["status"] = "too_few_points"
        out["n_valid"] = int((grown & valid).sum())
        return out
    outer = grown | tail | skin
    ys, xs = np.nonzero(_ring(outer, 2) | outer)
    y0, y1, x0, x1 = max(ys.min(), 0), min(ys.max() + 1, H), max(xs.min(), 0), min(xs.max() + 1, W)
    side = max(y1 - y0, x1 - x0)                      # square window for the pictures
    cy_, cx_ = (y0 + y1) // 2, (x0 + x1) // 2
    y0, x0 = max(cy_ - side // 2, 0), max(cx_ - side // 2, 0)
    y1, x1 = min(y0 + side, H), min(x0 + side, W)
    box = (x0, y0, x1, y1)

    # 1. original surface: height above the skin plane (positive towards the camera)
    sy, sx = np.nonzero(skin)
    nrm, dd, _ = fit_plane(backproject(depth_mm, sy, sx, K))
    win_valid = valid[y0:y1, x0:x1]
    wy, wx = np.nonzero(win_valid)
    pts = backproject(depth_mm, wy + y0, wx + x0, K)
    os_ = np.full((y1 - y0, x1 - x0), np.nan)
    os_[wy, wx] = pts @ nrm + dd

    # 2. no-vulva surface: quadratic fitted to the skin ring, filling the grown seed
    sc = float(side) / 2
    U, V = np.meshgrid((np.arange(x0, x1) - cx_) / sc, (np.arange(y0, y1) - cy_) / sc)
    sk = skin[y0:y1, x0:x1] & np.isfinite(os_)
    coef, skin_rmse = fit_quadratic(U[sk], V[sk], os_[sk])
    filled = _quad_eval(coef, U, V)
    g_win = grown[y0:y1, x0:x1]
    nvs = os_.copy()
    nvs[g_win] = filled[g_win]

    # 3. vulva-only surface
    vos = np.where(g_win & np.isfinite(os_), os_ - filled, 0.0)
    vos = np.nan_to_num(vos)

    # 4. bump inside the grown seed, the piece overlapping the seed most, ellipse fit
    s_win = seed[y0:y1, x0:x1]
    inside = vos[g_win & win_valid]
    peak = float(np.percentile(inside, 98)) if inside.size else 0.0
    thr = max(min_rise, rel_rise * peak)
    out.update(peak_mm=round(peak, 2), thr_mm=round(thr, 2), skin_rmse_mm=round(skin_rmse, 2),
               n_valid=int((g_win & win_valid).sum()))
    out.update(os=os_, nvs=nvs, vos=vos, grown=g_win, box=box)
    if peak < MIN_PEAK_MM:
        out["status"] = "flat"
        return out
    bump = ((vos >= thr) & g_win & win_valid).astype(np.uint8)
    bump = cv2.morphologyEx(bump, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bump, connectivity=8)
    if n <= 1:
        out["status"] = "no_bump"
        return out
    overlap = [(lab == i)[s_win].sum() for i in range(1, n)]
    best = 1 + int(np.argmax(overlap)) if max(overlap) > 0 else 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (lab == best).astype(np.uint8)
    comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 5:
        out["status"] = "no_bump"
        return out
    (ex, ey), (ea, eb), ang = cv2.fitEllipse(c)
    ell = np.zeros((H, W), np.uint8)
    cv2.ellipse(ell, ((ex + x0, ey + y0), (ea, eb), ang), 1, -1)
    ell = ell.astype(bool)
    area = float(comp.sum())
    ell_area = np.pi * ea * eb / 4
    edge = cv2.morphologyEx(comp, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
    g_edge = cv2.morphologyEx(g_win.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
    out.update(status="ok", mask=ell, bump=comp.astype(bool),
               ellipse=(round(ex + x0, 1), round(ey + y0, 1), round(ea / 2, 2), round(eb / 2, 2), round(ang, 1)),
               roundness=round(min(area / max(ell_area, 1e-6), ell_area / max(area, 1e-6)), 3),
               iou_seed=round(float((ell & seed).sum() / max((ell | seed).sum(), 1)), 3),
               seed_cover=round(float((ell & seed).sum() / max(seed.sum(), 1)), 3),
               clipped=round(float((edge & g_edge).sum() / max(edge.sum(), 1)), 3))
    return out


# ---------------------------------------------------------------- first look on a labelled set

def _jet(a, lo, hi):
    v = np.clip((np.nan_to_num(a, nan=lo) - lo) / max(hi - lo, 1e-6), 0, 1)
    img = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_JET)
    img[~np.isfinite(a)] = (40, 40, 40)
    return img


def _contour(img, mask, colour, scale, thick=2):
    if mask is None or not mask.any():
        return
    cnts, _ = cv2.findContours(cv2.resize(mask.astype(np.uint8), None, fx=scale, fy=scale,
                                          interpolation=cv2.INTER_NEAREST), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cnts, -1, colour, thick, cv2.LINE_AA)


def _label(img, text):
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def panel_row(colour, res, seed, P=200):
    """colour | original surface | no-vulva surface | vulva-only surface, one window."""
    x0, y0, x1, y1 = res["box"]
    s = P / max(x1 - x0, y1 - y0)
    def fit(img):
        im = cv2.resize(img, (P, P), interpolation=cv2.INTER_NEAREST if img.dtype != np.uint8 or img.ndim == 3 else cv2.INTER_LINEAR)
        return im
    col = cv2.resize(colour[y0:y1, x0:x1], (P, P), interpolation=cv2.INTER_CUBIC)
    sw = seed[y0:y1, x0:x1]
    _contour(col, sw, (60, 220, 60), s)
    if res.get("mask") is not None:
        _contour(col, res["mask"][y0:y1, x0:x1], (0, 165, 255), s)
    lo, hi = np.nanpercentile(res["os"], 2), np.nanpercentile(res["os"], 98)
    a = fit(_jet(res["os"], lo, hi))
    b = fit(_jet(res["nvs"], lo, hi))
    _contour(b, res["grown"], (255, 255, 255), s, 1)
    c = fit(_jet(res["vos"], 0, max(res["peak_mm"], 5)))
    if res.get("bump") is not None:
        _contour(c, res["bump"], (255, 255, 255), s, 1)
    if res.get("mask") is not None:
        _contour(c, res["mask"][y0:y1, x0:x1], (0, 165, 255), s)
    _label(col, "colour: hand / depth")
    _label(a, "original surface")
    _label(b, "no-vulva surface")
    _label(c, f"vulva only, 0-{max(res['peak_mm'], 5):.0f} mm")
    return np.hstack([col, a, b, c])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels_csv")
    ap.add_argument("--out", default="outputs/depth_outline_look")
    ap.add_argument("--n", type=int, default=20, help="frames drawn on the sheets (all present frames are measured)")
    ap.add_argument("--frames-root", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-rise", type=float, default=MIN_RISE_MM, help="mm a bump pixel must rise above the filled skin")
    ap.add_argument("--rel-rise", type=float, default=REL_RISE, help="... and this share of the peak")
    ap.add_argument("--grow", type=float, default=GROW, help="seed grown by this share of its radius")
    args = ap.parse_args()

    from eval_seg import read_rows
    from measure3d import load_depth_raw, measure
    from seg_model import read_ir, read_mask
    rows = [r for r in read_rows(args.labels_csv, [p for p in args.frames_root.split(";") if p.strip()])
            if r["label"] == "present"]
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    by = defaultdict(list)
    for r in rows:
        by[r["pig_id"]].append(r)
    pigs = sorted(by)
    rng.shuffle(pigs)
    for p in pigs:
        rng.shuffle(by[p])
    order = []
    while any(by.values()):                          # round-robin over sows: varied first frames
        for p in pigs:
            if by[p]:
                order.append(by[p].pop())

    table, sheet_rows = [], []
    for r in order:
        seed = read_mask(r["mask_path"])
        depth = load_depth_raw(r["depth_path"])
        if seed is None or depth is None:
            continue
        seed = seed > 0
        res = depth_outline(depth, seed, grow=args.grow, min_rise=args.min_rise, rel_rise=args.rel_rise)
        hand = measure(depth, seed)
        dm = measure(depth, res["mask"]) if res["status"] == "ok" else {}
        table.append({"pig_id": r["pig_id"], "pig_timestamp": r["pig_timestamp"], "status": res["status"],
                      "peak_mm": res["peak_mm"], "roundness": res["roundness"], "iou_with_hand": res["iou_seed"],
                      "clipped": res["clipped"], "skin_rmse_mm": res["skin_rmse_mm"],
                      "hand_width_mm": hand.get("width_mm", ""), "depth_width_mm": dm.get("width_mm", ""),
                      "hand_length_mm": hand.get("length_mm", ""), "depth_length_mm": dm.get("length_mm", "")})
        if len(sheet_rows) < args.n and "os" in res:
            ir = read_ir(r["ir_path"])
            reg = r["ir_path"].replace("_ir_vis_", "_rgbreg_")
            colour = cv2.imread(reg) if os.path.exists(reg) else None
            if colour is None or colour.shape[:2] != depth.shape:
                colour = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
            row = panel_row(colour, res, seed)
            head = np.full((22, row.shape[1], 3), 255, np.uint8)
            hw, dw = hand.get("width_mm", ""), dm.get("width_mm", "")
            txt = (f"sow {r['pig_id']}  {r['pig_timestamp']}  {res['status']}  peak {res['peak_mm']} mm  "
                   f"IoU with hand {res['iou_seed']}  width hand {hw} -> depth {dw} mm")
            cv2.putText(head, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
            sheet_rows.append(np.vstack([head, row]))

    with open(os.path.join(args.out, "depth_outline.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    for k in range(0, len(sheet_rows), 10):
        cv2.imwrite(os.path.join(args.out, f"sheet_{k // 10 + 1:02d}.jpg"), np.vstack(sheet_rows[k:k + 10]),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

    ok = [t for t in table if t["status"] == "ok"]
    st = defaultdict(int)
    for t in table:
        st[t["status"]] += 1
    print(f"{len(table)} hand-outlined frames: " + ", ".join(f"{k} {v}" for k, v in sorted(st.items())))
    if ok:
        f = lambda k: np.array([float(t[k]) for t in ok if t[k] != ""])  # noqa: E731
        print(f"bump peak median {np.median(f('peak_mm')):.1f} mm; roundness median {np.median(f('roundness')):.2f}; "
              f"IoU with the hand outline median {np.median(f('iou_with_hand')):.2f}; clipped by the grown seed "
              f"(>30% of its edge) on {int((f('clipped') > 0.3).sum())} of {len(ok)}")
        both = [t for t in ok if t["hand_width_mm"] != "" and t["depth_width_mm"] != ""]
        if both:
            hw = np.array([float(t["hand_width_mm"]) for t in both]); dw = np.array([float(t["depth_width_mm"]) for t in both])
            hl = np.array([float(t["hand_length_mm"]) for t in both]); dl = np.array([float(t["depth_length_mm"]) for t in both])
            print(f"width  median hand {np.median(hw):.1f} mm, depth {np.median(dw):.1f} mm (depth/hand {np.median(dw / hw):.2f})")
            print(f"length median hand {np.median(hl):.1f} mm, depth {np.median(dl):.1f} mm (depth/hand {np.median(dl / hl):.2f})")
    print(f"sheets and depth_outline.csv -> {args.out}")


if __name__ == "__main__":
    main()
