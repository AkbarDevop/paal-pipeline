#!/usr/bin/env python3
r"""Look at what run_vulva.py did, frame by frame: pictures, numbers and 3D.

Takes the vulva_frames.csv of a run, picks frames (random, seeded), runs the same model
again on just those frames and writes, under --out-dir:

  panels/<pig>_<timestamp>.jpg   one row per frame:
      colour image | IR image | height above the skin (depth) | 3D view of the surface
      with the model's outline (yellow; orange if the confidence check dropped it) and the
      frame's numbers underneath: width, length, protrusion, volume, confidence, distance
  index.jpg                      all panels of this review, smaller, one under another
  ply/<pig>_<timestamp>.ply      (--ply N) the frame's 3D points around the vulva as a point
                                 cloud: vulva red, skin grey. Open in MeshLab or CloudCompare,
                                 or: python vulva/view_pointcloud.py <file>.ply

Which frames (--which): ok = measured, low_confidence = outline dropped by the confidence
check, not_visible = no outline, all. The colour image is a separate camera matched onto the
IR (rgb_align.py); the model itself only sees the IR image.

Usage:
    python vulva/review_run.py outputs\vulva_r7_day1\vulva_frames.csv --model models\vulva_seg_r7.pth
    python vulva/review_run.py outputs\vulva_r7_day1\vulva_frames.csv --model models\vulva_seg_r7.pth --which low_confidence --n 30
    python vulva/review_run.py outputs\vulva_r7_day1\vulva_frames.csv --model models\vulva_seg_r7.pth --n 10 --ply 5
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from depth_views import views_for  # noqa: E402
from measure3d import DEFAULT_INTRINSICS as K  # noqa: E402
from measure3d import backproject, fit_plane, load_depth_raw  # noqa: E402
from rgb_align import register  # noqa: E402
from seg_model import ir_channel, load_checkpoint, predict, read_input  # noqa: E402
from select_frames import depth_path_for  # noqa: E402

P = 300            # panel size, px
HALF = 80          # half-width of the view around the vulva, IR px


def colour_for(ir_path, gray, centre):
    d, name = os.path.split(ir_path)
    for ext in (".jpg", ".png"):
        p = os.path.join(d, name.replace("_ir_vis_", "_rgb_").rsplit(".", 1)[0] + ext)
        if os.path.exists(p):
            rgb = cv2.imread(p, cv2.IMREAD_COLOR)
            if rgb is not None:
                return register(gray, rgb, centre)[0]
    return None


def crop(img, cx, cy, half=HALF):
    h, w = img.shape[:2]
    x1, y1 = int(min(max(cx - half, 0), w - 2 * half)), int(min(max(cy - half, 0), h - 2 * half))
    return img[y1:y1 + 2 * half, x1:x1 + 2 * half], (x1, y1)


def outlined(img, mask, x1y1, colour):
    x1, y1 = x1y1
    t = cv2.resize(img, (P, P), interpolation=cv2.INTER_CUBIC)
    if mask is not None and mask.any():
        sc = P / (2 * HALF)
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts = [np.round((c - [x1, y1]) * sc).astype(np.int32) for c in cnts]
        cv2.drawContours(t, cnts, -1, colour, 2)
    return t


def label(img, text):
    cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def local_surface(depth, mask, cx, cy, half=HALF, step=2):
    """3D points in a window around the vulva, as (u, v, height above the skin plane) in mm,
    plus which of them are vulva. None without usable depth."""
    if depth is None or mask is None or not mask.any():
        return None
    valid = (depth > 300) & (depth < 3000)
    m8 = mask.astype(np.uint8)
    ring = cv2.dilate(m8, np.ones((3, 3), np.uint8), iterations=12).astype(bool) & \
        ~cv2.dilate(m8, np.ones((3, 3), np.uint8), iterations=3).astype(bool) & valid
    if ring.sum() < 40:
        return None
    ys, xs = np.nonzero(ring)
    nrm, d, _ = fit_plane(backproject(depth, ys, xs, K))
    win = np.zeros_like(valid)
    (y1, y2), (x1, x2) = (max(cy - half, 0), min(cy + half, depth.shape[0])), (max(cx - half, 0), min(cx + half, depth.shape[1]))
    win[y1:y2:step, x1:x2:step] = True
    ys, xs = np.nonzero(win & valid)
    pts = backproject(depth, ys, xs, K)
    h = pts @ nrm + d
    u_ref = np.array([1.0, 0.0, 0.0]) - nrm[0] * nrm
    u_ref /= np.linalg.norm(u_ref)
    v_ref = np.cross(nrm, u_ref)
    c = pts.mean(axis=0)
    return (pts - c) @ u_ref, (pts - c) @ v_ref, h, mask[ys, xs].astype(bool), pts, ys, xs


def surface_panel(surf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(3, 3), dpi=P // 3)
    ax = fig.add_subplot(projection="3d")
    if surf is not None:
        u, v, h, vul, *_ = surf
        hc = np.clip(h, -6, 12)
        ax.scatter(u[~vul], -v[~vul], hc[~vul], c=hc[~vul], cmap="viridis", vmin=-6, vmax=12, s=1.2, linewidths=0)
        ax.scatter(u[vul], -v[vul], hc[vul], c="#ff3030", s=2.0, linewidths=0)
        ax.set_zlim(-6, 12)
        ax.view_init(elev=28, azim=-62)
        ax.set_zlabel("mm above skin", fontsize=6, labelpad=-2)
    ax.tick_params(labelsize=5, pad=0)
    ax.set_title("3D (red = vulva)" if surf is not None else "3D (no depth / no outline)", fontsize=7)
    fig.subplots_adjust(0, 0, 1, 0.95)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (P, P))


def write_ply(path, surf, ir):
    _, _, _, vul, pts, ys, xs = surf
    g = ir[ys, xs].astype(int)
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\n"
                "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for (x, y, z), v, gi in zip(pts, vul, g):
            r, gg, b = (230, 40, 40) if v else (gi, gi, gi)
            f.write(f"{x:.1f} {-y:.1f} {-z:.1f} {r} {gg} {b}\n")


def fmt(row, k, unit="", digits=0):
    try:
        return f"{float(row[k]):.{digits}f}{unit}"
    except (KeyError, ValueError):
        return "-"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames_csv", help="vulva_frames.csv from run_vulva.py")
    ap.add_argument("--model", required=True, help="the checkpoint the run used")
    ap.add_argument("--which", default="ok", choices=("ok", "low_confidence", "not_visible", "all"))
    ap.add_argument("--n", type=int, default=24, help="frames to show (random, seeded)")
    ap.add_argument("--ply", type=int, default=0, help="also write 3D point clouds for this many frames")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="", help="default: <run folder>/review_<which>")
    args = ap.parse_args()

    with open(args.frames_csv, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f)
                if args.which == "all" or r.get("measure_status") == args.which]
    if not rows:
        sys.exit(f"No frames with measure_status {args.which!r} in {args.frames_csv}")
    rows = sorted(random.Random(args.seed).sample(rows, min(args.n, len(rows))), key=lambda r: r["pig_timestamp"])
    out = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.frames_csv)), f"review_{args.which}")
    os.makedirs(os.path.join(out, "panels"), exist_ok=True)
    if args.ply:
        os.makedirs(os.path.join(out, "ply"), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model, ck = load_checkpoint(args.model, device)
    thr, min_area, cut = ck.get("threshold", 0.5), ck.get("min_area", 25), ck.get("presence_threshold", 0.0)
    inputs = ck.get("inputs", "ir")
    smalls, n_ply = [], 0
    for r in rows:
        ir_path = r["ir_path"]
        dpath = depth_path_for(ir_path)
        img = read_input(ir_path, dpath, inputs)
        if img is None:
            continue
        gray = ir_channel(img)
        mask, _pmax, _area, conf = predict(model, [img], device, thr, min_area)[0]
        has = mask.any()
        if has:
            ys, xs = np.nonzero(mask)
            cx, cy = int(xs.mean()), int(ys.mean())
        else:
            cx, cy = 310, 255
        col = (0, 215, 255) if has and conf >= cut else (0, 140, 255)
        depth = load_depth_raw(dpath)
        rgb = colour_for(ir_path, gray, (cx, cy) if has else None)
        tiles = []
        if rgb is not None:
            c, o = crop(rgb, cx, cy)
            tiles.append(label(outlined(c, mask, o, col), "colour (matched to IR)"))
        else:
            tiles.append(label(np.full((P, P, 3), 40, np.uint8), "no colour image"))
        # local contrast boost for display only (as in the labelling tool); a global histogram
        # equalisation washes the sow out to white, because the dark stall dominates the frame
        disp = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
        c, o = crop(cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR), cx, cy)
        tiles.append(label(outlined(c, mask, o, col), "IR (contrast-boosted)"))
        v = views_for(dpath)
        if v is not None:
            c, o = crop(cv2.cvtColor(v[1], cv2.COLOR_RGB2BGR), cx, cy)
            tiles.append(label(outlined(c, mask, o, (255, 255, 255)), "height above skin, +/-8 mm"))
        else:
            tiles.append(label(np.full((P, P, 3), 40, np.uint8), "no depth"))
        surf = local_surface(depth, mask, cx, cy)
        tiles.append(surface_panel(surf))
        ts = r["pig_timestamp"]
        bar = np.full((44, P * 4, 3), 245, np.uint8)
        line1 = (f"pig {r['pig_id']}  {ts[:4]}-{ts[4:6]}-{ts[6:8]} {ts[9:11]}:{ts[12:14]}   "
                 f"{r.get('measure_status', '')}   conf {conf:.3f} (cut {cut:.3f})")
        line2 = (f"width {fmt(r, 'width_mm', ' mm')}   length {fmt(r, 'length_mm', ' mm')}   "
                 f"protrudes {fmt(r, 'peak_height_mm', ' mm', 1)}   volume {fmt(r, 'volume_mm3', ' mm3')}   "
                 f"distance {fmt(r, 'distance_mm', ' mm')}   view angle {fmt(r, 'view_angle_deg', ' deg')}")
        cv2.putText(bar, line1, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(bar, line2, (8, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
        panel = np.vstack([np.hstack(tiles), bar])
        name = f"pig{r['pig_id']}_{ts}"
        cv2.imwrite(os.path.join(out, "panels", name + ".jpg"), panel, [cv2.IMWRITE_JPEG_QUALITY, 90])
        smalls.append(cv2.resize(panel, (panel.shape[1] // 2, panel.shape[0] // 2), interpolation=cv2.INTER_AREA))
        if args.ply and n_ply < args.ply and surf is not None:
            write_ply(os.path.join(out, "ply", name + ".ply"), surf, gray)
            n_ply += 1
    if smalls:
        cv2.imwrite(os.path.join(out, "index.jpg"), np.vstack(smalls), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"{len(smalls)} frames ({args.which}): panels in {os.path.join(out, 'panels')}, overview "
          f"{os.path.join(out, 'index.jpg')}" + (f", {n_ply} point clouds in {os.path.join(out, 'ply')}" if n_ply else ""))


if __name__ == "__main__":
    main()
