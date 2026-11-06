#!/usr/bin/env python3
r"""The check step, second try (plan E2b): a small CNN looks at a crop around the finder's
candidate and says whether it is a real, fully visible vulva.

Why: the feature check (E2, check_view.py) failed. At 95% precision it kept a third of the real
vulvas on Feb and 1 of 28 on the January sows. Its strongest inputs were blob size, distance and
position, which change between batches and camera set-ups, and a check that prefers big blobs
would bias the very size we want to measure. Most false candidates are vulvas partly covered by
the tail, so the check has to see the tail and the vulva together.

What the CNN sees: a square window 3 x the candidate's length, centred on it, resized to
128 x 128. Every candidate fills the same share of its window, so the check cannot prefer big (or
near) vulvas, and the tail root and anything crossing the vulva are in view. Layers:

  ir      the IR image, normalised within the window (no brightness shortcut between batches)
  blob    the candidate (the finder's blob), so the CNN knows which object it judges
  tail    the finder's tail probability (3-class finders; empty for 2-class)
  height  height above the local skin from the ToF depth (+/-20 mm; a tail lies in front of the
          vulva); neutral when the frame has no depth

Model: ResNet-18 (ImageNet weights). The IR goes into its three colour inputs; the other layers are
extra inputs whose first-layer weights start at zero (as in the fusion finder), so training starts
from an IR image classifier.

fit: candidates come from a finder run's held-out predictions (<run>/jobs/fold*.npz, as in
check_view.py), so each one was made by a finder that never saw that sow. One CNN per sow fold;
their out-of-fold scores give the cut (lowest score that keeps the precision target) and the
cross-validated numbers. The saved check is the fold CNNs together: a candidate's score is their
average, each on the crop and its mirror image.

    python vulva\check_crop.py fit --run outputs\vulva_seg_r10 --labels C:\Users\aakdbx\vulva_dataset_v1\labels.csv --split train --out models\checkcnn_r10.pth

eval: finder + check on a test split, against the finder's own keep rule. Also reports whether the
kept vulvas are bigger or smaller than all visible ones, and writes a sheet of the mistakes
(colour | what the check saw: candidate yellow, hand outline green, finder's tail red).

    python vulva\check_crop.py eval --model models\vulva_seg_r10.pth --check models\checkcnn_r10.pth --labels C:\Users\aakdbx\vulva_dataset_v1\labels.csv --split test_newdays
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_view import frame_key, pick_cut, pr_at, size_line  # noqa: E402
from seg_model import (FULL_H, FULL_W, blob_confidence, covered_by_tail, crop_box,  # noqa: E402
                       depth_features, parts_from_probs, read_ir, read_mask, to_full_frame,
                       to_model_space, to_tensor)

CHANNELS = ("ir", "blob", "tail", "height")
K_WINDOW = 3.0                 # window side = this x the candidate's length (full-frame pixels)
MIN_SIDE, MAX_SIDE = 72, 288   # full-frame pixels
WIN = 128                      # what the CNN sees
MARGIN = 1.5                   # stored patches are this much wider than the window (room to rotate / zoom)
STORE = int(round(WIN * MARGIN))
BORDER = {"ir": None, "blob": 0, "tail": 0, "height": 128}   # None = repeat the edge pixels


def parse_channels(text):
    ch = tuple(c.strip() for c in str(text).split(",") if c.strip())
    if not ch or ch[0] != "ir" or "blob" not in ch or any(c not in CHANNELS for c in ch) or len(set(ch)) < len(ch):
        sys.exit(f"--channels: 'ir' first and 'blob' included, from {','.join(CHANNELS)}; got {text!r}")
    return ch


# ------------------------------------------------------------------ crops

def prob_to_full_frame(p_small):
    """Model-space probability map -> full 640 x 480 frame (zeros outside the stall crop)."""
    x1, y1, x2, y2 = crop_box()
    full = np.zeros((FULL_H, FULL_W), np.float32)
    full[y1:y2, x1:x2] = cv2.resize(p_small.astype(np.float32), (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
    return full


def window_for(blob_full, k=K_WINDOW):
    """Candidate blob (full frame) -> window centre x, y and side, in full-frame pixels."""
    m = (blob_full > 0).astype(np.uint8)
    ys, xs = np.nonzero(m)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    length = max(cv2.minAreaRect(max(cnts, key=cv2.contourArea))[1])
    side = float(np.clip(k * max(length, 1.0), MIN_SIDE, MAX_SIDE))
    return float(xs.mean()), float(ys.mean()), side


def frame_layers(ir, blob_full, tail_prob_full=None, depth=None, channels=CHANNELS):
    """Full-frame uint8 layers in channel order."""
    out = []
    for c in channels:
        if c == "ir":
            out.append(ir)
        elif c == "blob":
            out.append((blob_full > 0).astype(np.uint8) * 255)
        elif c == "tail":
            out.append(np.zeros(ir.shape, np.uint8) if tail_prob_full is None
                       else np.clip(tail_prob_full * 255 + 0.5, 0, 255).astype(np.uint8))
        else:
            out.append(np.full(ir.shape, 128, np.uint8) if depth is None else depth_features(depth)[0])
    return out


def _warp(layer, A, size, border):
    if border is None:
        return cv2.warpAffine(layer, A, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.warpAffine(layer, A, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=border)


def extract_patch(layers, cx, cy, side, channels=CHANNELS, size=STORE, margin=MARGIN):
    """Full-frame layers -> C x size x size uint8. The window (side pixels around cx, cy) fills the
    middle size / margin pixels; the rest is context for augmentation."""
    s = size / (side * margin)
    A = np.float32([[s, 0, size / 2 - s * cx], [0, s, size / 2 - s * cy]])
    sigma = 0.5 * (1 / s - 1)                         # shrinking: smooth first, so fine texture does not alias
    out = []
    for name, lay in zip(channels, layers):
        if sigma > 0.3 and name != "blob":
            lay = cv2.GaussianBlur(lay, (0, 0), sigma)
        out.append(_warp(lay, A, size, BORDER[name]))
    return np.stack(out)


def centre(p, win=WIN):
    o = (p.shape[-1] - win) // 2
    return p[:, o:o + win, o:o + win]


def augment(p, rng, channels=CHANNELS):
    """Stored patch -> randomly turned (+/-15 deg), zoomed (0.87-1.15), shifted, mirrored window, with IR
    gamma, blur and noise. No occlusion: hiding part of the view could change the right answer."""
    S = p.shape[-1]
    A = cv2.getRotationMatrix2D((S / 2, S / 2), rng.uniform(-15, 15), rng.uniform(0.87, 1.15))
    A[:, 2] += rng.uniform(-0.05, 0.05, 2) * WIN
    q = np.stack([_warp(ch, A, S, BORDER[name]) for name, ch in zip(channels, p)])
    if rng.random() < 0.5:
        q = q[:, :, ::-1]
    q = centre(q).copy()
    ir = q[0].astype(np.float32) / 255
    ir = ir ** rng.uniform(0.8, 1.25)
    if rng.random() < 0.25:
        ir = cv2.GaussianBlur(ir, (0, 0), rng.uniform(0.6, 1.2))
    q[0] = np.clip(ir * 255 + rng.normal(0, 3, ir.shape), 0, 255).astype(np.uint8)
    return q


def patch_tensor(q, channels=CHANNELS):
    """C x WIN x WIN uint8 window -> float tensor. IR: zero mean and unit spread within the window,
    copied to 3 channels; candidate and tail: 0..255 -> -1..1; height: 128 = level -> 0."""
    xs = []
    for name, ch in zip(channels, q):
        x = ch.astype(np.float32)
        if name == "ir":
            x = (x - x.mean()) / (x.std() + 4.0)
            xs += [x, x, x]
        elif name == "height":
            xs.append((x / 255 - 0.5) / 0.25)
        else:
            xs.append(x / 127.5 - 1)
    return torch.from_numpy(np.stack(xs))


# ------------------------------------------------------------------ model

class CropCheck(nn.Module):
    def __init__(self, channels=CHANNELS, pretrained=True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        old = net.conv1
        net.conv1 = nn.Conv2d(3 + len(channels) - 1, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            net.conv1.weight.zero_()
            net.conv1.weight[:, :3] = old.weight
        net.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 1))
        self.net, self.channels = net, tuple(channels)

    def forward(self, x):
        return self.net(x)[:, 0]


def device_name():
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def train_check(P, y, channels=CHANNELS, epochs=30, device="cpu", pretrained=True, lr=3e-4, batch_size=32,
                seed=0, log=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = CropCheck(channels, pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(epochs * int(np.ceil(len(y) / batch_size)), 1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    lossf = nn.BCEWithLogitsLoss()
    yt = torch.tensor(np.asarray(y), dtype=torch.float32)
    for ep in range(epochs):
        model.train()
        order, tot = rng.permutation(len(y)), 0.0
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            if len(idx) < 2:                      # batch norm needs two or more
                continue
            x = torch.stack([patch_tensor(augment(P[j], rng, channels), channels) for j in idx]).to(device)
            loss = lossf(model(x), yt[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        if log and (ep % 10 == 9 or ep == epochs - 1):
            log(f"    epoch {ep + 1}/{epochs}  loss {tot / len(y):.4f}")
    return model.eval()


@torch.no_grad()
def score_patches(models, P, channels=CHANNELS, device="cpu", batch_size=64):
    """Mean probability over the models, each on the window and its mirror image."""
    out = np.zeros(len(P))
    for m in models:
        m.eval()
    for i in range(0, len(P), batch_size):
        x = torch.stack([patch_tensor(centre(p), channels) for p in P[i:i + batch_size]]).to(device)
        s = 0
        for m in models:
            s = s + torch.sigmoid(m(x)) + torch.sigmoid(m(torch.flip(x, dims=[3])))
        out[i:i + len(x)] = (s / (2 * len(models))).cpu().numpy()
    return out


def auc(scores, y):
    """Chance that a real candidate scores above a false one (ties count half)."""
    s, y = np.asarray(scores, float), np.asarray(y).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def save_check(path, states, meta):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    half = [{n: (t.detach().cpu().half() if t.is_floating_point() else t.detach().cpu()) for n, t in s.items()}
            for s in states]
    torch.save({"kind": "crop_check", "state_dicts": half, **meta}, path)


def load_check(path, device="cpu"):
    chk = torch.load(path, map_location="cpu", weights_only=False)
    if chk.get("kind") != "crop_check":
        sys.exit(f"{path} is not a crop check (check_crop.py fit)")
    models = []
    for s in chk["state_dicts"]:
        m = CropCheck(tuple(chk["channels"]), pretrained=False)
        m.load_state_dict({n: (t.float() if t.is_floating_point() else t) for n, t in s.items()})
        models.append(m.to(device).eval())
    return models, chk


# ------------------------------------------------------------------ fit

def candidates_from_run(run_dir, rows, channels=CHANNELS, threshold=0.5, min_area=25, iou_real=0.2):
    """Held-out probabilities of a finder run -> (patches N x C x STORE x STORE, labels, folds, keys,
    hand-outline sizes in pixels (sqrt of the area; 0 when not visible))."""
    from measure3d import load_depth_raw
    from train_seg import full_probs
    by_key = {frame_key(r["ir_path"]): r for r in rows}
    files = sorted(glob.glob(os.path.join(run_dir, "jobs", "fold*.npz")))
    if not files:
        sys.exit(f"No held-out predictions in {os.path.join(run_dir, 'jobs')} (train the finder with --jobs)")
    P, ys, folds, keys, sizes = [], [], [], [], []
    for fpath in files:
        k = int(re.search(r"fold(\d+)", os.path.basename(fpath)).group(1))
        d = np.load(fpath, allow_pickle=True)
        for key, pr in zip(d["keys"], d["probs"]):
            r = by_key.get(frame_key(key))
            if r is None:
                continue
            p = full_probs(pr)
            vul = parts_from_probs(p, threshold, min_area)[0]
            if not vul.any():
                continue                                   # nothing to check
            ir = read_ir(r["ir_path"])
            if ir is None:
                continue
            vf = to_full_frame(vul) > 0
            tp = prob_to_full_frame(p[2]) if p.ndim == 3 else None
            depth = load_depth_raw(r.get("depth_path", "")) if "height" in channels else None
            cx, cy, side = window_for(vf)
            P.append(extract_patch(frame_layers(ir, vf, tp, depth, channels), cx, cy, side, channels))
            real, size = 0, 0.0
            if r["label"] == "present":
                hm = read_mask(r["mask_path"])
                if hm is not None and hm.any():
                    hm = hm > 0
                    size = float(np.sqrt(hm.sum()))
                    real = int((vf & hm).sum() / max((vf | hm).sum(), 1) >= iou_real)
            ys.append(real)
            folds.append(k)
            keys.append(r["ir_path"])
            sizes.append(size)
    if not P:
        sys.exit(f"No candidates: none of the frames in {run_dir}\\jobs matched the labels, or the finder drew nothing")
    return np.stack(P), np.array(ys), np.array(folds), keys, np.array(sizes)


def cmd_fit(args):
    from eval_seg import read_rows
    from train_seg import require_split, split_set
    channels = parse_channels(args.channels)
    require_split(args.labels, args.split)
    rows = read_rows(args.labels, [p for p in args.frames_root.split(";") if p.strip()], split_set(args.split))
    t0 = time.time()
    P, y, folds, keys, sizes = candidates_from_run(args.run, rows, channels, args.threshold, args.min_area, args.iou)
    if len(set(y.tolist())) < 2:
        sys.exit(f"Need both real and false candidates; got {len(y)} candidates, {int(y.sum())} real")
    device = device_name()
    log = lambda s: print(s, flush=True)  # noqa: E731
    log(f"{len(y)} held-out candidates from {args.run}: {int(y.sum())} real vulvas, {int((y == 0).sum())} false "
        f"(crops made in {time.time() - t0:.0f} s; training on {device})")
    oof, states = np.zeros(len(y)), []
    for k in sorted(set(folds.tolist())):
        tr, te = folds != k, folds == k
        if len(set(y[tr].tolist())) < 2:
            sys.exit(f"fold {k}: the other folds hold only one kind of candidate")
        t1 = time.time()
        log(f"  fold {k}: train on {int(tr.sum())} ({int(y[tr].sum())} real), score {int(te.sum())}")
        m = train_check(P[tr], y[tr], channels, args.epochs, device, not args.no_pretrained, args.lr,
                        args.batch_size, args.seed + k, log)
        oof[te] = score_patches([m], P[te], channels, device)
        states.append(m.state_dict())
        log(f"    {time.time() - t1:.0f} s")
    cut = pick_cut(oof, y, args.precision)
    tp, fp, n_real, n_false = pr_at(oof, y, cut)
    a = auc(oof, y)
    others = []
    for target in (0.90, 0.95, 0.98):
        c = pick_cut(oof, y, target)
        t_, f_, _, _ = pr_at(oof, y, c)
        others.append(f"precision {target:.2f}: cut {c:.3f}, keeps {t_} real and {f_} false")
    kept = (oof >= cut) & (y == 1)
    lines = [f"{len(y)} held-out candidates from {args.run}: {n_real} real vulvas, {n_false} false "
             f"(tails, covered or wrong spot)",
             f"crop check, cross-validated by the same sow folds: AUC {a:.3f}",
             f"  cut {cut:.3f} (precision target {args.precision}): keeps {tp} of {n_real} real ({tp / max(n_real, 1):.0%}) "
             f"and {fp} of {n_false} false",
             "  other targets: " + "; ".join(others)]
    sl = size_line(sizes[kept], sizes[y == 1])
    if sl:
        lines.append("  " + sl)
    meta = {"channels": list(channels), "k_window": K_WINDOW, "min_side": MIN_SIDE, "max_side": MAX_SIDE,
            "win": WIN, "store": STORE, "margin": MARGIN, "cut": cut, "precision_target": args.precision,
            "iou_real": args.iou, "threshold": args.threshold, "min_area": args.min_area,
            "run": os.path.abspath(args.run), "candidates": int(len(y)), "auc": a, "epochs": args.epochs,
            "cv": {"real": n_real, "false": n_false, "kept_real": tp, "kept_false": fp}}
    save_check(args.out, states, meta)
    cv_csv = os.path.splitext(args.out)[0] + "_cv.csv"
    with open(cv_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ir_path", "fold", "real", "score", "hand_size_px"])
        for row in zip(keys, folds.tolist(), y.tolist(), np.round(oof, 4).tolist(), np.round(sizes, 1).tolist()):
            w.writerow(row)
    print("\n".join(lines))
    print(f"-> {args.out} ({len(states)} fold CNNs; out-of-fold scores {cv_csv})")


# ------------------------------------------------------------------ eval

@torch.no_grad()
def finder_probs(model, imgs, device, batch_size=16):
    """Full-frame inputs -> model-space probabilities per frame (3 x H x W bg / vulva / tail, or H x W),
    None where there is no input."""
    out = []
    n_out, inputs = getattr(model, "n_out", 1), getattr(model, "inputs", None)
    for i in range(0, len(imgs), batch_size):
        chunk = imgs[i:i + batch_size]
        live = [g for g in chunk if g is not None]
        probs = []
        if live:
            x = torch.stack([to_tensor(to_model_space(g)[0], inputs) for g in live]).to(device)
            lg = model(x)
            probs = list((torch.softmax(lg, 1) if n_out == 3 else torch.sigmoid(lg)[:, 0]).float().cpu().numpy())
        probs.reverse()
        for g in chunk:
            out.append(None if g is None else probs.pop())
    return out


def colour_window(r, ir, cx, cy):
    """The frame's colour image on the IR grid (registered near the candidate), or None."""
    reg = r.get("rgb_reg", "")
    if reg and os.path.exists(reg):
        im = cv2.imread(reg, cv2.IMREAD_COLOR)
        if im is not None and im.shape[:2] == ir.shape[:2]:
            return im
    plain = r.get("rgb_plain", "")
    if plain and os.path.exists(plain):
        rgb = cv2.imread(plain, cv2.IMREAD_COLOR)
        if rgb is not None:
            from rgb_align import register
            return register(ir, rgb, (cx, cy))[0]
    return None


def error_tile(r, ir, vf, tail_prob, hm, cx, cy, side, text, size=160):
    """Colour | IR, both showing the check's window: candidate yellow, hand outline green, tail red."""
    s = size / side
    A = np.float32([[s, 0, size / 2 - s * cx], [0, s, size / 2 - s * cy]])

    def win(img, nearest=False):
        flags = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
        return cv2.warpAffine(img, A, (size, size), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    irw = cv2.cvtColor(cv2.equalizeHist(win(ir)), cv2.COLOR_GRAY2BGR)
    col = colour_window(r, ir, cx, cy)
    left = win(col) if col is not None else irw.copy()
    marks = [(vf, (0, 220, 255)), (hm, (60, 200, 60))]
    if tail_prob is not None:
        marks.append((tail_prob >= 0.5, (60, 60, 255)))
    for m, colr in marks:
        if m is None or not m.any():
            continue
        cnts, _ = cv2.findContours(win(m.astype(np.uint8), nearest=True), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for t in (left, irw):
            cv2.drawContours(t, cnts, -1, colr, 1)
    t = np.hstack([left, irw])
    for y, line in ((14, text[0]), (size - 8, text[1])):
        cv2.putText(t, line, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(t, line, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def save_tiles(tiles, path, cols=4):
    if not tiles:
        return False
    blank = np.zeros_like(tiles[0])
    rows = [tiles[i:i + cols] + [blank] * (cols - len(tiles[i:i + cols])) for i in range(0, len(tiles), cols)]
    cv2.imwrite(path, np.vstack([np.hstack(r) for r in rows]), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return True


def check_frames(finder, ckpt, models, chk, rows, device="cpu", sheet=True):
    """Finder + crop check on labelled rows -> (one record per frame, [(sort key, tile)] for the mistakes)."""
    from measure3d import load_depth_raw
    from seg_model import read_input
    channels = tuple(chk["channels"])
    inputs = ckpt.get("inputs", "ir")
    thr = chk.get("threshold", ckpt.get("threshold", 0.5))
    min_area = chk.get("min_area", 25)
    presence = float(ckpt.get("presence_threshold", 0.0) or 0.0)
    out, errors = [], []
    for i in range(0, len(rows), 16):
        chunk = rows[i:i + 16]
        imgs = [read_input(r["ir_path"], r.get("depth_path") or None, inputs) for r in chunk]
        for r, p in zip(chunk, finder_probs(finder, imgs, device)):
            hm = read_mask(r["mask_path"]) if r["label"] == "present" else None
            hm = hm > 0 if hm is not None and hm.any() else None
            rec = {"ir_path": r["ir_path"], "label": r["label"], "candidate": 0, "iou": 0.0, "real": 0,
                   "check_score": 0.0, "check_keep": 0, "rule_keep": 0,
                   "hand_size_px": round(float(np.sqrt(hm.sum())), 1) if hm is not None else 0.0}
            vul = parts_from_probs(p, thr, min_area) if p is not None else None
            if vul is not None and vul[0].any():
                vm, pv, tail, _ = vul
                vf = to_full_frame(vm) > 0
                iou = float((vf & hm).sum() / max((vf | hm).sum(), 1)) if hm is not None else 0.0
                rule = (not covered_by_tail(vm, tail)) if tail is not None else blob_confidence(pv, vm) >= presence
                ir = read_ir(r["ir_path"])
                tp = prob_to_full_frame(p[2]) if p.ndim == 3 else None
                depth = load_depth_raw(r.get("depth_path", "")) if "height" in channels else None
                cx, cy, side = window_for(vf, chk.get("k_window", K_WINDOW))
                patch = extract_patch(frame_layers(ir, vf, tp, depth, channels), cx, cy, side, channels,
                                      chk.get("store", STORE), chk.get("margin", MARGIN))
                score = float(score_patches(models, patch[None], channels, device)[0])
                rec.update(candidate=1, iou=round(iou, 3), real=int(r["label"] == "present" and iou >= chk["iou_real"]),
                           check_score=round(score, 4), check_keep=int(score >= chk["cut"]), rule_keep=int(rule))
                if sheet and rec["check_keep"] != rec["real"]:
                    ts = frame_key(r["ir_path"])
                    head = f"p{ts[0]} {ts[1][4:6]}/{ts[1][6:8]} {ts[1][9:11]}:{ts[1][12:14]}" if ts else ""
                    what = "kept, not real" if rec["check_keep"] else "dropped, real"
                    errors.append((-score if rec["check_keep"] else score,
                                   error_tile(r, ir, vf, tp, hm, cx, cy, side, (head, f"{what} {score:.2f}"))))
            out.append(rec)
    return out, errors


def cmd_eval(args):
    from eval_seg import read_rows
    from seg_model import load_checkpoint
    from train_seg import require_split, split_set
    require_split(args.labels, args.split)
    device = device_name()
    models, chk = load_check(args.check, device)
    rows = read_rows(args.labels, [p for p in args.frames_root.split(";") if p.strip()], split_set(args.split))
    finder, ckpt = load_checkpoint(args.model, device)
    out, errors = check_frames(finder, ckpt, models, chk, rows, device, args.sheet)
    n_vis = sum(1 for o in out if o["label"] == "present")
    lines = [f"{args.model} + crop check {args.check} on {args.labels} (split {args.split}): {len(out)} frames, "
             f"{n_vis} with a visible vulva, {len(out) - n_vis} without"]
    all_sizes = np.array([o["hand_size_px"] for o in out if o["label"] == "present"])
    for name, k in (("finder's own rule", "rule_keep"), ("with the crop check", "check_keep")):
        kr = sum(1 for o in out if o[k] and o["real"])
        kf = sum(1 for o in out if o[k] and not o["real"])
        lines.append(f"  {name}: keeps {kr} of {n_vis} real vulvas ({kr / max(n_vis, 1):.0%}) and {kf} false "
                     f"(precision {kr / max(kr + kf, 1):.3f})")
        sl = size_line(np.array([o["hand_size_px"] for o in out if o[k] and o["real"]]), all_sizes)
        if sl:
            lines.append("    " + sl)
    print("\n".join(lines))
    od = args.out_dir or os.path.join("outputs", "checkcnn_" + os.path.splitext(os.path.basename(args.check))[0]
                                      + (f"_{args.split}" if args.split else ""))
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, "check_frames.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0]))
        wr.writeheader()
        wr.writerows(out)
    with open(os.path.join(od, "check_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    extra = ""
    if args.sheet and save_tiles([t for _, t in sorted(errors, key=lambda e: e[0])][:48],
                                 os.path.join(od, "check_mistakes.jpg")):
        extra = " (check_mistakes.jpg: kept but not real, and real but dropped; worst first)"
    print(f"-> {od}{extra}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("fit", help="train the crop check on a finder run's held-out candidates")
    a.add_argument("--run", required=True, help="the finder's --out-dir (with jobs/fold*.npz)")
    a.add_argument("--labels", required=True, help="the dataset table")
    a.add_argument("--split", default="")
    a.add_argument("--frames-root", default="")
    a.add_argument("--out", required=True, help="check model (.pth)")
    a.add_argument("--channels", default=",".join(CHANNELS))
    a.add_argument("--precision", type=float, default=0.95, help="target precision of kept candidates on CV")
    a.add_argument("--iou", type=float, default=0.2, help="overlap with the hand outline that counts as real")
    a.add_argument("--threshold", type=float, default=0.5)
    a.add_argument("--min-area", type=int, default=25)
    a.add_argument("--epochs", type=int, default=30)
    a.add_argument("--lr", type=float, default=3e-4)
    a.add_argument("--batch-size", type=int, default=32)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--no-pretrained", action="store_true", help="start from random weights (tests)")
    b = sub.add_parser("eval", help="finder + crop check on a labelled (test) split")
    b.add_argument("--model", required=True, help="the finder checkpoint")
    b.add_argument("--check", required=True)
    b.add_argument("--labels", required=True)
    b.add_argument("--split", default="")
    b.add_argument("--frames-root", default="")
    b.add_argument("--out-dir", default="")
    b.add_argument("--no-sheet", dest="sheet", action="store_false", help="skip the sheet of mistakes")
    args = ap.parse_args()
    cmd_fit(args) if args.cmd == "fit" else cmd_eval(args)


if __name__ == "__main__":
    main()
