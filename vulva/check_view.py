#!/usr/bin/env python3
r"""The check step: is the finder's candidate a real, fully visible vulva? (Plan E2.)

The finder's own confidence (mean probability in its blob) does not separate tails from
vulvas. This is a small, explainable model that does: logistic regression on measurable facts
about each candidate.

  conf, pmax        the finder's confidence and peak probability
  log_area          blob size
  elongation, fill  ellipse length / width, and how well the blob fills its ellipse
                    (a tail is long and thin; a vulva is round)
  tail_share        share of the blob's ellipse covered by the finder's tail class (3-class finders)
  tail_near         tail pixels in a ring just around the ellipse
  ir_contrast       ring minus blob brightness in IR, in units of the ring's spread
  dist_m, peak_mm   distance from the camera; how far the blob rises above the skin
  dome, ridge       share of blob pixels whose depth shape is a dome (vulva) or a ridge (tail)
  cx, cy            where the blob is in the frame
  no_depth          1 when the frame has no depth file (depth features neutral)

fit: learns from a finder run's held-out predictions (<run>/jobs/fold*.npz, written by
train_seg.py --jobs), so every candidate comes from a model that never saw that sow. A
candidate counts as real when the frame is labelled visible and the blob overlaps the hand
outline (IoU >= 0.2). The check is cross-validated with the same sow folds, and the cut is the
lowest score that keeps precision at the target on those held-out scores; then it is refitted on
all candidates and saved as JSON.

    python vulva\check_view.py fit --run outputs\vulva_seg_r10 --labels C:\Users\aakdbx\vulva_dataset_v1\labels.csv --split train --out models\check_r10.json

eval: finder + check on a test split, against the finder's own keep rule.

    python vulva\check_view.py eval --model models\vulva_seg_r10.pth --check models\check_r10.json --labels C:\Users\aakdbx\vulva_dataset_v1\labels.csv --split test_newdays
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seg_model import (blob_confidence, depth_features, parts_from_probs,  # noqa: E402
                       read_ir, read_mask, to_full_frame)

FEATURES = ["conf", "pmax", "log_area", "elongation", "fill", "tail_share", "tail_near", "ir_contrast",
            "dist_m", "peak_mm", "dome", "ridge", "cx", "cy", "no_depth"]
NAME = re.compile(r"pig(\d+)_ir_vis_(.+)\.(jpg|png)$", re.IGNORECASE)


def frame_key(path):
    m = NAME.search(os.path.basename(str(path).replace("\\", "/")))
    return (int(m.group(1)), m.group(2)) if m else None


def candidate_features(vul, tail, conf, pmax, ir, depth):
    """vul, tail: full-frame masks (tail may be None); ir: grey 480 x 640; depth: mm or None."""
    vul = vul.astype(bool)
    f = {"conf": float(conf), "pmax": float(pmax), "log_area": float(np.log(max(vul.sum(), 1)))}
    cnts, _ = cv2.findContours(vul.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    ell = np.zeros(vul.shape, np.uint8)
    if len(c) >= 5:
        (ex, ey), (w, h), ang = cv2.fitEllipse(c)
        cv2.ellipse(ell, ((ex, ey), (w, h), ang), 1, -1)
        a, b = max(w, h), max(min(w, h), 1e-3)
        f["elongation"] = float(a / b)
        f["fill"] = float(vul.sum() / max(np.pi * w * h / 4, 1e-3))
    else:
        cv2.drawContours(ell, [cv2.convexHull(c)], -1, 1, -1)
        f["elongation"], f["fill"] = 1.0, 1.0
    ell = ell.astype(bool)
    ring = cv2.dilate(ell.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool) & ~ell
    t = tail.astype(bool) if tail is not None else np.zeros_like(vul)
    f["tail_share"] = float((ell & t).sum() / max(ell.sum(), 1))
    f["tail_near"] = float((ring & t).sum() / max(ring.sum(), 1))
    irf = ir.astype(np.float32)
    f["ir_contrast"] = float((irf[ring].mean() - irf[vul].mean()) / (irf[ring].std() + 1.0)) if ring.any() else 0.0
    ys, xs = np.nonzero(vul)
    f["cx"], f["cy"] = float(xs.mean() / vul.shape[1]), float(ys.mean() / vul.shape[0])
    if depth is None:
        f.update(dist_m=np.nan, peak_mm=np.nan, dome=np.nan, ridge=np.nan, no_depth=1.0)
        return f
    h8, s8, _ = depth_features(depth)
    height = h8.astype(np.float32) / 255 * 40 - 20
    si = s8.astype(np.float32) / 255 * 2 - 1
    d = depth[vul].astype(np.float32)
    d = d[(d > 300) & (d < 3000)]
    f["dist_m"] = float(np.median(d) / 1000) if d.size else np.nan
    f["peak_mm"] = float(np.percentile(height[vul], 95))
    f["dome"] = float((si[vul] > 0.7).mean())
    f["ridge"] = float(((si[vul] > 0.3) & (si[vul] <= 0.7)).mean())
    f["no_depth"] = 0.0
    return f


# ------------------------------------------------------------------ logistic regression (numpy)

def standardise(X, mean=None, std=None):
    if mean is None:
        mean = np.nanmean(X, 0)
        std = np.nanstd(X, 0)
    mean, std = np.nan_to_num(mean), np.where(np.nan_to_num(std) < 1e-6, 1.0, np.nan_to_num(std))
    Z = (X - mean) / std
    return np.nan_to_num(Z), mean, std            # missing values -> the mean (0 after scaling)


def fit_lr(Z, y, lam=1.0, iters=60):
    """L2 logistic regression by Newton steps -> weights (intercept first)."""
    Xb = np.hstack([np.ones((len(Z), 1)), Z])
    w = np.zeros(Xb.shape[1])
    R = lam * np.eye(Xb.shape[1])
    R[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))
        g = Xb.T @ (p - y) + R @ w
        H = (Xb * (p * (1 - p) + 1e-6)[:, None]).T @ Xb + R
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    return w


def score_lr(Z, w):
    return 1 / (1 + np.exp(-np.clip(np.hstack([np.ones((len(Z), 1)), Z]) @ w, -30, 30)))


def pick_cut(scores, y, precision):
    """Lowest cut whose kept set reaches the precision target (most real vulvas kept). The cut sits on
    a real vulva's score, so no false candidate below the last kept real one is let in."""
    order = np.argsort(-scores)
    s, t = scores[order], y[order]
    tp = np.cumsum(t)
    prec = tp / np.arange(1, len(t) + 1)
    ok = np.nonzero((prec >= precision) & (t == 1))[0]
    if not len(ok):
        return float(s[0]) + 1e-9
    k = ok.max()
    return float(s[k])


def pr_at(scores, y, cut):
    kept = scores >= cut
    tp, fp = int((kept & (y == 1)).sum()), int((kept & (y == 0)).sum())
    return tp, fp, int((y == 1).sum()), int((y == 0).sum())


def size_line(kept_sizes, all_sizes):
    """Are the kept real vulvas bigger or smaller than all visible ones? Sizes: sqrt of the hand
    outline's area in pixels. A check that prefers big blobs would bias the size measurement
    (and, keeping few frames on days the vulva is small, flatten the rise at heat)."""
    k, a = np.asarray(kept_sizes, float), np.asarray(all_sizes, float)
    k, a = k[k > 0], a[a > 0]
    if not len(k) or not len(a):
        return ""
    mk, ma = float(np.median(k)), float(np.median(a))
    return (f"size of the kept real vulvas (hand outline, sqrt of area): median {mk:.1f} px vs {ma:.1f} px "
            f"for all visible (x{mk / ma:.2f})")


# ------------------------------------------------------------------ fit

def load_candidates(run_dir, rows, threshold=0.5, min_area=25, iou_real=0.2):
    """Held-out probabilities of a finder run -> (feature dicts, labels, folds, keys)."""
    from train_seg import full_probs
    by_key = {frame_key(r["ir_path"]): r for r in rows}
    feats, ys, folds, keys = [], [], [], []
    files = sorted(glob.glob(os.path.join(run_dir, "jobs", "fold*.npz")))
    if not files:
        sys.exit(f"No held-out predictions in {run_dir}\\jobs (train the finder with --jobs)")
    for fpath in files:
        k = int(re.search(r"fold(\d+)", os.path.basename(fpath)).group(1))
        d = np.load(fpath, allow_pickle=True)
        for key, pr in zip(d["keys"], d["probs"]):
            r = by_key.get(frame_key(key))
            if r is None:
                continue
            vul, pv, tail, _ = parts_from_probs(full_probs(pr), threshold, min_area)
            if not vul.any():
                continue                                   # nothing to check
            vf, tf = to_full_frame(vul) > 0, (to_full_frame(tail) > 0 if tail is not None else None)
            ir = read_ir(r["ir_path"])
            if ir is None:
                continue
            from measure3d import load_depth_raw
            depth = load_depth_raw(r.get("depth_path", ""))
            f = candidate_features(vf, tf, blob_confidence(pv, vul), pv.max(), ir, depth)
            real = 0
            if r["label"] == "present":
                hm = read_mask(r["mask_path"])
                if hm is not None and hm.any():
                    hm = hm > 0
                    real = int((vf & hm).sum() / max((vf | hm).sum(), 1) >= iou_real)
            feats.append(f)
            ys.append(real)
            folds.append(k)
            keys.append(r["ir_path"])
    return feats, np.array(ys), np.array(folds), keys


def cmd_fit(args):
    from eval_seg import read_rows
    from train_seg import require_split, split_set
    require_split(args.labels, args.split)
    rows = read_rows(args.labels, [p for p in args.frames_root.split(";") if p.strip()], split_set(args.split))
    feats, y, folds, _ = load_candidates(args.run, rows, args.threshold, args.min_area, args.iou)
    if len(set(y.tolist())) < 2:
        sys.exit(f"Need both real and false candidates; got {len(y)} candidates, {int(y.sum())} real")
    X = np.array([[f[n] for n in FEATURES] for f in feats], float)
    oof = np.zeros(len(y))
    for k in sorted(set(folds.tolist())):
        tr, te = folds != k, folds == k
        Z, m, s = standardise(X[tr])
        w = fit_lr(Z, y[tr], args.lam)
        oof[te] = score_lr(standardise(X[te], m, s)[0], w)
    cut = pick_cut(oof, y, args.precision)
    tp, fp, n_real, n_false = pr_at(oof, y, cut)
    Z, m, s = standardise(X)
    w = fit_lr(Z, y, args.lam)
    model = {"features": FEATURES, "mean": m.tolist(), "std": s.tolist(), "weights": w.tolist(), "cut": cut,
             "precision_target": args.precision, "iou_real": args.iou, "threshold": args.threshold,
             "min_area": args.min_area, "run": os.path.abspath(args.run), "candidates": len(y),
             "cv": {"real": n_real, "false": n_false, "kept_real": tp, "kept_false": fp}}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(model, f, indent=2)
    lines = [f"{len(y)} held-out candidates from {args.run}: {n_real} real vulvas, {n_false} false "
             f"(tails, covered or wrong spot)",
             f"check, cross-validated by the same sow folds, cut {cut:.3f} (precision target {args.precision}): "
             f"keeps {tp} of {n_real} real ({tp / max(n_real, 1):.0%}) and {fp} of {n_false} false",
             "weights (on standardised features; + = more likely a real vulva): "
             + ", ".join(f"{n} {wv:+.2f}" for n, wv in sorted(zip(FEATURES, w[1:]), key=lambda t: -abs(t[1])))]
    print("\n".join(lines))
    print(f"-> {args.out}")


# ------------------------------------------------------------------ eval

def cmd_eval(args):
    import torch
    from eval_seg import read_rows
    from measure3d import load_depth_raw
    from seg_model import load_checkpoint, predict_parts, read_input
    from train_seg import require_split, split_set
    require_split(args.labels, args.split)
    with open(args.check) as f:
        chk = json.load(f)
    rows = read_rows(args.labels, [p for p in args.frames_root.split(";") if p.strip()], split_set(args.split))
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model, ckpt = load_checkpoint(args.model, device)
    inputs = ckpt.get("inputs", "ir")
    thr = chk.get("threshold", ckpt.get("threshold", 0.5))
    presence = float(ckpt.get("presence_threshold", 0.0) or 0.0)
    m, s, w = np.array(chk["mean"]), np.array(chk["std"]), np.array(chk["weights"])
    out = []
    for i in range(0, len(rows), 16):
        chunk = rows[i:i + 16]
        imgs = [read_input(r["ir_path"], r.get("depth_path") or None, inputs) for r in chunk]
        for r, d in zip(chunk, predict_parts(model, imgs, device, thr, chk.get("min_area", 25))):
            vul = d["vulva"] > 0
            hm = read_mask(r["mask_path"]) if r["label"] == "present" else None
            iou = float((vul & (hm > 0)).sum() / max((vul | (hm > 0)).sum(), 1)) if hm is not None and vul.any() else 0.0
            real = r["label"] == "present" and iou >= chk.get("iou_real", 0.2)
            score, keep_rule = 0.0, False
            if vul.any():
                ir = read_ir(r["ir_path"])
                depth = load_depth_raw(r.get("depth_path", ""))
                tail = d["tail"] > 0 if d["tail"] is not None else None
                f = candidate_features(vul, tail, d["conf"], d["pmax"], ir, depth)
                X = np.array([[f[n] for n in chk["features"]]], float)
                score = float(score_lr(standardise(X, m, s)[0], w)[0])
                keep_rule = (not d["covered_by_tail"]) if tail is not None else d["conf"] >= presence
            size = float(np.sqrt((hm > 0).sum())) if hm is not None else 0.0
            out.append({"ir_path": r["ir_path"], "label": r["label"], "candidate": int(vul.any()), "iou": round(iou, 3),
                        "real": int(real), "check_score": round(score, 4), "check_keep": int(vul.any() and score >= chk["cut"]),
                        "rule_keep": int(vul.any() and keep_rule), "hand_size_px": round(size, 1)})
    n_vis = sum(1 for o in out if o["label"] == "present")
    n_not = len(out) - n_vis
    lines = [f"{args.model} + check {args.check} on {args.labels} (split {args.split}): {len(out)} frames, "
             f"{n_vis} with a visible vulva, {n_not} without"]
    all_sizes = [o["hand_size_px"] for o in out if o["label"] == "present"]
    for name, k in (("finder's own rule", "rule_keep"), ("with the check", "check_keep")):
        kr = sum(1 for o in out if o[k] and o["real"])
        kf = sum(1 for o in out if o[k] and not o["real"])
        lines.append(f"  {name}: keeps {kr} of {n_vis} real vulvas ({kr / max(n_vis, 1):.0%}) and {kf} false "
                     f"(precision {kr / max(kr + kf, 1):.3f})")
        sl = size_line([o["hand_size_px"] for o in out if o[k] and o["real"]], all_sizes)
        if sl:
            lines.append("    " + sl)
    print("\n".join(lines))
    od = args.out_dir or os.path.join("outputs", "check_" + os.path.splitext(os.path.basename(args.model))[0]
                                      + (f"_{args.split}" if args.split else ""))
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, "check_frames.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0]))
        wr.writeheader()
        wr.writerows(out)
    with open(os.path.join(od, "check_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"-> {od}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("fit", help="learn the check from a finder run's held-out predictions")
    a.add_argument("--run", required=True, help="the finder's --out-dir (with jobs/fold*.npz)")
    a.add_argument("--labels", nargs="+", required=True)
    a.add_argument("--split", default="")
    a.add_argument("--frames-root", default="")
    a.add_argument("--out", required=True, help="check model, JSON")
    a.add_argument("--precision", type=float, default=0.95, help="target precision of kept frames on CV")
    a.add_argument("--iou", type=float, default=0.2, help="overlap with the hand outline that counts as real")
    a.add_argument("--threshold", type=float, default=0.5)
    a.add_argument("--min-area", type=int, default=25)
    a.add_argument("--lam", type=float, default=1.0, help="L2 strength")
    b = sub.add_parser("eval", help="finder + check on a labelled (test) split")
    b.add_argument("--model", required=True)
    b.add_argument("--check", required=True)
    b.add_argument("--labels", required=True)
    b.add_argument("--split", default="")
    b.add_argument("--frames-root", default="")
    b.add_argument("--out-dir", default="")
    args = ap.parse_args()
    if args.cmd == "fit":
        if len(args.labels) > 1:
            sys.exit("give one labels CSV (the dataset table) for fit")
        args.labels = args.labels[0]
        cmd_fit(args)
    else:
        cmd_eval(args)


if __name__ == "__main__":
    main()
