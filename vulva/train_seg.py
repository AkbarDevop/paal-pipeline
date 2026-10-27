#!/usr/bin/env python3
r"""Train the vulva segmentation U-Net, with cross-validation grouped by sow.

1. K-fold CV where every sow's frames sit in one fold only. The out-of-fold
   predictions give honest numbers on sows the model never saw:
     - presence (is a vulva visible?): precision, recall, F1, confusion matrix
     - mask quality on frames where both agree it is visible: IoU and Dice
     - mean |predicted - labelled| mask area
   plus a confidence cut for "vulva visible" (mean probability inside the blob) set so
   out-of-fold precision reaches --presence-precision: a frame without a visible vulva
   that is called visible adds a wrong size to that sow's day, while a few dropped real
   frames cost little (each sow has many standing frames a day).
2. A final model trained on all labelled frames -> models/vulva_seg.pth.

Usage:
    python vulva/train_seg.py --labels labels/vulva_seg_labels.csv
    python vulva/train_seg.py --labels labels/vulva_seg_labels.csv --folds 5 --epochs 60
    python vulva/train_seg.py --labels <label set>/labels.csv --inputs ir+depth   # IR + height + shading
    # lab PC: --jobs 6 trains the 5 folds and the final model at the same time (split over the GPUs);
    # --copy-to also puts the model, report and per-frame CV file on the Mac share when done
    python vulva\train_seg.py --jobs 6 --labels \\tsclient\paal_exchange\vulva_label_set\labels_r4.csv --out-model models\vulva_seg_r5.pth --out-dir outputs\vulva_seg_r5 --copy-to \\tsclient\paal_exchange\vulva_label_set
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seg_model import (INPUTS, SEG_H, SEG_W, VulvaUNet, blob_confidence, largest_component,  # noqa: E402
                       read_input, read_mask, save_checkpoint, to_model_space, to_tensor)

MIN_AREA = 25    # model pixels; smaller blobs count as "no vulva"


def load_labels(path, inputs="ir"):
    """Label CSV from make_seg_labels.py or label_ellipse.py. Relative paths are
    resolved against the CSV's folder, so a label set can move between computers.
    With inputs="ir+depth" the depth file is taken from a depth_path column or,
    without one, from the IR file name (pigN_depth_<ts>.raw next to the IR)."""
    base = os.path.dirname(os.path.abspath(path))
    res = lambda p: p if (not p or os.path.isabs(p) or p.startswith("\\\\")) else os.path.join(base, p)  # noqa: E731
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["label"] in ("present", "absent")]
    items, unreadable = [], []
    for r in rows:
        r["ir_path"], r["mask_path"] = res(r["ir_path"]), res(r.get("mask_path", ""))
        img = None
        for attempt in range(3):          # network shares (\\tsclient) sometimes fail a read under load
            img = read_input(r["ir_path"], res(r.get("depth_path", "")) or None, inputs)
            if img is not None or not os.path.exists(r["ir_path"]):
                break
            time.sleep(0.5 * (attempt + 1))
        if img is None:
            unreadable.append(r["ir_path"])
            continue
        if r["label"] == "present":
            m = read_mask(r["mask_path"])
            if m is None or m.sum() == 0:
                continue
        else:
            m = np.zeros(img.shape[:2], np.uint8)
        g, mk = to_model_space(img, m)
        items.append({"image": g, "mask": mk, "pig_id": int(float(r["pig_id"])), "row": r})
    if unreadable:
        print(f"WARNING: {len(unreadable)} labelled frames could not be read and are left out, "
              f"e.g. {unreadable[0]}", flush=True)
    return items


def save_items(items, path):
    """Loaded frames -> one .npz, so --jobs workers train on exactly these without re-reading the share."""
    np.savez(path, images=np.stack([it["image"] for it in items]), masks=np.stack([it["mask"] for it in items]),
             pigs=np.array([it["pig_id"] for it in items]), keys=np.array([it["row"]["ir_path"] for it in items]))


def load_items(path):
    d = np.load(path)
    images, masks, pigs, keys = d["images"], d["masks"], d["pigs"], d["keys"]
    return [{"image": images[i], "mask": masks[i], "pig_id": int(pigs[i]), "row": {"ir_path": str(keys[i])}}
            for i in range(len(keys))]


def assign_folds(pigs, k, seed=42):
    pigs = sorted(set(pigs))
    rng = random.Random(seed)
    rng.shuffle(pigs)
    return {p: i % k for i, p in enumerate(pigs)}


class SegData(Dataset):
    def __init__(self, items, augment):
        self.items, self.augment = items, augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        g, m = self.items[i]["image"].copy(), self.items[i]["mask"].copy()
        if self.augment:
            if random.random() < 0.5:
                g, m = g[:, ::-1].copy(), m[:, ::-1].copy()
            ang = random.uniform(-8, 8)
            sc = random.uniform(0.9, 1.1)
            tx, ty = random.uniform(-0.05, 0.05) * SEG_W, random.uniform(-0.05, 0.05) * SEG_H
            A = cv2.getRotationMatrix2D((SEG_W / 2, SEG_H / 2), ang, sc)
            A[:, 2] += (tx, ty)
            g = cv2.warpAffine(g, A, (SEG_W, SEG_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            m = cv2.warpAffine(m, A, (SEG_W, SEG_H), flags=cv2.INTER_NEAREST, borderValue=0)
            a, b = random.uniform(0.7, 1.3), random.uniform(-25, 25)
            if g.ndim == 2:
                g = np.clip(g.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
            else:                                  # brightness/contrast only on the IR channel
                g[..., 0] = np.clip(g[..., 0].astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        return to_tensor(g), torch.from_numpy(m.astype(np.float32))[None]


def loss_fn(logits, target, pos_weight=1.0):
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.tensor(pos_weight, device=logits.device))
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    denom = p.sum((1, 2, 3)) + target.sum((1, 2, 3))
    has = (target.sum((1, 2, 3)) > 0).float()
    dice = 1 - (2 * inter + 1) / (denom + 1)
    dice = (dice * has).sum() / has.sum().clamp(min=1)
    return bce + dice


def train_model(train_items, epochs, device, pretrained=True, lr=3e-4, batch_size=8, log=None, pos_weight=1.0):
    model = VulvaUNet(pretrained=pretrained).to(device)
    pos = [it for it in train_items if it["mask"].any()]
    neg = [it for it in train_items if not it["mask"].any()]
    # balance: present frames are the minority
    weights = [1.0 / max(len(pos), 1) if it["mask"].any() else 1.0 / max(len(neg), 1) for it in train_items]
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(train_items), replacement=True)
    dl = DataLoader(SegData(train_items, augment=True), batch_size=batch_size, sampler=sampler, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(1, epochs * len(dl))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y, pos_weight)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(x)
        if log and (ep % 10 == 9 or ep == epochs - 1):
            log(f"    epoch {ep + 1}/{epochs}  loss {tot / len(dl.dataset):.4f}")
    return model


@torch.no_grad()
def predict_probs(model, items, device):
    model.eval()
    out = []
    for i in range(0, len(items), 16):
        x = torch.stack([to_tensor(it["image"]) for it in items[i:i + 16]]).to(device)
        out.extend(torch.sigmoid(model(x))[:, 0].cpu().numpy().astype(np.float16))
    return out


def masks_from_probs(probs, threshold=0.5):
    out = []
    for p in probs:
        m = largest_component((p.astype(np.float32) >= threshold).astype(np.uint8))
        if m.sum() < MIN_AREA:
            m[:] = 0
        out.append(m)
    return out


def predict_items(model, items, device, threshold=0.5):
    return masks_from_probs(predict_probs(model, items, device), threshold)


def pick_threshold(items, probs, grid=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)):
    """Threshold with the best out-of-fold IoU while presence F1 stays within 0.02 of its best.
    Over- or under-sized masks bias width and length, so this is tuned on held-out sows."""
    scored = [(t, score(items, masks_from_probs(probs, t))) for t in grid]
    best_f1 = max(s["presence_f1"] for _, s in scored)
    ok = [(t, s) for t, s in scored if s["presence_f1"] >= best_f1 - 0.02 and s["iou_mean"] is not None]
    t, s = max(ok, key=lambda ts: ts[1]["iou_mean"]) if ok else (0.5, dict(scored)[0.5])
    return t, s, scored


def pick_presence_cut(items, masks, confs, target=0.9):
    """Lowest blob-confidence cut whose out-of-fold precision on "vulva visible" reaches target
    (the lowest such cut keeps the most real frames). 0.0 = no cut; None = target not reachable."""
    if target <= 0:
        return 0.0
    truth = [bool(it["mask"].any()) for it in items]
    called = [bool(m.any()) for m in masks]
    for c in sorted({round(float(x), 4) for x, v in zip(confs, called) if v}):
        tp = sum(1 for t, v, x in zip(truth, called, confs) if t and v and x >= c)
        fp = sum(1 for t, v, x in zip(truth, called, confs) if not t and v and x >= c)
        if tp + fp and tp / (tp + fp) >= target:
            return c
    return None


def score(items, preds):
    tp = fp = fn = tn = 0
    ious, dices, area_err = [], [], []
    for it, pm in zip(items, preds):
        t, p = it["mask"].astype(bool), pm.astype(bool)
        tv, pv = t.any(), p.any()
        if tv and pv:
            tp += 1
            inter, union = (t & p).sum(), (t | p).sum()
            ious.append(inter / union)
            dices.append(2 * inter / (t.sum() + p.sum()))
            area_err.append(abs(int(p.sum()) - int(t.sum())) / t.sum())
        elif pv:
            fp += 1
        elif tv:
            fn += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"frames": len(items), "present": tp + fn, "absent": tn + fp,
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "presence_precision": round(prec, 3), "presence_recall": round(rec, 3), "presence_f1": round(f1, 3),
            "presence_accuracy": round((tp + tn) / max(len(items), 1), 3),
            "iou_mean": round(float(np.mean(ious)), 3) if ious else None,
            "dice_mean": round(float(np.mean(dices)), 3) if dices else None,
            "area_error_mean": round(float(np.mean(area_err)), 3) if area_err else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default=os.path.join("labels", "vulva_seg_labels.csv"))
    ap.add_argument("--folds", type=int, default=5, help="0 = skip cross-validation")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pos-weight", type=float, default=1.0,
                    help="BCE weight on vulva pixels; above 1 grows masks (Dice already handles imbalance)")
    ap.add_argument("--inputs", choices=INPUTS, default="ir",
                    help='"ir" or "ir+depth" (IR, height above the skin and 3D shading as the 3 channels)')
    ap.add_argument("--presence-precision", type=float, default=0.9,
                    help="tune the 'vulva visible' confidence cut to this out-of-fold precision (0 = no cut)")
    ap.add_argument("--no-pretrained", action="store_true", help="random init (tests / offline)")
    ap.add_argument("--out-model", default=os.path.join("models", "vulva_seg.pth"))
    ap.add_argument("--out-dir", default=os.path.join("outputs", "vulva_seg"))
    ap.add_argument("--copy-to", default="",
                    help="also copy <model>.pth, <model>_report.txt and <model>_cv.csv here when done "
                         "(e.g. \\\\tsclient\\paal_exchange\\vulva_label_set; PowerShell copy is refused there)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="train the CV folds and the final model in this many processes at once "
                         "(GPUs shared round-robin; 6 = all at once for 5 folds; lab PC only, at most 2 without CUDA)")
    ap.add_argument("--job", type=int, default=None, help=argparse.SUPPRESS)       # worker: fold k, or folds = final
    ap.add_argument("--job-dir", default="", help=argparse.SUPPRESS)
    ap.add_argument("--items-cache", default="", help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.job is not None:
        return run_job(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    log_lines = []

    def log(s):
        print(s, flush=True)
        log_lines.append(s)

    t0 = time.time()
    items = load_labels(args.labels, args.inputs)
    pigs = sorted({it["pig_id"] for it in items})
    n_pos = int(sum(bool(it["mask"].any()) for it in items))
    log(f"{len(items)} frames from {len(pigs)} sows: {n_pos} with a vulva mask, {len(items) - n_pos} without"
        f"  (inputs: {args.inputs})")
    if n_pos == 0:
        sys.exit("No vulva masks found; check --labels and the mask paths.")

    report = {"labels": args.labels, "frames": len(items), "present": n_pos, "sows": pigs,
              "epochs": args.epochs, "device": device, "inputs": args.inputs}
    if args.folds and len(pigs) >= args.folds:
        folds = assign_folds([it["pig_id"] for it in items], args.folds, args.seed)
        report["folds"] = {str(k): sorted(p for p, f in folds.items() if f == k) for k in range(args.folds)}
        if args.jobs > 2 and not torch.cuda.is_available():
            # each worker holds its own copy of the frames and the model: a laptop runs out of memory and disk (swap)
            log(f"--jobs {args.jobs} needs CUDA GPUs; using 2 at a time here")
            args.jobs = 2
        if args.jobs > 1:
            oof_probs, per_fold, final_state = run_parallel(args, items, folds, report, log)
        else:
            oof_probs, per_fold = [None] * len(items), []
            for k in range(args.folds):
                tr = [it for it in items if folds[it["pig_id"]] != k]
                te_idx = [i for i, it in enumerate(items) if folds[it["pig_id"]] == k]
                log(f"Fold {k + 1}/{args.folds}: test sows {report['folds'][str(k)]} "
                    f"({len(te_idx)} frames, train {len(tr)})")
                model = train_model(tr, args.epochs, device, not args.no_pretrained, args.lr, args.batch_size, log,
                                    args.pos_weight)
                probs = predict_probs(model, [items[i] for i in te_idx], device)
                for i, pr in zip(te_idx, probs):
                    oof_probs[i] = pr
                s = score([items[i] for i in te_idx], masks_from_probs(probs, 0.5))
                per_fold.append(s)
                log(f"    presence F1 {s['presence_f1']}  IoU {s['iou_mean']}  Dice {s['dice_mean']}")
        threshold, presence = finish_cv(args, items, folds, oof_probs, per_fold, report, log)
    else:
        threshold, presence = 0.5, 0.0
    log("")
    log(f"Final model on all {len(items)} frames ...")
    if args.jobs > 1 and args.folds and len(pigs) >= args.folds:
        model = VulvaUNet(pretrained=False)
        model.load_state_dict(final_state)
    else:
        model = train_model(items, args.epochs, device, not args.no_pretrained, args.lr, args.batch_size, log,
                            args.pos_weight)
    save_checkpoint(model, args.out_model, {"min_area": MIN_AREA, "threshold": threshold, "inputs": args.inputs,
                                            "presence_threshold": presence,
                                            "trained_on": len(items), "sows": pigs,
                                            "date": time.strftime("%Y-%m-%d")})
    log(f"Saved {args.out_model}  ({time.time() - t0:.0f} s)")
    report["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(args.out_dir, "seg_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(args.out_dir, "seg_report.txt"), "w") as f:
        f.write("\n".join(log_lines) + "\n")
    if args.copy_to:
        copy_results(args.out_model, args.out_dir, args.copy_to)



def run_job(args):
    """Worker for --jobs: train CV fold args.job (or, when args.job == args.folds, the final model
    on all frames) and save the held-out probabilities / the weights into args.job_dir."""
    k = args.job
    seed = args.seed + 1 + k
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    log = lambda s: print(s, flush=True)  # noqa: E731
    items = load_items(args.items_cache) if args.items_cache else load_labels(args.labels, args.inputs)
    if k == args.folds:
        log(f"Final model on all {len(items)} frames ({device})")
        model = train_model(items, args.epochs, device, not args.no_pretrained, args.lr, args.batch_size, log,
                            args.pos_weight)
        torch.save(model.state_dict(), os.path.join(args.job_dir, "final.pt"))
        return
    folds = assign_folds([it["pig_id"] for it in items], args.folds, args.seed)
    tr = [it for it in items if folds[it["pig_id"]] != k]
    te = [it for it in items if folds[it["pig_id"]] == k]
    log(f"Fold {k + 1}/{args.folds}: {len(te)} test frames, train {len(tr)} ({device})")
    model = train_model(tr, args.epochs, device, not args.no_pretrained, args.lr, args.batch_size, log,
                        args.pos_weight)
    probs = predict_probs(model, te, device)
    np.savez(os.path.join(args.job_dir, f"fold{k}.npz"), keys=np.array([it["row"]["ir_path"] for it in te]),
             probs=np.stack(probs))


def run_parallel(args, items, folds, report, log):
    """--jobs N: the CV folds and the final model are independent, so train them as separate
    processes, N at a time, each on the next GPU. Progress is printed as one line per change;
    each worker's own log is in <out-dir>/jobs/."""
    job_dir = os.path.join(args.out_dir, "jobs")
    os.makedirs(job_dir, exist_ok=True)
    for f in os.listdir(job_dir):
        if f.endswith((".npz", ".pt", ".log")):
            os.remove(os.path.join(job_dir, f))
    cmd = [sys.executable, os.path.abspath(__file__), "--labels", args.labels, "--folds", str(args.folds),
           "--epochs", str(args.epochs), "--batch-size", str(args.batch_size), "--lr", str(args.lr),
           "--pos-weight", str(args.pos_weight), "--inputs", args.inputs, "--seed", str(args.seed),
           "--job-dir", job_dir] + (["--no-pretrained"] if args.no_pretrained else [])
    cache = os.path.join(job_dir, "items.npz")    # the frames as loaded here: workers do not touch the share
    save_items(items, cache)
    cmd += ["--items-cache", cache]
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    names = {k: f"fold{k + 1}" for k in range(args.folds)}
    names[args.folds] = "final"
    todo, running, state = list(range(args.folds + 1)), {}, {}
    log(f"Training {len(todo)} models ({args.folds} folds + final), {min(args.jobs, len(todo))} at a time"
        f"{f' on {n_gpu} GPUs' if n_gpu else ''}; worker logs in {job_dir}")
    t0, last = time.time(), ""
    while todo or running:
        while todo and len(running) < args.jobs:
            k = todo.pop(0)
            env = dict(os.environ, PYTHONUNBUFFERED="1")
            if n_gpu:
                env["CUDA_VISIBLE_DEVICES"] = str(k % n_gpu)
            lf = open(os.path.join(job_dir, f"{names[k]}.log"), "w")
            running[k] = (subprocess.Popen(cmd + ["--job", str(k)], stdout=lf, stderr=subprocess.STDOUT, env=env), lf)
            state[k] = "loading"
        time.sleep(2)
        for k, (proc, lf) in list(running.items()):
            with open(os.path.join(job_dir, f"{names[k]}.log"), errors="replace") as f:
                ep = [ln.split()[1] for ln in f if ln.strip().startswith("epoch ")]
            if ep:
                state[k] = ep[-1]
            if proc.poll() is not None:
                lf.close()
                del running[k]
                state[k] = "done" if proc.returncode == 0 else "FAILED"
        line = "  ".join(f"{names[k]} {state[k]}" for k in sorted(state))
        if line != last:
            print(f"  {(time.time() - t0) / 60:5.1f} min  {line}", flush=True)
            last = line
    bad = [k for k in state if state[k] == "FAILED"]
    if bad:
        for k in bad:
            with open(os.path.join(job_dir, f"{names[k]}.log"), errors="replace") as f:
                print(f"--- {names[k]} log (end) ---\n" + "".join(f.readlines()[-15:]))
        sys.exit(f"{len(bad)} training job(s) failed; see {job_dir}")
    log(f"All {args.folds + 1} models trained in {(time.time() - t0) / 60:.1f} min")

    where = {it["row"]["ir_path"]: i for i, it in enumerate(items)}
    oof_probs, per_fold = [None] * len(items), []
    for k in range(args.folds):
        d = np.load(os.path.join(job_dir, f"fold{k}.npz"))
        pairs = [(where[key], pr) for key, pr in zip(d["keys"], d["probs"]) if key in where]
        for i, pr in pairs:
            oof_probs[i] = pr
        s = score([items[i] for i, _ in pairs], masks_from_probs([pr for _, pr in pairs], 0.5))
        per_fold.append(s)
        log(f"Fold {k + 1}/{args.folds}: test sows {report['folds'][str(k)]} ({len(pairs)} frames)  "
            f"presence F1 {s['presence_f1']}  IoU {s['iou_mean']}  Dice {s['dice_mean']}")
    missing = [items[i]["row"]["ir_path"] for i, pr in enumerate(oof_probs) if pr is None]
    if missing:
        sys.exit(f"{len(missing)} frames have no held-out prediction (a worker could not read them), "
                 f"e.g. {missing[0]}; rerun, ideally from a local copy of the label set")
    return oof_probs, per_fold, torch.load(os.path.join(job_dir, "final.pt"), map_location="cpu")


def finish_cv(args, items, folds, oof_probs, per_fold, report, log):
    """Out-of-fold probabilities -> outline threshold, confidence cut, report and per-frame CSV."""
    threshold, overall, by_thr = pick_threshold(items, oof_probs)
    oof = masks_from_probs(oof_probs, threshold)
    report["cv"] = {"overall": overall, "per_fold_at_0.5": per_fold, "threshold": threshold,
                    "by_threshold": {str(t): {k: s[k] for k in ("presence_f1", "iou_mean", "area_error_mean")}
                                     for t, s in by_thr}}
    c = overall["confusion"]
    log("")
    log(f"Cross-validation (sows held out), {overall['frames']} frames, mask threshold {threshold}:")
    log(f"  Vulva visible: precision {overall['presence_precision']}, recall {overall['presence_recall']}, "
        f"F1 {overall['presence_f1']}, accuracy {overall['presence_accuracy']}")
    log(f"  Confusion (rows true present/absent): [[{c['tp']}, {c['fn']}], [{c['fp']}, {c['tn']}]]")
    log(f"  Mask on frames both call visible: IoU {overall['iou_mean']}, Dice {overall['dice_mean']}, "
        f"area error {overall['area_error_mean']}")
    peaks = [float(p.max()) for p in oof_probs]
    confs = [blob_confidence(p, m) for p, m in zip(oof_probs, oof)]
    presence = pick_presence_cut(items, oof, confs, args.presence_precision)
    if presence is None:
        log(f"  No confidence cut reaches precision {args.presence_precision}; using none")
        presence = 0.0
    kept = [m if c >= presence else np.zeros_like(m) for m, c in zip(oof, confs)]
    cut = score(items, kept)
    report["cv"]["presence_cut"] = {"score": "mean prob in blob", "threshold": presence,
                                    "target_precision": args.presence_precision, "overall": cut}
    c = cut["confusion"]
    spec = c["tn"] / max(c["tn"] + c["fp"], 1)
    log(f"With confidence cut {presence} (target precision {args.presence_precision}; used by run_vulva):")
    log(f"  Vulva visible: precision {cut['presence_precision']}, recall {cut['presence_recall']}, "
        f"specificity {spec:.3f}, F1 {cut['presence_f1']}")
    log(f"  Confusion (rows true present/absent): [[{c['tp']}, {c['fn']}], [{c['fp']}, {c['tn']}]]")
    log(f"  Mask on frames both call visible: IoU {cut['iou_mean']}, Dice {cut['dice_mean']}, "
        f"area error {cut['area_error_mean']}")
    with open(os.path.join(args.out_dir, "cv_predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        # per frame, so results can be split later (e.g. hand-drawn vs accepted model suggestions)
        # pred_area_px / iou are before the confidence cut; peak_prob and blob_conf let it be re-tuned
        w.writerow(["timestamp_folder", "pig_id", "pig_timestamp", "label", "how", "fold",
                    "pred_area_px", "true_area_px", "iou", "peak_prob", "blob_conf"])
        for it, p, pk, cf in zip(items, oof, peaks, confs):
            r = it["row"]
            t, q = it["mask"].astype(bool), p.astype(bool)
            iou = round(float((t & q).sum() / (t | q).sum()), 4) if (t.any() and q.any()) else ""
            w.writerow([r["timestamp_folder"], r["pig_id"], r["pig_timestamp"], r["label"], r.get("how", ""),
                        folds[it["pig_id"]], int(p.sum()), int(it["mask"].sum()), iou,
                        round(pk, 4), round(cf, 4)])
    return threshold, presence

def copy_results(model_path, out_dir, dest):
    """Plain read/write copy (Windows CopyFile, used by PowerShell copy, is refused on \\\\tsclient shares).
    A failed copy only prints a warning: the local files stay."""
    stem = os.path.splitext(os.path.basename(model_path))[0]
    for src, name in ((model_path, f"{stem}.pth"), (os.path.join(out_dir, "seg_report.txt"), f"{stem}_report.txt"),
                      (os.path.join(out_dir, "cv_predictions.csv"), f"{stem}_cv.csv")):
        if not os.path.exists(src):
            continue
        try:
            with open(src, "rb") as fi, open(os.path.join(dest, name), "wb") as fo:
                shutil.copyfileobj(fi, fo)
            print(f"Copied {name} to {dest}")
        except OSError as e:
            print(f"Could not copy {name} to {dest}: {e} (kept in {os.path.dirname(src) or '.'})")


if __name__ == "__main__":
    main()
