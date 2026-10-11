#!/usr/bin/env python3
"""Train the vulva segmentation U-Net, with cross-validation grouped by sow.

1. K-fold CV where every sow's frames sit in one fold only. The out-of-fold
   predictions give honest numbers on sows the model never saw:
     - presence (is a vulva visible?): precision, recall, F1, confusion matrix
     - mask quality on frames where both agree it is visible: IoU and Dice
     - mean |predicted - labelled| mask area
2. A final model trained on all labelled frames -> models/vulva_seg.pth.

Usage:
    python vulva/train_seg.py --labels labels/vulva_seg_labels.csv
    python vulva/train_seg.py --labels labels/vulva_seg_labels.csv --folds 5 --epochs 60
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seg_model import (SEG_H, SEG_W, VulvaUNet, largest_component, read_ir, read_mask,  # noqa: E402
                       save_checkpoint, to_model_space, to_tensor)

MIN_AREA = 25    # model pixels; smaller blobs count as "no vulva"


def load_labels(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["label"] in ("present", "absent")]
    items = []
    for r in rows:
        img = read_ir(r["ir_path"])
        if img is None:
            continue
        if r["label"] == "present":
            m = read_mask(r["mask_path"])
            if m is None or m.sum() == 0:
                continue
        else:
            m = np.zeros_like(img)
        g, mk = to_model_space(img, m)
        items.append({"gray": g, "mask": mk, "pig_id": int(float(r["pig_id"])), "row": r})
    return items


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
        g, m = self.items[i]["gray"].copy(), self.items[i]["mask"].copy()
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
            g = np.clip(g.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
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
        x = torch.stack([to_tensor(it["gray"]) for it in items[i:i + 16]]).to(device)
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
    ap.add_argument("--no-pretrained", action="store_true", help="random init (tests / offline)")
    ap.add_argument("--out-model", default=os.path.join("models", "vulva_seg.pth"))
    ap.add_argument("--out-dir", default=os.path.join("outputs", "vulva_seg"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

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
    items = load_labels(args.labels)
    pigs = sorted({it["pig_id"] for it in items})
    n_pos = int(sum(bool(it["mask"].any()) for it in items))
    log(f"{len(items)} frames from {len(pigs)} sows: {n_pos} with a vulva mask, {len(items) - n_pos} without")
    if n_pos == 0:
        sys.exit("No vulva masks found; check --labels and the mask paths.")

    report = {"labels": args.labels, "frames": len(items), "present": n_pos, "sows": pigs,
              "epochs": args.epochs, "device": device}
    if args.folds and len(pigs) >= args.folds:
        folds = assign_folds([it["pig_id"] for it in items], args.folds, args.seed)
        report["folds"] = {str(k): sorted(p for p, f in folds.items() if f == k) for k in range(args.folds)}
        oof = [None] * len(items)
        oof_probs = [None] * len(items)
        per_fold = []
        for k in range(args.folds):
            tr = [it for it in items if folds[it["pig_id"]] != k]
            te_idx = [i for i, it in enumerate(items) if folds[it["pig_id"]] == k]
            log(f"Fold {k + 1}/{args.folds}: test sows {report['folds'][str(k)]} "
                f"({len(te_idx)} frames, train {len(tr)})")
            model = train_model(tr, args.epochs, device, not args.no_pretrained, args.lr, args.batch_size, log, args.pos_weight)
            probs = predict_probs(model, [items[i] for i in te_idx], device)
            preds = masks_from_probs(probs, 0.5)
            for i, p, pr in zip(te_idx, preds, probs):
                oof[i] = p
                oof_probs[i] = pr
            s = score([items[i] for i in te_idx], preds)
            per_fold.append(s)
            log(f"    presence F1 {s['presence_f1']}  IoU {s['iou_mean']}  Dice {s['dice_mean']}")
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
        with open(os.path.join(args.out_dir, "cv_predictions.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp_folder", "pig_id", "pig_timestamp", "label", "pred_area_px", "true_area_px"])
            for it, p in zip(items, oof):
                r = it["row"]
                w.writerow([r["timestamp_folder"], r["pig_id"], r["pig_timestamp"], r["label"],
                            int(p.sum()), int(it["mask"].sum())])
    else:
        threshold = 0.5
    log("")
    log(f"Final model on all {len(items)} frames ...")
    model = train_model(items, args.epochs, device, not args.no_pretrained, args.lr, args.batch_size, log, args.pos_weight)
    save_checkpoint(model, args.out_model, {"min_area": MIN_AREA, "threshold": threshold,
                                            "trained_on": len(items), "sows": pigs,
                                            "date": time.strftime("%Y-%m-%d")})
    log(f"Saved {args.out_model}  ({time.time() - t0:.0f} s)")
    report["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(args.out_dir, "seg_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(args.out_dir, "seg_report.txt"), "w") as f:
        f.write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
