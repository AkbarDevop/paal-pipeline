#!/usr/bin/env python3
r"""Score a vulva model end to end on labelled frames it was never trained on.

For every labelled frame the model runs exactly as in run_vulva.py (the checkpoint's
outline threshold, minimum area and confidence cut), and both the model's mask and the
hand-drawn one are measured in mm from the same depth frame (measure3d.py). So this
checks the whole chain the daily values come from:

  - visible or not: precision, recall, specificity, confusion matrix, with and without the cut
  - outline: IoU on frames both call visible
  - size: width and length in mm from the model's mask minus from the hand-drawn mask

Writes, under --out-dir:
  eval_frames.csv   one row per frame (verdict, confidence, IoU, both sizes)
  eval_report.txt   the numbers above
  eval_sheet.jpg    every frame, mistakes first; hand outline green, model outline yellow
                    (orange when the confidence cut dropped it)

Label the test frames by hand without model suggestions, or the test is not independent:
    python vulva/label_ellipse.py ~/Downloads/paal_exchange/vulva_test_feb12

Usage:
    python vulva/eval_seg.py --model models/vulva_seg_r5.pth --labels ~/Downloads/paal_exchange/vulva_test_feb12/labels.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from measure3d import load_depth_raw, load_intrinsics, measure  # noqa: E402
from seg_model import default_depth_path, ir_channel, load_checkpoint, predict, read_input, read_mask  # noqa: E402

FIELDS = ["timestamp_folder", "pig_id", "pig_timestamp", "label", "how", "verdict", "model_visible",
          "model_visible_before_cut", "blob_conf", "iou", "human_width_mm", "model_width_mm",
          "human_length_mm", "model_length_mm", "human_status", "model_status", "ir_path"]
ORDER = {"false alarm": 0, "missed": 1, "dropped by cut": 2, "both visible": 3, "both not visible": 4}


def read_rows(labels_csv):
    base = os.path.dirname(os.path.abspath(labels_csv))
    res = lambda p: p if (not p or os.path.isabs(p) or p.startswith("\\\\")) else os.path.join(base, p)  # noqa: E731
    with open(labels_csv, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("label") in ("present", "absent")]
    for r in rows:
        r["ir_path"], r["mask_path"] = res(r["ir_path"]), res(r.get("mask_path", ""))
        r["depth_path"] = res(r.get("depth_path", "")) or default_depth_path(r["ir_path"])
    return rows


def evaluate(model, ckpt, rows, device, K=None):
    thr, min_area = ckpt.get("threshold", 0.5), ckpt.get("min_area", 25)
    cut, inputs = ckpt.get("presence_threshold", 0.0), ckpt.get("inputs", "ir")
    out = []
    for r in rows:
        img = read_input(r["ir_path"], r["depth_path"], inputs)
        if img is None:
            continue
        human = read_mask(r["mask_path"]) if r["label"] == "present" else None
        if r["label"] == "present" and (human is None or not human.any()):
            continue
        mask, _pmax, _area, conf = predict(model, [img], device, thr, min_area)[0]
        before = bool(mask.any())
        visible = before and conf >= cut
        truth = r["label"] == "present"
        verdict = ("both visible" if truth and visible else "dropped by cut" if truth and before else
                   "missed" if truth else "false alarm" if visible else "both not visible")
        depth = load_depth_raw(r["depth_path"]) if os.path.exists(r["depth_path"]) else None
        hm = measure(depth, human, K) if truth else {}
        mm = measure(depth, mask, K) if before else {}
        iou = ""
        if truth and before:
            t, p = human.astype(bool), mask.astype(bool)
            iou = round(float((t & p).sum() / (t | p).sum()), 4)
        out.append({"timestamp_folder": r.get("timestamp_folder", ""), "pig_id": r.get("pig_id", ""),
                    "pig_timestamp": r.get("pig_timestamp", ""), "label": r["label"], "how": r.get("how", ""),
                    "verdict": verdict, "model_visible": int(visible), "model_visible_before_cut": int(before),
                    "blob_conf": round(conf, 4), "iou": iou,
                    "human_width_mm": hm.get("width_mm", ""), "model_width_mm": mm.get("width_mm", ""),
                    "human_length_mm": hm.get("length_mm", ""), "model_length_mm": mm.get("length_mm", ""),
                    "human_status": hm.get("measure_status", ""), "model_status": mm.get("measure_status", ""),
                    "ir_path": r["ir_path"], "_ir": ir_channel(img), "_human": human, "_model": mask})
    return out


def summarise(frames, cut):
    def conf_at(key):
        tp = sum(1 for f in frames if f["label"] == "present" and f[key])
        fn = sum(1 for f in frames if f["label"] == "present" and not f[key])
        fp = sum(1 for f in frames if f["label"] == "absent" and f[key])
        tn = sum(1 for f in frames if f["label"] == "absent" and not f[key])
        d = lambda a, b: a / b if b else float("nan")  # noqa: E731
        p, r = d(tp, tp + fp), d(tp, tp + fn)
        return tp, fn, fp, tn, p, r, d(tn, tn + fp), d(2 * p * r, p + r)

    lines = [f"{len(frames)} labelled frames: {sum(f['label'] == 'present' for f in frames)} with a visible vulva, "
             f"{sum(f['label'] == 'absent' for f in frames)} without"]
    for name, key in ((f"With the confidence cut ({cut}, as run_vulva.py uses it)", "model_visible"),
                      ("Before the cut", "model_visible_before_cut")):
        tp, fn, fp, tn, p, r, s, f1 = conf_at(key)
        lines += [f"{name}:",
                  f"  vulva visible: precision {p:.3f}, recall {r:.3f}, specificity {s:.3f}, F1 {f1:.3f}",
                  f"  confusion (rows true present/absent): [[{tp}, {fn}], [{fp}, {tn}]]"]
    both = [f for f in frames if f["verdict"] == "both visible"]
    ious = [f["iou"] for f in both if f["iou"] != ""]
    if ious:
        lines.append(f"Outline on frames both call visible ({len(ious)}): IoU mean {np.mean(ious):.3f}, "
                     f"median {np.median(ious):.3f}, below 0.5 on {sum(i < 0.5 for i in ious)}")
    ok = [f for f in both if f["human_status"] == "ok" and f["model_status"] == "ok"]
    for k in ("width_mm", "length_mm"):
        diff = np.array([float(f["model_" + k]) - float(f["human_" + k]) for f in ok])
        hum = np.array([float(f["human_" + k]) for f in ok])
        if len(diff):
            lines.append(f"{k.split('_')[0].capitalize()} (model minus hand, {len(diff)} frames measured on both): "
                         f"bias {diff.mean():+.1f} mm, mean abs {np.abs(diff).mean():.1f} mm, "
                         f"median abs {np.median(np.abs(diff)):.1f} mm; hand median {np.median(hum):.1f} mm")
    return lines


def tile(f, size=200, half=110):
    ir, hm, md = f["_ir"], f["_human"], f["_model"]
    union = np.zeros(ir.shape, bool)
    for m in (hm, md):
        if m is not None and m.any():
            union |= m.astype(bool)
    if union.any():
        ys, xs = np.nonzero(union)
        cy, cx = int(ys.mean()), int(xs.mean())
    else:
        cy, cx = ir.shape[0] // 2 + 20, ir.shape[1] // 2
    y1, x1 = max(cy - half, 0), max(cx - half, 0)
    y2, x2 = min(y1 + 2 * half, ir.shape[0]), min(x1 + 2 * half, ir.shape[1])
    crop = cv2.equalizeHist(ir[y1:y2, x1:x2])
    t = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    for m, col in ((hm, (60, 200, 60)), (md, (0, 220, 255) if f["model_visible"] else (0, 140, 255))):
        if m is not None and m.any():
            cnts, _ = cv2.findContours(m[y1:y2, x1:x2].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(t, cnts, -1, col, 1)
    t = cv2.resize(t, (size, size), interpolation=cv2.INTER_LINEAR)
    ts = str(f["pig_timestamp"])
    top = f"p{f['pig_id']} {ts[4:6]}/{ts[6:8]} {ts[9:11]}:{ts[12:14]}"
    v = f["verdict"] + (f" IoU {float(f['iou']):.2f}" if f["verdict"] == "both visible" and f["iou"] != "" else "")
    for y, text in ((14, top), (size - 8, v)):
        cv2.putText(t, text, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(t, text, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def save_sheet(frames, path, cols=8):
    key = lambda f: (ORDER[f["verdict"]], float(f["iou"]) if f["iou"] != "" else 1.0)  # noqa: E731
    tiles = [tile(f) for f in sorted(frames, key=key)]
    if not tiles:
        return
    blank = np.zeros_like(tiles[0])
    rows = [tiles[i:i + cols] + [blank] * (cols - len(tiles[i:i + cols])) for i in range(0, len(tiles), cols)]
    cv2.imwrite(path, np.vstack([np.hstack(r) for r in rows]), [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.join("models", "vulva_seg.pth"))
    ap.add_argument("--labels", required=True, help="labels.csv of a label set the model was not trained on")
    ap.add_argument("--out-dir", default="", help="default: outputs/eval_<model name>")
    ap.add_argument("--calibration", default="", help="calibration.json with tof_intrinsics")
    args = ap.parse_args()

    name = os.path.splitext(os.path.basename(args.model))[0]
    out_dir = args.out_dir or os.path.join("outputs", f"eval_{name}")
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model, ckpt = load_checkpoint(args.model, device)
    rows = read_rows(args.labels)
    if not rows:
        sys.exit(f"No labelled frames in {args.labels}")
    frames = evaluate(model, ckpt, rows, device, load_intrinsics(args.calibration))
    lines = [f"Model {args.model} on {args.labels}"] + summarise(frames, ckpt.get("presence_threshold", 0.0))
    print("\n".join(lines))
    with open(os.path.join(out_dir, "eval_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "eval_frames.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(frames)
    save_sheet(frames, os.path.join(out_dir, "eval_sheet.jpg"))
    print(f"Outputs in {out_dir} (eval_sheet.jpg: mistakes first; hand outline green, model yellow, "
          f"orange = dropped by the confidence cut)")


if __name__ == "__main__":
    main()
