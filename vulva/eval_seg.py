#!/usr/bin/env python3
r"""Score a vulva model end to end on labelled frames it was never trained on.

For every labelled frame the model runs exactly as in run_vulva.py (the checkpoint's
outline threshold, minimum area and confidence cut), and both the model's mask and the
hand-drawn one are measured in mm from the same depth frame (measure3d.py). So this
checks the whole chain the daily values come from:

  - visible or not: precision, recall, specificity, confusion matrix, with and without the keep
    decision (keep_rule.py: the confidence cut, plus the bump check when a rule is tuned)
  - outline: IoU on frames both call visible
  - size: width and length in mm from the model's mask minus from the hand-drawn mask

Writes, under --out-dir:
  eval_frames.csv   one row per frame (verdict, confidence, IoU, both sizes)
  eval_report.txt   the numbers above
  eval_sheet.jpg    every frame, mistakes first; hand outline green, model outline yellow
                    (orange when the keep decision dropped it), drawn on the colour image when
                    the set has one (register_rgb.py, or matched on the fly), else on the IR

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
from rgb_align import register  # noqa: E402
from keep_rule import keep, rule_from_checkpoint  # noqa: E402
from seg_model import default_depth_path, ir_channel, load_checkpoint, predict_parts, read_input, read_mask  # noqa: E402
from train_seg import require_split, split_set  # noqa: E402

FIELDS = ["timestamp_folder", "pig_id", "pig_timestamp", "label", "note", "how", "verdict", "model_visible",
          "model_visible_before_cut", "blob_conf", "iou", "human_width_mm", "model_width_mm",
          "human_length_mm", "model_length_mm", "human_peak_height_mm", "model_peak_height_mm",
          "model_mean_height_mm", "model_area_mm2", "model_distance_mm", "human_status", "model_status",
          "covered_by_tail", "model_tail_px", "ir_path"]
ORDER = {"false alarm": 0, "missed": 1, "dropped by cut": 2, "dropped (covered by tail)": 2, "both visible": 3,
         "both not visible": 4}


def read_rows(labels_csv, frames_roots=None, splits=None):
    from select_frames import index_folders
    from train_seg import relocate_row
    index = index_folders(frames_roots) if frames_roots else {}
    base = os.path.dirname(os.path.abspath(labels_csv))
    res = lambda p: p if (not p or os.path.isabs(p) or p.startswith("\\\\")) else os.path.join(base, p)  # noqa: E731
    with open(labels_csv, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("label") in ("present", "absent")
                and (splits is None or r.get("split") in splits)]
    colour = {}                                   # colour images listed in the set's label_set.csv
    ls = os.path.join(base, "label_set.csv")
    if os.path.exists(ls):
        with open(ls, newline="", encoding="utf-8-sig") as f:
            colour = {fr["ir_path"]: (fr.get("rgb_aligned_path", ""), fr.get("rgb_path", "")) for fr in csv.DictReader(f)}
    for r in rows:
        reg, plain = colour.get(r["ir_path"], ("", ""))
        r["rgb_reg"], r["rgb_plain"] = res(reg), res(plain)
        r["ir_path"], r["mask_path"] = res(r["ir_path"]), res(r.get("mask_path", ""))
        r["depth_path"] = res(r.get("depth_path", ""))
        relocate_row(r, index)
        r["depth_path"] = r["depth_path"] or default_depth_path(r["ir_path"])
        if not (r["rgb_reg"] or r["rgb_plain"]):          # colour image next to the IR, if any
            d, name = os.path.split(r["ir_path"])
            cand = os.path.join(d, name.replace("_ir_vis_", "_rgb_").rsplit(".", 1)[0] + ".jpg")
            r["rgb_plain"] = cand if os.path.exists(cand) else ""
    return rows


def evaluate(model, ckpt, rows, device, K=None):
    thr, min_area = ckpt.get("threshold", 0.5), ckpt.get("min_area", 25)
    rule, inputs = rule_from_checkpoint(ckpt), ckpt.get("inputs", "ir")
    out = []
    for r in rows:
        img = read_input(r["ir_path"], r["depth_path"], inputs)
        if img is None:
            continue
        human = read_mask(r["mask_path"]) if r["label"] == "present" else None
        if r["label"] == "present" and (human is None or not human.any()):
            continue
        d = predict_parts(model, [img], device, thr, min_area)[0]
        mask, conf, tail_touch = d["vulva"], d["conf"], d["covered_by_tail"]
        before = bool(mask.any())
        truth = r["label"] == "present"
        depth = load_depth_raw(r["depth_path"]) if os.path.exists(r["depth_path"]) else None
        hm = measure(depth, human, K) if truth else {}
        mm = measure(depth, mask, K) if before else {}
        measured = mm.get("measure_status") == "ok"
        visible = (before and not tail_touch
                   and keep(conf, mm.get("peak_height_mm") if measured else None, measured, rule))
        verdict = ("both visible" if truth and visible else
                   ("dropped (covered by tail)" if tail_touch else "dropped by cut") if truth and before else
                   "missed" if truth else "false alarm" if visible else "both not visible")
        iou = ""
        if truth and before:
            t, p = human.astype(bool), mask.astype(bool)
            iou = round(float((t & p).sum() / (t | p).sum()), 4)
        out.append({"timestamp_folder": r.get("timestamp_folder", ""), "pig_id": r.get("pig_id", ""),
                    "pig_timestamp": r.get("pig_timestamp", ""), "label": r["label"], "note": r.get("note", ""),
                    "how": r.get("how", ""),
                    "verdict": verdict, "model_visible": int(visible), "model_visible_before_cut": int(before),
                    "blob_conf": round(conf, 6), "iou": iou,
                    "human_width_mm": hm.get("width_mm", ""), "model_width_mm": mm.get("width_mm", ""),
                    "human_length_mm": hm.get("length_mm", ""), "model_length_mm": mm.get("length_mm", ""),
                    "human_peak_height_mm": hm.get("peak_height_mm", ""),
                    "model_peak_height_mm": mm.get("peak_height_mm", ""),
                    "model_mean_height_mm": mm.get("mean_height_mm", ""), "model_area_mm2": mm.get("area_mm2", ""),
                    "model_distance_mm": mm.get("distance_mm", ""), "covered_by_tail": int(tail_touch),
                    "model_tail_px": int(d["tail"].sum()) if d["tail"] is not None else "",
                    "human_status": hm.get("measure_status", ""), "model_status": mm.get("measure_status", ""),
                    "ir_path": r["ir_path"], "_ir": ir_channel(img), "_human": human, "_model": mask,
                    "_rgb": (r.get("rgb_reg", ""), r.get("rgb_plain", "")), "_tail": d["tail"]})
    return out


def colour_view(f):
    """The frame's colour image on the IR grid, or None."""
    reg, plain = f.get("_rgb", ("", ""))
    if reg and os.path.exists(reg):
        im = cv2.imread(reg, cv2.IMREAD_COLOR)
        if im is not None and im.shape[:2] == f["_ir"].shape[:2]:
            return im
    if plain and os.path.exists(plain):
        rgb = cv2.imread(plain, cv2.IMREAD_COLOR)
        if rgb is not None:
            centre = None
            for m in (f["_human"], f["_model"]):
                if m is not None and m.any():
                    ys, xs = m.nonzero()
                    centre = (xs.mean(), ys.mean())
                    break
            return register(f["_ir"], rgb, centre)[0]
    return None


IOU_THRESHOLDS = [round(0.5 + 0.05 * k, 2) for k in range(10)]      # 0.50, 0.55, ..., 0.95 (COCO)


def mask_ap(frames, thr):
    """COCO-style average precision of the vulva masks at one IoU threshold. Every outline the model
    drew (before any keep decision) is a detection scored by its blob confidence; a detection is right
    when the frame has a hand outline and the two overlap by IoU >= thr (one vulva per frame, so at most
    one match). Precision is interpolated at 101 recall points, as in the COCO evaluation."""
    n_gt = sum(1 for f in frames if f["label"] == "present")
    dets = sorted((f for f in frames if f["model_visible_before_cut"]), key=lambda f: -float(f["blob_conf"]))
    if not n_gt:
        return float("nan")
    tp = fp = 0
    prec, rec = [], []
    for f in dets:
        ok = f["label"] == "present" and f["iou"] != "" and float(f["iou"]) >= thr
        tp, fp = tp + ok, fp + (not ok)
        prec.append(tp / (tp + fp))
        rec.append(tp / n_gt)
    for i in range(len(prec) - 2, -1, -1):          # precision envelope (non-increasing)
        prec[i] = max(prec[i], prec[i + 1])
    total = 0.0
    for r in np.linspace(0, 1, 101):
        p = [pv for pv, rv in zip(prec, rec) if rv >= r - 1e-12]
        total += p[0] if p else 0.0
    return total / 101


def map_scores(frames):
    """mAP50 and mAP50-95 (mean over IoU 0.50:0.05:0.95) of the vulva masks."""
    aps = [mask_ap(frames, t) for t in IOU_THRESHOLDS]
    return aps[0], float(np.mean(aps)), dict(zip(IOU_THRESHOLDS, aps))


def summarise(frames, rule):
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
    m50, m5095, by_t = map_scores(frames)
    lines.append(f"Mask mAP50 {m50:.3f}, mAP50-95 {m5095:.3f}  (every outline drawn, ranked by confidence; "
                 f"AP at IoU 0.75 {by_t[0.75]:.3f}, at 0.90 {by_t[0.9]:.3f})")
    how = (f"conf >= {rule['conf_high']}, or conf >= {rule['conf_low']} and bump >= {rule['min_peak_mm']} mm"
           if rule["conf_low"] < rule["conf_high"] else f"confidence cut {rule['conf_high']}")
    if any(f.get("covered_by_tail") for f in frames) or any(f.get("model_tail_px") not in ("", None) for f in frames):
        how += ", and not covered by a tail"
    for name, key in ((f"Kept ({how}, as run_vulva.py decides)", "model_visible"),
                      ("Before the cut", "model_visible_before_cut")):
        tp, fn, fp, tn, p, r, s, f1 = conf_at(key)
        lines += [f"{name}:",
                  f"  vulva visible: precision {p:.3f}, recall {r:.3f}, specificity {s:.3f}, F1 {f1:.3f}",
                  f"  confusion (rows true present/absent): [[{tp}, {fn}], [{fp}, {tn}]]"]
    both = [f for f in frames if f["verdict"] == "both visible"]
    tt = sum(1 for f in frames if f["verdict"] == "dropped (covered by tail)")
    if tt:
        lines.append(f"Vulvas dropped because the model drew a tail cutting into them: {tt}")
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


def tile(f, size=200, half=110, colour=True):
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
    rgb = colour_view(f) if colour else None
    t = rgb[y1:y2, x1:x2].copy() if rgb is not None else cv2.cvtColor(cv2.equalizeHist(ir[y1:y2, x1:x2]),
                                                                       cv2.COLOR_GRAY2BGR)
    for m, col in ((f.get("_tail"), (60, 60, 255)), (hm, (60, 200, 60)),
                   (md, (0, 220, 255) if f["model_visible"] else (0, 140, 255))):
        if m is not None and m.any():
            cnts, _ = cv2.findContours(m[y1:y2, x1:x2].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(t, cnts, -1, col, 2 if rgb is not None else 1)
    t = cv2.resize(t, (size, size), interpolation=cv2.INTER_LINEAR)
    ts = str(f["pig_timestamp"])
    top = f"p{f['pig_id']} {ts[4:6]}/{ts[6:8]} {ts[9:11]}:{ts[12:14]}"
    v = f["verdict"] + (f" IoU {float(f['iou']):.2f}" if f["verdict"] == "both visible" and f["iou"] != "" else "")
    for y, text in ((14, top), (size - 8, v)):
        cv2.putText(t, text, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(t, text, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def save_sheet(frames, path, cols=8, colour=True):
    key = lambda f: (ORDER[f["verdict"]], float(f["iou"]) if f["iou"] != "" else 1.0)  # noqa: E731
    tiles = [tile(f, colour=colour) for f in sorted(frames, key=key)]
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
    ap.add_argument("--ir-sheet", action="store_true", help="draw the sheet on the IR image even if colour exists")
    ap.add_argument("--frames-root", default="",
                    help="';'-separated raw folders, for labelled frames whose images are not next to the labels")
    ap.add_argument("--split", default="", help="for a dataset table (build_dataset.py): which split(s) to score, "
                    "comma-separated, e.g. test_newdays")
    args = ap.parse_args()
    require_split(args.labels, args.split)

    name = os.path.splitext(os.path.basename(args.model))[0]
    out_dir = args.out_dir or os.path.join("outputs", f"eval_{name}")
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model, ckpt = load_checkpoint(args.model, device)
    rows = read_rows(args.labels, [r for r in args.frames_root.split(";") if r.strip()], split_set(args.split))
    if not rows:
        sys.exit(f"No labelled frames in {args.labels}")
    frames = evaluate(model, ckpt, rows, device, load_intrinsics(args.calibration))
    lines = [f"Model {args.model} on {args.labels}" + (f" (split {args.split})" if args.split else "")] + summarise(frames, rule_from_checkpoint(ckpt))
    print("\n".join(lines))
    with open(os.path.join(out_dir, "eval_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "eval_frames.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(frames)
    save_sheet(frames, os.path.join(out_dir, "eval_sheet.jpg"), colour=not args.ir_sheet)
    print(f"Outputs in {out_dir} (eval_sheet.jpg: mistakes first; hand outline green, model yellow, "
          f"orange = dropped by the keep decision)")


if __name__ == "__main__":
    main()
