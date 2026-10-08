"""Before/after benchmark for the posture classifier (run on the lab PC).

What it does, all on the labeled frames in labels/labels_posture3.csv:
  1. Re-scores the shipped model (models/posture3_ir_best.pth) on its held-out
     test pigs 16-19, with full per-class metrics.
  2. Grouped K-fold CV by pig for the v1.0 recipe ("baseline") and the v1.1
     recipe ("improved"), same folds and seeds, pooled out-of-fold metrics.
  3. Temporal smoothing check on the out-of-fold predictions: accuracy and,
     more importantly, how many posture transitions are predicted versus how
     many really happen in the labels (fake transitions inflate the estrus
     signal).
Writes outputs/benchmark_report.md + outputs/benchmark_report.json.

Usage (lab PC, repo root):
  python posture/benchmark.py --data-root "C:\\PAAL_data\\Fed_pig\\Pictures_OAK"
  python posture/benchmark.py --data-root D:\\labeled --cv 5 --backbone mobilenet_v2
"""

import argparse
import json
import os
import sys
import pathlib
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import LABELS_POSTURE3_CSV, MODEL_DIR, OUTPUT_DIR, POSTURE3_CLASSES, TEST_PIG_IDS, IMG_SIZE
from data_loader import cache_images, read_label_rows
from models import SingleModalModel
from preprocess import ckpt_preprocessing, pick_device
from temporal import parse_ts, smooth_sequence
import train as trainmod

NAMES = [POSTURE3_CLASSES[i] for i in range(3)]


def score_shipped_model(model_path, data_roots, device, labels_csv=LABELS_POSTURE3_CSV):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    normalize, crop = ckpt_preprocessing(ckpt)
    model = SingleModalModel(in_channels=ckpt.get("in_channels", 3), num_classes=ckpt.get("num_classes", 3),
                             pretrained=False, backbone=ckpt.get("backbone", "mobilenet_v2")).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    samples = read_label_rows(labels_csv, TEST_PIG_IDS, None, data_roots)
    images, found = cache_images(samples, ckpt.get("modality", "ir"), IMG_SIZE, crop=crop)
    keep = np.where(found)[0]
    labels = np.array([samples[i]["label"] for i in keep])
    probs = trainmod.predict(model, images[keep], device, normalize)
    return trainmod.compute_metrics(labels, probs.argmax(1), 3), int(len(samples) - len(keep))


def transitions(seq_labels, seq_times, max_gap):
    n = 0
    for i in range(1, len(seq_labels)):
        gap = (seq_times[i] - seq_times[i - 1]).total_seconds() / 60.0
        if gap <= max_gap and seq_labels[i] != seq_labels[i - 1]:
            n += 1
    return n


def smoothing_check(oof, window, max_gap):
    """Compare raw vs smoothed out-of-fold predictions on per-pig time sequences."""
    by_pig = defaultdict(list)
    for r in oof:
        t = parse_ts(r["pig_timestamp"])
        if t is not None:
            by_pig[r["pig_id"]].append((t, r))
    y, raw, smooth = [], [], []
    true_tr = raw_tr = smooth_tr = 0
    for rows in by_pig.values():
        rows.sort(key=lambda x: x[0])
        times = [t for t, _ in rows]
        lab = [NAMES[r["label"]] for _, r in rows]
        pred = [NAMES[r["pred"]] for _, r in rows]
        probs = [dict(zip(NAMES, r["probs"])) for _, r in rows]
        sm = smooth_sequence(pred, times, window=window, max_gap_minutes=max_gap, probs=probs)
        y += lab; raw += pred; smooth += sm
        true_tr += transitions(lab, times, max_gap)
        raw_tr += transitions(pred, times, max_gap)
        smooth_tr += transitions(sm, times, max_gap)
    idx = {n: i for i, n in enumerate(NAMES)}
    m_raw = trainmod.compute_metrics([idx[a] for a in y], [idx[a] for a in raw], 3)
    m_sm = trainmod.compute_metrics([idx[a] for a in y], [idx[a] for a in smooth], 3)
    return {"window": window, "max_gap_min": max_gap, "raw": m_raw, "smoothed": m_sm,
            "true_transitions": true_tr, "raw_transitions": raw_tr, "smoothed_transitions": smooth_tr}


def pct(x):
    return f"{x * 100:.2f}%"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", action="append", required=True,
                    help="Folder(s) that contain the labeled timestamp folders")
    ap.add_argument("--labels-csv", default=LABELS_POSTURE3_CSV)
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--backbone", default="mobilenet_v2")
    ap.add_argument("--modality", default="ir")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smooth-window", type=int, default=3)
    ap.add_argument("--max-gap", type=float, default=30.0)
    ap.add_argument("--shipped-model", default=os.path.join(MODEL_DIR, "posture3_ir_best.pth"))
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    device = pick_device()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report = {"date": time.strftime("%Y-%m-%d %H:%M"), "device": str(device), "cv": args.cv,
              "backbone": args.backbone, "modality": args.modality, "seed": args.seed}

    if os.path.exists(args.shipped_model):
        m, missing = score_shipped_model(args.shipped_model, args.data_root, device, args.labels_csv)
        report["shipped_model_test_pigs"] = {"metrics": m, "missing_frames": missing}
        print(f"\nShipped model on test pigs {TEST_PIG_IDS}:\n{trainmod.format_metrics(m, NAMES)}")

    for recipe in ("baseline", "improved"):
        targs = trainmod.build_parser().parse_args([
            "--class-set", "posture3", "--labels-csv", args.labels_csv, "--modality", args.modality, "--backbone", args.backbone,
            "--recipe", recipe, "--cv", str(args.cv), "--seed", str(args.seed), "--no-save",
            "--out-json", os.path.join(OUTPUT_DIR, f"benchmark_cv_{recipe}.json")]
            + sum((["--data-root", r] for r in args.data_root), [])
            + (["--epochs", str(args.epochs)] if args.epochs else [])
            + (["--no-pretrained"] if args.no_pretrained else []))
        res = trainmod.main(targs)
        report[recipe] = {"pooled": res["cv"]["pooled"], "fold_accuracy": res["cv"]["fold_accuracy"],
                          "fold_macro_f1": res["cv"]["fold_macro_f1"], "folds": res["cv"]["folds"],
                          "smoothing": smoothing_check(res["oof_predictions"], args.smooth_window, args.max_gap)}

    with open(os.path.join(OUTPUT_DIR, "benchmark_report.json"), "w") as f:
        json.dump(report, f, indent=1)

    lines = [f"# Posture classifier benchmark ({report['date']})", "",
             f"Device {report['device']}, backbone {args.backbone}, modality {args.modality}, "
             f"{args.cv}-fold CV grouped by pig, seed {args.seed}.", ""]
    if "shipped_model_test_pigs" in report:
        m = report["shipped_model_test_pigs"]["metrics"]
        lines += ["## Shipped model (posture3_ir_best.pth) on test pigs 16-19", "",
                  f"Frames {m['n']}, accuracy {pct(m['accuracy'])}, macro-F1 {pct(m['macro_f1'])}, "
                  f"sitting recall {pct(m['per_class'][1]['recall'])} on {m['per_class'][1]['support']} sitting frames.", ""]
    lines += ["## Recipe comparison (pooled out-of-fold, every pig tested once)", "",
              "| Recipe | Accuracy | Macro-F1 | Balanced acc | Sitting F1 | Fold macro-F1 (mean ± sd) |",
              "|---|---|---|---|---|---|"]
    for recipe in ("baseline", "improved"):
        p = report[recipe]["pooled"]
        f1s = np.array(report[recipe]["fold_macro_f1"])
        lines.append(f"| {recipe} | {pct(p['accuracy'])} | {pct(p['macro_f1'])} | {pct(p['balanced_accuracy'])} | "
                     f"{pct(p['per_class'][1]['f1'])} | {f1s.mean()*100:.2f} ± {f1s.std()*100:.2f} |")
    lines += ["", "## Temporal smoothing on out-of-fold predictions", "",
              f"Window {args.smooth_window}, frames more than {args.max_gap:.0f} min apart are never smoothed together.", "",
              "| Recipe | Raw acc | Smoothed acc | True transitions | Predicted (raw) | Predicted (smoothed) |",
              "|---|---|---|---|---|---|"]
    for recipe in ("baseline", "improved"):
        s = report[recipe]["smoothing"]
        lines.append(f"| {recipe} | {pct(s['raw']['accuracy'])} | {pct(s['smoothed']['accuracy'])} | "
                     f"{s['true_transitions']} | {s['raw_transitions']} | {s['smoothed_transitions']} |")
    lines += ["", "Confusion matrices and per-class numbers are in benchmark_report.json."]
    out_md = os.path.join(OUTPUT_DIR, "benchmark_report.md")
    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nSaved {out_md}")


if __name__ == "__main__":
    main()
