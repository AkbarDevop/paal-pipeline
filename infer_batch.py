#!/usr/bin/env python3
"""Batch inference — run trained posture model on a new (unlabeled) OAK-D dataset.

Scans timestamp folders, runs depth prefilter, predicts postures,
and saves results to CSV + optional summary visuals.

Usage:
    # Run on default data/ folder with best model
    python3 infer_batch.py

    # Run on a specific folder (e.g. new Google Drive data)
    python3 infer_batch.py --data-dir /path/to/new_data

    # Use cropped images, specific modality
    python3 infer_batch.py --modality ir --use-cropped

    # Save prediction grid image
    python3 infer_batch.py --save-grid
"""

import argparse
import csv
import os
import re
import sys

import cv2
import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

from config import (
    CROP_TOF,
    DATA_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    POSTURE3_CLASSES,
    IMG_SIZE,
)
from models import SingleModalModel
from data_loader import MODALITY_CHANNELS, load_and_preprocess

# ── Depth prefilter (inline, no CSV needed) ────────────────────────────────

TOF_W, TOF_H = 640, 480
DEPTH_THRESHOLD = 1463
BAR_MARGIN = 50


def load_depth_raw(path):
    """Load depth raw file (uint16, 640x480, optional 8-byte header)."""
    if not path or not os.path.exists(path):
        return None
    size = os.path.getsize(path)
    expected = TOF_W * TOF_H * 2
    if size == expected + 8:
        raw = np.fromfile(path, dtype=np.uint16, offset=8)
    elif size == expected:
        raw = np.fromfile(path, dtype=np.uint16)
    else:
        return None
    if raw.size != TOF_W * TOF_H:
        return None
    return raw.reshape((TOF_H, TOF_W))


def check_pig_present(depth_raw_path):
    """Return (is_present, median_depth) using depth prefilter logic."""
    depth = load_depth_raw(depth_raw_path)
    if depth is None:
        return True, 0.0  # no depth data → assume present

    x1, y1, x2, y2 = CROP_TOF
    crop = depth[y1:y2, x1:x2].copy()

    # Mask stall bars
    crop[:, :BAR_MARGIN] = 0
    crop[:, -BAR_MARGIN:] = 0

    # Valid depth range
    valid = crop[(crop > 200) & (crop < 5000)]
    if len(valid) == 0:
        return False, 0.0

    median = float(np.median(valid))
    return median < DEPTH_THRESHOLD, median


# ── Scan data folder ───────────────────────────────────────────────────────

FILENAME_RE = re.compile(
    r"^pig(\d+)_(depth_vis|depth|ir_vis|ir|rgb_aligned|rgb)_"
    r"(\d{8}-\d{2}-\d{2}-\d{2})(_cropped)?\.(jpg|raw)$"
)
FOLDER_RE = re.compile(r"^\d{8}-\d{2}-\d{2}-\d{2}$")


def scan_folder(data_dir):
    """Scan OAK timestamp folders and return list of frame dicts."""
    folders = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and FOLDER_RE.match(d)
    )
    if not folders:
        print(f"No timestamp folders found in {data_dir}")
        return []

    print(f"Found {len(folders)} timestamp folders in {data_dir}")

    records = {}
    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        for fname in os.listdir(folder_path):
            m = FILENAME_RE.match(fname)
            if not m:
                continue
            pig_id = int(m.group(1))
            modality = m.group(2)
            timestamp = m.group(3)
            is_cropped = m.group(4) is not None
            ext = m.group(5)
            fpath = os.path.join(folder_path, fname)

            key = (folder, pig_id, timestamp)
            if key not in records:
                records[key] = {
                    "timestamp_folder": folder,
                    "pig_id": pig_id,
                    "pig_timestamp": timestamp,
                }

            col_name = f"{modality}{'_cropped' if is_cropped else ''}_{ext}"
            records[key][col_name] = fpath

    frames = sorted(records.values(), key=lambda r: (r["timestamp_folder"], r["pig_id"]))
    print(f"  Total frames: {len(frames)}")
    return frames


# ── Image loading ──────────────────────────────────────────────────────────

def get_image(frame, modality, use_cropped):
    """Get image path for the given modality, preferring cropped if requested."""
    key = f"{modality}_jpg"
    cropped_key = f"{modality}_cropped_jpg"

    if use_cropped:
        p = frame.get(cropped_key, "")
        if p and os.path.exists(p):
            return p

    p = frame.get(key, "")
    if p and os.path.exists(p):
        return p
    return ""


def load_frame_tensor(frame, modality, use_cropped):
    """Load and preprocess frame into a model-ready tensor."""
    imgs = []

    if modality in ("rgb", "rgb_depth", "rgb_ir", "all"):
        if modality in ("rgb_depth", "rgb_ir", "all"):
            path = get_image(frame, "rgb_aligned", use_cropped) or get_image(frame, "rgb", use_cropped)
        else:
            path = get_image(frame, "rgb", use_cropped)
        img = load_and_preprocess(path)
        imgs.append(img if img is not None else np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32))

    if modality in ("ir", "rgb_ir", "all"):
        path = get_image(frame, "ir", use_cropped)
        img = load_and_preprocess(path)
        imgs.append(img if img is not None else np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32))

    if modality in ("depth", "rgb_depth", "all"):
        path = get_image(frame, "depth", use_cropped)
        img = load_and_preprocess(path)
        imgs.append(img if img is not None else np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32))

    if not imgs:
        return None

    x = np.concatenate(imgs, axis=2)
    return torch.from_numpy(x).permute(2, 0, 1)


# ── Main ───────────────────────────────────────────────────────────────────

def main(args):
    data_dir = args.data_dir or DATA_DIR
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # Load model
    model_path = os.path.join(MODEL_DIR, f"{args.model_prefix}_{args.modality}_best.pth")
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print(f"Available models in {MODEL_DIR}:")
        if os.path.exists(MODEL_DIR):
            for f in sorted(os.listdir(MODEL_DIR)):
                if f.endswith(".pth"):
                    print(f"  {f}")
        return

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    in_channels = ckpt.get("in_channels", MODALITY_CHANNELS[args.modality])
    num_classes = ckpt.get("num_classes", 3)
    backbone = ckpt.get("backbone", "mobilenet_v2")

    model = SingleModalModel(
        in_channels=in_channels,
        num_classes=num_classes,
        pretrained=False,
        backbone=backbone,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model: {model_path} ({backbone}, {num_classes} classes)")

    class_names = POSTURE3_CLASSES

    # Scan data
    frames = scan_folder(data_dir)
    if not frames:
        return

    # Run inference
    results = []
    skipped_no_pig = 0
    skipped_no_image = 0

    print(f"\nRunning inference (modality={args.modality}, cropped={args.use_cropped})...\n")

    with torch.no_grad():
        for i, frame in enumerate(frames):
            pid = frame["pig_id"]
            ts = frame["pig_timestamp"]

            # Depth prefilter
            if not args.skip_prefilter:
                depth_raw = frame.get("depth_raw", "")
                present, median_d = check_pig_present(depth_raw)
                if not present:
                    skipped_no_pig += 1
                    continue
            else:
                median_d = 0.0

            # Load image
            x = load_frame_tensor(frame, args.modality, args.use_cropped)
            if x is None:
                skipped_no_image += 1
                continue

            # Predict
            logits = model(x.unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))
            conf = float(probs[pred])
            pred_name = class_names.get(pred, str(pred))

            results.append({
                "timestamp_folder": frame["timestamp_folder"],
                "pig_id": pid,
                "pig_timestamp": ts,
                "prediction": pred,
                "prediction_name": pred_name,
                "confidence": round(conf, 4),
                "median_depth": round(median_d, 1),
                "image_path": get_image(frame, args.modality, args.use_cropped),
            })

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(frames)} frames...")

    # Summary
    print(f"\n{'='*50}")
    print(f"Total frames scanned: {len(frames)}")
    print(f"Skipped (no pig):     {skipped_no_pig}")
    print(f"Skipped (no image):   {skipped_no_image}")
    print(f"Predicted:            {len(results)}")

    if results:
        from collections import Counter
        counts = Counter(r["prediction_name"] for r in results)
        print(f"\nPosture distribution:")
        for name in ["standing", "sitting", "lying"]:
            c = counts.get(name, 0)
            pct = c / len(results) * 100
            print(f"  {name:<10}: {c:>5} ({pct:.1f}%)")

        avg_conf = np.mean([r["confidence"] for r in results])
        print(f"\nAvg confidence: {avg_conf:.3f}")

    # Save CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_csv = args.output or os.path.join(OUTPUT_DIR, "batch_predictions.csv")
    fields = ["timestamp_folder", "pig_id", "pig_timestamp",
              "prediction", "prediction_name", "confidence",
              "median_depth", "image_path"]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved predictions: {out_csv}")

    # Optional: save grid of sample predictions
    if args.save_grid and HAS_PLT and results:
        import random
        random.seed(42)
        n = min(16, len(results))
        samples = random.sample(results, n)

        fig, axes = plt.subplots(4, 4, figsize=(16, 12))
        for ax, r in zip(axes.flat, samples):
            img = cv2.imread(r["image_path"])
            if img is not None:
                img = cv2.cvtColor(cv2.resize(img, (320, 240)), cv2.COLOR_BGR2RGB)
                ax.imshow(img)
            ax.set_title(
                f"pig{r['pig_id']} | {r['prediction_name']} ({r['confidence']:.2f})",
                fontsize=9,
            )
            ax.axis("off")
        for ax in axes.flat[n:]:
            ax.axis("off")

        plt.suptitle(f"Batch Predictions — {args.modality} / {backbone}", fontsize=14)
        plt.tight_layout()
        grid_path = os.path.join(OUTPUT_DIR, "batch_predictions_grid.png")
        plt.savefig(grid_path, dpi=150)
        plt.close()
        print(f"Saved grid: {grid_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run posture model on new (unlabeled) OAK-D data")
    p.add_argument("--data-dir", default=None,
                    help="Path to data folder (default: config.DATA_DIR)")
    p.add_argument("--modality", choices=["rgb", "ir", "depth", "rgb_depth", "rgb_ir", "all"],
                    default="ir", help="Which image modality to use (default: ir)")
    p.add_argument("--model-prefix", default="posture3",
                    help="Model prefix (default: posture3)")
    p.add_argument("--use-cropped", action="store_true",
                    help="Use cropped images (must run crop_images.py first)")
    p.add_argument("--skip-prefilter", action="store_true",
                    help="Skip depth prefilter (predict on all frames)")
    p.add_argument("--save-grid", action="store_true",
                    help="Save a 4x4 grid of sample predictions")
    p.add_argument("--output", default=None,
                    help="Output CSV path (default: outputs/batch_predictions.csv)")
    main(p.parse_args())
