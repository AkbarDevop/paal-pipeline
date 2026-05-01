#!/usr/bin/env python3
"""Filter existing predictions.csv using the trained pig-presence detector.

Loads each prediction's image, runs pig_detector, drops predictions where
detector says no pig. Faster than re-running full inference since we
don't re-run the posture model.

Usage:
    python filter_predictions.py outputs/predictions_feb_combined.csv outputs/predictions_feb_filtered.csv
"""

import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np
import torch
import torchvision.models as tv_models

NETWORK_BASE = "//umad.umsystem.edu/rde/zhoujianf-lab/Pig project/Sow Data/Pictures_OAK"
DETECTOR_PATH = "models/pig_detector.pth"
IMG_SIZE = 224
THRESHOLD = 0.5  # p_present >= 0.5 means pig present


def remap_path(original_path):
    """Map any old path to network share path.

    Network share has Feb 13+ data with both `_cropped.jpg` and original `.jpg`,
    but early Feb (06-11) only has originals. Try cropped first, fall back to
    original.
    """
    import re
    norm = original_path.replace("\\", "/")
    m = re.search(r"(\d{8}-\d{2}-\d{2}-\d{2})/([^/]+\.jpg)$", norm)
    if not m:
        return original_path
    folder = m.group(1)
    fname = m.group(2)
    cropped_path = f"{NETWORK_BASE}/{folder}/{fname}"
    if os.path.exists(cropped_path):
        return cropped_path
    # Fall back to non-cropped version
    fname_raw = fname.replace("_cropped", "")
    raw_path = f"{NETWORK_BASE}/{folder}/{fname_raw}"
    return raw_path


def load_detector(device):
    m = tv_models.mobilenet_v2(weights=None)
    m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, 2)
    ckpt = torch.load(DETECTOR_PATH, map_location=device, weights_only=False)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval().to(device)
    return m, ckpt.get("val_acc", "?")


CROP_TOF = (120, 30, 500, 480)
TOF_W, TOF_H = 640, 480


def crop_box_for_size(w, h):
    x1, y1, x2, y2 = CROP_TOF
    sx, sy = w / TOF_W, h / TOF_H
    return int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)


def preprocess(path):
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    # If it's a raw (uncropped) image, crop it. Cropped images have "_cropped" in path.
    if "_cropped" not in path:
        h, w = img.shape[:2]
        x1, y1, x2, y2 = crop_box_for_size(w, h)
        img = img[y1:y2, x1:x2]
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help="p_present threshold to keep a frame (default: 0.5)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    detector, val_acc = load_detector(device)
    print(f"Detector loaded (val_acc={val_acc})")

    with open(args.input_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames + ["p_present"]
        rows = list(reader)
    print(f"Loaded {len(rows)} predictions")

    kept = []
    dropped = 0
    missing = 0
    start = time.time()

    with torch.no_grad():
        for i, row in enumerate(rows, 1):
            net_path = remap_path(row["image_path"])
            x = preprocess(net_path)
            if x is None:
                missing += 1
                continue
            x = x.to(device)
            probs = torch.softmax(detector(x), dim=1).cpu().numpy()[0]
            p_present = float(probs[1])
            row["p_present"] = round(p_present, 4)
            if p_present >= args.threshold:
                kept.append(row)
            else:
                dropped += 1

            if i % 500 == 0:
                elapsed = time.time() - start
                rate = i / elapsed
                eta = (len(rows) - i) / rate / 60
                print(f"  {i}/{len(rows)} ({rate:.1f}/s, ~{eta:.1f} min remaining)  "
                      f"kept={len(kept)} dropped={dropped} missing={missing}")

    print(f"\nFinal: kept {len(kept)} / dropped {dropped} / missing {missing}")
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)
    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
