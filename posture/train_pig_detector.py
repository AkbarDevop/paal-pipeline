#!/usr/bin/env python3
"""Train a binary pig-presence CNN (MobileNetV2).

Pipeline:
  1. Read auto-labels CSV, take balanced sample (N empty + N present)
  2. Copy training images from network share to local disk
  3. Train MobileNetV2 (2-class) on cropped IR images
  4. Save weights to models/pig_detector.pth

Usage:
    python train_pig_detector.py
    python train_pig_detector.py --n-per-class 2000 --epochs 8
"""

import argparse
import csv
import os
import random
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import BASE_DIR, MODEL_DIR

LABELS_CSV = os.path.join(BASE_DIR, "labels/pig_presence/auto_labels_v3.csv")
LOCAL_DATA = Path(BASE_DIR) / "labels/pig_presence/data"
MODEL_OUT = os.path.join(MODEL_DIR, "pig_detector.pth")
IMG_SIZE = 224
NETWORK_BASE = "//umad.umsystem.edu/rde/zhoujianf-lab/Pig project/Sow Data/Pictures_OAK"


def remap_path(original_path):
    """Map any old path to network share path. Try cropped, fall back to raw."""
    import re
    norm = original_path.replace("\\", "/")
    m = re.search(r"(\d{8}-\d{2}-\d{2}-\d{2})/([^/]+\.jpg)$", norm)
    if not m:
        return original_path
    folder, fname = m.group(1), m.group(2)
    cropped = f"{NETWORK_BASE}/{folder}/{fname}"
    if os.path.exists(cropped):
        return cropped
    return f"{NETWORK_BASE}/{folder}/{fname.replace('_cropped', '')}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-class", type=int, default=2000,
                   help="Frames per class to sample (default: 2000)")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-copy", action="store_true",
                   help="Skip image copy step (use already-copied data)")
    return p.parse_args()


def sample_and_copy(n_per_class, seed, skip_copy):
    """Sample balanced training set and copy from network to local disk."""
    random.seed(seed)
    rows = []
    with open(LABELS_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    by_label = {"empty": [r for r in rows if r["label"] == "empty"],
                "present": [r for r in rows if r["label"] == "present"]}
    print(f"Available: empty={len(by_label['empty'])}, present={len(by_label['present'])}")

    sampled = []
    for label in ("empty", "present"):
        random.shuffle(by_label[label])
        sampled.extend([(r, label) for r in by_label[label][:n_per_class]])
    print(f"Sampled balanced subset: {len(sampled)} total ({n_per_class} per class)")

    LOCAL_DATA.mkdir(parents=True, exist_ok=True)
    (LOCAL_DATA / "empty").mkdir(exist_ok=True)
    (LOCAL_DATA / "present").mkdir(exist_ok=True)

    if skip_copy:
        print("Skipping copy step — using existing local files")
    else:
        print(f"\nCopying images from network share to {LOCAL_DATA}/ ...")
        print("(this is the slow part — ~25 min for 4000 images)")
        copied = 0
        skipped = 0
        failed = 0
        start = time.time()
        for i, (row, label) in enumerate(sampled, 1):
            src = remap_path(row["image_path"])
            ts = row["timestamp"]
            pid = row["pig_id"]
            # Suffix with "_raw" if it's an uncropped file so Dataset knows to crop
            is_raw = "_cropped" not in src
            suffix = "_raw" if is_raw else ""
            dst = LOCAL_DATA / label / f"pid{pid}_{ts}{suffix}.jpg"
            if dst.exists():
                skipped += 1
                continue
            try:
                shutil.copy(src, dst)
                copied += 1
            except Exception:
                failed += 1
            if i % 100 == 0:
                elapsed = time.time() - start
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(sampled) - i) / rate if rate > 0 else 0
                print(f"  {i}/{len(sampled)}  ({rate:.1f} img/s, ~{eta/60:.1f} min remaining)")
        print(f"\nCopied: {copied}, already local: {skipped}, failed: {failed}")

    # Build local label list
    local_rows = []
    for label_idx, label_name in enumerate(("empty", "present")):
        for f in (LOCAL_DATA / label_name).glob("*.jpg"):
            local_rows.append((str(f), label_idx))
    random.shuffle(local_rows)
    print(f"\nLocal training data: {len(local_rows)} total")
    return local_rows


CROP_TOF = (120, 30, 500, 480)
TOF_W, TOF_H = 640, 480


def crop_box_for_size(w, h):
    x1, y1, x2, y2 = CROP_TOF
    sx, sy = w / TOF_W, h / TOF_H
    return int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)


class IRDataset(Dataset):
    def __init__(self, rows, augment=False):
        self.rows = rows
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        path, label = self.rows[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        # If raw (uncropped) image (suffix "_raw"), crop to stall region first
        if "_raw" in path:
            h, w = img.shape[:2]
            x1, y1, x2, y2 = crop_box_for_size(w, h)
            img = img[y1:y2, x1:x2]
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.augment and random.random() < 0.5:
            img = cv2.flip(img, 1)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img, label


def build_model(device):
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    return m.to(device)


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def train(local_rows, epochs, batch_size, lr, device):
    n = len(local_rows)
    split = int(n * 0.8)
    train_rows = local_rows[:split]
    val_rows = local_rows[split:]
    print(f"Train: {len(train_rows)}, Val: {len(val_rows)}")

    train_loader = DataLoader(IRDataset(train_rows, augment=True),
                              batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(IRDataset(val_rows), batch_size=batch_size,
                            shuffle=False, num_workers=0)

    model = build_model(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0
        correct = total = 0
        for i, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            if i % 50 == 0:
                print(f"  Epoch {epoch} batch {i}/{len(train_loader)} loss={loss.item():.4f}")

        train_acc = correct / total
        val_acc = evaluate(model, val_loader, device)
        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch}: train_loss={running_loss/len(train_loader):.4f} "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} ({elapsed:.0f}s)")

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "classes": ["empty", "present"],
                "img_size": IMG_SIZE,
            }, MODEL_OUT)
            print(f"  Saved best model (val_acc={val_acc:.4f})")

    print(f"\nBest val_acc: {best_acc:.4f}")
    print(f"Saved to: {MODEL_OUT}")


def main():
    args = parse_args()
    print(f"Pig presence detector training")
    print(f"  N per class: {args.n_per_class}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR: {args.lr}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if device.type == "cpu":
        print("  (CPU training — will be slower; GPU recommended for production)")

    local_rows = sample_and_copy(args.n_per_class, args.seed, args.skip_copy)
    if not local_rows:
        print("No training data — aborting")
        return
    train(local_rows, args.epochs, args.batch_size, args.lr, device)


if __name__ == "__main__":
    main()
