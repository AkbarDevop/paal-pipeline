"""Pre-train posture classifier on sowbot 2022 IR dataset.

Trains our SingleModalModel on the sowbot folder-based dataset
(5 folders → 3 classes: standing/sitting/lying) so it can later
be fine-tuned on OAK data via train.py --init-weights.
"""

import argparse
import json
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from config import IMG_SIZE, MODEL_DIR, OUTPUT_DIR, POSTURE3_CLASSES
from models import BACKBONES, SingleModalModel

# Sowbot folder → PAAL 3-class mapping
# Folders 0,1,3 = lying variants → class 2 (lying)
# Folder 2 = sitting → class 1
# Folder 4 = standing → class 0
SOWBOT_FOLDER_TO_CLASS = {
    "0": 2,  # lying
    "1": 2,  # lying
    "2": 1,  # sitting
    "3": 2,  # lying
    "4": 0,  # standing
}

SOWBOT_DATA_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "sowbot", "posture_data_2000",
)


class SowbotDataset(Dataset):
    """Load sowbot IR images from folder structure, map to 3 classes."""

    def __init__(self, samples, size=IMG_SIZE):
        self.samples = samples
        self.size = size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.size, self.size), dtype=np.uint8)
        img = cv2.resize(img, (self.size, self.size))
        # Grayscale → 3 channels (repeat) so model architecture matches OAK 3-ch
        img = np.stack([img, img, img], axis=2).astype(np.float32) / 255.0
        x = torch.from_numpy(img).permute(2, 0, 1)
        return x, torch.tensor(label, dtype=torch.long)


def load_sowbot_samples(data_root):
    """Scan sowbot folder structure, return list of (path, label)."""
    samples = []
    for folder_name, label in SOWBOT_FOLDER_TO_CLASS.items():
        folder_path = os.path.join(data_root, folder_name)
        if not os.path.isdir(folder_path):
            print(f"  Warning: folder not found: {folder_path}")
            continue
        for fname in os.listdir(folder_path):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                samples.append((os.path.join(folder_path, fname), label))
    return samples


def split_samples(samples, train_ratio=0.70, val_ratio=0.15, seed=42):
    """Random 70/15/15 split."""
    rng = random.Random(seed)
    rng.shuffle(samples)
    n = len(samples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return samples[:n_train], samples[n_train:n_train + n_val], samples[n_train + n_val:]


def run_epoch(model, loader, criterion, device, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    with torch.set_grad_enabled(train_mode):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if train_mode:
                optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if train_mode:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


def main(args):
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    data_root = args.data_root or os.path.abspath(SOWBOT_DATA_ROOT)
    print(f"Sowbot data: {data_root}")

    samples = load_sowbot_samples(data_root)
    print(f"Total samples: {len(samples)}")

    from collections import Counter
    counts = Counter(label for _, label in samples)
    for cls_id in sorted(counts):
        print(f"  {POSTURE3_CLASSES[cls_id]}: {counts[cls_id]}")

    train_s, val_s, test_s = split_samples(samples, seed=args.seed)
    print(f"\nSplit: train={len(train_s)}, val={len(val_s)}, test={len(test_s)}")

    train_set = SowbotDataset(train_s)
    val_set = SowbotDataset(val_s)
    test_set = SowbotDataset(test_s)

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False, num_workers=0)

    model = SingleModalModel(
        in_channels=3, num_classes=3, pretrained=True, backbone=args.backbone,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.backbone} | Params: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_tag = f"sowbot_{args.backbone}"
    model_path = os.path.join(MODEL_DIR, f"{model_tag}_best.pth")
    hist_path = os.path.join(OUTPUT_DIR, f"history_{model_tag}.json")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val = -1.0

    print(f"\nTraining {args.backbone} on sowbot, epochs={args.epochs}, lr={args.lr}\n")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        saved = ""
        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "in_channels": 3,
                    "num_classes": 3,
                    "backbone": args.backbone,
                    "source": "sowbot_pretrain",
                },
                model_path,
            )
            saved = " * saved"

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train {train_acc:.3f} ({train_loss:.4f}) | "
            f"Val {val_acc:.3f} ({val_loss:.4f}) | "
            f"{time.time() - t0:.1f}s{saved}"
        )

    # Test
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_acc = run_epoch(model, test_loader, criterion, device)

    history["test_loss"] = test_loss
    history["test_acc"] = test_acc
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nPre-training complete")
    print(f"Best val accuracy: {best_val:.4f}")
    print(f"Test accuracy:     {test_acc:.4f}")
    print(f"Model:   {model_path}")
    print(f"History: {hist_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train on sowbot 2022 data")
    parser.add_argument("--backbone", default="mobilenet_v2", choices=BACKBONES)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", default=None, help="Override sowbot data path")
    main(parser.parse_args())
