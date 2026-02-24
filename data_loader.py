"""Dataset and dataloaders for posture classification."""

import csv
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from collections import Counter

from config import BATCH_SIZE, IMG_SIZE, LABELS_CSV, TEST_PIG_IDS, TRAIN_PIG_IDS, VAL_PIG_IDS


MODALITY_CHANNELS = {
    "rgb": 3,
    "ir": 3,
    "depth": 3,
    "rgb_depth": 6,
    "rgb_ir": 6,
    "all": 9,
}


def load_and_preprocess(path, size=IMG_SIZE):
    if not path or not os.path.exists(path):
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img


class SowPostureDataset(Dataset):
    def __init__(
        self,
        modality="rgb",
        labels_csv=LABELS_CSV,
        pig_ids=None,
        allowed_labels=None,
        transform=None,
        silent=False,
    ):
        self.modality = modality
        self.transform = transform
        self.samples = []
        self.label_map = None
        if allowed_labels is not None:
            allowed_sorted = sorted(set(int(x) for x in allowed_labels))
            self.label_map = {orig: idx for idx, orig in enumerate(allowed_sorted)}

        if not os.path.exists(labels_csv):
            raise FileNotFoundError(f"Missing labels file: {labels_csv}")

        with open(labels_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not (row.get("label") or "").strip():
                    continue

                pid = int(row["pig_id"])
                if pig_ids is not None and pid not in pig_ids:
                    continue

                label = int(row["label"])
                if allowed_labels is not None and label not in allowed_labels:
                    continue
                if self.label_map is not None:
                    label = self.label_map[label]

                self.samples.append(
                    {
                        "rgb_jpg": row.get("rgb_jpg", ""),
                        "ir_jpg": row.get("ir_jpg", ""),
                        "depth_jpg": row.get("depth_jpg", ""),
                        "label": label,
                        "pig_id": pid,
                    }
                )

        if not silent:
            pigs = sorted({s["pig_id"] for s in self.samples})
            label_counts = Counter(s["label"] for s in self.samples)
            print(f"  {len(self.samples)} samples | pigs: {pigs} | label_counts: {dict(sorted(label_counts.items()))}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = sample["label"]
        imgs = []

        if self.modality in ("rgb", "rgb_depth", "rgb_ir", "all"):
            rgb = load_and_preprocess(sample["rgb_jpg"])
            imgs.append(rgb if rgb is not None else np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32))

        if self.modality in ("ir", "rgb_ir", "all"):
            ir = load_and_preprocess(sample["ir_jpg"])
            imgs.append(ir if ir is not None else np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32))

        if self.modality in ("depth", "rgb_depth", "all"):
            depth = load_and_preprocess(sample["depth_jpg"])
            imgs.append(depth if depth is not None else np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32))

        x = np.concatenate(imgs, axis=2)
        x = torch.from_numpy(x).permute(2, 0, 1)
        if self.transform:
            x = self.transform(x)
        return x, torch.tensor(label, dtype=torch.long)


def get_dataloaders(modality="rgb", batch_size=BATCH_SIZE, labels_csv=LABELS_CSV, allowed_labels=None):
    print(f"\nLoading data for modality='{modality}':")
    print(f"  Train pigs: {TRAIN_PIG_IDS}")
    train_set = SowPostureDataset(
        modality=modality,
        labels_csv=labels_csv,
        pig_ids=TRAIN_PIG_IDS,
        allowed_labels=allowed_labels,
    )

    print(f"  Val pigs:   {VAL_PIG_IDS}")
    val_set = SowPostureDataset(
        modality=modality,
        labels_csv=labels_csv,
        pig_ids=VAL_PIG_IDS,
        allowed_labels=allowed_labels,
    )

    print(f"  Test pigs:  {TEST_PIG_IDS}")
    test_set = SowPostureDataset(
        modality=modality,
        labels_csv=labels_csv,
        pig_ids=TEST_PIG_IDS,
        allowed_labels=allowed_labels,
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    channels = MODALITY_CHANNELS[modality]

    print(f"\n  Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}, Channels: {channels}")
    return train_loader, val_loader, test_loader, channels
