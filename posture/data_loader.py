"""Dataset and dataloaders for posture classification.

v1.1: images are cropped to the stall in memory (same code path as inference,
see preprocess.py), so training no longer depends on *_cropped.jpg files
written into the raw data folders. Label CSVs store absolute paths from the
machine they were made on; pass --data-root (one or more folders that contain
the timestamp folders) and files are found by name on any machine.
"""

import csv
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import BATCH_SIZE, IMG_SIZE, LABELS_CSV, TEST_PIG_IDS, TRAIN_PIG_IDS, VAL_PIG_IDS, resolve_path
from preprocess import crop_stall, read_image, resize_rgb_uint8, uint8_to_tensor


_fix_path = resolve_path


MODALITY_CHANNELS = {
    "rgb": 3,
    "ir": 3,
    "depth": 3,
    "rgb_depth": 6,
    "rgb_ir": 6,
    "all": 9,
}

MODALITY_PARTS = {
    "rgb": ["rgb"],
    "ir": ["ir"],
    "depth": ["depth"],
    "rgb_depth": ["rgb", "depth"],
    "rgb_ir": ["rgb", "ir"],
    "all": ["rgb", "ir", "depth"],
}


def load_and_preprocess(path, size=IMG_SIZE):
    """v1.0 helper kept for older scripts: whole image, no crop, HWC float 0..1."""
    img = read_image(path, min_bytes=0)
    if img is None:
        return None
    return resize_rgb_uint8(img, size).astype(np.float32) / 255.0


# ── path resolution ─────────────────────────────────────────────────────────

TS_FOLDER_RE = re.compile(r"^\d{8}-\d{2}-\d{2}-\d{2}$")


def timestamp_folders(root):
    """{timestamp folder name: [paths]} directly under root or one level down
    (round folders such as E:\\Pig_data\\Pictures_OAK\\2026-01-23_02-03\\<timestamp>).
    Lists directories only, so it stays fast on the network share."""
    out = {}
    try:
        entries = list(os.scandir(root))
    except OSError:
        return out
    for e in entries:
        if not e.is_dir():
            continue
        if TS_FOLDER_RE.match(e.name):
            out.setdefault(e.name, []).append(e.path)
            continue
        try:
            for e2 in os.scandir(e.path):
                if e2.is_dir() and TS_FOLDER_RE.match(e2.name):
                    out.setdefault(e2.name, []).append(e2.path)
        except OSError:
            pass
    return out


class FileIndex:
    """Find label-CSV images by file name under one or more data roots."""

    def __init__(self, roots=None):
        self.roots = [r for r in (roots or []) if r]
        self._index = None
        self._folders = None

    def _build(self):
        self._index = {}
        for root in self.roots:
            for dirpath, _, files in os.walk(root):
                for name in files:
                    if name.endswith(".jpg"):
                        self._index.setdefault(name, os.path.join(dirpath, name))

    def resolve(self, csv_path):
        if not csv_path:
            return ""
        if os.path.exists(csv_path):
            return csv_path
        if self.roots:
            # fast path: find the timestamp folder (directory listing only), then the file,
            # so a big network share is never walked file by file
            parts = [x for x in csv_path.replace("\\", "/").split("/") if x]
            if len(parts) >= 2:
                if self._folders is None:
                    self._folders = {}
                    for root in self.roots:
                        for name, paths in timestamp_folders(root).items():
                            self._folders.setdefault(name, []).extend(paths)
                folders = self._folders.get(parts[-2])
                if folders:
                    for d in folders:
                        cand = os.path.join(d, parts[-1])
                        if os.path.exists(cand):
                            return cand
                    return _fix_path(csv_path)   # folder is there but this file is not: no full walk
                if self._folders:
                    # the roots do hold timestamp folders, just not this one: report it missing
                    # instead of walking every file on a (possibly network) drive
                    return _fix_path(csv_path)
            if self._index is None:
                self._build()
            hit = self._index.get(os.path.basename(csv_path.replace("\\", "/")))
            if hit:
                return hit
        return _fix_path(csv_path)


def read_label_rows(labels_csv=LABELS_CSV, pig_ids=None, allowed_labels=None, data_roots=None):
    """Read labeled frames -> list of sample dicts with resolved image paths."""
    if not os.path.exists(labels_csv):
        raise FileNotFoundError(f"Missing labels file: {labels_csv}")
    label_map = None
    if allowed_labels is not None:
        label_map = {orig: i for i, orig in enumerate(sorted(set(int(x) for x in allowed_labels)))}
    index = FileIndex(data_roots)
    samples = []
    with open(labels_csv, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if not (row.get("label") or "").strip():
                continue
            pid = int(row["pig_id"])
            if pig_ids is not None and pid not in pig_ids:
                continue
            label = int(row["label"])
            if label_map is not None:
                if label not in label_map:
                    continue
                label = label_map[label]
            sample = {
                "label": label,
                "pig_id": pid,
                "timestamp_folder": row.get("timestamp_folder", ""),
                "pig_timestamp": row.get("pig_timestamp", ""),
            }
            for key in ("rgb_jpg", "rgb_aligned_jpg", "ir_jpg", "depth_jpg",
                        "rgb_cropped_jpg", "rgb_aligned_cropped_jpg", "ir_cropped_jpg", "depth_cropped_jpg"):
                sample[key] = index.resolve(row.get(key, ""))
            samples.append(sample)
    return samples


def _part_path(sample, part, modality, use_cropped):
    if part == "rgb" and modality in ("rgb_depth", "rgb_ir", "all"):
        keys = ["rgb_aligned_jpg", "rgb_jpg"]   # aligned RGB shares the ToF frame
    else:
        keys = [f"{part}_jpg"]
    if use_cropped:
        keys = [k.replace("_jpg", "_cropped_jpg") for k in keys] + keys
    for k in keys:
        p = sample.get(k, "")
        if p and os.path.exists(p):
            return p
    return ""


def load_sample_uint8(sample, modality="ir", size=IMG_SIZE, crop=True, use_cropped=False):
    """Return HWC uint8 (C = 3 x parts) and whether every part was found."""
    parts, ok = [], True
    for part in MODALITY_PARTS[modality]:
        path = _part_path(sample, part, modality, use_cropped)
        img = read_image(path, min_bytes=0)
        if img is None:
            ok = False
            parts.append(np.zeros((size, size, 3), dtype=np.uint8))
            continue
        if crop:
            img = crop_stall(img, path)
        parts.append(resize_rgb_uint8(img, size))
    return np.concatenate(parts, axis=2), ok


def cache_images(samples, modality="ir", size=IMG_SIZE, crop=True, use_cropped=False, workers=8):
    """Decode every labeled image once -> (N,H,W,C) uint8 array + found mask."""
    def load(s):
        return load_sample_uint8(s, modality, size, crop, use_cropped)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(load, samples))
    arr = np.stack([o[0] for o in out]) if out else np.zeros((0, size, size, 3), np.uint8)
    found = np.array([o[1] for o in out], dtype=bool)
    return arr, found


class SowPostureDataset(Dataset):
    """Labeled frames for one set of pigs.

    New in v1.1: crop (in-memory stall crop, default True), normalize
    (None for v1.0-style 0..1 inputs, "imagenet" for v1.1 models), data_roots.
    """

    def __init__(
        self,
        modality="rgb",
        labels_csv=LABELS_CSV,
        pig_ids=None,
        allowed_labels=None,
        transform=None,
        silent=False,
        use_cropped=False,
        crop=True,
        normalize=None,
        data_roots=None,
    ):
        self.modality = modality
        self.transform = transform
        self.use_cropped = use_cropped
        self.crop = crop
        self.normalize = normalize
        self.samples = read_label_rows(labels_csv, pig_ids, allowed_labels, data_roots)
        if not silent:
            pigs = sorted({s["pig_id"] for s in self.samples})
            label_counts = Counter(s["label"] for s in self.samples)
            print(f"  {len(self.samples)} samples | pigs: {pigs} | label_counts: {dict(sorted(label_counts.items()))}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img, _ = load_sample_uint8(sample, self.modality, crop=self.crop, use_cropped=self.use_cropped)
        x = uint8_to_tensor(img, None)
        if self.transform:
            x = self.transform(x)
        if self.normalize:
            from preprocess import normalize_batch
            x = normalize_batch(x.unsqueeze(0), self.normalize)[0]
        return x, torch.tensor(sample["label"], dtype=torch.long)


def get_dataloaders(modality="rgb", batch_size=BATCH_SIZE, labels_csv=LABELS_CSV, allowed_labels=None,
                    use_cropped=False, crop=True, normalize=None, data_roots=None):
    print(f"\nLoading data for modality='{modality}'{' (cropped)' if use_cropped else ''}:")
    sets = []
    for name, pigs in (("Train", TRAIN_PIG_IDS), ("Val", VAL_PIG_IDS), ("Test", TEST_PIG_IDS)):
        print(f"  {name} pigs: {pigs}")
        sets.append(SowPostureDataset(modality=modality, labels_csv=labels_csv, pig_ids=pigs,
                                      allowed_labels=allowed_labels, use_cropped=use_cropped,
                                      crop=crop, normalize=normalize, data_roots=data_roots))
    train_set, val_set, test_set = sets
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    channels = MODALITY_CHANNELS[modality]
    print(f"\n  Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}, Channels: {channels}")
    return train_loader, val_loader, test_loader, channels
