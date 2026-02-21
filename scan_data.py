"""Scan OAK data folders and build metadata CSV."""

import csv
import os
import re

from config import DATA_DIR, LABEL_DIR, METADATA_CSV


PATTERN = re.compile(r"^pig(\d+)_(depth_vis|depth|ir_vis|ir|rgb_aligned|rgb)_(\d{8}-\d{2}-\d{2}-\d{2})\.(jpg|raw)$")


def parse_filename(name):
    m = PATTERN.match(name)
    if not m:
        return None
    return {"pig_id": int(m.group(1)), "modality": m.group(2), "timestamp": m.group(3), "ext": m.group(4)}


def scan_data():
    os.makedirs(LABEL_DIR, exist_ok=True)

    folders = sorted(
        d
        for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d)) and re.match(r"^\d{8}-\d{2}-\d{2}-\d{2}$", d)
    )
    print(f"Found {len(folders)} timestamp folders in {DATA_DIR}")
    if not folders:
        print("No data folders found.")
        return

    records = {}
    for folder in folders:
        folder_path = os.path.join(DATA_DIR, folder)
        for fname in os.listdir(folder_path):
            parsed = parse_filename(fname)
            if not parsed:
                continue

            key = (folder, parsed["pig_id"])
            if key not in records:
                records[key] = {
                    "timestamp_folder": folder,
                    "pig_id": parsed["pig_id"],
                    "pig_timestamp": parsed["timestamp"],
                    "rgb_jpg": "",
                    "rgb_aligned_jpg": "",
                    "ir_jpg": "",
                    "depth_jpg": "",
                    "rgb_raw": "",
                    "ir_raw": "",
                    "depth_raw": "",
                }

            path = os.path.join(folder_path, fname)
            mod, ext = parsed["modality"], parsed["ext"]
            if mod == "rgb_aligned" and ext == "jpg":
                records[key]["rgb_aligned_jpg"] = path
            elif mod == "rgb" and ext == "jpg":
                records[key]["rgb_jpg"] = path
            elif mod == "rgb" and ext == "raw":
                records[key]["rgb_raw"] = path
            elif mod == "ir_vis" and ext == "jpg":
                records[key]["ir_jpg"] = path
            elif mod == "ir" and ext == "raw":
                records[key]["ir_raw"] = path
            elif mod == "depth_vis" and ext == "jpg":
                records[key]["depth_jpg"] = path
            elif mod == "depth" and ext == "raw":
                records[key]["depth_raw"] = path

    rows = sorted(records.values(), key=lambda r: (r["timestamp_folder"], r["pig_id"]))
    fields = [
        "timestamp_folder",
        "pig_id",
        "pig_timestamp",
        "rgb_jpg",
        "rgb_aligned_jpg",
        "ir_jpg",
        "depth_jpg",
        "rgb_raw",
        "ir_raw",
        "depth_raw",
    ]

    with open(METADATA_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    pig_ids = sorted({r["pig_id"] for r in rows})
    n_rgb = sum(1 for r in rows if r["rgb_jpg"])
    n_aligned = sum(1 for r in rows if r["rgb_aligned_jpg"])
    n_ir = sum(1 for r in rows if r["ir_jpg"])
    n_depth = sum(1 for r in rows if r["depth_jpg"])

    print(f"\nMetadata saved: {METADATA_CSV}")
    print(f"Timestamp folders: {len(folders)}")
    print(f"Total pig-frame records: {len(rows)}")
    print(f"Pig IDs found: {pig_ids}")
    print(f"Records with RGB jpg:         {n_rgb}")
    print(f"Records with RGB aligned jpg: {n_aligned}")
    print(f"Records with IR jpg:          {n_ir}")
    print(f"Records with Depth jpg:       {n_depth}")


if __name__ == "__main__":
    scan_data()
