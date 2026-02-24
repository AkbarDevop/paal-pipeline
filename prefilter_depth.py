"""Pig presence and image quality prefilter.

Adapted from Xue et al. (sowbot) occlusion detection approach:
images with excessive dark pixels are filtered as occluded/no-pig.
Also uses depth raw data validity as a secondary signal.
"""

import argparse
import csv
import os

import cv2
import numpy as np

from config import METADATA_CSV, PRESENCE_CSV


def is_occluded(image_path, dark_thresh=26, dark_ratio_thresh=0.4):
    """Check if image is occluded using dark pixel ratio (sowbot method)."""
    if not image_path or not os.path.exists(image_path):
        return True, 1.0
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return True, 1.0
    dark_ratio = float((img < dark_thresh).mean())
    return dark_ratio > dark_ratio_thresh, dark_ratio


def check_depth_valid(path, min_valid_ratio=0.3):
    """Check if depth raw file has enough valid pixels."""
    if not path or not os.path.exists(path):
        return False, 0.0
    size = os.path.getsize(path)
    expected = 640 * 480 * 2
    if size == expected + 8:
        raw = np.fromfile(path, dtype=np.uint16, offset=8)
    elif size == expected:
        raw = np.fromfile(path, dtype=np.uint16)
    else:
        return False, 0.0
    if raw.size != 640 * 480:
        return False, 0.0
    depth = raw.reshape(480, 640)
    valid = (depth > 200) & (depth < 5000)
    valid_ratio = float(valid.mean())
    return valid_ratio >= min_valid_ratio, valid_ratio


def main(args):
    if not os.path.exists(args.metadata_csv):
        print(f"Missing metadata: {args.metadata_csv}")
        return

    with open(args.metadata_csv, "r") as f:
        rows = list(csv.DictReader(f))

    out_rows = []
    present_count = 0
    occluded_count = 0
    depth_fail_count = 0

    for r in rows:
        if args.force_all_present:
            present_count += 1
            out_rows.append({
                "timestamp_folder": r["timestamp_folder"],
                "pig_id": r["pig_id"],
                "pig_timestamp": r.get("pig_timestamp", ""),
                "pig_present": 1,
                "ir_dark_ratio": "",
                "depth_valid_ratio": "",
                "reason": "force_all_present",
            })
            continue

        ir_occluded, ir_dark_ratio = is_occluded(
            r.get("ir_jpg", ""), args.dark_thresh, args.dark_ratio_thresh,
        )
        depth_ok, depth_valid_ratio = check_depth_valid(
            r.get("depth_raw", ""), args.min_depth_valid,
        )

        if ir_occluded:
            occluded_count += 1
        if not depth_ok:
            depth_fail_count += 1

        present = not ir_occluded or depth_ok
        if present:
            present_count += 1

        reason = "ok"
        if ir_occluded and not depth_ok:
            reason = "occluded+no_depth"
        elif ir_occluded:
            reason = "ir_occluded_but_depth_ok"
        elif not depth_ok:
            reason = "depth_invalid_but_ir_ok"

        out_rows.append({
            "timestamp_folder": r["timestamp_folder"],
            "pig_id": r["pig_id"],
            "pig_timestamp": r.get("pig_timestamp", ""),
            "pig_present": int(present),
            "ir_dark_ratio": round(ir_dark_ratio, 4),
            "depth_valid_ratio": round(depth_valid_ratio, 4),
            "reason": reason,
        })

    fields = [
        "timestamp_folder", "pig_id", "pig_timestamp",
        "pig_present", "ir_dark_ratio", "depth_valid_ratio", "reason",
    ]
    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    print(f"Saved: {args.output_csv}")
    print(f"Total frames: {len(out_rows)}")
    print(f"Pig present: {present_count}")
    print(f"No pig: {len(out_rows) - present_count}")
    print(f"IR occluded: {occluded_count}")
    print(f"Depth invalid: {depth_fail_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pig presence prefilter")
    parser.add_argument("--metadata-csv", default=METADATA_CSV)
    parser.add_argument("--output-csv", default=PRESENCE_CSV)
    parser.add_argument("--dark-thresh", type=int, default=26)
    parser.add_argument("--dark-ratio-thresh", type=float, default=0.4)
    parser.add_argument("--min-depth-valid", type=float, default=0.3)
    parser.add_argument("--force-all-present", action="store_true")
    main(parser.parse_args())
