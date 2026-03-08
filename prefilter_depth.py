"""Pig presence prefilter using depth-based median threshold.

Classifies each frame as pig-present or no-pig by computing the
median depth in the cropped stall region (with stall bars masked).

Rule: median_depth < threshold → pig present
      median_depth >= threshold → no pig

Camera is side-mounted (vulva side):
  - Pig present: body blocks view → lower median depth
  - No pig: see through to far wall → higher median depth (~1500mm+)
"""

import argparse
import csv
import os

import numpy as np

from config import CROP_TOF, METADATA_CSV, PRESENCE_CSV

TOF_W, TOF_H = 640, 480
THRESHOLD = 1463      # median depth in mm
BAR_MARGIN = 50       # pixels to mask on left/right edges


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
    return raw.reshape(TOF_H, TOF_W)


def check_pig_present(depth_path, crop, bar_margin, threshold):
    """Check if pig is present using median depth threshold."""
    depth = load_depth_raw(depth_path)
    if depth is None:
        return False, 0.0

    x1, y1, x2, y2 = crop
    cropped = depth[y1:y2, x1:x2]
    ch, cw = cropped.shape

    # Mask stall bars on left/right edges
    bar_mask = np.ones((ch, cw), dtype=bool)
    bar_mask[:, :bar_margin] = False
    bar_mask[:, cw - bar_margin:] = False

    valid = (cropped > 200) & (cropped < 5000) & bar_mask
    masked = cropped[valid]

    if masked.size == 0:
        return False, 0.0

    median_d = float(np.median(masked))
    return median_d < threshold, median_d


def main(args):
    if not os.path.exists(args.metadata_csv):
        print(f"Missing metadata: {args.metadata_csv}")
        return

    with open(args.metadata_csv, "r") as f:
        rows = list(csv.DictReader(f))

    out_rows = []
    present_count = 0

    for r in rows:
        present, median_d = check_pig_present(
            r.get("depth_raw", ""), CROP_TOF, args.bar_margin, args.threshold,
        )
        if present:
            present_count += 1

        out_rows.append({
            "timestamp_folder": r["timestamp_folder"],
            "pig_id": r["pig_id"],
            "pig_timestamp": r.get("pig_timestamp", ""),
            "pig_present": int(present),
            "median_depth": round(median_d, 1),
            "threshold": args.threshold,
        })

    fields = [
        "timestamp_folder", "pig_id", "pig_timestamp",
        "pig_present", "median_depth", "threshold",
    ]
    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    print(f"Saved: {args.output_csv}")
    print(f"Total frames: {len(out_rows)}")
    print(f"Pig present: {present_count}")
    print(f"No pig: {len(out_rows) - present_count}")
    print(f"Threshold: {args.threshold}mm | Bar margin: {args.bar_margin}px")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pig presence prefilter (depth median)")
    parser.add_argument("--metadata-csv", default=METADATA_CSV)
    parser.add_argument("--output-csv", default=PRESENCE_CSV)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--bar-margin", type=int, default=BAR_MARGIN)
    main(parser.parse_args())
