"""Rule-based no_pig detector from depth raw files."""

import argparse
import csv
import os

import cv2
import numpy as np

from config import METADATA_CSV, PRESENCE_CSV


def load_depth_raw(path):
    if not path or not os.path.exists(path):
        return None
    size = os.path.getsize(path)
    expected = 640 * 480 * 2
    if size == expected + 8:
        raw = np.fromfile(path, dtype=np.uint16, offset=8)
    elif size == expected:
        raw = np.fromfile(path, dtype=np.uint16)
    else:
        return None
    if raw.size != 640 * 480:
        return None
    return raw.reshape(480, 640)


def largest_blob_area(mask_u8):
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n <= 1:
        return 0
    return int(stats[1:, cv2.CC_STAT_AREA].max())


def detect_present(depth, args):
    h, w = depth.shape
    x0, x1 = int(args.roi_x0 * w), int(args.roi_x1 * w)
    y0, y1 = int(args.roi_y0 * h), int(args.roi_y1 * h)
    roi = depth[y0:y1, x0:x1]

    valid = roi > 0
    valid_ratio = float(valid.mean())
    if valid_ratio < args.min_valid_ratio:
        return False, valid_ratio, 0.0, 0.0, 0

    valid_vals = roi[valid]
    local_median = float(np.median(valid_vals))

    fg = (roi > 0) & (roi < (local_median - args.delta_mm))
    fg_ratio = float(fg.mean())
    blob = largest_blob_area((fg.astype(np.uint8) * 255))

    cx0, cx1 = int(args.center_x0 * w), int(args.center_x1 * w)
    cy0, cy1 = int(args.center_y0 * h), int(args.center_y1 * h)
    center = depth[cy0:cy1, cx0:cx1]
    c_valid = center > 0
    if c_valid.any():
        c_median = float(np.median(center[c_valid]))
        c_fg = (center > 0) & (center < (c_median - args.delta_mm))
        center_fg_ratio = float(c_fg.mean())
    else:
        center_fg_ratio = 0.0

    present = (
        fg_ratio >= args.min_fg_ratio
        and center_fg_ratio >= args.min_center_fg_ratio
        and blob >= args.min_blob_area
    )
    return present, valid_ratio, fg_ratio, center_fg_ratio, blob


def main(args):
    if not os.path.exists(args.metadata_csv):
        print(f"Missing metadata: {args.metadata_csv}")
        return

    with open(args.metadata_csv, "r") as f:
        rows = list(csv.DictReader(f))

    out_rows = []
    present_count = 0
    for r in rows:
        if args.force_all_present:
            present_count += 1
            out_rows.append(
                {
                    "timestamp_folder": r["timestamp_folder"],
                    "pig_id": r["pig_id"],
                    "pig_timestamp": r.get("pig_timestamp", ""),
                    "pig_present": 1,
                    "valid_ratio": "",
                    "fg_ratio": "",
                    "center_fg_ratio": "",
                    "largest_blob_area": "",
                    "reason": "force_all_present",
                }
            )
            continue

        depth = load_depth_raw(r.get("depth_raw", ""))
        if depth is None:
            out_rows.append(
                {
                    "timestamp_folder": r["timestamp_folder"],
                    "pig_id": r["pig_id"],
                    "pig_timestamp": r.get("pig_timestamp", ""),
                    "pig_present": 0,
                    "valid_ratio": 0.0,
                    "fg_ratio": 0.0,
                    "center_fg_ratio": 0.0,
                    "largest_blob_area": 0,
                    "reason": "missing_depth_raw",
                }
            )
            continue

        present, valid_ratio, fg_ratio, center_fg_ratio, blob = detect_present(depth, args)
        if present:
            present_count += 1
        out_rows.append(
            {
                "timestamp_folder": r["timestamp_folder"],
                "pig_id": r["pig_id"],
                "pig_timestamp": r.get("pig_timestamp", ""),
                "pig_present": int(present),
                "valid_ratio": round(valid_ratio, 6),
                "fg_ratio": round(fg_ratio, 6),
                "center_fg_ratio": round(center_fg_ratio, 6),
                "largest_blob_area": int(blob),
                "reason": "ok",
            }
        )

    fields = [
        "timestamp_folder",
        "pig_id",
        "pig_timestamp",
        "pig_present",
        "valid_ratio",
        "fg_ratio",
        "center_fg_ratio",
        "largest_blob_area",
        "reason",
    ]
    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    print(f"Saved: {args.output_csv}")
    print(f"Total frames: {len(out_rows)}")
    print(f"Pig present: {present_count}")
    print(f"No pig: {len(out_rows) - present_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depth-based no_pig prefilter")
    parser.add_argument("--metadata-csv", default=METADATA_CSV)
    parser.add_argument("--output-csv", default=PRESENCE_CSV)
    parser.add_argument("--min-depth-mm", type=int, default=300)
    parser.add_argument("--max-depth-mm", type=int, default=1800)
    parser.add_argument("--min-valid-ratio", type=float, default=0.03)
    parser.add_argument("--delta-mm", type=int, default=80)
    parser.add_argument("--min-fg-ratio", type=float, default=0.08)
    parser.add_argument("--min-center-fg-ratio", type=float, default=0.05)
    parser.add_argument("--min-blob-area", type=int, default=4500)
    parser.add_argument("--roi-x0", type=float, default=0.28)
    parser.add_argument("--roi-x1", type=float, default=0.72)
    parser.add_argument("--roi-y0", type=float, default=0.58)
    parser.add_argument("--roi-y1", type=float, default=0.98)
    parser.add_argument("--center-x0", type=float, default=0.40)
    parser.add_argument("--center-x1", type=float, default=0.60)
    parser.add_argument("--center-y0", type=float, default=0.68)
    parser.add_argument("--center-y1", type=float, default=0.98)
    parser.add_argument("--force-all-present", action="store_true")
    main(parser.parse_args())
