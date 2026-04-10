#!/usr/bin/env python3
"""Detect pig ID shifts by comparing per-stall depth baselines.

Uses predictions.csv (from run_inference.py) to build depth fingerprints
per stall position. When a pig is lying, depth reflects the stall floor
distance — which is physically fixed. Shifts in this pattern indicate
the camera's sequential counting skipped a stall.

Usage:
    python detect_id_shifts.py                           # uses outputs/predictions.csv
    python detect_id_shifts.py --csv path/to/predictions.csv
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np

from config import OUTPUT_DIR


def load_predictions(csv_path):
    """Load predictions CSV into list of dicts."""
    rows = []
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            row["pig_id"] = int(row["pig_id"])
            row["median_depth"] = float(row["median_depth"])
            row["confidence"] = float(row["confidence"])
            rows.append(row)
    return rows


def build_stall_baselines(rows, min_folders=50):
    """Build per-stall depth baselines from lying pig frames.

    When a pig is lying, depth is close to the stall floor — a fixed
    physical property. We collect lying-pig depths per cam_pig_id across
    all folders to build a baseline per stall.

    Returns: {pig_id: median_lying_depth}
    """
    # Collect lying-pig depths per pig_id
    lying_depths = defaultdict(list)
    for r in rows:
        if r["prediction_name"] == "lying" and r["median_depth"] > 0:
            lying_depths[r["pig_id"]].append(r["median_depth"])

    baselines = {}
    for pid in sorted(lying_depths):
        depths = lying_depths[pid]
        if len(depths) >= min_folders:
            baselines[pid] = {
                "median": float(np.median(depths)),
                "std": float(np.std(depths)),
                "count": len(depths),
                "q25": float(np.percentile(depths, 25)),
                "q75": float(np.percentile(depths, 75)),
            }

    return baselines


def build_folder_profiles(rows):
    """Build per-folder depth profiles using lying pig frames.

    Returns: {folder: {pig_id: median_lying_depth}}
    """
    # Collect lying depths per (folder, pig_id)
    folder_pig_depths = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["prediction_name"] == "lying" and r["median_depth"] > 0:
            folder_pig_depths[r["timestamp_folder"]][r["pig_id"]].append(r["median_depth"])

    profiles = {}
    for folder in folder_pig_depths:
        profiles[folder] = {}
        for pid, depths in folder_pig_depths[folder].items():
            profiles[folder][pid] = float(np.median(depths))

    return profiles


def detect_shifts(baselines, folder_profiles):
    """Detect ID shifts by comparing folder profiles against baselines.

    For each folder, try offset 0 (no shift), offset 1 (pig 0 missing),
    offset 2 (pigs 0-1 missing), etc. Pick the offset with lowest error.

    Returns list of (folder, best_offset, score, details)
    """
    baseline_pids = sorted(baselines.keys())
    if not baseline_pids:
        return []

    baseline_vec = np.array([baselines[p]["median"] for p in baseline_pids])

    results = []
    for folder, profile in sorted(folder_profiles.items()):
        pids = sorted(profile.keys())
        if len(pids) < 10:  # too few lying pigs to compare
            continue

        # Try offsets 0-3
        best_offset = 0
        best_score = float('inf')
        scores = {}

        for offset in range(4):
            error = 0.0
            count = 0
            for cam_pid in pids:
                physical_pid = cam_pid + offset
                if physical_pid in baselines and cam_pid in profile:
                    diff = abs(profile[cam_pid] - baselines[physical_pid]["median"])
                    # Normalize by std to account for natural variation
                    std = max(baselines[physical_pid]["std"], 50)
                    error += diff / std
                    count += 1

            if count > 0:
                avg_error = error / count
                scores[offset] = round(avg_error, 3)
                if avg_error < best_score:
                    best_score = avg_error
                    best_offset = offset

        if best_offset > 0:
            results.append({
                "folder": folder,
                "best_offset": best_offset,
                "score_offset0": scores.get(0, -1),
                "score_best": scores.get(best_offset, -1),
                "num_lying_pigs": len(pids),
                "all_scores": scores,
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="Detect pig ID shifts via depth baselines")
    parser.add_argument("--csv", default=os.path.join(OUTPUT_DIR, "predictions.csv"),
                        help="Path to predictions CSV")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: {args.csv} not found. Run run_inference.py first.")
        return

    print(f"Loading predictions from {args.csv}...")
    rows = load_predictions(args.csv)
    print(f"  {len(rows)} frames loaded")

    # Step 1: Build baselines
    print("\nBuilding per-stall depth baselines (lying pigs only)...")
    baselines = build_stall_baselines(rows)
    print(f"\n{'='*60}")
    print("STALL DEPTH BASELINES (lying pig median depth)")
    print(f"{'='*60}")
    for pid in sorted(baselines):
        b = baselines[pid]
        print(f"  pig {pid:>2}: median={b['median']:>7.1f}mm  std={b['std']:>6.1f}  "
              f"IQR=[{b['q25']:.0f}-{b['q75']:.0f}]  n={b['count']}")

    # Check if there's a clear pattern/gradient
    medians = [baselines[p]["median"] for p in sorted(baselines)]
    if len(medians) >= 2:
        gradient = medians[-1] - medians[0]
        print(f"\n  Depth gradient (pig 0 → pig {max(baselines)}): {gradient:+.0f}mm")
        if abs(gradient) > 100:
            print("  ^ Clear gradient detected — stalls get progressively closer/farther")
        else:
            print("  ^ Flat gradient — stalls are roughly equidistant from camera")

    # Step 2: Build folder profiles
    print("\nBuilding per-folder depth profiles...")
    profiles = build_folder_profiles(rows)
    print(f"  {len(profiles)} folders with lying pig data")

    # Step 3: Detect shifts
    print("\nDetecting ID shifts...")
    shifts = detect_shifts(baselines, profiles)

    if not shifts:
        print("\n  No shifted folders detected! All folders match baseline at offset 0.")
    else:
        print(f"\n{'='*60}")
        print(f"SHIFTED FOLDERS DETECTED: {len(shifts)}")
        print(f"{'='*60}")
        for s in shifts[:50]:
            print(f"\n  {s['folder']}:")
            print(f"    Best offset: {s['best_offset']} (physical pig {s['best_offset']} was first missing)")
            print(f"    Scores: {s['all_scores']}")
            print(f"    Lying pigs compared: {s['num_lying_pigs']}")

    # Save report
    out_dir = os.path.join(OUTPUT_DIR, "pig_audit")
    os.makedirs(out_dir, exist_ok=True)

    # Save baselines
    baseline_path = os.path.join(out_dir, "stall_depth_baselines.csv")
    with open(baseline_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pig_id", "median_depth", "std", "q25", "q75", "count"])
        for pid in sorted(baselines):
            b = baselines[pid]
            writer.writerow([pid, b["median"], round(b["std"], 1),
                             b["q25"], b["q75"], b["count"]])
    print(f"\nSaved baselines: {baseline_path}")

    # Save shift detections
    if shifts:
        shift_path = os.path.join(out_dir, "id_shift_detections.csv")
        with open(shift_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_folder", "detected_offset", "score_offset0",
                             "score_best", "num_lying_pigs"])
            for s in shifts:
                writer.writerow([s["folder"], s["best_offset"], s["score_offset0"],
                                 s["score_best"], s["num_lying_pigs"]])
        print(f"Saved shift detections: {shift_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
