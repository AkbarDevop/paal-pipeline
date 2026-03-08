"""Validate depth-based pig detection threshold.

Runs median-depth threshold on all frames and produces:
  1. CSV with all frame predictions + depth stats
  2. Edge cases shown on screen for visual check
  3. Histogram saved as PNG

Usage:
  python3 validate_pig_detection.py
  python3 validate_pig_detection.py --threshold 1463 --bar-margin 50
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from config import CROP_TOF, METADATA_CSV, LABEL_DIR, OUTPUT_DIR

TOF_W, TOF_H = 640, 480
THRESHOLD = 1463  # median depth in mm
BAR_MARGIN = 50

RESULTS_CSV = os.path.join(LABEL_DIR, "pig_detection_results.csv")


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


def compute_stats(depth_2d, crop, bar_margin):
    """Compute masked depth stats for cropped region."""
    x1, y1, x2, y2 = crop
    cropped = depth_2d[y1:y2, x1:x2]
    ch, cw = cropped.shape
    bar_mask = np.ones((ch, cw), dtype=bool)
    bar_mask[:, :bar_margin] = False
    bar_mask[:, cw - bar_margin:] = False
    valid = (cropped > 200) & (cropped < 5000) & bar_mask
    masked = cropped[valid]
    if masked.size == 0:
        return {"median": 9999.0, "mean": 9999.0, "std": 0.0,
                "iqr": 0.0, "valid_pct": 0.0}
    return {
        "median": float(np.median(masked)),
        "mean": float(masked.mean()),
        "std": float(masked.std()),
        "iqr": float(np.percentile(masked, 75) - np.percentile(masked, 25)),
        "valid_pct": float(valid.mean() * 100),
    }


def check_ground_truth(results, gt_csv):
    """Check predictions against ground truth labels."""
    if not os.path.exists(gt_csv):
        return None, None, None

    gt = {}
    with open(gt_csv) as f:
        for r in csv.DictReader(f):
            key = (r["timestamp_folder"], r["pig_id"])
            gt[key] = int(r["pig_present"])

    correct = 0
    total = 0
    mismatches = []
    for r in results:
        key = (r["timestamp_folder"], r["pig_id"])
        if key in gt:
            total += 1
            if int(r["pig_present"]) == gt[key]:
                correct += 1
            else:
                mismatches.append((r, gt[key]))

    return correct, total, mismatches


def main(args):
    if not os.path.exists(args.metadata_csv):
        print(f"Missing: {args.metadata_csv}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(args.metadata_csv) as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("depth_raw") and os.path.exists(r["depth_raw"])]

    print(f"Running pig detection on {len(rows)} frames...")
    print(f"  Threshold: {args.threshold}mm")
    print(f"  Bar margin: {args.bar_margin}px")
    print()

    # ── Classify all frames and collect stats ──
    results = []
    for r in rows:
        depth = load_depth_raw(r.get("depth_raw", ""))
        if depth is None:
            continue
        stats = compute_stats(depth, CROP_TOF, args.bar_margin)
        pig_present = 1 if stats["median"] < args.threshold else 0
        results.append({
            "timestamp_folder": r["timestamp_folder"],
            "pig_id": r["pig_id"],
            "pig_present": pig_present,
            "median_depth": f"{stats['median']:.1f}",
            "mean_depth": f"{stats['mean']:.1f}",
            "depth_std": f"{stats['std']:.1f}",
            "depth_iqr": f"{stats['iqr']:.1f}",
            "valid_pct": f"{stats['valid_pct']:.1f}",
            "threshold": args.threshold,
        })

    # ── Save results CSV ──
    fields = ["timestamp_folder", "pig_id", "pig_present",
              "median_depth", "mean_depth", "depth_std",
              "depth_iqr", "valid_pct", "threshold"]
    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    pig_results = [r for r in results if int(r["pig_present"]) == 1]
    nopig_results = [r for r in results if int(r["pig_present"]) == 0]

    # ── Summary ──
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total frames:  {len(results)}")
    print(f"Pig present:   {len(pig_results)}")
    print(f"No pig:        {len(nopig_results)}")
    print(f"Threshold:     {args.threshold}mm (median depth)")
    print(f"Bar margin:    {args.bar_margin}px")
    print()

    pig_medians = [float(r["median_depth"]) for r in pig_results]
    nopig_medians = [float(r["median_depth"]) for r in nopig_results]
    if pig_medians:
        print(f"Pig median range:    {min(pig_medians):.0f} - {max(pig_medians):.0f}mm")
    if nopig_medians:
        print(f"No-pig median range: {min(nopig_medians):.0f} - {max(nopig_medians):.0f}mm")
    if pig_medians and nopig_medians:
        gap = min(nopig_medians) - max(pig_medians)
        print(f"Gap:                 {gap:.0f}mm")
    print()

    print(f"Results saved: {args.output_csv}")
    print()

    # ── Ground truth check ──
    gt_csv = os.path.join(LABEL_DIR, "pig_detection_gt.csv")
    correct, total, mismatches = check_ground_truth(results, gt_csv)
    if total is not None:
        print(f"Ground truth accuracy: {correct}/{total} = {correct/total*100:.1f}%")
        if mismatches:
            print(f"  MISMATCHES ({len(mismatches)}):")
            for r, actual in mismatches:
                print(f"    pig{r['pig_id']} | {r['timestamp_folder']} | "
                      f"median={r['median_depth']}mm | predicted={'PIG' if int(r['pig_present']) else 'NO PIG'} | "
                      f"actual={'PIG' if actual else 'NO PIG'}")
        else:
            print("  No mismatches!")
    print()

    # ── Histogram (save + show) ──
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(pig_medians, bins=40, alpha=0.7, color="orange", label=f"Pig ({len(pig_results)})")
    ax.hist(nopig_medians, bins=20, alpha=0.7, color="blue", label=f"No pig ({len(nopig_results)})")
    ax.axvline(x=args.threshold, color="red", linestyle="--", linewidth=2,
               label=f"Threshold={args.threshold}mm")
    ax.set_title("Median Depth Distribution — All Frames", fontsize=14)
    ax.set_xlabel("Median Depth (mm)")
    ax.set_ylabel("Frame count")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    hist_path = os.path.join(OUTPUT_DIR, "pig_detection_histogram.png")
    fig.savefig(hist_path, dpi=150, bbox_inches="tight")
    print(f"Histogram saved: {hist_path}")
    plt.show()

    print("\nValidation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate pig detection threshold")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--bar-margin", type=int, default=BAR_MARGIN)
    parser.add_argument("--metadata-csv", default=METADATA_CSV)
    parser.add_argument("--output-csv", default=RESULTS_CSV)
    args = parser.parse_args()
    main(args)
