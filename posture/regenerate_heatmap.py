#!/usr/bin/env python3
"""Regenerate heatmaps from existing predictions.csv without re-running inference.

Saves ~30 minutes by skipping crop + predict steps. Just reads the CSV
and rebuilds heatmaps. Run this after fixing pig IDs in the data and
re-running inference, or to tweak heatmap settings.

Usage:
    python regenerate_heatmap.py                              # from default outputs/predictions.csv
    python regenerate_heatmap.py --csv path/to/predictions.csv
    python regenerate_heatmap.py --no-normalize               # show raw pig IDs (no modulo)
"""

import argparse
import csv
import os
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
except ImportError:
    print("matplotlib required: pip install matplotlib")
    sys.exit(1)

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import OUTPUT_DIR, POSTURE3_CLASSES
from datetime import datetime, timedelta
from collections import defaultdict


def parse_timestamp(ts_str):
    try:
        return datetime.strptime(ts_str, "%Y%m%d-%H-%M-%S")
    except ValueError:
        return None


def load_csv(csv_path):
    rows = []
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            row["pig_id"] = int(row["pig_id"])
            row["confidence"] = float(row["confidence"])
            rows.append(row)
    return rows


def generate_heatmap(results, out_path, title="Posture Timeline Heatmap",
                     normalize=True, fixed_range=None):
    """Generate heatmap from prediction rows."""
    if normalize:
        for r in results:
            r["pig_id"] = r["pig_id"] % 20

    all_pids = sorted(set(r["pig_id"] for r in results))
    if normalize:
        pig_ids = list(range(20))
    else:
        pig_ids = list(range(max(all_pids) + 1))

    timestamps = [(r, parse_timestamp(r["pig_timestamp"])) for r in results]
    valid = [(r, t) for r, t in timestamps if t is not None]
    if not valid:
        print("No valid timestamps")
        return

    min_t = min(t for _, t in valid)
    max_t = max(t for _, t in valid)
    hours = []
    t = min_t.replace(minute=0, second=0)
    while t <= max_t:
        hours.append(t)
        t += timedelta(hours=1)

    posture_map = {"standing": 0, "sitting": 1, "lying": 2}
    votes = defaultdict(list)
    for r, t in valid:
        hi = int((t - hours[0]).total_seconds() / 3600)
        hi = min(hi, len(hours) - 1)
        votes[(r["pig_id"], hi)].append(posture_map.get(r["prediction_name"], -1))

    pig_idx = {pid: i for i, pid in enumerate(pig_ids)}
    grid = np.full((len(pig_ids), len(hours)), -1, dtype=float)
    for (pid, hi), postures in votes.items():
        if pid in pig_idx:
            grid[pig_idx[pid], hi] = max(set(postures), key=postures.count)

    colors = ["#d4d4d4", "#22c55e", "#f97316", "#3b82f6"]
    cmap = ListedColormap(colors)
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(max(14, len(hours) * 0.3), max(8, len(pig_ids) * 0.4)))
    ax.pcolormesh(grid, cmap=cmap, norm=norm, edgecolors="white", linewidth=0.5)

    labels = []
    for pid in pig_ids:
        if not normalize and pid >= 20:
            labels.append(f"pig {pid} (overflow)")
        else:
            labels.append(f"pig {pid}")
    ax.set_yticks(np.arange(len(pig_ids)) + 0.5)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_ylim(0, len(pig_ids))

    if not normalize and max(pig_ids) >= 20:
        ax.axhline(y=20, color="red", linewidth=2, linestyle="--")

    step = max(1, 6)
    ax.set_xticks(np.arange(0, len(hours), step) + 0.5)
    ax.set_xticklabels(
        [hours[i].strftime("%m/%d %H:%M") for i in range(0, len(hours), step)],
        rotation=45, ha="right", fontsize=7,
    )
    ax.set_xlim(0, len(hours))

    ax.set_xlabel("Time")
    ax.set_ylabel("Pig ID")
    ax.set_title(title)

    legend_elements = [
        Patch(facecolor="#22c55e", label="Standing"),
        Patch(facecolor="#f97316", label="Sitting"),
        Patch(facecolor="#3b82f6", label="Lying"),
        Patch(facecolor="#d4d4d4", label="No data"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Regenerate heatmaps from predictions CSV")
    parser.add_argument("--csv", default=os.path.join(OUTPUT_DIR, "predictions.csv"),
                        help="Path to predictions CSV")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Show raw pig IDs without modulo normalization")
    parser.add_argument("--output", default=None, help="Output PNG path")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: {args.csv} not found")
        return

    print(f"Loading {args.csv}...")
    rows = load_csv(args.csv)
    print(f"  {len(rows)} frames loaded")

    # Always generate both heatmaps
    out_norm = args.output or os.path.join(OUTPUT_DIR, "posture_heatmap.png")
    generate_heatmap(list(rows), out_norm, normalize=True)

    out_raw = os.path.join(OUTPUT_DIR, "posture_heatmap_raw_ids.png")
    rows_raw = load_csv(args.csv)
    title_raw = "Posture Heatmap — RAW Pig IDs (overflow visible above red line)"
    generate_heatmap(rows_raw, out_raw, title=title_raw, normalize=False)


if __name__ == "__main__":
    main()
