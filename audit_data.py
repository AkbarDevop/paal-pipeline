#!/usr/bin/env python3
"""Audit OAK-D data folders — detect missing pigs and file count anomalies.

Scans each timestamp folder and reports:
- Total file count per folder
- Which pig IDs are present
- Which pig IDs are missing (if < 20)
- File type breakdown per pig

Usage:
    python audit_data.py "C:\PAAL_data\Fed_pig\Pictures_OAk"
    python audit_data.py /path/to/data
"""

import argparse
import csv
import os
import re
from collections import Counter, defaultdict

FOLDER_RE = re.compile(r"^\d{8}-\d{2}-\d{2}-\d{2}$")
FILENAME_RE = re.compile(
    r"^pig(\d+)_(depth_vis|depth|ir_vis|ir|rgb_aligned|rgb)_"
    r"(\d{8}-\d{2}-\d{2}-\d{2})(_cropped)?\.(jpg|raw)$"
)

ALL_PIGS = set(range(20))


def audit(data_dir, output_csv=None):
    folders = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and FOLDER_RE.match(d)
    )
    print(f"Scanning {len(folders)} timestamp folders in {data_dir}\n")

    # Track stats
    file_count_dist = Counter()  # file_count -> num_folders
    missing_pig_tracker = defaultdict(list)  # pig_id -> [folders where missing]
    rows = []

    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        all_files = os.listdir(folder_path)

        # Count total files (excluding cropped)
        original_files = [f for f in all_files if "_cropped" not in f]
        total = len(original_files)

        # Parse pig IDs from filenames
        pig_ids = set()
        files_per_pig = defaultdict(int)
        modalities_per_pig = defaultdict(set)

        for fname in original_files:
            m = FILENAME_RE.match(fname)
            if m:
                pid = int(m.group(1))
                modality = m.group(2)
                pig_ids.add(pid)
                files_per_pig[pid] += 1
                modalities_per_pig[pid].add(modality)

        # Detect missing pigs (only check 0-19 range)
        max_expected = max(pig_ids) + 1 if pig_ids else 20
        expected = set(range(min(20, max_expected)))
        missing = sorted(expected - pig_ids)
        overflow = sorted(pid for pid in pig_ids if pid >= 20)

        file_count_dist[total] += 1

        for pid in missing:
            missing_pig_tracker[pid].append(folder)

        rows.append({
            "timestamp_folder": folder,
            "total_files": total,
            "num_pigs": len(pig_ids),
            "pig_ids": sorted(pig_ids),
            "missing_pigs": missing,
            "overflow_pigs": overflow,
            "files_per_pig": dict(files_per_pig),
        })

    # ── Report ──────────────────────────────────────────────────────────

    # Determine "normal" file count (most common)
    normal_count = file_count_dist.most_common(1)[0][0]

    print("=" * 60)
    print("FILE COUNT DISTRIBUTION (how many folders have N files)")
    print("=" * 60)
    for count in sorted(file_count_dist):
        marker = " <-- expected" if count == normal_count else ""
        print(f"  {count:>4} files: {file_count_dist[count]:>5} folders{marker}")

    # Abnormal file count folders
    abnormal_count = [r for r in rows if r["total_files"] != normal_count]
    print(f"\n{'=' * 60}")
    print(f"FOLDERS WITH ABNORMAL FILE COUNT (!= {normal_count})")
    print(f"{'=' * 60}")
    if not abnormal_count:
        print(f"  All {len(rows)} folders have exactly {normal_count} files.")
    else:
        print(f"  {'Folder':<25} {'Files':>6} {'Diff':>6}  Details")
        print(f"  {'-'*25} {'-'*6} {'-'*6}  {'-'*30}")
        for r in abnormal_count:
            diff = r["total_files"] - normal_count
            sign = "+" if diff > 0 else ""
            missing_str = f"missing pigs: {r['missing_pigs']}" if r["missing_pigs"] else ""
            overflow_str = f"overflow IDs: {r['overflow_pigs']}" if r["overflow_pigs"] else ""
            extra = " | ".join(filter(None, [missing_str, overflow_str]))
            print(f"  {r['timestamp_folder']:<25} {r['total_files']:>6} {sign}{diff:>5}  {extra}")
        print(f"\n  Total abnormal: {len(abnormal_count)} / {len(rows)} folders")

    print(f"\n{'=' * 60}")
    print("PIG COUNT DISTRIBUTION (how many pigs per folder)")
    print("=" * 60)
    pig_count_dist = Counter(r["num_pigs"] for r in rows)
    for count in sorted(pig_count_dist):
        print(f"  {count:>2} pigs: {pig_count_dist[count]:>5} folders")

    print(f"\n{'=' * 60}")
    print("MISSING PIG SUMMARY")
    print("=" * 60)
    if not missing_pig_tracker:
        print("  No missing pigs detected.")
    else:
        for pid in sorted(missing_pig_tracker):
            folders_missing = missing_pig_tracker[pid]
            if len(folders_missing) <= 5:
                print(f"  pig {pid:>2}: missing in {len(folders_missing)} folders — {folders_missing}")
            else:
                print(f"  pig {pid:>2}: missing in {len(folders_missing)} folders — {folders_missing[0]} ... {folders_missing[-1]}")

    # Show folders with abnormal pig counts
    print(f"\n{'=' * 60}")
    print("FOLDERS WITH != 20 PIGS (first 50)")
    print("=" * 60)
    abnormal = [r for r in rows if r["num_pigs"] != 20]
    for r in abnormal[:50]:
        missing_str = f"missing: {r['missing_pigs']}" if r["missing_pigs"] else ""
        overflow_str = f"overflow: {r['overflow_pigs']}" if r["overflow_pigs"] else ""
        extra = " | ".join(filter(None, [missing_str, overflow_str]))
        print(f"  {r['timestamp_folder']}: {r['num_pigs']} pigs, {r['total_files']} files | {extra}")

    if len(abnormal) > 50:
        print(f"  ... and {len(abnormal) - 50} more")

    print(f"\n  Total abnormal: {len(abnormal)} / {len(rows)} folders")

    # ── Save detailed CSV ───────────────────────────────────────────────
    if output_csv:
        fields = ["timestamp_folder", "total_files", "num_pigs", "missing_pigs", "overflow_pigs"]
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in rows:
                writer.writerow({
                    "timestamp_folder": r["timestamp_folder"],
                    "total_files": r["total_files"],
                    "num_pigs": r["num_pigs"],
                    "missing_pigs": str(r["missing_pigs"]),
                    "overflow_pigs": str(r["overflow_pigs"]),
                })
        print(f"\nSaved audit CSV: {output_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Audit OAK-D data folders for missing pigs")
    p.add_argument("data_dir", help="Path to OAK-D data folder")
    p.add_argument("--output", default=None, help="Save detailed audit to CSV")
    args = p.parse_args()
    audit(args.data_dir, args.output)
