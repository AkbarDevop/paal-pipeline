#!/usr/bin/env python3
"""Fix pig IDs in predictions.csv to match file renames.

Applies the same corrections as fix_pig_ids.py but to the CSV,
so you don't need to re-run 30 min of inference.

Usage:
    python fix_predictions_csv.py
    python fix_predictions_csv.py --csv path/to/predictions.csv
"""

import argparse
import csv
import os

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import OUTPUT_DIR


# Same rules as fix_pig_ids.py
CORRECTIONS = [
    {
        "name": "Camera magnet issue — pig19 duplication (02/13)",
        "start": "20260213-13-41-49",
        "end": "20260213-19-22-53",
        "delete_pids": [19],       # pig19 = dup of pig18 → delete
        "rename": {20: 19},        # pig20 = real pig19 → rename
        "delete_gte": 21,          # pig21+ = duplicates → delete
    },
    {
        "name": "Camera counter overflow (02/17)",
        "start": "20260217-06-01-04",
        "end": "20260217-06-01-04",
        "delete_pids": [19],       # pig19 dup
        "rename": {20: 19},        # pig20 → pig19
        "delete_gte": 21,
    },
    {
        "name": "Camera counter overflow — morning (02/18)",
        "start": "20260218-10-54-39",
        "end": "20260218-11-15-52",
        "delete_pids": [19, 20],   # pig19 + pig20 dups
        "rename": {21: 19},        # pig21 → pig19
        "delete_gte": 22,
    },
    {
        "name": "Camera counter overflow — afternoon (02/18)",
        "start": "20260218-15-33-04",
        "end": "20260218-19-00-39",
        "delete_pids": [19],       # pig19 = dup of pig18 → delete
        "rename": {20: 19},        # pig20 = real pig19 → rename
        "delete_gte": 21,          # pig21+ = duplicates → delete
    },
    {
        "name": "Camera counter overflow — pig19=pig17, pig20=pig18 (02/19)",
        "start": "20260219-15-09-15",
        "end": "20260219-17-58-36",
        "delete_pids": [19, 20],   # pig19=pig17 dup, pig20=pig18 dup
        "rename": {21: 19},        # pig21 → pig19 (real pig19)
        "delete_gte": 22,          # remove pig22+ rows (extra duplicates)
    },
    {
        "name": "Camera counter overflow (02/20)",
        "start": "20260220-16-36-37",
        "end": "20260220-16-36-37",
        "delete_pids": [],
        "rename": {},
        "delete_gte": 20,          # delete pig20+ rows
    },
    {
        "name": "Camera counter overflow (03/08)",
        "start": "20260308-05-39-19",
        "end": "20260308-05-39-19",
        "delete_pids": [],
        "rename": {},
        "delete_gte": 20,
    },
    {
        "name": "Camera counter overflow (03/10-03/11)",
        "start": "20260310-17-43-30",
        "end": "20260311-14-11-47",
        "delete_pids": [],
        "rename": {},
        "delete_gte": 20,
    },
]


def main():
    parser = argparse.ArgumentParser(description="Fix pig IDs in predictions CSV")
    parser.add_argument("--csv", default=os.path.join(OUTPUT_DIR, "predictions.csv"))
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: {args.csv} not found")
        return

    # Read all rows
    with open(args.csv, "r") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)

    print(f"Loaded {len(rows)} rows from {args.csv}")

    total_deleted = 0
    total_renamed = 0

    fixed_rows = []
    for row in rows:
        folder = row["timestamp_folder"]
        pid = int(row["pig_id"])
        keep = True

        for rule in CORRECTIONS:
            if folder < rule["start"] or folder > rule["end"]:
                continue

            # Delete specific pig IDs
            if pid in rule["delete_pids"]:
                keep = False
                total_deleted += 1
                break

            # Delete pig IDs >= threshold
            if pid >= rule["delete_gte"]:
                keep = False
                total_deleted += 1
                break

            # Rename pig IDs
            if pid in rule["rename"]:
                row["pig_id"] = str(rule["rename"][pid])
                total_renamed += 1

        if keep:
            fixed_rows.append(row)

    # Write back
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(fixed_rows)

    print(f"Deleted {total_deleted} rows, renamed {total_renamed} pig IDs")
    print(f"Saved {len(fixed_rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
