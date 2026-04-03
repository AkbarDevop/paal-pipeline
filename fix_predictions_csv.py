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

from config import OUTPUT_DIR


# Same rules as fix_pig_ids.py
CORRECTIONS = [
    {
        "name": "Camera magnet issue — pig19 duplication",
        "start": "20260213-13-41-49",
        "end": "20260213-19-01-34",
        "delete_pids": [19],       # remove pig19 rows (not real pig19)
        "rename": {20: 19},        # pig20 → pig19
        "delete_gte": 21,          # remove pig21+ rows (duplicates)
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
