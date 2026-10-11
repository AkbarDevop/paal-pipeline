#!/usr/bin/env python3
"""Fix pig IDs in predictions.csv to match file renames.

Only needed for predictions.csv files made before v1.1 — run_inference.py
now applies these corrections itself (see id_corrections.py). Not idempotent
on old CSVs either: run it once per file.

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


# Single source of truth for the correction rules (shared with run_inference.py)
from id_corrections import CORRECTIONS  # noqa: E402


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

    if "id_status" in (fields or []):
        print("This CSV was written by run_inference.py v1.1+, which already applied the "
              "corrections. Running them again would delete the real pig 19 rows. Nothing changed.")
        return

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
