"""Copy rgb_aligned_jpg paths from metadata.csv into labels.csv."""

import csv
import os

from config import LABELS_CSV, METADATA_CSV


def update_labels():
    if not os.path.exists(LABELS_CSV):
        print("No labels.csv found.")
        return
    if not os.path.exists(METADATA_CSV):
        print("No metadata.csv found.")
        return

    lookup = {}
    with open(METADATA_CSV, "r") as f:
        for row in csv.DictReader(f):
            lookup[(row["timestamp_folder"], row["pig_id"])] = row.get("rgb_aligned_jpg", "")

    with open(LABELS_CSV, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = list(reader.fieldnames or [])

    if "rgb_aligned_jpg" not in fields:
        idx = fields.index("rgb_jpg") + 1 if "rgb_jpg" in fields else len(fields)
        fields.insert(idx, "rgb_aligned_jpg")

    updated = 0
    for r in rows:
        v = lookup.get((r.get("timestamp_folder", ""), r.get("pig_id", "")), "")
        r["rgb_aligned_jpg"] = v
        if v:
            updated += 1

    with open(LABELS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"Updated {updated}/{len(rows)} rows with rgb_aligned_jpg.")
    print(f"Saved: {LABELS_CSV}")


if __name__ == "__main__":
    update_labels()
