"""Normalize legacy posture3 labels to strict 3-class labels.

Input legacy labels may include:
  0=no_pig, 1=standing, 2=sitting, 3=lying

Output strict posture3:
  0=standing, 1=sitting, 2=lying

Rows with label 0 (no_pig/ambiguous) are dropped.
"""

import argparse
import csv
import os
from collections import Counter

from config import LABELS_POSTURE3_CSV


def main(args):
    if not os.path.exists(args.input_csv):
        print(f"Missing input: {args.input_csv}")
        return

    with open(args.input_csv, "r") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    out_rows = []
    in_counts = Counter()
    out_counts = Counter()

    for r in rows:
        lab = (r.get("label") or "").strip()
        if lab == "":
            continue
        in_counts[lab] += 1
        if lab == "1":
            r["label"] = "0"  # standing
        elif lab == "2":
            r["label"] = "1"  # sitting
        elif lab == "3":
            r["label"] = "2"  # lying
        else:
            continue  # drop legacy no_pig/unknown
        out_rows.append(r)
        out_counts[r["label"]] += 1

    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    print(f"Input:  {args.input_csv}")
    print(f"Output: {args.output_csv}")
    print(f"Input label counts:  {dict(sorted(in_counts.items(), key=lambda x:int(x[0])))}")
    print(f"Output label counts: {dict(sorted(out_counts.items(), key=lambda x:int(x[0])))}")
    print(f"Rows kept: {len(out_rows)} / {len(rows)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize posture3 labels to strict 3-class schema")
    parser.add_argument("--input-csv", default=LABELS_POSTURE3_CSV)
    parser.add_argument("--output-csv", default="labels/labels_posture3_clean.csv")
    main(parser.parse_args())
