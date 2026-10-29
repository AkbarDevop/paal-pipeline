#!/usr/bin/env python3
r"""Split a labels CSV by sow (stall) into a test part and a training part, so the test
sows are never seen in training.

Writes labels_test.csv and labels_train.csv next to the input (same columns, so the
relative image and mask paths still work) and prints which sows went where.

Usage:
    python vulva/split_by_sow.py C:\Users\aakdbx\vulva_jan_set\labels.csv
    python vulva/split_by_sow.py <labels.csv> --test-fraction 0.5 --seed 42
    python vulva/split_by_sow.py <labels.csv> --test-sows 1,4,7,10
"""

from __future__ import annotations

import argparse
import csv
import os
import random


def split(rows, test_fraction=0.5, seed=42, test_sows=None):
    sows = sorted({int(float(r["pig_id"])) for r in rows})
    if test_sows is None:
        order = sows[:]
        random.Random(seed).shuffle(order)
        test_sows = set(order[:max(1, round(test_fraction * len(sows)))])
    test = [r for r in rows if int(float(r["pig_id"])) in test_sows]
    train = [r for r in rows if int(float(r["pig_id"])) not in test_sows]
    return test, train, sorted(test_sows), sorted(set(sows) - set(test_sows))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels")
    ap.add_argument("--test-fraction", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-sows", default="", help="comma list of pig ids for the test part (overrides the draw)")
    args = ap.parse_args()
    with open(args.labels, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    chosen = {int(v) for v in args.test_sows.split(",") if v.strip()} or None
    test, train, ts, tr = split(rows, args.test_fraction, args.seed, chosen)
    base = os.path.dirname(os.path.abspath(args.labels))
    for name, part in (("labels_test.csv", test), ("labels_train.csv", train)):
        with open(os.path.join(base, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(part)
    for name, part, sw in (("test", test, ts), ("train", train, tr)):
        pres = sum(r["label"] == "present" for r in part)
        print(f"{name}: sows {sw} -> {len(part)} frames ({pres} visible) in labels_{name}.csv")


if __name__ == "__main__":
    main()
