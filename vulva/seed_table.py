#!/usr/bin/env python3
r"""One table from many eval_seg.py runs: each recipe trained with several seeds, scored on each test.

Why: a single training run per recipe cannot show whether a 0.01-0.02 mAP difference is real; the
same recipe trained with another seed moves by about that much. So every comparison is made on the
mean over seeds, and the spread over seeds (max - min) is the noise floor (plan E1b).

Folder names say what each eval is, as written by the E1b / E10 run:

    <dir>/eval_<recipe>_s<seed>_<split>/eval_report.txt      e.g. eval_ir_s1_test_newsows

    python vulva/seed_table.py outputs\seeds                  # all recipes, against "ir"
    python vulva/seed_table.py outputs\seeds --ref ir --out outputs\seeds\seed_table.csv

Prints, per test split and recipe: the seeds, mAP50-95 and mAP50 per seed, their mean and spread,
and the difference of means against its baseline next to the larger of the two spreads. The baseline
is the reference recipe (--ref), except for a recipe started from a pretrained model, pre_<X>, which
is compared with <X> (pre_ir with ir, pre_fusion with fusion). Writes the per-seed rows to
seed_table.csv; --copy-to also copies the table and every eval report there (e.g. the Mac share).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
from collections import defaultdict

NAME = re.compile(r"^eval_(?P<recipe>.+?)_s(?P<seed>\d+)_(?P<split>.+)$")
MAP = re.compile(r"mAP50 ([\d.]+), mAP50-95 ([\d.]+)")
AP75 = re.compile(r"AP at IoU 0\.75 ([\d.]+)")
IOU = re.compile(r"IoU mean ([\d.]+)")
FIELDS = ["recipe", "seed", "split", "map50", "map50_95", "ap75", "iou_mean", "report"]


def read_report(path):
    """eval_report.txt -> {map50, map50_95, ap75, iou_mean} (None where a line is missing)."""
    text = open(path, encoding="utf-8", errors="replace").read()
    m, a, i = MAP.search(text), AP75.search(text), IOU.search(text)
    return {"map50": float(m.group(1)) if m else None, "map50_95": float(m.group(2)) if m else None,
            "ap75": float(a.group(1)) if a else None, "iou_mean": float(i.group(1)) if i else None}


def collect(root):
    rows = []
    for name in sorted(os.listdir(root)):
        m = NAME.match(name)
        rep = os.path.join(root, name, "eval_report.txt")
        if not m or not os.path.isfile(rep):
            continue
        r = read_report(rep)
        if r["map50_95"] is None:
            print(f"  {name}: no mAP line in eval_report.txt, left out", file=sys.stderr)
            continue
        rows.append({"recipe": m["recipe"], "seed": int(m["seed"]), "split": m["split"], **r, "report": rep})
    return rows


def baseline_of(recipe, recs, ref):
    """pre_<X> -> <X> when <X> was run, else the reference recipe; the reference itself -> None."""
    if recipe.startswith("pre_") and recipe[4:] in recs:
        return recipe[4:]
    return ref if recipe != ref and ref in recs else None


def summarise(rows, ref):
    """-> {split: {recipe: {seeds, map50_95 [..], map50 [..], mean / spread of both}}}"""
    out = defaultdict(dict)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["split"], r["recipe"])].append(r)
    for (split, recipe), rs in groups.items():
        rs.sort(key=lambda r: r["seed"])
        s = {"seeds": [r["seed"] for r in rs]}
        for k in ("map50_95", "map50"):
            v = [r[k] for r in rs]
            s[k] = v
            s[k + "_mean"] = sum(v) / len(v)
            s[k + "_spread"] = max(v) - min(v)
        out[split][recipe] = s
    for split, recs in out.items():
        for recipe, s in recs.items():
            name = baseline_of(recipe, recs, ref)
            if name is None:
                continue
            base = recs[name]
            s["vs"] = name
            for k in ("map50_95", "map50"):
                s[k + "_diff"] = s[k + "_mean"] - base[k + "_mean"]
                s[k + "_noise"] = max(s[k + "_spread"], base[k + "_spread"])
    return out


def table_text(summary):
    lines = []
    for split in sorted(summary):
        lines.append(f"\n{split}")
        lines.append(f"  {'recipe':<12} {'seeds':<12} {'mAP50-95 per seed':<26} {'mean':>6} {'spread':>7}"
                     f"   {'mAP50 mean':>10} {'spread':>7}   difference of means")
        recs = summary[split]
        for recipe in sorted(recs, key=lambda r: ("vs" in recs[r], r)):
            s = recs[recipe]
            per = " ".join(f"{v:.3f}" for v in s["map50_95"])
            line = (f"  {recipe:<12} {','.join(map(str, s['seeds'])):<12} {per:<26} {s['map50_95_mean']:6.3f}"
                    f" {s['map50_95_spread']:7.3f}   {s['map50_mean']:10.3f} {s['map50_spread']:7.3f}")
            if "vs" in s:
                line += (f"   vs {s['vs']}: mAP50-95 {s['map50_95_diff']:+.3f} (noise {s['map50_95_noise']:.3f}),"
                         f" mAP50 {s['map50_diff']:+.3f} (noise {s['map50_noise']:.3f})")
            lines.append(line)
    lines.append("\nnoise = the larger of the two recipes' spreads over seeds (max - min)")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="folder holding eval_<recipe>_s<seed>_<split> folders")
    ap.add_argument("--ref", default="ir", help="recipe the others are compared with (default ir)")
    ap.add_argument("--out", default="", help="per-seed CSV (default <root>/seed_table.csv)")
    ap.add_argument("--copy-to", default="", help="also copy the table and the eval reports here "
                    "(e.g. \\\\tsclient\\paal_exchange\\seeds)")
    args = ap.parse_args()
    rows = collect(args.root)
    if not rows:
        sys.exit(f"no eval_<recipe>_s<seed>_<split>/eval_report.txt under {args.root}")
    summary = summarise(rows, args.ref)
    text = table_text(summary)
    print(text)
    out = args.out or os.path.join(args.root, "seed_table.csv")
    with open(os.path.splitext(out)[0] + ".txt", "w") as f:
        f.write(text.lstrip("\n") + "\n")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} evals -> {out}")
    if args.copy_to:
        copy_out(rows, out, args.copy_to)
    return summary


def copy_out(rows, table, dest):
    """Plain read/write copies (Windows CopyFile, used by PowerShell copy, is refused on \\\\tsclient shares).
    A failed copy only prints a warning: the local files stay."""
    pairs = [(table, os.path.basename(table))]
    txt = os.path.splitext(table)[0] + ".txt"
    if os.path.exists(txt):
        pairs.append((txt, os.path.basename(txt)))
    for r in rows:
        d = os.path.dirname(r["report"])
        for f in ("eval_report.txt", "eval_frames.csv"):
            pairs.append((os.path.join(d, f), os.path.join(os.path.basename(d), f)))
    n = 0
    for src, rel in pairs:
        if not os.path.exists(src):
            continue
        try:
            os.makedirs(os.path.dirname(os.path.join(dest, rel)) or dest, exist_ok=True)
            with open(src, "rb") as fi, open(os.path.join(dest, rel), "wb") as fo:
                shutil.copyfileobj(fi, fo)
            n += 1
        except OSError as e:
            print(f"Could not copy {rel} to {dest}: {e}")
    print(f"Copied {n} files to {dest}")


if __name__ == "__main__":
    main()
