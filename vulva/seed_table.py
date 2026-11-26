#!/usr/bin/env python3
r"""One table from many eval_seg.py runs: each recipe trained with several seeds, scored on each test.

Why: a single training run per recipe cannot show whether a 0.01-0.02 mAP difference is real; the
same recipe trained with another seed moves by about that much. So every comparison is made on the
mean over seeds, and the spread over seeds (max - min) is the noise floor (plan E1b).

Folder names say what each eval is, as written by the E1b / E10 run:

    <dir>/eval_<recipe>_s<seed>_<split>/eval_report.txt      e.g. eval_ir_s1_test_newsows
    <dir>/eval_<recipe>_sens_<split>/eval_report.txt         the recipe's seeds as one ensemble (eval_seg.py
                                                             with several --model), shown next to the seeds

    python vulva/seed_table.py outputs\seeds                  # all recipes, against "ir"
    python vulva/seed_table.py outputs\seeds --ref ir --out outputs\seeds\seed_table.csv

Prints, per test split and recipe: the seeds, mAP50-95 and mAP50 per seed, their mean and spread,
and the difference of means against its baseline next to the larger of the two spreads. The baseline
is the reference recipe (--ref), except for a recipe started from a pretrained model, pre_<X>, which
is compared with <X> (pre_ir with ir, pre_fusion with fusion), and so are <X>_m (mirror averaged) and
<X>_v2 (the same recipe trained on dataset v2). Writes the per-seed rows to
seed_table.csv; --copy-to also copies the table and every eval report there (e.g. the Mac share).
Below the table: outline size against the hand outlines (seed means), and, from each eval's
eval_frames.csv, which visible vulvas every seed finds, some find and none finds, and the frames that
no model at all finds.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
from collections import defaultdict

NAME = re.compile(r"^eval_(?P<recipe>.+?)_s(?P<seed>\d+|ens)_(?P<split>.+)$")
MAP = re.compile(r"mAP50 ([\d.]+), mAP50-95 ([\d.]+)")
AP75 = re.compile(r"AP at IoU 0\.75 ([\d.]+)")
IOU = re.compile(r"IoU mean ([\d.]+)")
SIZE = re.compile(r"^(Width|Length) \(model minus hand.*?bias (-?[\d.]+) mm.*?hand median ([\d.]+) mm", re.M)
FIELDS = ["recipe", "seed", "split", "map50", "map50_95", "ap75", "iou_mean", "width_bias", "length_bias",
          "hand_width", "hand_length", "report"]


def read_report(path):
    """eval_report.txt -> {map50, map50_95, ap75, iou_mean, width_bias, length_bias (model minus hand, mm),
    hand_width, hand_length (median mm)} (None where a line is missing)."""
    text = open(path, encoding="utf-8", errors="replace").read()
    m, a, i = MAP.search(text), AP75.search(text), IOU.search(text)
    out = {"map50": float(m.group(1)) if m else None, "map50_95": float(m.group(2)) if m else None,
           "ap75": float(a.group(1)) if a else None, "iou_mean": float(i.group(1)) if i else None,
           "width_bias": None, "length_bias": None, "hand_width": None, "hand_length": None}
    for what, bias, hand in SIZE.findall(text):
        out[f"{what.lower()}_bias"], out[f"hand_{what.lower()}"] = float(bias), float(hand)
    return out


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
        seed = m["seed"] if m["seed"] == "ens" else int(m["seed"])
        rows.append({"recipe": m["recipe"], "seed": seed, "split": m["split"], **r, "report": rep})
    return rows


def baseline_of(recipe, recs, ref):
    """pre_<X> (started from a pretrained model) and <X>_m (mirror averaged) -> <X> when <X> was run,
    else the reference recipe; the reference itself -> None."""
    if recipe.startswith("pre_") and recipe[4:] in recs:
        return recipe[4:]
    if recipe.endswith("_m") and recipe[:-2] in recs:
        return recipe[:-2]
    if recipe.endswith("_v2") and recipe[:-3] in recs:              # same recipe on the next dataset version
        return recipe[:-3]
    return ref if recipe != ref and ref in recs else None


def summarise(rows, ref):
    """-> {split: {recipe: {seeds, map50_95 [..], map50 [..], mean / spread of both, size means, ens}}}"""
    out = defaultdict(dict)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["split"], r["recipe"])].append(r)
    for (split, recipe), rs in groups.items():
        seeds = sorted((r for r in rs if r["seed"] != "ens"), key=lambda r: r["seed"])
        ens = [r for r in rs if r["seed"] == "ens"]
        s = {"seeds": [r["seed"] for r in seeds]}
        if ens:
            s["ens"] = {k: ens[0][k] for k in ("map50_95", "map50", "iou_mean", "width_bias", "length_bias")}
        if seeds:
            for k in ("map50_95", "map50"):
                v = [r[k] for r in seeds]
                s[k] = v
                s[k + "_mean"] = sum(v) / len(v)
                s[k + "_spread"] = max(v) - min(v)
            for k in ("iou_mean", "width_bias", "length_bias", "hand_width", "hand_length"):
                v = [r[k] for r in seeds if r[k] is not None]
                s[k + "_mean"] = sum(v) / len(v) if v else None
        out[split][recipe] = s
    for split, recs in out.items():
        for recipe, s in recs.items():
            name = baseline_of(recipe, recs, ref)
            if name is None or not s["seeds"] or not recs[name]["seeds"]:
                continue
            base = recs[name]
            s["vs"] = name
            for k in ("map50_95", "map50"):
                s[k + "_diff"] = s[k + "_mean"] - base[k + "_mean"]
                s[k + "_noise"] = max(s[k + "_spread"], base[k + "_spread"])
    return out


def _f(v, fmt):
    return "-" if v is None else format(v, fmt)


def table_text(summary, agreement=None):
    lines = []
    for split in sorted(summary):
        recs = summary[split]
        order = sorted(recs, key=lambda r: ("vs" in recs[r], r))
        lines.append(f"\n{split}")
        lines.append(f"  {'recipe':<12} {'seeds':<12} {'mAP50-95 per seed':<26} {'mean':>6} {'spread':>7}"
                     f"   {'mAP50 mean':>10} {'spread':>7}   {'ensemble':>13}   difference of means")
        for recipe in order:
            s = recs[recipe]
            e = s.get("ens")
            ens = f"{e['map50_95']:.3f} / {e['map50']:.3f}" if e else "-"
            if s["seeds"]:
                per = " ".join(f"{v:.3f}" for v in s["map50_95"])
                line = (f"  {recipe:<12} {','.join(map(str, s['seeds'])):<12} {per:<26} {s['map50_95_mean']:6.3f}"
                        f" {s['map50_95_spread']:7.3f}   {s['map50_mean']:10.3f} {s['map50_spread']:7.3f}   {ens:>13}")
            else:
                line = f"  {recipe:<12} {'-':<12} {'-':<26} {'-':>6} {'-':>7}   {'-':>10} {'-':>7}   {ens:>13}"
            if "vs" in s:
                line += (f"   vs {s['vs']}: mAP50-95 {s['map50_95_diff']:+.3f} (noise {s['map50_95_noise']:.3f}),"
                         f" mAP50 {s['map50_diff']:+.3f} (noise {s['map50_noise']:.3f})")
            lines.append(line)
        hands = [recs[r] for r in order if recs[r].get("hand_width_mean") is not None]
        if hands:
            h = hands[0]
            lines.append(f"  outline against the hand outlines (seed means; hand median {_f(h['hand_width_mean'], '.1f')} x "
                         f"{_f(h['hand_length_mean'], '.1f')} mm):")
            for recipe in order:
                s = recs[recipe]
                if s["seeds"]:
                    lines.append(f"    {recipe:<12} IoU {_f(s['iou_mean_mean'], '.3f')}   width {_f(s['width_bias_mean'], '+.1f')} mm"
                                 f"   length {_f(s['length_bias_mean'], '+.1f')} mm")
        if agreement and split in agreement:
            lines.extend(agreement[split])
    lines.append("\nnoise = the larger of the two recipes' spreads over seeds (max - min); "
                 "ensemble = mAP50-95 / mAP50 of the recipe's seeds averaged into one model")
    return "\n".join(lines)


def read_frames(path):
    """eval_frames.csv -> {frame key: (label, found before any cut (0/1), IoU or None)}"""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = f"{r.get('timestamp_folder', '')}/{os.path.basename(r['ir_path'].replace(chr(92), '/'))}"
            out[key] = (r["label"], int(r.get("model_visible_before_cut") or 0),
                        float(r["iou"]) if r.get("iou") not in ("", None) else None)
    return out


def frame_agreement(rows, max_list=12):
    """Per split: for each recipe, how many visible vulvas all its seeds find, some find, none finds,
    and how much the outline's IoU moves between seeds on the frames all find; then the visible
    vulvas that no model at all finds. -> {split: [lines]}"""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        p = os.path.join(os.path.dirname(r["report"]), "eval_frames.csv")
        if r["seed"] != "ens" and os.path.exists(p):
            by[r["split"]][r["recipe"]].append(read_frames(p))
    out = {}
    for split, recs in by.items():
        lines = [f"  which visible vulvas are found (before any cut), per recipe over its seeds:"]
        never = None
        for recipe in sorted(recs):
            runs = recs[recipe]
            keys = [k for k, v in runs[0].items() if v[0] == "present"]
            found = [sum(run.get(k, ("", 0, None))[1] for run in runs) for k in keys]
            n = len(runs)
            spreads = []
            for k, c in zip(keys, found):
                ious = [run[k][2] for run in runs if k in run and run[k][2] is not None]
                if c == n and len(ious) == n:
                    spreads.append(max(ious) - min(ious))
            lines.append(f"    {recipe:<12} {len(keys)} visible: all {n} seeds {sum(c == n for c in found)}, some "
                         f"{sum(0 < c < n for c in found)}, none {sum(c == 0 for c in found)};  IoU change between "
                         f"seeds on frames all find: median {_f(sorted(spreads)[len(spreads) // 2] if spreads else None, '.3f')}")
            missed = {k for k, c in zip(keys, found) if c == 0}
            never = missed if never is None else never & missed
        n_models = sum(len(v) for v in recs.values())
        never = sorted(never or [])
        lines.append(f"    missed by all {n_models} models: {len(never)}" + (": " + ", ".join(never[:max_list])
                     + (" ..." if len(never) > max_list else "") if never else ""))
        out[split] = lines
    return out


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
    text = table_text(summary, frame_agreement(rows))
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
