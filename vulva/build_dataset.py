#!/usr/bin/env python3
r"""One versioned vulva dataset from the separate label sets: copies of every labelled frame
(IR, depth, colour), the masks, and ONE table with a row per frame and a split column.

Why: the labels lived in five CSVs on two computers (Feb training, January halves, the hard
frames, the Feb blind test). Training and testing from one table with a split column makes it
impossible to train on a test frame by accident, and the dataset can be frozen (v1) and
copied whole. The raw share is only read, never written.

    python vulva\build_dataset.py C:\Users\aakdbx\vulva_dataset_v1 --frames-root $RAW `
        --train-from outputs\vulva_seg_r10\seg_report.json `
        --add test_newdays  feb12    "$FEB\vulva_test_feb12\labels.csv" `
        --add test_newsows  jan_test C:\Users\aakdbx\vulva_jan_set\labels_test.csv

(--train-from takes the training label files from an earlier run's report, so the train split is
exactly what that model was trained on; --add SPLIT NAME CSV adds any set by hand.)

Writes OUT/labels.csv (the master table), OUT/tail_labels.csv, OUT/label_set.csv (colour
images), OUT/frames/<timestamp folder>/<OAK file names>, OUT/masks/vulva|tail/<frame_id>.png
and OUT/README.md (counts by split, month, time of day and sow, and the leak checks).
Then train and test by split:

    python vulva\train_seg.py --labels C:\Users\aakdbx\vulva_dataset_v1\labels.csv --split train ...
    python vulva\eval_seg.py  --labels C:\Users\aakdbx\vulva_dataset_v1\labels.csv --split test_newdays ...

Animals: stall numbers repeat across batches of sows, so an animal is (batch, stall). Batches
start at the dates in --batch-starts (default 2026-02-05, when Lucas's February sows were moved
in; check with Lucas). The leak checks stop the build if a frame, or an animal on the same day,
is both in train and in a test split.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FIELDS = ["frame_id", "split", "source", "batch", "animal_id", "date", "hour",
          "ir_path", "mask_path", "depth_path", "label", "pig_id", "timestamp_folder", "pig_timestamp",
          "cx", "cy", "half_length_px", "half_width_px", "angle_deg", "how", "note", "review"]
TAIL_FIELDS = ["ir_path", "pig_id", "timestamp_folder", "pig_timestamp", "tail", "tail_mask_path",
               "covers_vulva", "how", "width_px", "date"]


def pig_int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return -1


def batch_of(date, starts):
    """date 'YYYYMMDD' -> 'b1', 'b2', ... by the batch start dates (sorted 'YYYY-MM-DD')."""
    k = 1
    for s in starts:
        if date >= s.replace("-", ""):
            k += 1
    return f"b{k}"


def read_tails(labels_csv):
    """tail_labels.csv next to a labels CSV -> {ir_path as written there: row (mask path made absolute)}."""
    p = os.path.join(os.path.dirname(os.path.abspath(labels_csv)), "tail_labels.csv")
    if not os.path.exists(p):
        return {}
    base = os.path.dirname(p)
    out = {}
    with open(p, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            mp = r.get("tail_mask_path", "")
            r["_mask_abs"] = (mp if os.path.isabs(mp) else os.path.join(base, mp)) if mp else ""
            out[r["ir_path"]] = r
    return out


def collect(sources, frames_roots, starts):
    """sources: [(split, name, labels_csv)] -> (frames {key: row}, problems [str])."""
    from eval_seg import read_rows
    frames, problems = {}, []
    for split, name, path in sources:
        with open(path, newline="", encoding="utf-8-sig") as f:
            raw_ir = [r["ir_path"] for r in csv.DictReader(f) if r.get("label") in ("present", "absent")]
        rows = read_rows(path, frames_roots)
        tails = read_tails(path)
        for r, orig in zip(rows, raw_ir):
            pid, ts = pig_int(r.get("pig_id")), r.get("pig_timestamp", "")
            key = (pid, ts)
            date, hour = ts[:8], (int(ts[9:11]) if len(ts) >= 11 and ts[9:11].isdigit() else -1)
            row = dict(r, split=split, source=name, date=date, hour=hour, pig_id=pid,
                       batch=batch_of(date, starts), _tail=tails.get(orig))
            row["animal_id"] = f"{row['batch']}_s{pid:02d}"
            row["frame_id"] = f"{r.get('timestamp_folder', '')}_pig{pid}_{ts}"
            if key in frames:
                prev = frames[key]
                if prev["split"] != split:
                    problems.append(f"frame {row['frame_id']} is in {prev['split']} ({prev['source']}) "
                                    f"and in {split} ({name})")
                continue                                   # same split: keep the first copy
            frames[key] = row
    return frames, problems


def leak_checks(frames):
    """-> (errors, notes). Errors: an animal on the same day in train and in a test split."""
    errors, notes = [], []
    by_split = defaultdict(list)
    for r in frames.values():
        by_split[r["split"]].append(r)
    train = by_split.get("train", [])
    train_animal_days = {(r["animal_id"], r["date"]) for r in train}
    train_animals = {r["animal_id"] for r in train}
    train_dates = {r["date"] for r in train}
    for split, rows in sorted(by_split.items()):
        if split == "train":
            continue
        shared = sorted({(r["animal_id"], r["date"]) for r in rows} & train_animal_days)
        if shared:
            errors.append(f"{split}: {len(shared)} animal-days also in train, e.g. {shared[:3]}")
        a = {r["animal_id"] for r in rows}
        d = {r["date"] for r in rows}
        notes.append(f"{split}: {len(a)} animals ({len(a & train_animals)} also in train), "
                     f"{len(d)} days ({len(d & train_dates)} also in train)")
    return errors, notes


def copy_file(src, dst, dry):
    if not src or not os.path.exists(src):
        return False
    if dry:
        return True
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not (os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src)):
        shutil.copy2(src, dst)
    return True


def build(out_dir, frames, dry=False):
    """Copy files and write the tables. Returns (rows written, missing files count)."""
    rows, tails, colour, missing = [], [], [], 0
    for r in sorted(frames.values(), key=lambda r: (r["split"], r["pig_timestamp"], r["pig_id"])):
        folder = r.get("timestamp_folder") or r["date"]
        ir_rel = f"frames/{folder}/{os.path.basename(r['ir_path'].replace(os.sep, '/'))}"
        if not copy_file(r["ir_path"], os.path.join(out_dir, ir_rel), dry):
            missing += 1
            continue
        dep = r.get("depth_path", "")
        dep_rel = f"frames/{folder}/{os.path.basename(dep.replace(os.sep, '/'))}" if dep else ""
        if dep_rel and not copy_file(dep, os.path.join(out_dir, dep_rel), dry):
            dep_rel = ""
        mask_rel = ""
        if r["label"] == "present":
            mask_rel = f"masks/vulva/{r['frame_id']}.png"
            if not copy_file(r.get("mask_path", ""), os.path.join(out_dir, mask_rel), dry):
                missing += 1
                continue
        reg_rel = plain_rel = ""
        for key, kind in (("rgb_reg", "reg"), ("rgb_plain", "plain")):
            src = r.get(key, "")
            if src:
                rel = f"frames/{folder}/{os.path.basename(src.replace(os.sep, '/'))}"
                if copy_file(src, os.path.join(out_dir, rel), dry):
                    if kind == "reg":
                        reg_rel = rel
                    else:
                        plain_rel = rel
        reg_guess = r["ir_path"].replace("_ir_vis_", "_rgbreg_")
        if not reg_rel and os.path.exists(reg_guess):
            reg_rel = f"frames/{folder}/{os.path.basename(reg_guess.replace(os.sep, '/'))}"
            copy_file(reg_guess, os.path.join(out_dir, reg_rel), dry)
        if reg_rel or plain_rel:
            colour.append({"ir_path": ir_rel, "rgb_aligned_path": reg_rel, "rgb_path": plain_rel})
        t = r.get("_tail")
        if t is not None:
            tm_rel = ""
            if t.get("_mask_abs"):
                tm_rel = f"masks/tail/{r['frame_id']}.png"
                if not copy_file(t["_mask_abs"], os.path.join(out_dir, tm_rel), dry):
                    tm_rel = ""
            tails.append({**{k: t.get(k, "") for k in TAIL_FIELDS}, "ir_path": ir_rel, "tail_mask_path": tm_rel})
        rows.append({**{k: r.get(k, "") for k in FIELDS}, "ir_path": ir_rel, "mask_path": mask_rel,
                     "depth_path": dep_rel})
    if not dry:
        os.makedirs(out_dir, exist_ok=True)
        for name, fields, data in (("labels.csv", FIELDS, rows), ("tail_labels.csv", TAIL_FIELDS, tails),
                                   ("label_set.csv", ["ir_path", "rgb_aligned_path", "rgb_path"], colour)):
            with open(os.path.join(out_dir, name), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                w.writerows(data)
    return rows, tails, missing


def summary(rows, tails, notes, errors, starts):
    lines = []
    sp = defaultdict(Counter)
    for r in rows:
        sp[r["split"]][r["label"]] += 1
    lines.append("| Split | Frames | Vulva visible | Not visible | Animals | Days | Months | Night (18:00-06:00) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for s in sorted(sp):
        rs = [r for r in rows if r["split"] == s]
        months = sorted({r["date"][:6] for r in rs})
        night = sum(1 for r in rs if r["hour"] >= 18 or 0 <= r["hour"] < 6)
        lines.append(f"| {s} | {len(rs)} | {sp[s]['present']} | {sp[s]['absent']} | "
                     f"{len({r['animal_id'] for r in rs})} | {len({r['date'] for r in rs})} | "
                     f"{', '.join(m[:4] + '-' + m[4:] for m in months)} | {night} |")
    lines.append(f"| all | {len(rows)} | {sum(c['present'] for c in sp.values())} | "
                 f"{sum(c['absent'] for c in sp.values())} | {len({r['animal_id'] for r in rows})} | "
                 f"{len({r['date'] for r in rows})} | | |")
    src = Counter((r["split"], r["source"]) for r in rows)
    lines += ["", "Sources: " + "; ".join(f"{s} <- {n} ({c})" for (s, n), c in sorted(src.items()))]
    lines.append(f"Tail looked at in {len(tails)} frames, outlined in {sum(1 for t in tails if str(t.get('tail')) == '1')}.")
    lines.append(f"Batches start {', '.join(starts) or '(none given)'}; animal = batch + stall.")
    lines += ["", "Leak checks:"] + [f"- {n}" for n in notes] + [f"- ERROR {e}" for e in errors]
    if not errors:
        lines.append("- no animal-day is both in train and in a test split")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--add", nargs=3, action="append", metavar=("SPLIT", "NAME", "LABELS_CSV"), default=[],
                    help="a label set: its split (train, test_newdays, ...), a short name and its labels CSV")
    ap.add_argument("--train-from", default="", help="seg_report.json of a training run: its label files become "
                    "the train split (named after their folders)")
    ap.add_argument("--frames-root", default="", help="';'-separated raw folders (images not next to the labels)")
    ap.add_argument("--batch-starts", default="2026-02-05", help="comma dates where a new batch of sows starts")
    ap.add_argument("--dry-run", action="store_true", help="check and count only, copy nothing")
    args = ap.parse_args()
    starts = sorted(s.strip() for s in args.batch_starts.split(",") if s.strip())
    roots = [p for p in args.frames_root.split(";") if p.strip()]
    sources = [tuple(a) for a in args.add]
    if args.train_from:
        import json
        with open(args.train_from) as f:
            for p in json.load(f)["labels"]:
                sources.insert(0, ("train", os.path.basename(os.path.dirname(os.path.abspath(p))), p))
    if not sources:
        sys.exit("Nothing to build: give --train-from and/or --add SPLIT NAME LABELS_CSV")
    frames, problems = collect(sources, roots, starts)
    errors, notes = leak_checks(frames)
    errors = problems + errors
    rows, tails, missing = build(args.out_dir, frames, dry=args.dry_run or bool(errors))
    lines = summary(rows, tails, notes, errors, starts)
    if missing:
        lines.append(f"{missing} frames left out: image or mask not found")
    print("\n".join(lines))
    if errors:
        sys.exit("Not built: fix the leaks above (a frame or animal-day in both train and test).")
    if not args.dry_run:
        with open(os.path.join(args.out_dir, "README.md"), "w") as f:
            f.write("# Vulva dataset\n\nBuilt by `vulva/build_dataset.py`. Copies only; the raw share is untouched.\n"
                    "Train with `--split train`; test splits are frozen and never trained or tuned on.\n\n"
                    + "\n".join(lines) + "\n")
        print(f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
