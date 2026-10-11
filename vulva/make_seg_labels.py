#!/usr/bin/env python3
r"""Build the segmentation label index from the vulva masks we already have.

Reads the manual measurement table (labels/vulva_measurements_manual_dataset.csv,
one row per frame with status measured / no_vulva / failed and a mask_path) and
writes labels/vulva_seg_labels.csv:

    ir_path, mask_path, label, pig_id, timestamp_folder, pig_timestamp

label = present (mask drawn), absent (looked at, no vulva visible). Rows that
failed are left out. Absent frames matter: they teach the model to output an
empty mask when the tail or posture hides the vulva.

Paths in the manual table were written on another PC, so the IR image is found
again under --data-root (flat timestamp folders or round folders one level down)
and the mask under --mask-root by file name.

Usage:
    python vulva/make_seg_labels.py --manual labels/vulva_measurements_manual_dataset.csv \
        --data-root "\\umad.umsystem.edu\rde\zhoujianf-lab\Pig project\Sow Data\Pictures_OAK" \
        --mask-root labels/vulva_masks_manual
"""

from __future__ import annotations

import argparse
import csv
import os
import re

FIELDS = ["ir_path", "mask_path", "label", "pig_id", "timestamp_folder", "pig_timestamp"]
TS_RE = re.compile(r"^\d{8}-\d{2}-\d{2}-\d{2}$")


def folder_index(root):
    """timestamp folder name -> path, for flat and one-level-nested layouts."""
    idx = {}
    if not root or not os.path.isdir(root):
        return idx
    for e in os.scandir(root):
        if not e.is_dir():
            continue
        if TS_RE.match(e.name):
            idx.setdefault(e.name, e.path)
        else:
            try:
                for s in os.scandir(e.path):
                    if s.is_dir() and TS_RE.match(s.name):
                        idx.setdefault(s.name, s.path)
            except OSError:
                pass
    return idx


def find_ir(row, folders):
    name = f"pig{int(float(row['pig_id']))}_ir_vis_{row['pig_timestamp']}.jpg"
    folder = folders.get(row["timestamp_folder"])
    if folder:
        p = os.path.join(folder, name)
        if os.path.exists(p):
            return p
    old = (row.get("texture_input") or "").replace("\\", "/")
    return old if old and os.path.exists(old) else ""


def find_mask(row, mask_root):
    old = (row.get("mask_path") or "").replace("\\", "/")
    if not old:
        return ""
    for cand in (old, os.path.join(mask_root or "", os.path.basename(old))):
        if cand and os.path.exists(cand):
            return cand
    return ""


def build(manual_rows, data_root, mask_root):
    folders = folder_index(data_root)
    out, missing = [], {"ir": 0, "mask": 0, "skipped": 0}
    for r in manual_rows:
        status = (r.get("status") or "").strip()
        if status == "measured":
            label = "present"
        elif status == "no_vulva":
            label = "absent"
        else:
            missing["skipped"] += 1
            continue
        ir = find_ir(r, folders)
        if not ir:
            missing["ir"] += 1
            continue
        mask = ""
        if label == "present":
            mask = find_mask(r, mask_root)
            if not mask:
                missing["mask"] += 1
                continue
        out.append({"ir_path": ir, "mask_path": mask, "label": label,
                    "pig_id": int(float(r["pig_id"])), "timestamp_folder": r["timestamp_folder"],
                    "pig_timestamp": r["pig_timestamp"]})
    return out, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manual", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--mask-root", default=os.path.join("labels", "vulva_masks_manual"))
    ap.add_argument("--out", default=os.path.join("labels", "vulva_seg_labels.csv"))
    args = ap.parse_args()

    with open(args.manual, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out, miss = build(rows, args.data_root, args.mask_root)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out)
    pres = sum(r["label"] == "present" for r in out)
    print(f"{len(out)} labelled frames ({pres} present, {len(out) - pres} absent) -> {args.out}")
    print(f"Left out: {miss['skipped']} failed/other, {miss['ir']} IR image not found, {miss['mask']} mask not found")


if __name__ == "__main__":
    main()
