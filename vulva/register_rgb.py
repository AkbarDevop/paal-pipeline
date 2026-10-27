#!/usr/bin/env python3
r"""Add colour images on the IR pixel grid to a label set (rgb_align.py), so the labelling tool
can show and click the colour image like the IR one, and result sheets can be drawn in colour.

For every frame with a plain colour image (export_label_set.py --rgb / --add-rgb) this writes
<folder>/pigN_rgbreg_<ts>.jpg (640 x 480, same pixels as the IR) and fills the label set's
rgb_aligned_path column with it; rgb_shift records the per-frame shift, its match score and
which window it came from (vulva / rear / default). The frame's vulva centre is taken from
labels.csv (or --labels) when the frame is labelled visible, which gives the best match.
label_set.csv is backed up to label_set_before_rgbreg.csv the first time. Frames, IR, depth
and labels are not changed.

Usage:
    python vulva/register_rgb.py ~/Downloads/paal_exchange/vulva_label_set
    python vulva/register_rgb.py ~/Downloads/paal_exchange/vulva_test_feb12
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rgb_align import register  # noqa: E402
from seg_model import read_ir  # noqa: E402


def vulva_centres(labels_csv):
    out = {}
    if not os.path.exists(labels_csv):
        return out
    with open(labels_csv, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                if r.get("label") == "present":
                    out[r["ir_path"]] = (float(r["cx"]), float(r["cy"]))
            except (KeyError, ValueError):
                pass
    return out


def register_set(set_dir, labels_csv=None):
    path = os.path.join(set_dir, "label_set.csv")
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields, frames = list(reader.fieldnames), list(reader)
    centres = vulva_centres(labels_csv or os.path.join(set_dir, "labels.csv"))
    counts = {"vulva": 0, "rear": 0, "default": 0, "none": 0}
    for fr in frames:
        src = fr.get("rgb_path", "")
        ir = read_ir(os.path.join(set_dir, fr["ir_path"]))
        rgb = cv2.imread(os.path.join(set_dir, src), cv2.IMREAD_COLOR) if src else None
        if ir is None or rgb is None:
            counts["none"] += 1
            continue
        img, shift, score, how = register(ir, rgb, centres.get(fr["ir_path"]))
        rel = fr["ir_path"].replace("_ir_vis_", "_rgbreg_").rsplit(".", 1)[0] + ".jpg"
        cv2.imwrite(os.path.join(set_dir, rel), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        fr["rgb_aligned_path"] = rel
        fr["rgb_shift"] = f"{shift[0]},{shift[1]},{score:.2f},{how}"
        counts[how] += 1
    for k in ("rgb_aligned_path", "rgb_shift"):
        if k not in fields:
            fields.append(k)
    backup = os.path.join(set_dir, "label_set_before_rgbreg.csv")
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(frames)
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("set_dir")
    ap.add_argument("--labels", default="", help="labels CSV with vulva centres (default: <set>/labels.csv)")
    args = ap.parse_args()
    set_dir = os.path.abspath(os.path.expanduser(args.set_dir))
    c = register_set(set_dir, os.path.expanduser(args.labels) if args.labels else None)
    print(f"Colour on the IR grid for {c['vulva'] + c['rear'] + c['default']} frames "
          f"(shift matched at the vulva {c['vulva']}, over the rear {c['rear']}, typical shift {c['default']}); "
          f"{c['none']} without a colour image")


if __name__ == "__main__":
    main()
