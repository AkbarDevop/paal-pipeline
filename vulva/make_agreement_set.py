#!/usr/bin/env python3
r"""Make a small image set so other people (Lucas, Pushpa, ...) can outline the same frames in
CVAT Online, to measure how much human outlines agree (the ceiling for any model's score).

From a labelled set (default: the Feb 12-15 blind test) it picks N frames, mostly with a visible
vulva and some without, spread over sows, and writes for each one picture that CVAT shows:

    left half   the colour image matched onto the IR pixel grid (rgb_align.py)
    right half  the IR image, contrast-boosted
    both        cropped to the stall and enlarged 2x, so the vulva is easy to see

An annotator draws one ellipse around the whole vulva on the IR half (the colour half is for
reference: its registration can be ~10 px off), or tags the frame "not_visible" / "tail_covers".
import_cvat.py maps the ellipses back onto the IR pixel grid (either half) and records the side.
The hand labels of the set are NOT drawn on the pictures, so the outlines are independent.

Writes OUT/images/*.jpg, OUT/manifest.csv (how to map back), OUT/cvat_labels.json (paste into the
CVAT project's label constructor, "Raw") and OUT/HOW_TO_LABEL.md (instructions for the annotators).

    python vulva\make_agreement_set.py "$FEB\vulva_test_feb12\labels.csv" C:\Users\aakdbx\agreement_set --frames-root $RAW
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rgb_align import register  # noqa: E402

CROP = (120, 30, 500, 480)       # stall crop on the 640 x 480 IR grid (config.CROP_TOF)
SCALE = 2                        # enlargement in the exported pictures
CVAT_LABELS = [
    {"name": "vulva", "type": "ellipse", "color": "#ffe600", "attributes": []},
    {"name": "not_visible", "type": "tag", "color": "#33ddff", "attributes": []},
    {"name": "tail_covers", "type": "tag", "color": "#ff4040", "attributes": []},
]
HOW_TO = """# How to outline the vulva (about 30 minutes for 50 pictures)

Each picture shows the same moment twice: **colour on the left, infrared (black and white) on the right**.
**Always draw on the right (infrared) side.** Look at the colour side to see what is what, but draw
the oval on the infrared one (the two sides are not lined up perfectly). At night the colour side is dark.

## For each picture, do ONE of these

1. **Vulva visible:** draw one oval, on the infrared side, around the **whole vulva: its outer edge,
   lips and dark centre together** (not only the dark centre).
2. **The tail covers any part of it:** do not draw. Add the tag `tail_covers`.
3. **Not visible for another reason** (sow turned, lying, blurred, dirty): add the tag `not_visible`.

Only one oval per picture. If you are not sure, make your best guess: we want your own judgement.

## In CVAT (https://app.cvat.ai)

- Open **Jobs** (top menu) and click your job. `F` goes to the next picture, `D` to the previous one
  (or use the arrows at the top). The picture number is shown at the top.
- **Draw an oval:** on the left toolbar click the **ellipse** icon (an oval). Label `vulva`, drawing method
  **2 Points**, click **Shape**. Then click once at the top-left corner of the vulva and once at its
  bottom-right corner: the oval fills the box between the two clicks. If the vulva is clearly tilted,
  choose drawing method **Rotated** instead and click 5 or more points around its edge, then press `N`.
- **Add a tag:** on the left toolbar click the **tag** icon (the luggage-label shape just below the
  ellipse icon), pick `not_visible` or `tail_covers`, and press the blue **+** button.
- **Fix a mistake:** click the oval and press `Delete`, or drag its edges.
- Press **Save** (top left, or `Ctrl+S`) every few pictures and when you finish.
- When all pictures are done, tell Akbar. He exports the result; you do not need to send files.
"""


def pick_frames(rows, n, n_absent, seed=0):
    """n frames: n - n_absent with a visible vulva and n_absent without, spread round-robin over sows."""
    rng = random.Random(seed)

    def spread(pool, k):
        by = defaultdict(list)
        for r in pool:
            by[r["pig_id"]].append(r)
        pigs = sorted(by)
        rng.shuffle(pigs)
        for p in pigs:
            rng.shuffle(by[p])
        out = []
        while len(out) < k and any(by.values()):
            for p in pigs:
                if by[p] and len(out) < k:
                    out.append(by[p].pop())
        return out
    present = [r for r in rows if r["label"] == "present"]
    absent = [r for r in rows if r["label"] == "absent"]
    picked = spread(present, n - n_absent) + spread(absent, n_absent)
    rng.shuffle(picked)                       # mixed order, so the answer is not given away by position
    return picked


def colour_on_grid(row, ir):
    """Colour image on the IR grid: the set's registered one, else matched on the fly, else None."""
    for key in ("rgb_reg",):
        p = row.get(key, "")
        if p and os.path.exists(p):
            im = cv2.imread(p, cv2.IMREAD_COLOR)
            if im is not None and im.shape[:2] == ir.shape[:2]:
                return im
    p = row.get("rgb_plain", "")
    if p and os.path.exists(p):
        rgb = cv2.imread(p, cv2.IMREAD_COLOR)
        if rgb is not None:
            centre = None
            try:
                centre = (float(row["cx"]), float(row["cy"])) if row.get("label") == "present" else None
            except (KeyError, TypeError, ValueError):
                pass
            return register(ir, rgb, centre)[0]
    return None


def compose(ir, colour):
    """Side-by-side picture: colour | IR, both cropped to the stall and enlarged."""
    x1, y1, x2, y2 = CROP
    disp = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(ir)
    right = cv2.cvtColor(disp[y1:y2, x1:x2], cv2.COLOR_GRAY2BGR)
    left = colour[y1:y2, x1:x2] if colour is not None else np.full_like(right, 30)
    w, h = (x2 - x1) * SCALE, (y2 - y1) * SCALE
    left = cv2.resize(left, (w, h), interpolation=cv2.INTER_CUBIC)
    right = cv2.resize(right, (w, h), interpolation=cv2.INTER_CUBIC)
    for im, text in ((left, "colour" if colour is not None else "no colour image"), (right, "infrared")):
        cv2.putText(im, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(im, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return np.hstack([left, right])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels_csv", help="labelled set to take frames from (the reference labels)")
    ap.add_argument("out_dir")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--absent", type=int, default=10, help="how many frames without a visible vulva")
    ap.add_argument("--frames-root", default="", help="';'-separated raw folders, for frames not next to the labels")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from eval_seg import read_rows
    from seg_model import read_ir
    rows = read_rows(args.labels_csv, [r for r in args.frames_root.split(";") if r.strip()])
    picked = pick_frames(rows, args.n, args.absent, args.seed)
    out_img = os.path.join(args.out_dir, "images")
    os.makedirs(out_img, exist_ok=True)
    manifest, missing = [], 0
    for k, r in enumerate(picked):
        ir = read_ir(r["ir_path"])
        if ir is None:
            missing += 1
            continue
        name = f"{k + 1:03d}.jpg"           # neutral names: no pig, date or label in them
        cv2.imwrite(os.path.join(out_img, name), compose(ir, colour_on_grid(r, ir)), [cv2.IMWRITE_JPEG_QUALITY, 95])
        manifest.append({"image": name, "pig_id": r["pig_id"], "pig_timestamp": r["pig_timestamp"],
                         "timestamp_folder": r.get("timestamp_folder", ""), "ir_path": r["ir_path"],
                         "crop_x": CROP[0], "crop_y": CROP[1], "scale": SCALE,
                         "half_width": (CROP[2] - CROP[0]) * SCALE})
    with open(os.path.join(args.out_dir, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)
    with open(os.path.join(args.out_dir, "cvat_labels.json"), "w") as f:
        json.dump(CVAT_LABELS, f, indent=2)
    with open(os.path.join(args.out_dir, "HOW_TO_LABEL.md"), "w") as f:
        f.write(HOW_TO)
    n_pres = sum(1 for r in picked if r["label"] == "present")
    print(f"Wrote {len(manifest)} pictures ({n_pres} of the picked frames have a hand outline, "
          f"{len(picked) - n_pres} not; {missing} unreadable) to {out_img}, from "
          f"{len({r['pig_id'] for r in picked})} sows. Upload the images folder to a CVAT task; "
          f"labels: cvat_labels.json; instructions: HOW_TO_LABEL.md")


if __name__ == "__main__":
    main()
