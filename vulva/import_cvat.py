#!/usr/bin/env python3
r"""Turn a CVAT export of an agreement set (make_agreement_set.py) into a labels CSV + masks in our
format, on the IR pixel grid, so it can be compared with other people's labels (compare_labels.py)
or trained on.

In CVAT: open the task -> Actions -> Export task dataset -> format "CVAT for images 1.1"
(annotations only is enough) -> download the zip. Pass the zip (or the annotations.xml in it):

    python vulva\import_cvat.py lucas_export.zip C:\Users\aakdbx\agreement_set --who lucas

Writes <set>/labels_<who>.csv and <set>/masks_<who>/*.png. A picture with an ellipse labelled
"vulva" is "present" (the largest ellipse if there are several); tag "tail_covers" makes it
"absent" with note "tail"; tag "not_visible" makes it "absent"; a picture with nothing is left out
(not labelled). The "how" column says which side the oval was drawn on (cvat:<who>:ir or :colour).
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import os
import sys
import xml.etree.ElementTree as ET
import zipfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FULL_W, FULL_H = 640, 480
FIELDS = ["ir_path", "mask_path", "label", "pig_id", "timestamp_folder", "pig_timestamp",
          "cx", "cy", "half_length_px", "half_width_px", "angle_deg", "how", "note", "review"]


def read_xml(path):
    """annotations.xml from a CVAT 'CVAT for images 1.1' export (zip or xml) -> ElementTree root."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            name = next((n for n in z.namelist() if n.endswith(".xml")), None)
            if name is None:
                sys.exit(f"No .xml in {path}: export with the format 'CVAT for images 1.1'")
            return ET.parse(io.BytesIO(z.read(name))).getroot()
    return ET.parse(path).getroot()


def parse_images(root):
    """-> {image file name: {"ellipses": [(cx, cy, rx, ry, rotation)], "tags": set()}} in picture pixels."""
    out = {}
    for im in root.iter("image"):
        name = os.path.basename(im.get("name", ""))
        rec = {"ellipses": [], "tags": set()}
        for e in im.iter("ellipse"):
            if e.get("label") != "vulva":
                continue
            rec["ellipses"].append(tuple(float(e.get(k, 0) or 0) for k in ("cx", "cy", "rx", "ry", "rotation")))
        for t in im.iter("tag"):
            rec["tags"].add(t.get("label", ""))
        out[name] = rec
    return out


def to_ir_grid(ell, crop_x, crop_y, scale, half_width):
    """Ellipse drawn on the side-by-side picture -> the same ellipse on the 640 x 480 IR grid
    (either half: colour on the left, IR on the right, both the same crop and enlargement)."""
    cx, cy, rx, ry, rot = ell
    if cx >= half_width:
        cx -= half_width
    return cx / scale + crop_x, cy / scale + crop_y, rx / scale, ry / scale, rot


def ellipse_mask(cx, cy, rx, ry, rot):
    m = np.zeros((FULL_H, FULL_W), np.uint8)
    cv2.ellipse(m, (int(round(cx)), int(round(cy))), (max(int(round(rx)), 1), max(int(round(ry)), 1)),
                rot, 0, 360, 255, -1)
    return m


def import_export(export_path, set_dir, who):
    with open(os.path.join(set_dir, "manifest.csv"), newline="", encoding="utf-8-sig") as f:
        manifest = {r["image"]: r for r in csv.DictReader(f)}
    images = parse_images(read_xml(export_path))
    mask_dir = os.path.join(set_dir, f"masks_{who}")
    os.makedirs(mask_dir, exist_ok=True)
    rows, unknown, empty, on_colour = [], 0, 0, 0
    for name, rec in sorted(images.items()):
        m = manifest.get(name)
        if m is None:
            unknown += 1
            continue
        base = {"ir_path": m["ir_path"], "pig_id": m["pig_id"], "timestamp_folder": m.get("timestamp_folder", ""),
                "pig_timestamp": m["pig_timestamp"], "how": f"cvat:{who}", "mask_path": "", "note": "", "review": ""}
        if rec["ellipses"]:
            e = max(rec["ellipses"], key=lambda t: t[2] * t[3])
            cx, cy, rx, ry, rot = to_ir_grid(e, float(m["crop_x"]), float(m["crop_y"]), float(m["scale"]),
                                             float(m["half_width"]))
            rel = os.path.join(f"masks_{who}", name.rsplit(".", 1)[0] + ".png").replace("\\", "/")
            cv2.imwrite(os.path.join(set_dir, rel), ellipse_mask(cx, cy, rx, ry, rot))
            a, b = (rx, ry) if rx >= ry else (ry, rx)
            ang = rot if rx >= ry else rot + 90
            side = "ir" if e[0] >= float(m["half_width"]) else "colour"
            on_colour += side == "colour"
            rows.append(dict(base, label="present", mask_path=rel, cx=round(cx, 1), cy=round(cy, 1),
                             how=f"cvat:{who}:{side}",
                             half_length_px=round(a, 2), half_width_px=round(b, 2),
                             angle_deg=round(math.fmod(ang, 360), 1)))
        elif "tail_covers" in rec["tags"]:
            rows.append(dict(base, label="absent", note="tail"))
        elif "not_visible" in rec["tags"]:
            rows.append(dict(base, label="absent"))
        else:
            empty += 1
    out = os.path.join(set_dir, f"labels_{who}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return out, rows, unknown, empty, on_colour


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export", help="CVAT export zip (or annotations.xml), format 'CVAT for images 1.1'")
    ap.add_argument("set_dir", help="the folder from make_agreement_set.py (has manifest.csv)")
    ap.add_argument("--who", required=True, help="annotator name, used in the file names (e.g. lucas)")
    args = ap.parse_args()
    out, rows, unknown, empty, on_colour = import_export(args.export, args.set_dir, args.who.lower())
    pres = sum(1 for r in rows if r["label"] == "present")
    tail = sum(1 for r in rows if r.get("note") == "tail")
    print(f"{args.who}: {len(rows)} pictures labelled ({pres} vulva outlined, {tail} tail covers, "
          f"{len(rows) - pres - tail} not visible); {empty} left empty; {unknown} not in the manifest -> {out}")
    if on_colour:
        print(f"Note: {on_colour} ovals were drawn on the colour side; they are mapped over, but the colour "
              f"image can sit a few pixels off the IR one, so those outlines may be slightly shifted")


if __name__ == "__main__":
    main()
