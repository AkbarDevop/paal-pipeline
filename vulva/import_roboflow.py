#!/usr/bin/env python3
r"""Roboflow `vulva_seg` export (COCO Segmentation) -> a label set train_seg.py reads, for pretraining.

The export (universe.roboflow.com/sow-xychd/vulva_seg, CC BY 4.0) holds the lab's 2022 robot images:
colour, 480 x 480 (Roboflow cut 80 px off each side of the 640 x 480 original, no rescaling), one hand
outline of the vulva per image, plus a rectangle around the same vulva and a rectangle around the tail.
Kept: the vulva outlines. Dropped: every rectangle (filled as a mask, a rectangle would teach the model
square outlines, and a tail box covers skin), so the tail counts as "not outlined" and the trainer only
learns vulva-or-not on these images, as on our own frames without a tail outline.

Each image is made to look like one of our OAK infrared frames:
  grey      0.299 R + 0.587 G + 0.114 B (OpenCV's standard conversion)
  scale     zoomed --zoom times (default 1.4: their vulvas are about 22 x 31 px, ours about 27 x 52 px;
            1.4 matches the area) and placed on a 640 x 480 frame so that their median vulva position
            lands on ours, so our stall crop (config.CROP_TOF) sees it where it sees our vulvas
  blur      a light Gaussian blur (--blur, px), since our infrared is softer than their colour images
Masks are drawn from the outline polygons on the same 640 x 480 frame.

Written: OUT/<timestamp folder>/pig<ear tag>_ir_vis_<time>.jpg (OAK names, so every tool reads them),
OUT/masks/*.png, OUT/labels.csv (ir_path, mask_path, label, pig_id = ear tag, timestamp_folder,
pig_timestamp, source_name, roboflow_split, size of the outline), and OUT/outline_sheet.jpg: their
outline on the colour image | what the model will see, for a check by eye. Roboflow's own
train/valid/test split is by image (the same sow in all three), so it is recorded but not used.

    python vulva\import_roboflow.py C:\Users\aakdbx\Downloads\vulva_seg.v2i.coco-segmentation.zip C:\Users\aakdbx\vulva_roboflow_set
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import statistics as st
import sys
import zipfile
from collections import Counter

import cv2
import numpy as np

FULL_W, FULL_H = 640, 480
OUR_CENTRE = (327.0, 253.0)          # median centre of our hand outlines (Feb training + blind test, IR px)
OUR_SIZE = (27.0, 52.0)              # median width x length of our hand outlines (IR px)
NAME = re.compile(r"^(\d+)_(\d{4})_(\d{1,2})_(\d{1,2})_(\d{1,2})_(\d{1,2})_(\d{1,2})")
FIELDS = ["ir_path", "mask_path", "label", "pig_id", "timestamp_folder", "pig_timestamp", "source_name",
          "roboflow_split", "cx", "cy", "outline_w_px", "outline_l_px"]


class Source:
    """The export as a zip or as an unpacked folder."""

    def __init__(self, path):
        self.zip = zipfile.ZipFile(path) if zipfile.is_zipfile(path) else None
        self.root = None if self.zip else path
        if not self.zip and not os.path.isdir(path):
            sys.exit(f"{path}: neither a zip nor a folder")

    def names(self):
        if self.zip:
            return self.zip.namelist()
        out = []
        for d, _, files in os.walk(self.root):
            out += [os.path.relpath(os.path.join(d, f), self.root).replace("\\", "/") for f in files]
        return out

    def read(self, name):
        if self.zip:
            return self.zip.read(name)
        with open(os.path.join(self.root, name), "rb") as f:
            return f.read()


def is_rectangle(poly):
    """A flat list x1, y1, x2, y2, ... that only traces an axis-aligned box (how Roboflow writes boxes)."""
    xs, ys = [round(v, 1) for v in poly[0::2]], [round(v, 1) for v in poly[1::2]]
    return len(set(xs)) <= 2 and len(set(ys)) <= 2


def parse_name(name):
    """'131_2022_8_16_8_22_19.png' -> (ear tag, 'YYYYMMDD-HH-MM-SS') or None."""
    m = NAME.match(os.path.basename(name))
    if not m:
        return None
    tag, y, mo, d, h, mi, s = (int(g) for g in m.groups())
    return tag, f"{y:04d}{mo:02d}{d:02d}-{h:02d}-{mi:02d}-{s:02d}"


def read_export(src):
    """-> list of dicts: file, split, original name, size, vulva outlines (lists of (x, y)), counts of what was dropped."""
    images, dropped = [], Counter()
    for js in sorted(n for n in src.names() if n.endswith("_annotations.coco.json")):
        split = js.split("/")[0] if "/" in js else ""
        d = json.loads(src.read(js))
        cats = {c["id"]: c["name"].strip().lower() for c in d.get("categories", [])}
        anns = {}
        for a in d.get("annotations", []):
            anns.setdefault(a["image_id"], []).append(a)
        folder = os.path.dirname(js)
        for im in d.get("images", []):
            outlines = []
            for a in anns.get(im["id"], []):
                cat = cats.get(a["category_id"], "")
                segs = a.get("segmentation") or []
                segs = segs if isinstance(segs, list) else []
                if cat != "vulva":
                    dropped[f"{cat or 'other'} annotations"] += 1
                    continue
                kept = [s for s in segs if isinstance(s, list) and len(s) >= 6 and not is_rectangle(s)]
                if not kept:
                    dropped["vulva rectangles"] += 1
                    continue
                for s in kept:
                    outlines.append([(float(s[i]), float(s[i + 1])) for i in range(0, len(s) - 1, 2)])
            images.append({"file": f"{folder}/{im['file_name']}" if folder else im["file_name"], "split": split,
                           "name": (im.get("extra") or {}).get("name") or im["file_name"],
                           "w": int(im["width"]), "h": int(im["height"]), "outlines": outlines})
    return images, dropped


def to_frame(img_bgr, outlines, zoom, offset, blur):
    """Colour image + outlines -> (grey 640 x 480 frame, vulva mask, outlines on the frame)."""
    A = np.float32([[zoom, 0, offset[0]], [0, zoom, offset[1]]])
    grey = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    frame = cv2.warpAffine(grey, A, (FULL_W, FULL_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                           borderValue=0)
    if blur > 0:
        frame = cv2.GaussianBlur(frame, (0, 0), blur)
    mask = np.zeros((FULL_H, FULL_W), np.uint8)
    moved = []
    for poly in outlines:
        p = np.array([[zoom * x + offset[0], zoom * y + offset[1]] for x, y in poly], np.float32)
        moved.append(p)
        cv2.fillPoly(mask, [np.round(p).astype(np.int32)], 255)
    return frame, mask, moved


def outline_size(mask):
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    (cx, cy), (a, b), _ = cv2.fitEllipse(c) if len(c) >= 5 else (cv2.minAreaRect(c)[0], cv2.minAreaRect(c)[1], 0)
    return cx, cy, min(a, b), max(a, b)


def sheet_tile(img_bgr, outlines, frame, moved, title, size=200):
    """Their outline on the colour image | the grey frame the model sees with the same outline."""
    tiles = []
    for im, polys, is_frame in ((img_bgr, [np.array(p, np.float32) for p in outlines], False), (frame, moved, True)):
        pts = np.vstack(polys)
        cx, cy = pts.mean(0)
        half = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1]))) * 1.6 + 10
        s = size / (2 * half)
        A = np.float32([[s, 0, size / 2 - s * cx], [0, s, size / 2 - s * cy]])
        view = cv2.warpAffine(im if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR), A, (size, size),
                              flags=cv2.INTER_CUBIC)
        for p in polys:
            q = np.round(p * s + [A[0, 2], A[1, 2]]).astype(np.int32)
            cv2.polylines(view, [q], True, (60, 220, 60), 1, cv2.LINE_AA)
        label = "model sees" if is_frame else title
        cv2.putText(view, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(view, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(view)
    return np.hstack(tiles)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export", help="the Roboflow zip (COCO Segmentation) or its unpacked folder")
    ap.add_argument("out_dir")
    ap.add_argument("--zoom", type=float, default=1.4, help="scale from their pixels to ours (default 1.4)")
    ap.add_argument("--blur", type=float, default=1.0, help="Gaussian blur after zooming, px (0 = none)")
    ap.add_argument("--sheet", type=int, default=48, help="images on the outline sheet (0 = none)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src = Source(args.export)
    images, dropped = read_export(src)
    if not images:
        sys.exit(f"No COCO annotation files (_annotations.coco.json) in {args.export}")
    with_vulva = [im for im in images if im["outlines"]]
    no_vulva = len(images) - len(with_vulva)
    if not with_vulva:
        sys.exit("No vulva outlines (only rectangles?): export the dataset as 'COCO Segmentation', not a detection format")
    centres = [np.vstack(im["outlines"]).mean(0) for im in with_vulva]
    their_centre = (float(np.median([c[0] for c in centres])), float(np.median([c[1] for c in centres])))
    offset = (OUR_CENTRE[0] - args.zoom * their_centre[0], OUR_CENTRE[1] - args.zoom * their_centre[1])

    os.makedirs(os.path.join(args.out_dir, "masks"), exist_ok=True)
    rows, seen, skipped, sizes_in, sizes_out = [], set(), Counter(), [], []
    rng = random.Random(args.seed)
    picks = set(rng.sample(range(len(with_vulva)), min(args.sheet, len(with_vulva))))
    tiles = []
    for k, im in enumerate(with_vulva):
        parsed = parse_name(im["name"])
        if parsed is None:
            skipped["name without ear tag and time"] += 1
            continue
        tag, ts = parsed
        if (tag, ts) in seen:
            skipped["same sow and second twice"] += 1
            continue
        img = cv2.imdecode(np.frombuffer(src.read(im["file"]), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            skipped["unreadable image"] += 1
            continue
        frame, mask, moved = to_frame(img, im["outlines"], args.zoom, offset, args.blur)
        size = outline_size(mask)
        if size is None or mask[30:480, 120:500].sum() < 0.9 * mask.sum():
            skipped["outline outside our stall crop"] += 1
            continue
        seen.add((tag, ts))
        rel_img = f"{ts}/pig{tag}_ir_vis_{ts}.jpg"
        rel_mask = f"masks/{ts}_pig{tag}.png"
        os.makedirs(os.path.join(args.out_dir, ts), exist_ok=True)
        cv2.imwrite(os.path.join(args.out_dir, rel_img), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(os.path.join(args.out_dir, rel_mask), mask)
        orig = np.zeros((im["h"], im["w"]), np.uint8)
        for poly in im["outlines"]:
            cv2.fillPoly(orig, [np.round(np.array(poly, np.float32)).astype(np.int32)], 255)
        before = outline_size(orig)
        if before:
            sizes_in.append(before[2:])
        sizes_out.append(size[2:])
        rows.append({"ir_path": rel_img, "mask_path": rel_mask, "label": "present", "pig_id": tag,
                     "timestamp_folder": ts, "pig_timestamp": ts, "source_name": im["name"],
                     "roboflow_split": im["split"], "cx": round(size[0], 1), "cy": round(size[1], 1),
                     "outline_w_px": round(size[2], 1), "outline_l_px": round(size[3], 1)})
        if k in picks:
            tiles.append(sheet_tile(img, im["outlines"], frame, moved, f"sow {tag} {ts[4:6]}/{ts[6:8]}"))
    if not rows:
        sys.exit("Nothing written")
    rows.sort(key=lambda r: (r["pig_id"], r["pig_timestamp"]))
    with open(os.path.join(args.out_dir, "labels.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    if tiles:
        cols = 4
        blank = np.zeros_like(tiles[0])
        grid = [tiles[i:i + cols] + [blank] * (cols - len(tiles[i:i + cols])) for i in range(0, len(tiles), cols)]
        cv2.imwrite(os.path.join(args.out_dir, "outline_sheet.jpg"), np.vstack([np.hstack(g) for g in grid]),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

    sows = Counter(r["pig_id"] for r in rows)
    med = lambda v, i: st.median(x[i] for x in v) if v else float("nan")  # noqa: E731
    lines = [f"{len(images)} images in the export: {len(with_vulva)} with a vulva outline, {no_vulva} without (left out)",
             "dropped annotations: " + ", ".join(f"{v} {k}" for k, v in sorted(dropped.items())),
             f"written: {len(rows)} frames from {len(sows)} sows -> {args.out_dir}"
             + (" (skipped: " + ", ".join(f"{v} {k}" for k, v in skipped.items()) + ")" if skipped else ""),
             "frames per sow: " + ", ".join(f"{t}: {n}" for t, n in sows.most_common()),
             f"vulva outline, median width x length: {med(sizes_in, 0):.1f} x {med(sizes_in, 1):.1f} px in their images, "
             f"{med(sizes_out, 0):.1f} x {med(sizes_out, 1):.1f} px after zoom {args.zoom} "
             f"(our hand outlines: about {OUR_SIZE[0]:.0f} x {OUR_SIZE[1]:.0f} px)",
             f"placed so their median vulva centre ({their_centre[0]:.0f}, {their_centre[1]:.0f}) lands on ours "
             f"({OUR_CENTRE[0]:.0f}, {OUR_CENTRE[1]:.0f}); grey, blur {args.blur} px",
             "tails: not outlined in this set (rectangles only), so train_seg.py learns vulva-or-not here"]
    with open(os.path.join(args.out_dir, "import_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if tiles:
        print(f"check by eye: {os.path.join(args.out_dir, 'outline_sheet.jpg')} (their outline on the colour image | what the model sees)")


if __name__ == "__main__":
    main()
