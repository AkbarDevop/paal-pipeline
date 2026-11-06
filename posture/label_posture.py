#!/usr/bin/env python3
r"""Label sow postures with one key per picture: standing, sitting or lying.

The labelling tool behind labels/labels_posture3.csv (archive/label_tool.py), revived as one file that
needs only Python with opencv-python and numpy and no separate scan step. It reads either

  a data folder   the timestamp folders (YYYYMMDD-HH-MM-SS) directly in it or one level down, as the OAK
                  camera writes them: every sow picture, in time order; or
  predictions.csv from posture/run_inference.py: only pictures where a sow was found, each shown with the
                  model's guess. --first sitting shows the pictures the model thinks most likely sitting
                  first, so a rare posture is found in minutes instead of after hundreds of lying sows.

Each picture is shown as infrared | depth | colour with the stall, the time and (from a predictions file)
the model's guess. Keys:

  0 standing    1 sitting    2 lying    Enter = take the model's guess
  s skip        b back (undo the last answer)    q quit

Every answer is saved at once to --out, in the format of labels/labels_posture3.csv (timestamp_folder,
pig_id, pig_timestamp, rgb_jpg, ir_jpg, depth_jpg, label: 0 standing, 1 sitting, 2 lying), so you can stop
at any time and run the same command again to carry on. Pictures already in --out, or in a file given with
--already (e.g. labels/labels_posture3.csv), are not shown again.

    python posture\label_posture.py "E:\Pig_data\Pictures_OAK" --out labels\posture_new.csv
    python posture\label_posture.py "E:\Pig_data\outputs\review_depth\predictions.csv" --first sitting --pigs 0-11 --out labels\sitting_round.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import Counter

import cv2
import numpy as np

CLASSES = {0: "standing", 1: "sitting", 2: "lying"}
FIELDS = ["timestamp_folder", "pig_id", "pig_timestamp", "rgb_jpg", "ir_jpg", "depth_jpg", "label"]
TS = re.compile(r"^\d{8}-\d{2}-\d{2}-\d{2}$")
IR_NAME = re.compile(r"^pig(\d+)_ir_vis_(\d{8}-\d{2}-\d{2}-\d{2})\.jpg$", re.IGNORECASE)
KEYS = {ord("0"): 0, ord("1"): 1, ord("2"): 2}
WINDOW = "label_posture: 0 standing, 1 sitting, 2 lying, Enter = model, s skip, b back, q quit"


# ------------------------------------------------------------------ which pictures

def index_folders(root):
    """Timestamp folders in root (or root itself, or one level down) -> {name: path}."""
    out = {}
    name = os.path.basename(os.path.normpath(root))
    if TS.match(name) and os.path.isdir(root):
        return {name: root}
    if not os.path.isdir(root):
        return out
    for e in sorted(os.scandir(root), key=lambda e: e.name):
        if not e.is_dir():
            continue
        if TS.match(e.name):
            out.setdefault(e.name, e.path)
            continue
        for e2 in sorted(os.scandir(e.path), key=lambda e: e.name):
            if e2.is_dir() and TS.match(e2.name):
                out.setdefault(e2.name, e2.path)
    return out


def companions(folder, pig, ts):
    """Infrared, depth and colour pictures of one capture (paths may not exist)."""
    rgb = os.path.join(folder, f"pig{pig}_rgb_{ts}.jpg")
    if not os.path.exists(rgb):
        aligned = os.path.join(folder, f"pig{pig}_rgb_aligned_{ts}.jpg")
        rgb = aligned if os.path.exists(aligned) else rgb
    depth = os.path.join(folder, f"pig{pig}_depth_vis_{ts}.jpg")
    if not os.path.exists(depth):
        raw = os.path.join(folder, f"pig{pig}_depth_{ts}.raw")
        depth = raw if os.path.exists(raw) else depth
    return {"ir_jpg": os.path.join(folder, f"pig{pig}_ir_vis_{ts}.jpg"), "depth_jpg": depth, "rgb_jpg": rgb}


def frames_from_folder(root):
    frames = []
    for name, path in sorted(index_folders(root).items()):
        for f in os.listdir(path):
            m = IR_NAME.match(f)
            if m:
                pig, ts = int(m.group(1)), m.group(2)
                frames.append({"timestamp_folder": name, "pig_id": pig, "pig_timestamp": ts, **companions(path, pig, ts)})
    frames.sort(key=lambda r: (r["timestamp_folder"], r["pig_timestamp"], r["pig_id"]))
    return frames


def frames_from_predictions(path, data_root=""):
    """run_inference.py's predictions.csv -> pictures with the model's guess and class probabilities.
    data_root: where the timestamp folders are now, if the pictures moved since the run."""
    index = index_folders(data_root) if data_root else {}
    frames = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            ts = r.get("pig_timestamp") or r["timestamp_folder"]
            raw = int(r.get("pig_id_raw") or r["pig_id"])          # the file name carries the camera's own number
            folder = os.path.dirname(r.get("image_path", ""))
            if not folder or not os.path.isdir(folder):
                folder = index.get(r["timestamp_folder"], folder)
            probs = {c: float(r[f"prob_{n}"]) for c, n in CLASSES.items() if r.get(f"prob_{n}") not in (None, "")}
            guess = r.get("prediction", "")
            guess = int(guess) if guess not in ("", None) else (max(probs, key=probs.get) if probs else None)
            frames.append({"timestamp_folder": r["timestamp_folder"], "pig_id": int(r["pig_id"]), "pig_timestamp": ts,
                           "guess": guess, "probs": probs, **companions(folder, raw, ts)})
    return frames


def parse_pigs(text):
    """'0-11,15' -> {0, ..., 11, 15}; '' -> None (all)."""
    if not text:
        return None
    out = set()
    for part in str(text).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return out


def select(frames, first="", pigs=None, limit=0):
    if pigs is not None:
        frames = [f for f in frames if f["pig_id"] in pigs]
    if first:
        c = {n: k for k, n in CLASSES.items()}[first]
        if frames and not any(f.get("probs") for f in frames):
            sys.exit("--first needs a predictions.csv from run_inference.py (a data folder has no model guesses)")
        frames = sorted(frames, key=lambda f: -f.get("probs", {}).get(c, 0.0))
    return frames[:limit] if limit else frames


def key_of(row):
    """One capture: its timestamp folder and infrared file name (paths differ between computers)."""
    return (row["timestamp_folder"], os.path.basename(str(row["ir_jpg"]).replace("\\", "/")))


def labelled_keys(paths):
    keys = set()
    for p in paths:
        if p and os.path.exists(p):
            with open(p, newline="", encoding="utf-8-sig") as f:
                keys.update(key_of(r) for r in csv.DictReader(f) if r.get("ir_jpg"))
    return keys


# ------------------------------------------------------------------ saving

class LabelFile:
    """Answers go to the CSV at once (stop any time); back removes this session's last one."""

    def __init__(self, path):
        self.path, self.written = path, []
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, newline="", encoding="utf-8-sig") as f:
                header = next(csv.reader(f), [])
            if header != FIELDS:
                sys.exit(f"{path} has other columns ({', '.join(header)}); give a new --out file")
        else:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    def add(self, frame, label):
        row = {k: frame.get(k, "") for k in FIELDS[:-1]}
        row["label"] = int(label)
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        self.written.append(row)

    def undo(self):
        if not self.written:
            return None
        last = self.written.pop()
        with open(self.path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        for i in range(len(rows) - 1, -1, -1):
            if key_of(rows[i]) == key_of(last):
                del rows[i]
                break
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        return last


# ------------------------------------------------------------------ showing

def read_depth_raw(path, w=640, h=480):
    """OAK ToF depth .raw (uint16 mm, with or without an 8-byte header) -> colour picture, near = red."""
    size = os.path.getsize(path)
    if size not in (w * h * 2, w * h * 2 + 8):
        return None
    d = np.fromfile(path, dtype=np.uint16, offset=size - w * h * 2).reshape(h, w).astype(np.float32)
    ok = (d > 300) & (d < 3000)
    if not ok.any():
        return None
    lo, hi = np.percentile(d[ok], (2, 98))
    x = np.clip((hi - d) / max(hi - lo, 1.0) * 255, 0, 255).astype(np.uint8)
    im = cv2.applyColorMap(x, cv2.COLORMAP_JET)
    im[~ok] = 0
    return im


def panel(path, title, h=360):
    im = None
    if path and os.path.exists(path):
        im = read_depth_raw(path) if path.lower().endswith(".raw") else cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        im = np.full((h, h * 4 // 3, 3), 35, np.uint8)
        cv2.putText(im, "no picture", (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2, cv2.LINE_AA)
    else:
        im = cv2.resize(im, (max(int(im.shape[1] * h / im.shape[0]), 1), h), interpolation=cv2.INTER_AREA)
    cv2.putText(im, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(im, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return im


def compose(frame, i, n, counts, max_width=1600):
    body = np.hstack([panel(frame["ir_jpg"], "infrared"), panel(frame["depth_jpg"], "depth"),
                      panel(frame["rgb_jpg"], "colour")])
    ts = frame["pig_timestamp"]
    when = f"{ts[4:6]}/{ts[6:8]} {ts[9:11]}:{ts[12:14]}" if len(ts) >= 14 else ts
    line1 = f"{i + 1} of {n}   stall {frame['pig_id']}   {when}"
    g = frame.get("guess")
    if g is not None:
        p = frame.get("probs", {}).get(g)
        line1 += f"   model: {CLASSES.get(g, g)}" + (f" ({p:.2f})" if p is not None else "")
    line2 = ("0 standing   1 sitting   2 lying   Enter = model   s skip   b back   q quit      done: "
             + ", ".join(f"{CLASSES[c]} {counts[c]}" for c in CLASSES))
    head = np.full((70, body.shape[1], 3), 25, np.uint8)
    cv2.putText(head, line1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(head, line2, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (190, 190, 190), 1, cv2.LINE_AA)
    img = np.vstack([head, body])
    if img.shape[1] > max_width:
        s = max_width / img.shape[1]
        img = cv2.resize(img, (max_width, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return img


def screen():
    """-> (show, wait_key) on an OpenCV window."""
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    def show(img):
        cv2.imshow(WINDOW, img)

    def wait_key():
        while True:
            k = cv2.waitKey(100)
            if k != -1:
                return k & 0xFF
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return ord("q")                          # window closed
    return show, wait_key


# ------------------------------------------------------------------ the loop

def label_loop(frames, out, show, wait_key, already=()):
    """Show each picture not yet labelled and save the answers. -> (answers per class, pictures left)."""
    lf = LabelFile(out)
    done = labelled_keys([out, *already])
    todo = [f for f in frames if key_of(f) not in done]
    counts, answered, i = Counter(), [], 0
    while i < len(todo):
        f = todo[i]
        show(compose(f, i, len(todo), counts))
        k = wait_key()
        if k in (ord("q"), 27):
            break
        if k == ord("s"):
            i += 1
            continue
        if k == ord("b"):
            last = lf.undo()
            if last is not None and answered:
                i = answered.pop()
                counts[int(last["label"])] -= 1
            continue
        label = KEYS.get(k)
        if label is None and k in (10, 13) and f.get("guess") is not None:
            label = f["guess"]
        if label is None:
            continue
        lf.add(f, label)
        counts[label] += 1
        answered.append(i)
        i += 1
    return counts, len(todo) - i


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="a data folder (timestamp folders) or a predictions.csv from run_inference.py")
    ap.add_argument("--out", default=os.path.join("labels", "posture_new_labels.csv"),
                    help="CSV the answers go to (created, or continued if it exists)")
    ap.add_argument("--already", nargs="*", default=[], help="label files whose pictures are not shown again")
    ap.add_argument("--first", choices=list(CLASSES.values()), default="",
                    help="predictions.csv only: the pictures most likely in this posture first")
    ap.add_argument("--pigs", default="", help="stalls to show, e.g. 0-11 or 3,5,7")
    ap.add_argument("--limit", type=int, default=0, help="show at most this many pictures")
    ap.add_argument("--data-root", default="", help="predictions.csv only: where the timestamp folders are now")
    args = ap.parse_args()
    if args.source.lower().endswith(".csv"):
        frames = frames_from_predictions(args.source, args.data_root)
        found = [f for f in frames if os.path.exists(f["ir_jpg"])]
        if frames and not found:
            sys.exit(f"None of the {len(frames)} pictures in {args.source} were found where the run saw them; "
                     "give --data-root <folder that holds their timestamp folders>")
        frames = found
    else:
        frames = frames_from_folder(args.source)
    if not frames:
        sys.exit(f"No sow pictures (pigN_ir_vis_<time>.jpg) found in {args.source}")
    frames = select(frames, args.first, parse_pigs(args.pigs), args.limit)
    print(f"{len(frames)} pictures; answers go to {args.out}")
    show, wait_key = screen()
    counts, left = label_loop(frames, args.out, show, wait_key, args.already)
    cv2.destroyAllWindows()
    print(f"Saved {sum(counts.values())} answers this session ("
          + ", ".join(f"{CLASSES[c]} {counts[c]}" for c in CLASSES) + f") to {args.out}; {left} pictures left")


if __name__ == "__main__":
    main()
