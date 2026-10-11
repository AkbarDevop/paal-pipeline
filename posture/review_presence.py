"""Contact sheets for checking the pig-presence decision by eye.

Two ways to pick frames:

  1. Ground-truth misses (after eval_ground_truth.py): stalls where Lucas recorded
     a sow but the pipeline produced no posture. Each tile shows what the camera
     saw, the recorded posture and why the frame was skipped.

       python posture/review_presence.py --misses outputs/gt_feb06/ground_truth_misses.csv

  2. Frames the depth gate called empty, near the threshold (needs no ground truth):

       python posture/review_presence.py --skipped outputs/gt_feb06/skipped_frames.csv \
           --min-depth 1463 --max-depth 1600 --n 96

Writes sheet_01.jpg, sheet_02.jpg, ... and review.csv (one row per tile with an
empty has_sow column to fill in) to --out. Reads images only; never writes
next to the raw data.
"""

import argparse
import csv
import os
import random
import sys
import pathlib

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from preprocess import crop_stall, read_image  # noqa: E402

TILE_W, TILE_H, CAPTION_H = 220, 260, 44


def load_rows(args):
    if args.misses:
        with open(args.misses, encoding="utf-8-sig") as f:
            rows = [r for r in csv.DictReader(f) if r.get("image_path")]
        if args.reason:
            rows = [r for r in rows if r["reason"] in args.reason]
        for r in rows:
            r["line1"] = f"pig {r['pig_id']}  {r['time'][5:]}"
            r["line2"] = f"GT {r['ground_truth']} | {r['reason']} {r.get('median_depth', '')}"
        return rows
    with open(args.skipped, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    reasons = args.reason or ["depth"]
    rows = [r for r in rows if r["reason"] in reasons and r.get("image_path")]
    if args.min_depth is not None or args.max_depth is not None:
        lo = args.min_depth if args.min_depth is not None else -1e9
        hi = args.max_depth if args.max_depth is not None else 1e9
        rows = [r for r in rows if r["median_depth"] and lo <= float(r["median_depth"]) <= hi]
    for r in rows:
        r["line1"] = f"pig {r['pig_id']}  {r['pig_timestamp'][4:13]}"
        r["line2"] = f"{r['reason']}  depth {r['median_depth']} mm"
    return rows


def tile(row):
    img = read_image(row["image_path"])
    canvas = np.full((TILE_H + CAPTION_H, TILE_W, 3), 255, np.uint8)
    if img is None:
        cv2.putText(canvas, "image missing", (10, TILE_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
    else:
        crop = crop_stall(img, row["image_path"])
        canvas[:TILE_H] = cv2.resize(crop, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
    cv2.putText(canvas, f"#{row['tile']} {row['line1']}", (4, TILE_H + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, row["line2"], (4, TILE_H + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (70, 70, 70), 1,
                cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--misses", help="ground_truth_misses.csv from eval_ground_truth.py")
    src.add_argument("--skipped", help="skipped_frames.csv from run_inference.py")
    ap.add_argument("--reason", nargs="*", choices=["depth", "detector", "corrupt", "id_policy", "no_frame"],
                    help="Only these skip reasons (default: all for --misses, depth for --skipped)")
    ap.add_argument("--min-depth", type=float, default=None)
    ap.add_argument("--max-depth", type=float, default=None)
    ap.add_argument("--n", type=int, default=96, help="Max tiles (random sample, fixed seed)")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--per-sheet", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="Output folder (default: presence_review/ next to the CSV)")
    args = ap.parse_args()

    rows = load_rows(args)
    if not rows:
        print("Nothing to show (no rows with an image path after filtering).")
        return
    random.Random(args.seed).shuffle(rows)
    rows = sorted(rows[:args.n], key=lambda r: (r["pig_id"].zfill(3), r.get("time") or r.get("pig_timestamp")))
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.misses or args.skipped)), "presence_review")
    os.makedirs(out, exist_ok=True)

    for i, r in enumerate(rows, 1):
        r["tile"] = i
    for s in range(0, len(rows), args.per_sheet):
        chunk = [tile(r) for r in rows[s:s + args.per_sheet]]
        while len(chunk) % args.cols:
            chunk.append(np.full_like(chunk[0], 255))
        grid = np.vstack([np.hstack(chunk[k:k + args.cols]) for k in range(0, len(chunk), args.cols)])
        path = os.path.join(out, f"sheet_{s // args.per_sheet + 1:02d}.jpg")
        cv2.imwrite(path, grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"  {path}")

    fields = ["tile", "pig_id", "time", "pig_timestamp", "frame", "ground_truth", "reason", "median_depth",
              "pig_present_prob", "image_path", "has_sow"]
    with open(os.path.join(out, "review.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(dict(r, has_sow=""))
    print(f"{len(rows)} tiles. Fill has_sow (y/n) in {os.path.join(out, 'review.csv')} while looking at the sheets.")


if __name__ == "__main__":
    main()
