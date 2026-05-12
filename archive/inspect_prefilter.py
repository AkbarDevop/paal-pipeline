"""Visual QA for depth prefilter results."""

import argparse
import csv
import os
import random

import cv2

from config import DATA_DIR, PRESENCE_CSV


def load_rows(csv_path, target):
    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))
    if target == "no_pig":
        return [r for r in rows if str(r.get("pig_present", "0")) == "0"]
    if target == "present":
        return [r for r in rows if str(r.get("pig_present", "0")) == "1"]
    return rows


def find_image(row, key):
    base = os.path.join(DATA_DIR, row["timestamp_folder"])
    ts = row.get("pig_timestamp", "")
    pid = row["pig_id"]
    if key == "rgb":
        name = f"pig{pid}_rgb_{ts}.jpg"
    elif key == "ir":
        name = f"pig{pid}_ir_vis_{ts}.jpg"
    else:
        name = f"pig{pid}_depth_vis_{ts}.jpg"
    path = os.path.join(base, name)
    return path if os.path.exists(path) else ""


def main(args):
    if not os.path.exists(args.csv):
        print(f"Missing CSV: {args.csv}")
        return

    rows = load_rows(args.csv, args.target)
    if not rows:
        print(f"No rows for target={args.target}")
        return

    random.seed(args.seed)
    random.shuffle(rows)
    rows = rows[: args.n]

    print(f"Showing {len(rows)} rows for target={args.target}")
    print("Press n=next, q=quit")

    for i, row in enumerate(rows, 1):
        rgb_p = find_image(row, "rgb")
        ir_p = find_image(row, "ir")
        depth_p = find_image(row, "depth")
        if not (rgb_p and ir_p and depth_p):
            continue

        rgb = cv2.imread(rgb_p)
        ir = cv2.imread(ir_p)
        depth = cv2.imread(depth_p)
        if rgb is None or ir is None or depth is None:
            continue

        h = min(rgb.shape[0], ir.shape[0], depth.shape[0])
        w = min(rgb.shape[1], ir.shape[1], depth.shape[1])
        rgb = cv2.resize(rgb, (w, h))
        ir = cv2.resize(ir, (w, h))
        depth = cv2.resize(depth, (w, h))
        panel = cv2.hconcat([rgb, ir, depth])

        title = (
            f"[{i}/{len(rows)}] {row['timestamp_folder']} pig{row['pig_id']} "
            f"present={row.get('pig_present')} ir_dark={row.get('ir_dark_ratio')} "
            f"depth_valid={row.get('depth_valid_ratio')} {row.get('reason', '')}"
        )
        cv2.putText(panel, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.imshow("Prefilter QA - RGB | IR | Depth", panel)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                cv2.destroyAllWindows()
                return
            if key == ord("n") or key == ord(" "):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect prefilter outputs visually")
    parser.add_argument("--csv", default=PRESENCE_CSV)
    parser.add_argument("--target", choices=["no_pig", "present", "all"], default="no_pig")
    parser.add_argument("-n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
