"""Manual labeling tool."""

import argparse
import csv
import os

import cv2

from config import LABELS_CSV, LABELS_POSTURE3_CSV, METADATA_CSV

CLASS_SPECS = {
    "binary": {
        "csv": LABELS_CSV,
        "prompt": "Keys: 1=standing, 0=not-standing, s=skip, q=quit",
        "keys": {ord("0"): 0, ord("1"): 1},
    },
    "posture3": {
        "csv": LABELS_POSTURE3_CSV,
        "prompt": "Keys: 0=standing, 1=sitting, 2=lying, s=skip, q=quit",
        "keys": {ord("0"): 0, ord("1"): 1, ord("2"): 2},
    },
}


def load_metadata():
    if not os.path.exists(METADATA_CSV):
        print("Missing metadata.csv. Run scan_data.py first.")
        return []
    with open(METADATA_CSV, "r") as f:
        return list(csv.DictReader(f))


def load_existing_keys(labels_csv):
    if not os.path.exists(labels_csv):
        return set()
    with open(labels_csv, "r") as f:
        return {(r["timestamp_folder"], r["pig_id"]) for r in csv.DictReader(f)}


def pick_image_path(row, primary_key):
    for k in [primary_key, "rgb_jpg", "ir_jpg", "depth_jpg"]:
        p = row.get(k, "")
        if p and os.path.exists(p):
            return p
    return ""


def main(args):
    rows = load_metadata()
    if not rows:
        return

    if args.present_only:
        if not args.presence_csv or not os.path.exists(args.presence_csv):
            print("Missing presence CSV. Run prefilter_depth.py first or pass --presence-csv.")
            return
        with open(args.presence_csv, "r") as f:
            present_keys = {
                (r["timestamp_folder"], r["pig_id"])
                for r in csv.DictReader(f)
                if str(r.get("pig_present", "0")).strip() in {"1", "true", "True"}
            }
        rows = [r for r in rows if (r["timestamp_folder"], r["pig_id"]) in present_keys]
        print(f"After present-only filter: {len(rows)}")

    if args.pig is not None:
        rows = [r for r in rows if int(r["pig_id"]) == args.pig]

    spec = CLASS_SPECS[args.class_set]
    labels_csv = spec["csv"]

    existing = load_existing_keys(labels_csv)
    rows = [r for r in rows if (r["timestamp_folder"], r["pig_id"]) not in existing]
    print(f"Remaining to label: {len(rows)}")
    if not rows:
        return

    primary_key = "depth_jpg" if args.show_depth else "ir_jpg" if args.show_ir else "rgb_jpg"
    mode_name = "DEPTH" if args.show_depth else "IR" if args.show_ir else "RGB"

    file_exists = os.path.exists(labels_csv)
    with open(labels_csv, "a", newline="") as f:
        fields = ["timestamp_folder", "pig_id", "pig_timestamp", "rgb_jpg", "ir_jpg", "depth_jpg", "label"]
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()

        labeled = 0
        print(spec["prompt"])
        for i, row in enumerate(rows, 1):
            path = pick_image_path(row, primary_key)
            if not path:
                continue

            img = cv2.imread(path)
            if img is None:
                continue

            h, w = img.shape[:2]
            scale = min(600 / h, 800 / w, 1.0)
            if scale < 1.0:
                img = cv2.resize(img, (int(w * scale), int(h * scale)))

            title = f"[{i}/{len(rows)}] pig{row['pig_id']} | {row['timestamp_folder']} | {mode_name}"
            cv2.imshow(title, img)

            label = None
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key in spec["keys"]:
                    label = spec["keys"][key]
                    break
                if key == ord("s"):
                    label = -1
                    break
                if key == ord("q"):
                    cv2.destroyAllWindows()
                    print(f"Labeled this session: {labeled}")
                    return

            cv2.destroyAllWindows()
            if label == -1:
                continue

            writer.writerow(
                {
                    "timestamp_folder": row["timestamp_folder"],
                    "pig_id": row["pig_id"],
                    "pig_timestamp": row["pig_timestamp"],
                    "rgb_jpg": row.get("rgb_jpg", ""),
                    "ir_jpg": row.get("ir_jpg", ""),
                    "depth_jpg": row.get("depth_jpg", ""),
                    "label": label,
                }
            )
            f.flush()
            labeled += 1

    print(f"Labeled this session: {labeled}")
    print(f"Saved: {labels_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label posture classes")
    parser.add_argument("--pig", type=int, default=None)
    parser.add_argument("--show-ir", action="store_true")
    parser.add_argument("--show-depth", action="store_true")
    parser.add_argument("--class-set", choices=["binary", "posture3"], default="binary")
    parser.add_argument("--present-only", action="store_true")
    parser.add_argument("--presence-csv", default=None)
    main(parser.parse_args())
