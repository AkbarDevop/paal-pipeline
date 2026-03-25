"""Crop images to remove neighbor pigs and ceiling hardware.

Uses a fixed crop box (defined in config.CROP_TOF) determined from stall bar
positions across all frames. The crop is in ToF coordinates (640x480) and
scaled proportionally for RGB images (1280x800).

Cropped files are saved alongside originals with '_cropped' before the
extension, e.g. pig0_ir_vis_20260123-14-47-04_cropped.jpg
"""

import argparse
import os
import re

import cv2

from config import CROP_TOF, DATA_DIR

TOF_W, TOF_H = 640, 480


def crop_box_for_size(w, h):
    """Scale the ToF crop box to match image dimensions."""
    x1, y1, x2, y2 = CROP_TOF
    sx, sy = w / TOF_W, h / TOF_H
    return int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)


def cropped_path(original_path):
    """Insert '_cropped' before the file extension."""
    base, ext = os.path.splitext(original_path)
    return base + "_cropped" + ext


def is_croppable(fname):
    """Match image files we want to crop (jpg only, skip raw and calibration)."""
    return (
        fname.endswith(".jpg")
        and re.match(r"^pig\d+_", fname)
        and "_cropped" not in fname
    )


def main(args):
    data_dir = args.data_dir or DATA_DIR
    folders = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d[0].isdigit()
    )
    print(f"Found {len(folders)} data folders")
    print(f"Crop box (ToF coords): x={CROP_TOF[0]}..{CROP_TOF[2]}, y={CROP_TOF[1]}..{CROP_TOF[3]}")

    total_cropped = 0
    total_skipped = 0

    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        files = sorted(f for f in os.listdir(folder_path) if is_croppable(f))
        folder_count = 0

        for fname in files:
            src_path = os.path.join(folder_path, fname)
            out_path = cropped_path(src_path)

            if os.path.exists(out_path) and not args.force:
                total_skipped += 1
                continue

            img = cv2.imread(src_path)
            if img is None:
                continue

            h, w = img.shape[:2]
            x1, y1, x2, y2 = crop_box_for_size(w, h)
            cropped = img[y1:y2, x1:x2]
            cv2.imwrite(out_path, cropped)
            total_cropped += 1
            folder_count += 1

        if folder_count > 0:
            print(f"  {folder}: cropped {folder_count} images")

    print(f"\nDone. New: {total_cropped}, Skipped (existing): {total_skipped}")
    print(f"Total cropped images: {total_cropped + total_skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop images to stall region")
    parser.add_argument("--data-dir", default=None, help="Path to data folder (default: config.DATA_DIR)")
    parser.add_argument("--force", action="store_true", help="Re-crop even if output exists")
    main(parser.parse_args())
