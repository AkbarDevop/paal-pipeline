"""Align RGB images to ToF frame using camera intrinsics.

OAK-D has RGB (1280x800) and ToF (640x480) sensors with known intrinsics.
For each ToF pixel, we compute the corresponding RGB pixel using:
  u_rgb = fx_rgb/fx_tof * (u_tof - cx_tof) + cx_rgb
  v_rgb = fy_rgb/fy_tof * (v_tof - cy_tof) + cy_rgb

This produces a 640x480 aligned RGB image matching the IR/depth frame.
Parallax from the 1.73cm baseline is <5px at typical depths, negligible
for classification purposes.
"""

import argparse
import json
import os
import random

import cv2
import numpy as np

from config import DATA_DIR


def load_calibration(cal_path):
    try:
        with open(cal_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None


def build_remap_tables(cal):
    """Build cv2.remap tables: for each ToF pixel, the corresponding RGB pixel."""
    tof = cal["tof_intrinsics"]
    rgb = cal["rgb_intrinsics"]

    h_t, w_t = tof["height"], tof["width"]
    u_t, v_t = np.meshgrid(np.arange(w_t, dtype=np.float32), np.arange(h_t, dtype=np.float32))

    u_rgb = (rgb["fx"] / tof["fx"]) * (u_t - tof["cx"]) + rgb["cx"]
    v_rgb = (rgb["fy"] / tof["fy"]) * (v_t - tof["cy"]) + rgb["cy"]

    return u_rgb, v_rgb


def align_rgb_to_tof(rgb_img, map_x, map_y):
    """Remap RGB image into the ToF coordinate frame."""
    return cv2.remap(rgb_img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def find_any_calibration():
    """Find first valid calibration.json (same camera for all folders)."""
    for folder in sorted(os.listdir(DATA_DIR)):
        folder_path = os.path.join(DATA_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        cal_path = os.path.join(folder_path, "calibration.json")
        cal = load_calibration(cal_path)
        if cal is not None:
            return cal
    return None


def main(args):
    folders = sorted(
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d)) and d[0].isdigit()
    )
    print(f"Found {len(folders)} data folders")

    cal = find_any_calibration()
    if cal is None:
        print("No valid calibration.json found anywhere")
        return

    map_x, map_y = build_remap_tables(cal)
    print(f"Remap: ToF {cal['tof_intrinsics']['width']}x{cal['tof_intrinsics']['height']} "
          f"<- RGB {cal['rgb_intrinsics']['width']}x{cal['rgb_intrinsics']['height']}")

    total_aligned = 0
    total_skipped = 0

    for folder in folders:
        folder_path = os.path.join(DATA_DIR, folder)
        files = os.listdir(folder_path)
        rgb_files = sorted(f for f in files if "_rgb_" in f and "aligned" not in f and f.endswith(".jpg"))
        folder_count = 0

        for rgb_name in rgb_files:
            out_name = rgb_name.replace("_rgb_", "_rgb_aligned_")
            out_path = os.path.join(folder_path, out_name)

            if os.path.exists(out_path) and not args.force:
                total_skipped += 1
                continue

            rgb_img = cv2.imread(os.path.join(folder_path, rgb_name))
            if rgb_img is None:
                continue

            aligned = align_rgb_to_tof(rgb_img, map_x, map_y)
            cv2.imwrite(out_path, aligned)
            total_aligned += 1
            folder_count += 1

        if folder_count > 0:
            print(f"  {folder}: aligned {folder_count} images")

    print(f"\nDone. New: {total_aligned}, Skipped (existing): {total_skipped}")
    print(f"Total aligned images: {total_aligned + total_skipped}")


def show_alignment(args):
    """Visual QA: show aligned RGB overlaid on IR to check alignment quality."""
    cal = find_any_calibration()
    if cal is None:
        print("No valid calibration.json found anywhere")
        return

    map_x, map_y = build_remap_tables(cal)

    folders = sorted(
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d)) and d[0].isdigit()
    )

    pairs = []
    for folder in folders:
        folder_path = os.path.join(DATA_DIR, folder)
        files = os.listdir(folder_path)
        rgb_files = sorted(f for f in files if "_rgb_" in f and "aligned" not in f and f.endswith(".jpg"))
        for rgb_name in rgb_files:
            ir_name = rgb_name.replace("_rgb_", "_ir_vis_")
            ir_path = os.path.join(folder_path, ir_name)
            if os.path.exists(ir_path):
                pairs.append((os.path.join(folder_path, rgb_name), ir_path, folder))
    if not pairs:
        print("No RGB+IR pairs found")
        return

    if args.random:
        random.shuffle(pairs)
    print(f"Found {len(pairs)} RGB+IR pairs. Showing {min(args.n, len(pairs))} ({'random' if args.random else 'sequential'}).")
    print("Keys: SPACE=next, q=quit")

    for i, (rgb_path, ir_path, folder) in enumerate(pairs[:args.n]):
        rgb_img = cv2.imread(rgb_path)
        ir_img = cv2.imread(ir_path)
        if rgb_img is None or ir_img is None:
            continue

        aligned = align_rgb_to_tof(rgb_img, map_x, map_y)

        ir_resized = cv2.resize(ir_img, (aligned.shape[1], aligned.shape[0]))
        h, w = aligned.shape[:2]

        # Edge detection on both images
        ir_gray = cv2.cvtColor(ir_resized, cv2.COLOR_BGR2GRAY)
        rgb_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        ir_edges = cv2.Canny(ir_gray, 50, 150) > 0
        rgb_edges = cv2.Canny(rgb_gray, 50, 150) > 0

        # Dilate edges slightly so nearby edges can overlap
        kernel = np.ones((3, 3), np.uint8)
        ir_dilated = cv2.dilate(ir_edges.astype(np.uint8), kernel, iterations=1) > 0
        rgb_dilated = cv2.dilate(rgb_edges.astype(np.uint8), kernel, iterations=1) > 0

        # Green = edges that overlap (aligned), Red = edges that don't match
        edge_map = np.zeros((h, w, 3), dtype=np.uint8)
        # Both edges present (aligned) -> green
        both = ir_edges & rgb_dilated | rgb_edges & ir_dilated
        edge_map[both] = (0, 255, 0)
        # IR edge only (misaligned) -> red
        ir_only = ir_edges & ~rgb_dilated
        edge_map[ir_only] = (0, 0, 255)
        # RGB edge only (misaligned) -> red
        rgb_only = rgb_edges & ~ir_dilated
        edge_map[rgb_only] = (0, 0, 255)

        # Overlay edges on blended background
        blend = cv2.addWeighted(aligned, 0.5, ir_resized, 0.5, 0)
        edge_mask = ir_edges | rgb_edges
        blend[edge_mask] = edge_map[edge_mask]

        # Count alignment quality
        total_edge_px = int(edge_mask.sum())
        green_px = int(both.sum())
        pct = green_px / total_edge_px * 100 if total_edge_px > 0 else 0

        # Panels: IR edges | RGB edges | overlay with green/red
        ir_edge_vis = cv2.cvtColor(ir_gray, cv2.COLOR_GRAY2BGR) // 2
        ir_edge_vis[ir_edges] = (0, 255, 0)

        rgb_edge_vis = aligned.copy() // 2
        rgb_edge_vis[rgb_edges] = (0, 255, 0)

        def make_label(text):
            lbl = np.zeros((30, w, 3), dtype=np.uint8)
            cv2.putText(lbl, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            return lbl

        canvas = np.hstack([
            np.vstack([make_label("IR edges"), ir_edge_vis]),
            np.vstack([make_label("Aligned RGB edges"), rgb_edge_vis]),
            np.vstack([make_label(f"Overlap: green=aligned, red=off ({pct:.0f}% match)"), blend]),
        ])

        title = f"[{i+1}/{min(args.n, len(pairs))}] {folder}"
        cv2.imshow(title, canvas)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if key == ord("q"):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align RGB to ToF frame")
    parser.add_argument("--force", action="store_true", help="Re-align even if output exists")
    parser.add_argument("--show", action="store_true", help="Visual QA: show aligned RGB vs IR")
    parser.add_argument("-n", type=int, default=10, help="Number of samples to show (with --show)")
    parser.add_argument("--random", action="store_true", help="Shuffle samples randomly (with --show)")
    args = parser.parse_args()
    if args.show:
        show_alignment(args)
    else:
        main(args)