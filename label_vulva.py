#!/usr/bin/env python3
"""Vulva polygon annotation tool for Task B.

Displays standing pig frames (from posture3 labels) and lets the user:
1. Classify frame quality: good / tail_closed / too_close / bad / skip
2. If good: draw a polygon around the vulva region using mouse clicks
3. Saves binary masks as PNG + polygon coords to CSV

Usage:
    python label_vulva.py                    # Label all standing frames
    python label_vulva.py --pig 5            # Label only pig 5
    python label_vulva.py --use-cropped      # Use cropped images
    python label_vulva.py --show-ir          # Show IR side-by-side with RGB
"""

import argparse
import ast
import csv
import json
import os

import cv2
import numpy as np

from config import (
    LABELS_POSTURE3_CSV,
    METADATA_CSV,
    VULVA_LABELS_CSV,
    VULVA_MASK_DIR,
    resolve_path,
)

WINDOW_NAME = "Vulva Labeling"

QUALITY_KEYS = {
    ord("g"): "good",
    ord("t"): "tail_closed",
    ord("c"): "too_close",
    ord("b"): "bad",
    ord("s"): "skip",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_metadata_index(metadata_csv):
    """Load full frame metadata keyed by (timestamp_folder, pig_id)."""
    if not os.path.exists(metadata_csv):
        return {}

    out = {}
    with open(metadata_csv, "r") as f:
        for row in csv.DictReader(f):
            key = (row.get("timestamp_folder", ""), row.get("pig_id", ""))
            out[key] = row
    return out


def load_standing_frames(labels_csv, pig_id=None, metadata_csv=METADATA_CSV):
    """Load standing frames (label=0) from posture3 labels."""
    if not os.path.exists(labels_csv):
        print(f"Missing labels file: {labels_csv}")
        return []

    metadata = load_metadata_index(metadata_csv)
    rows = []
    with open(labels_csv, "r") as f:
        for row in csv.DictReader(f):
            if row.get("label", "").strip() != "0":  # standing only
                continue
            pid = int(row["pig_id"])
            if pig_id is not None and pid != pig_id:
                continue
            key = (row.get("timestamp_folder", ""), row.get("pig_id", ""))
            merged = dict(metadata.get(key, {}))
            merged.update(row)
            rows.append(merged)
    return rows


def load_existing_keys(vulva_csv):
    """Load already-labeled (timestamp_folder, pig_id) pairs."""
    if not os.path.exists(vulva_csv):
        return set()
    with open(vulva_csv, "r") as f:
        return {(r["timestamp_folder"], r["pig_id"]) for r in csv.DictReader(f)}


def get_image_path(row, key, use_cropped=False):
    """Get image path, preferring cropped version if requested."""
    if use_cropped:
        crop_key = key.replace("_jpg", "_cropped_jpg")
        p = resolve_path(row.get(crop_key, ""))
        if p and os.path.exists(p):
            return p
    p = resolve_path(row.get(key, ""))
    if p and os.path.exists(p):
        return p
    return ""


def scale_for_display(img, max_h=700, max_w=900):
    """Scale image to fit display, return (scaled_img, scale_factor)."""
    h, w = img.shape[:2]
    scale = min(max_h / h, max_w / w, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img, scale


def make_mask(polygon_points, shape):
    """Create binary mask from polygon points."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if len(polygon_points) >= 3:
        pts = np.array(polygon_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
    return mask


def draw_polygon_overlay(img, points, closed=False):
    """Draw polygon points and lines on image copy."""
    vis = img.copy()
    for i, pt in enumerate(points):
        cv2.circle(vis, pt, 4, (0, 255, 0), -1)
        if i > 0:
            cv2.line(vis, points[i - 1], pt, (0, 255, 0), 2)
    if closed and len(points) >= 3:
        cv2.line(vis, points[-1], points[0], (0, 255, 0), 2)
    elif len(points) >= 3:
        # Draw dashed closing line
        cv2.line(vis, points[-1], points[0], (0, 255, 0), 1)
    return vis


# ── Polygon drawing ─────────────────────────────────────────────────────────

class PolygonDrawer:
    """Interactive polygon drawing via mouse clicks."""

    def __init__(self, img):
        self.img = img
        self.points = []
        self.done = False
        self.cancelled = False

    def mouse_callback(self, event, x, y, flags, param):
        if self.done:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            self._update_display()
        elif event == cv2.EVENT_RBUTTONDOWN and self.points:
            self.points.pop()
            self._update_display()

    def _update_display(self):
        vis = draw_polygon_overlay(self.img, self.points)
        n = len(self.points)
        cv2.putText(vis, f"{n} points | Enter=finish, R=reset, Right-click=undo, Q=cancel",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow(WINDOW_NAME, vis)

    def run(self):
        """Run interactive polygon drawing. Returns list of points or None."""
        cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback)
        self._update_display()

        while not self.done:
            key = cv2.waitKey(30) & 0xFF
            if key == 13:  # Enter
                if len(self.points) >= 3:
                    self.done = True
                else:
                    # Flash message: need at least 3 points
                    vis = self.img.copy()
                    cv2.putText(vis, "Need at least 3 points!", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow(WINDOW_NAME, vis)
                    cv2.waitKey(800)
                    self._update_display()
            elif key == ord("r"):  # Reset
                self.points = []
                self._update_display()
            elif key == ord("q"):  # Cancel
                self.cancelled = True
                self.done = True

        cv2.setMouseCallback(WINDOW_NAME, lambda *a: None)

        if self.cancelled:
            return None
        return self.points


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    rows = load_standing_frames(args.labels_csv, pig_id=args.pig)
    if not rows:
        print("No standing frames found.")
        return

    existing = load_existing_keys(VULVA_LABELS_CSV)
    rows = [r for r in rows if (r["timestamp_folder"], r["pig_id"]) not in existing]
    print(f"Standing frames to label: {len(rows)} (already done: {len(existing)})")
    if not rows:
        print("All standing frames already labeled!")
        return

    os.makedirs(VULVA_MASK_DIR, exist_ok=True)

    file_exists = os.path.exists(VULVA_LABELS_CSV)
    fields = [
        "timestamp_folder", "pig_id", "pig_timestamp",
        "rgb_jpg", "rgb_aligned_jpg", "ir_jpg", "depth_jpg",
        "rgb_cropped_jpg", "rgb_aligned_cropped_jpg", "ir_cropped_jpg", "depth_cropped_jpg",
        "rgb_raw", "ir_raw", "depth_raw",
        "quality", "mask_space", "polygon_points", "mask_path",
    ]

    with open(VULVA_LABELS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()

        labeled = 0
        print("\nQuality keys: g=good (draw polygon), t=tail_closed, c=too_close, b=bad, s=skip, q=quit")
        print("When drawing: Left-click=add point, Right-click=undo, Enter=finish, R=reset, Q=cancel\n")

        for i, row in enumerate(rows, 1):
            rgb_path = get_image_path(row, "rgb_jpg", args.use_cropped)
            if not rgb_path:
                continue

            rgb = cv2.imread(rgb_path)
            if rgb is None:
                continue

            orig_h, orig_w = rgb.shape[:2]
            display_rgb, scale = scale_for_display(rgb)

            # Build title
            pid = row["pig_id"]
            ts = row["timestamp_folder"]
            title_text = f"[{i}/{len(rows)}] pig{pid} | {ts} | g/t/c/b/s/q?"

            # Show RGB (and optionally IR/depth side by side)
            show_img = display_rgb.copy()
            cv2.putText(show_img, title_text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            if args.show_ir or args.show_depth:
                side_key = "ir_jpg" if args.show_ir else "depth_jpg"
                side_path = get_image_path(row, side_key, args.use_cropped)
                if side_path:
                    side_img = cv2.imread(side_path)
                    if side_img is not None:
                        side_img = cv2.resize(side_img, (show_img.shape[1], show_img.shape[0]))
                        label = "IR" if args.show_ir else "Depth"
                        cv2.putText(side_img, label, (10, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                        show_img = np.hstack([show_img, side_img])

            cv2.imshow(WINDOW_NAME, show_img)

            # Step 1: Quality classification
            quality = None
            while quality is None:
                key = cv2.waitKey(0) & 0xFF
                if key in QUALITY_KEYS:
                    quality = QUALITY_KEYS[key]
                elif key == ord("q"):
                    cv2.destroyAllWindows()
                    print(f"\nLabeled this session: {labeled}")
                    return

            if quality == "skip":
                continue

            polygon_str = ""
            mask_path = ""
            mask_space = "rgb_cropped" if args.use_cropped else "rgb"

            # Step 2: If good, draw polygon
            if quality == "good":
                # Show just RGB for drawing
                cv2.putText(display_rgb, f"pig{pid} | Draw vulva polygon",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                cv2.imshow(WINDOW_NAME, display_rgb)

                drawer = PolygonDrawer(display_rgb)
                display_points = drawer.run()

                if display_points is None:
                    # User cancelled drawing — treat as skip
                    continue

                # Scale points back to original resolution
                orig_points = [(int(x / scale), int(y / scale)) for x, y in display_points]

                # Generate and save mask at original resolution
                mask = make_mask(orig_points, (orig_h, orig_w))
                mask_filename = f"pig{pid}_{row['pig_timestamp']}_mask.png"
                mask_full_path = os.path.join(VULVA_MASK_DIR, mask_filename)
                cv2.imwrite(mask_full_path, mask)

                polygon_str = str(orig_points)
                mask_path = os.path.join("labels", "vulva_masks", mask_filename)

                # Show verification: polygon on IR and depth
                verify_imgs = []
                for vkey, vlabel in [("rgb_jpg", "RGB"), ("ir_jpg", "IR"), ("depth_jpg", "Depth")]:
                    vpath = get_image_path(row, vkey, args.use_cropped)
                    if vpath:
                        vimg = cv2.imread(vpath)
                        if vimg is not None:
                            vimg = cv2.resize(vimg, (int(orig_w * scale), int(orig_h * scale)))
                            vimg = draw_polygon_overlay(vimg, display_points, closed=True)
                            cv2.putText(vimg, vlabel, (10, 25),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                            verify_imgs.append(vimg)

                if verify_imgs:
                    # Resize all to same height
                    min_h = min(v.shape[0] for v in verify_imgs)
                    verify_imgs = [cv2.resize(v, (int(v.shape[1] * min_h / v.shape[0]), min_h))
                                   for v in verify_imgs]
                    verify = np.hstack(verify_imgs)
                    cv2.putText(verify, "Verification — press any key to continue",
                                (10, min_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.imshow(WINDOW_NAME, verify)
                    cv2.waitKey(0)

            # Save to CSV
            writer.writerow({
                "timestamp_folder": row["timestamp_folder"],
                "pig_id": row["pig_id"],
                "pig_timestamp": row["pig_timestamp"],
                "rgb_jpg": row.get("rgb_jpg", ""),
                "rgb_aligned_jpg": row.get("rgb_aligned_jpg", ""),
                "ir_jpg": row.get("ir_jpg", ""),
                "depth_jpg": row.get("depth_jpg", ""),
                "rgb_cropped_jpg": row.get("rgb_cropped_jpg", ""),
                "rgb_aligned_cropped_jpg": row.get("rgb_aligned_cropped_jpg", ""),
                "ir_cropped_jpg": row.get("ir_cropped_jpg", ""),
                "depth_cropped_jpg": row.get("depth_cropped_jpg", ""),
                "rgb_raw": row.get("rgb_raw", ""),
                "ir_raw": row.get("ir_raw", ""),
                "depth_raw": row.get("depth_raw", ""),
                "quality": quality,
                "mask_space": mask_space,
                "polygon_points": polygon_str,
                "mask_path": mask_path,
            })
            f.flush()
            labeled += 1

            status = f"{'+ mask saved' if quality == 'good' else quality}"
            print(f"  [{i}] pig{pid} — {status}")

    cv2.destroyAllWindows()
    print(f"\nLabeled this session: {labeled}")
    print(f"Saved: {VULVA_LABELS_CSV}")
    if os.path.exists(VULVA_MASK_DIR):
        mask_count = len([f for f in os.listdir(VULVA_MASK_DIR) if f.endswith(".png")])
        print(f"Masks: {mask_count} in {VULVA_MASK_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vulva polygon annotation tool (Task B)")
    parser.add_argument("--pig", type=int, default=None, help="Label only this pig ID")
    parser.add_argument("--use-cropped", action="store_true", help="Use cropped images")
    parser.add_argument("--show-ir", action="store_true", help="Show IR alongside RGB")
    parser.add_argument("--show-depth", action="store_true", help="Show depth alongside RGB")
    parser.add_argument("--labels-csv", default=LABELS_POSTURE3_CSV,
                        help="Posture labels CSV (to find standing frames)")
    main(parser.parse_args())
