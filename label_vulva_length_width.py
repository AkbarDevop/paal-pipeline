#!/usr/bin/env python3
"""Dataset-wide vulva length/width labeling on IR images.

Workflow per frame:
1. Review the IR image with RGB/depth context.
2. Choose a status:
   - p: vulva present, then click 2 points for length and 2 points for width
   - u: vulva not clear
   - s: skip for now
   - q: quit session
3. Save one CSV row with timeslot, pig ID, L, W, and estimated area.

Area is estimated from the two clicked measurements using `0.5 * L * W`.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import time

import cv2
import numpy as np

from config import IMAGE_DIR, VULVA_LENGTH_WIDTH_CSV
from point_cloud import load_calibration, load_depth_raw
from scan_data import parse_filename


WINDOW_NAME = "Vulva Length/Width Labeling"
STATUS_KEYS = {
    ord("p"): "present",
    ord("u"): "not_clear",
    ord("s"): "skip",
}
CSV_FIELDS = [
    "timeslot_name",
    "pig_id",
    "status",
    "L_mm",
    "GT_L_mm",
    "diff_L_mm",
    "W_mm",
    "GT_W_mm",
    "diff_W_mm",
    "area_mm2",
    "area_method",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parent


GROUND_TRUTH_CSV = repo_root() / "ground_truths" / "vulva_ground_truths.csv"


def to_repo_rel(path_str: str | Path | None) -> str:
    if not path_str:
        return ""
    path = Path(path_str)
    try:
        return path.resolve().relative_to(repo_root()).as_posix()
    except ValueError:
        return str(path)


def scan_live_rows(image_dir: str) -> list[dict]:
    records: dict[tuple[str, int], dict] = {}
    for folder in sorted(Path(image_dir).iterdir()):
        if not folder.is_dir():
            continue
        for file_path in folder.iterdir():
            parsed = parse_filename(file_path.name)
            if not parsed:
                continue

            key = (folder.name, parsed["pig_id"])
            record = records.setdefault(
                key,
                {
                    "timestamp_folder": folder.name,
                    "pig_id": parsed["pig_id"],
                    "pig_timestamp": parsed["timestamp"],
                    "rgb_jpg": "",
                    "ir_jpg": "",
                    "depth_jpg": "",
                    "rgb_raw": "",
                    "ir_raw": "",
                    "depth_raw": "",
                },
            )

            modality = parsed["modality"]
            ext = parsed["ext"]
            if parsed["cropped"]:
                continue

            file_rel = to_repo_rel(file_path)
            if modality == "rgb" and ext == "jpg":
                record["rgb_jpg"] = file_rel
            elif modality == "rgb" and ext == "raw":
                record["rgb_raw"] = file_rel
            elif modality == "ir_vis" and ext == "jpg":
                record["ir_jpg"] = file_rel
            elif modality == "ir" and ext == "raw":
                record["ir_raw"] = file_rel
            elif modality == "depth_vis" and ext == "jpg":
                record["depth_jpg"] = file_rel
            elif modality == "depth" and ext == "raw":
                record["depth_raw"] = file_rel

    return sorted(records.values(), key=lambda row: (row["timestamp_folder"], row["pig_id"]))


def load_existing_rows(csv_path: str) -> list[dict]:
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict) -> tuple[str, str]:
    return (row["timeslot_name"], str(row["pig_id"]))


def derive_measurement_date(timeslot_name: str) -> str:
    token = (timeslot_name or "").strip()
    if len(token) >= 8 and token[:8].isdigit():
        return f"{token[:4]}-{token[4:6]}-{token[6:8]}"
    return ""


def coerce_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_number(value, digits: int = 2):
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def load_ground_truth_index(csv_path: Path) -> dict[tuple[str, str], dict]:
    if not csv_path.exists():
        return {}

    gt_index: dict[tuple[str, str], dict] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            date_value = (row.get("date") or "").strip()
            pig_id = str(row.get("pig_id", "")).strip()
            if not date_value or not pig_id:
                continue
            gt_index[(date_value, pig_id)] = row
    return gt_index


def enrich_with_ground_truth(output: dict, ground_truth_index: dict[tuple[str, str], dict]):
    measurement_date = derive_measurement_date(output["timeslot_name"])
    gt_row = ground_truth_index.get((measurement_date, str(output["pig_id"])))

    gt_length = coerce_float(gt_row.get("gt_vl_mm")) if gt_row else None
    gt_width = coerce_float(gt_row.get("gt_vw_mm")) if gt_row else None
    length_mm = coerce_float(output.get("L_mm"))
    width_mm = coerce_float(output.get("W_mm"))

    output["GT_L_mm"] = format_number(gt_length, 2)
    output["GT_W_mm"] = format_number(gt_width, 2)
    output["diff_L_mm"] = format_number(length_mm - gt_length, 2) if length_mm is not None and gt_length is not None else ""
    output["diff_W_mm"] = format_number(width_mm - gt_width, 2) if width_mm is not None and gt_width is not None else ""
    return output


def normalize_saved_row(row: dict, ground_truth_index: dict[tuple[str, str], dict]):
    length_mm = coerce_float(row.get("L_mm"))
    width_mm = coerce_float(row.get("W_mm"))
    area_mm2 = coerce_float(row.get("area_mm2"))
    if area_mm2 is None and length_mm is not None and width_mm is not None:
        area_mm2 = estimate_area_mm2(length_mm, width_mm)

    normalized = {
        "timeslot_name": row.get("timeslot_name", ""),
        "pig_id": row.get("pig_id", ""),
        "status": row.get("status", ""),
        "L_mm": format_number(length_mm, 2),
        "GT_L_mm": "",
        "diff_L_mm": "",
        "W_mm": format_number(width_mm, 2),
        "GT_W_mm": "",
        "diff_W_mm": "",
        "area_mm2": format_number(area_mm2, 2),
        "area_method": row.get("area_method", "half_LxW") if area_mm2 is not None else "",
    }
    return enrich_with_ground_truth(normalized, ground_truth_index)


def write_rows(csv_path: str, rows: list[dict]):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    target = Path(csv_path)
    temp = target.with_suffix(target.suffix + ".tmp")
    last_error = None

    for attempt in range(8):
        try:
            with open(temp, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                writer.writeheader()
                for row in rows:
                    writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
            os.replace(temp, target)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.25 * (attempt + 1))
        except OSError as exc:
            last_error = exc
            time.sleep(0.25 * (attempt + 1))
        finally:
            try:
                if temp.exists():
                    temp.unlink()
            except OSError:
                pass

    raise PermissionError(
        f"Could not write {csv_path}. Close the CSV if it is open in Excel/another app and try again."
    ) from last_error


def load_image(path_str: str, flags: int) -> np.ndarray | None:
    if not path_str:
        return None
    path = repo_root() / path_str
    if not path.exists():
        return None
    return cv2.imread(str(path), flags)


def fit_for_annotation(img: np.ndarray, max_h: int = 960, max_w: int = 1280):
    height, width = img.shape[:2]
    scale = min(max_h / height, max_w / width)
    if scale <= 0:
        scale = 1.0
    if abs(scale - 1.0) < 1e-6:
        return img.copy(), 1.0
    resized = cv2.resize(
        img,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_CUBIC,
    )
    return resized, scale


def make_review_panel(row: dict):
    ir = load_image(row.get("ir_jpg", ""), cv2.IMREAD_GRAYSCALE)
    if ir is None:
        return None, None

    ir_bgr = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
    rgb = load_image(row.get("rgb_jpg", ""), cv2.IMREAD_COLOR)
    depth = load_image(row.get("depth_jpg", ""), cv2.IMREAD_COLOR)

    if rgb is None:
        rgb = np.full_like(ir_bgr, 30)
        cv2.putText(rgb, "RGB missing", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 180, 255), 2, cv2.LINE_AA)
    if depth is None:
        depth = np.full_like(ir_bgr, 30)
        cv2.putText(depth, "Depth preview missing", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2, cv2.LINE_AA)

    ir_main = cv2.resize(ir_bgr, (960, 720), interpolation=cv2.INTER_CUBIC)
    rgb_side = cv2.resize(rgb, (320, 240), interpolation=cv2.INTER_CUBIC)
    depth_side = cv2.resize(depth, (320, 240), interpolation=cv2.INTER_CUBIC)

    cv2.putText(ir_main, "IR image (click points here)", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(rgb_side, "RGB context", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(depth_side, "Depth context", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    right = np.vstack([rgb_side, depth_side, np.full((240, 320, 3), 18, dtype=np.uint8)])
    info = right[480:720]
    info_lines = [
        f"pig{row['pig_id']} | {row['timestamp_folder']}",
        f"frame time: {row['pig_timestamp']}",
        "Keys: p=present  u=not clear  s=skip  q=quit",
        "After p: click 2 points for L, then 2 points for W",
    ]
    y = 38
    for line in info_lines:
        cv2.putText(info, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        y += 34

    panel = np.hstack([ir_main, right])
    return panel, ir


def draw_measurement_overlay(img: np.ndarray, length_points: list[tuple[int, int]], width_points: list[tuple[int, int]]):
    if img.ndim == 2:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        vis = img.copy()

    def draw_pair(points: list[tuple[int, int]], color: tuple[int, int, int], prefix: str):
        for idx, point in enumerate(points, start=1):
            cv2.circle(vis, point, 5, color, -1)
            cv2.putText(vis, f"{prefix}{idx}", (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        if len(points) == 2:
            cv2.line(vis, points[0], points[1], color, 2, cv2.LINE_AA)

    draw_pair(length_points, (0, 0, 255), "L")
    draw_pair(width_points, (255, 0, 0), "W")
    return vis


class LinePairAnnotator:
    def __init__(self, img: np.ndarray):
        self.img = img
        self.length_points: list[tuple[int, int]] = []
        self.width_points: list[tuple[int, int]] = []
        self.cancelled = False

    def current_stage(self) -> str:
        if len(self.length_points) < 2:
            return "length"
        if len(self.width_points) < 2:
            return "width"
        return "done"

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            stage = self.current_stage()
            if stage == "length":
                self.length_points.append((x, y))
            elif stage == "width":
                self.width_points.append((x, y))
            self.show()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.width_points:
                self.width_points.pop()
            elif self.length_points:
                self.length_points.pop()
            self.show()

    def show(self, footer: str = ""):
        vis = draw_measurement_overlay(self.img, self.length_points, self.width_points)
        stage_text = {
            "length": "Click 2 points for length",
            "width": "Click 2 points for width",
            "done": "Press Enter to save, R to redraw, Q to cancel",
        }[self.current_stage()]
        cv2.putText(vis, stage_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, "Left=add point  Right=undo  R=reset  Q=cancel", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        if footer:
            cv2.putText(vis, footer, (12, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, vis)

    def run(self):
        cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback)
        self.show()
        while True:
            key = cv2.waitKey(30) & 0xFF
            if key == ord("r"):
                self.length_points = []
                self.width_points = []
                self.show()
            elif key == ord("q"):
                self.cancelled = True
                break
            elif key == 13:
                if len(self.length_points) == 2 and len(self.width_points) == 2:
                    break
                self.show("Need 2 length points and 2 width points")

        cv2.setMouseCallback(WINDOW_NAME, lambda *args: None)
        if self.cancelled:
            return None
        return self.length_points, self.width_points


def load_depth_context(row: dict):
    depth_raw_rel = row.get("depth_raw", "")
    if not depth_raw_rel:
        raise FileNotFoundError("depth_raw path missing for this frame")
    depth_raw_path = repo_root() / depth_raw_rel
    depth_mm = load_depth_raw(str(depth_raw_path))
    intrinsics = load_calibration(None, str(depth_raw_path.parent))
    return depth_mm, intrinsics


def nearest_valid_depth(depth_mm: np.ndarray, point: tuple[int, int], max_radius: int = 5):
    x, y = point
    height, width = depth_mm.shape
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("Clicked point is outside the depth image")

    for radius in range(max_radius + 1):
        x1 = max(0, x - radius)
        x2 = min(width - 1, x + radius)
        y1 = max(0, y - radius)
        y2 = min(height - 1, y + radius)
        patch = depth_mm[y1:y2 + 1, x1:x2 + 1]
        valid = (patch >= 200) & (patch <= 5000)
        if np.any(valid):
            ys, xs = np.where(valid)
            xs = xs + x1
            ys = ys + y1
            distances = (xs - x) ** 2 + (ys - y) ** 2
            best = int(np.argmin(distances))
            px = int(xs[best])
            py = int(ys[best])
            return (px, py), float(depth_mm[py, px])

    raise ValueError("No valid depth found near the clicked point")


def backproject_mm(point: tuple[int, int], depth_value_mm: float, intrinsics: dict):
    x, y = point
    z = float(depth_value_mm)
    x_mm = (float(x) - float(intrinsics["cx"])) * z / float(intrinsics["fx"])
    y_mm = (float(y) - float(intrinsics["cy"])) * z / float(intrinsics["fy"])
    return np.array([x_mm, y_mm, z], dtype=np.float32)


def line_distance_mm(depth_mm: np.ndarray, intrinsics: dict, p1: tuple[int, int], p2: tuple[int, int]):
    rp1, d1 = nearest_valid_depth(depth_mm, p1)
    rp2, d2 = nearest_valid_depth(depth_mm, p2)
    xyz1 = backproject_mm(rp1, d1, intrinsics)
    xyz2 = backproject_mm(rp2, d2, intrinsics)
    dist = float(np.linalg.norm(xyz2 - xyz1))
    return round(dist, 2)


def estimate_area_mm2(length_mm: float, width_mm: float):
    return round(0.5 * float(length_mm) * float(width_mm), 2)


def verify_measurements(row: dict, length_points: list[tuple[int, int]], width_points: list[tuple[int, int]], length_mm: float, width_mm: float, area_mm2: float):
    panels = []
    for label, path_key, flags in [
        ("IR", "ir_jpg", cv2.IMREAD_GRAYSCALE),
        ("RGB", "rgb_jpg", cv2.IMREAD_COLOR),
        ("Depth", "depth_jpg", cv2.IMREAD_COLOR),
    ]:
        image = load_image(row.get(path_key, ""), flags)
        if image is None:
            continue
        overlay = draw_measurement_overlay(image, length_points, width_points)
        cv2.putText(overlay, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(cv2.resize(overlay, (420, 315), interpolation=cv2.INTER_CUBIC))

    if not panels:
        return True

    verify = np.hstack(panels)
    canvas = np.full((verify.shape[0] + 86, verify.shape[1], 3), 18, dtype=np.uint8)
    canvas[86:, :] = verify
    cv2.putText(canvas, "Enter=save  R=redraw  Q=cancel", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"L={length_mm:.2f} mm  W={width_mm:.2f} mm  Area={area_mm2:.2f} mm^2", (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

    while True:
        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(0) & 0xFF
        if key == 13:
            return True
        if key == ord("r"):
            return False
        if key == ord("q"):
            return None


def draw_length_width_on_ir(row: dict, ir_gray: np.ndarray):
    ir_bgr = cv2.cvtColor(ir_gray, cv2.COLOR_GRAY2BGR)
    canvas, scale = fit_for_annotation(ir_bgr)
    annotator = LinePairAnnotator(canvas)
    result = annotator.run()
    if result is None:
        return None

    length_points_disp, width_points_disp = result
    length_points = [(int(round(x / scale)), int(round(y / scale))) for x, y in length_points_disp]
    width_points = [(int(round(x / scale)), int(round(y / scale))) for x, y in width_points_disp]

    depth_mm, intrinsics = load_depth_context(row)
    length_mm = line_distance_mm(depth_mm, intrinsics, length_points[0], length_points[1])
    width_mm = line_distance_mm(depth_mm, intrinsics, width_points[0], width_points[1])
    area_mm2 = estimate_area_mm2(length_mm, width_mm)

    verdict = verify_measurements(row, length_points, width_points, length_mm, width_mm, area_mm2)
    if verdict is True:
        return {
            "length_points": length_points,
            "width_points": width_points,
            "length_mm": length_mm,
            "width_mm": width_mm,
            "area_mm2": area_mm2,
        }
    if verdict is False:
        return draw_length_width_on_ir(row, ir_gray)
    return None


def build_output_row(row: dict, status: str, measurement: dict | None):
    output = {
        "timeslot_name": row["timestamp_folder"],
        "pig_id": row["pig_id"],
        "status": status,
        "L_mm": "",
        "GT_L_mm": "",
        "diff_L_mm": "",
        "W_mm": "",
        "GT_W_mm": "",
        "diff_W_mm": "",
        "area_mm2": "",
        "area_method": "",
    }
    if measurement is not None:
        output["L_mm"] = format_number(measurement["length_mm"], 2)
        output["W_mm"] = format_number(measurement["width_mm"], 2)
        output["area_mm2"] = format_number(measurement["area_mm2"], 2)
        output["area_method"] = "half_LxW"
    return output


def main(args):
    all_rows = scan_live_rows(IMAGE_DIR)
    rows = list(all_rows)
    if args.pig is not None:
        rows = [row for row in rows if int(row["pig_id"]) == args.pig]
    candidate_rows = list(rows)

    ground_truth_index = load_ground_truth_index(GROUND_TRUTH_CSV)
    existing_rows = load_existing_rows(VULVA_LENGTH_WIDTH_CSV)
    existing_rows = [normalize_saved_row(row, ground_truth_index) for row in existing_rows]
    existing_index = {row_key(row): idx for idx, row in enumerate(existing_rows)}
    existing_keys = set(existing_index)
    unlabeled_rows = [row for row in candidate_rows if (row["timestamp_folder"], str(row["pig_id"])) not in existing_keys]
    rows = candidate_rows if args.force_relabel else unlabeled_rows

    if args.start_at > 1:
        rows = rows[args.start_at - 1 :]
    if args.limit:
        rows = rows[: args.limit]

    print(f"Frames scanned in images/: {len(all_rows)}")
    if args.pig is not None:
        print(f"Frames after --pig {args.pig}: {len(candidate_rows)}")
    print(f"Existing L/W rows: {len(existing_rows)}")
    print(f"Ground-truth matches loaded: {len(ground_truth_index)}")
    print(f"Unlabeled rows after CSV filter: {len(unlabeled_rows)}")
    if args.force_relabel:
        print("Force relabel mode: existing rows will be reopened and overwritten.")
    if args.session_limit > 0:
        print(f"Session limit: stop after {args.session_limit} present measurements")
    print(f"Frames available to label: {len(rows)}")
    print("Statuses: present / not_clear / skip")
    if not rows:
        print(f"CSV: {VULVA_LENGTH_WIDTH_CSV}")
        if args.force_relabel or unlabeled_rows:
            print("No rows matched the current filters.")
        else:
            print("All rows already have saved length/width labels. Use --force-relabel to reopen them.")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    labeled = 0
    present_labeled = 0

    for idx, row in enumerate(rows, start=1):
        panel, ir_gray = make_review_panel(row)
        if panel is None or ir_gray is None:
            print(f"[{idx}] pig{row['pig_id']} skipped automatically: missing IR image")
            continue

        panel = panel.copy()
        cv2.putText(panel, f"[{idx}/{len(rows)}]", (14, panel.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, panel)

        status = None
        while status is None:
            key = cv2.waitKey(0) & 0xFF
            if key in STATUS_KEYS:
                status = STATUS_KEYS[key]
            elif key == ord("q"):
                cv2.destroyAllWindows()
                print(f"Labeled this session: {labeled}")
                print(f"Present measured this session: {present_labeled}")
                print(f"Saved labels: {VULVA_LENGTH_WIDTH_CSV}")
                return

        if status == "skip":
            continue

        measurement = None
        if status == "present":
            try:
                measurement = draw_length_width_on_ir(row, ir_gray)
            except Exception as exc:
                print(f"[{idx}] pig{row['pig_id']} measurement failed: {exc}")
                continue
            if measurement is None:
                continue

        output_row = build_output_row(row, status, measurement)
        enrich_with_ground_truth(output_row, ground_truth_index)
        key = row_key(output_row)
        replaced = key in existing_index
        if replaced:
            existing_rows[existing_index[key]] = output_row
        else:
            existing_index[key] = len(existing_rows)
            existing_rows.append(output_row)

        write_rows(VULVA_LENGTH_WIDTH_CSV, existing_rows)
        labeled += 1
        if status == "present":
            present_labeled += 1
        action = "updated" if replaced else "saved"
        if measurement is None:
            print(f"[{idx}] pig{row['pig_id']} -> {status} ({action})")
        else:
            print(
                f"[{idx}] pig{row['pig_id']} -> present ({action}) | "
                f"L={measurement['length_mm']:.2f} mm W={measurement['width_mm']:.2f} mm "
                f"A={measurement['area_mm2']:.2f} mm^2"
            )

        if args.session_limit > 0 and present_labeled >= args.session_limit:
            cv2.destroyAllWindows()
            print(f"Labeled this session: {labeled}")
            print(f"Reached session limit: {args.session_limit} present measurements")
            print(f"Saved labels: {VULVA_LENGTH_WIDTH_CSV}")
            return

    cv2.destroyAllWindows()
    print(f"Labeled this session: {labeled}")
    print(f"Present measured this session: {present_labeled}")
    print(f"Saved labels: {VULVA_LENGTH_WIDTH_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label vulva length and width from IR images")
    parser.add_argument("--pig", type=int, default=None, help="Only label this pig ID")
    parser.add_argument("--start-at", type=int, default=1, help="Start at this 1-based frame index after filtering")
    parser.add_argument("--limit", type=int, default=0, help="Only show this many frames")
    parser.add_argument("--session-limit", type=int, default=20, help="Stop after this many present measurements in the current session (0 = no limit)")
    parser.add_argument("--force-relabel", action="store_true", help="Reopen already-labeled frames and overwrite their CSV rows")
    main(parser.parse_args())
