#!/usr/bin/env python3
"""Build a pig depth-map bundle from OAK-D depth + IR.

This is the first 3D step for Task B:
depth raw -> pig mask -> point cloud -> textured mesh

The script can export either:
  - point-cloud-only mode: rear-view PLY point cloud, raw 3D color image data, and compact JSON summary
  - full mode: previews, mask, point cloud, mesh, bundle, masked pig RGB image, and summary

Examples:
    python point_cloud.py ^
      --depth-raw images\\20260211-09-17-32\\pig0_depth_20260211-09-17-49.raw ^
      --ir-img images\\20260211-09-17-32\\pig0_ir_vis_20260211-09-17-49.jpg

    python point_cloud.py ^
      --depth-raw images\\20260211-09-17-32\\pig0_depth_20260211-09-17-49.raw ^
      --mask labels\\vulva_masks\\pig0_20260211-09-17-49_mask.png
"""

import argparse
import json
import os
import re
import shutil
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

from config import CROP_TOF, OUTPUT_DIR

TOF_W, TOF_H = 640, 480
BAR_MARGIN = 50
FRAME_RE = re.compile(r"^(pig\d+)_(depth|depth_vis|ir|ir_vis|rgb|rgb_aligned)_(\d{8}-\d{2}-\d{2}-\d{2})(_cropped)?\.(jpg|raw)$")

DEFAULT_TOF_INTRINSICS = {
    "fx": 471.583,
    "fy": 471.430,
    "cx": 323.905,
    "cy": 246.816,
    "width": TOF_W,
    "height": TOF_H,
}


def load_depth_raw(path: str) -> np.ndarray:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Depth raw not found: {path}")

    size = os.path.getsize(path)
    expected = TOF_W * TOF_H * 2
    if size == expected + 8:
        raw = np.fromfile(path, dtype=np.uint16, offset=8)
    elif size == expected:
        raw = np.fromfile(path, dtype=np.uint16)
    else:
        raise ValueError(f"Unexpected depth raw size for {path}: {size} bytes")

    if raw.size != TOF_W * TOF_H:
        raise ValueError(f"Unexpected depth element count for {path}: {raw.size}")

    return raw.reshape((TOF_H, TOF_W))


def load_gray_raw(path: str, shape: Tuple[int, int]) -> np.ndarray:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Texture raw not found: {path}")

    h, w = shape
    expected = h * w
    size = os.path.getsize(path)
    if size == expected + 8:
        raw = np.fromfile(path, dtype=np.uint8, offset=8)
    elif size == expected:
        raw = np.fromfile(path, dtype=np.uint8)
    else:
        raise ValueError(f"Unexpected texture raw size for {path}: {size} bytes")

    if raw.size != expected:
        raise ValueError(f"Unexpected texture element count for {path}: {raw.size}")

    return raw.reshape((h, w))


def normalize_to_uint8(gray: np.ndarray) -> np.ndarray:
    gray = gray.astype(np.float32)
    if gray.size == 0:
        return np.zeros_like(gray, dtype=np.uint8)

    lo = float(np.percentile(gray, 1))
    hi = float(np.percentile(gray, 99))
    if hi <= lo:
        hi = lo + 1.0
    gray = np.clip((gray - lo) / (hi - lo), 0.0, 1.0)
    return (gray * 255.0).astype(np.uint8)


def infer_sibling_path(depth_raw: str, replacements) -> Optional[str]:
    folder = os.path.dirname(depth_raw)
    name = os.path.basename(depth_raw)
    match = FRAME_RE.match(name)
    if not match:
        return None

    pig_id = match.group(1)
    timestamp = match.group(3)
    for modality, ext in replacements:
        candidate = os.path.join(folder, f"{pig_id}_{modality}_{timestamp}.{ext}")
        if os.path.exists(candidate):
            return candidate
    return None


def load_texture_image(path: str, depth_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    if path.lower().endswith(".raw"):
        gray = load_gray_raw(path, depth_shape)
        gray = normalize_to_uint8(gray)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return rgb, gray

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to load texture image: {path}")

    if img.ndim == 2:
        gray = img
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if gray.shape[:2] != depth_shape:
        gray = cv2.resize(gray, (depth_shape[1], depth_shape[0]), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.resize(rgb, (depth_shape[1], depth_shape[0]), interpolation=cv2.INTER_LINEAR)
    return rgb, gray


def load_calibration(calibration_path: Optional[str], search_dir: str) -> Dict[str, float]:
    cal_path = calibration_path or os.path.join(search_dir, "calibration.json")
    if os.path.exists(cal_path):
        try:
            with open(cal_path, "r", encoding="utf-8") as f:
                cal = json.load(f)
            tof = cal.get("tof_intrinsics", {})
            if all(k in tof for k in ("fx", "fy", "cx", "cy")):
                return {
                    "fx": float(tof["fx"]),
                    "fy": float(tof["fy"]),
                    "cx": float(tof["cx"]),
                    "cy": float(tof["cy"]),
                    "width": int(tof.get("width", TOF_W)),
                    "height": int(tof.get("height", TOF_H)),
                    "calibration_path": cal_path,
                }
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            pass
    data = dict(DEFAULT_TOF_INTRINSICS)
    data["calibration_path"] = ""
    return data


def make_roi_mask(shape: Tuple[int, int], use_crop: bool) -> np.ndarray:
    mask = np.ones(shape, dtype=bool)
    if not use_crop:
        return mask

    x1, y1, x2, y2 = CROP_TOF
    out = np.zeros(shape, dtype=bool)
    left = min(x2, x1 + BAR_MARGIN)
    right = max(left, x2 - BAR_MARGIN)
    out[y1:y2, left:right] = True
    return out


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask.astype(np.uint8) * 255)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num <= 1:
        return mask

    best_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == best_idx


def select_body_component(mask: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask.astype(np.uint8) * 255)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num <= 1:
        return mask

    ys, xs = np.where(roi_mask)
    if len(xs) == 0:
        return largest_connected_component(mask)

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    body_seed = np.zeros_like(mask, dtype=bool)
    cy1 = int(y1 + 0.55 * max(1, y2 - y1))
    cx1 = int(x1 + 0.20 * max(1, x2 - x1))
    cx2 = int(x1 + 0.80 * max(1, x2 - x1))
    body_seed[cy1:y2 + 1, cx1:cx2 + 1] = True

    best_idx = -1
    best_score = -1
    for idx in range(1, num):
        component = labels == idx
        overlap = int((component & body_seed).sum())
        area = int(stats[idx, cv2.CC_STAT_AREA])
        score = overlap * 10 + area
        if overlap > 0 and score > best_score:
            best_idx = idx
            best_score = score

    if best_idx < 0:
        return largest_connected_component(mask)
    return labels == best_idx


def auto_segment_pig(
    depth_mm: np.ndarray,
    roi_mask: np.ndarray,
    min_depth: int,
    max_depth: int,
    foreground_margin_mm: int,
    min_pixels: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    valid = roi_mask & (depth_mm >= min_depth) & (depth_mm <= max_depth)
    values = depth_mm[valid]
    if values.size == 0:
        raise ValueError("No valid depth pixels inside the selected ROI")

    wall_depth = float(np.percentile(values, 90))
    cutoff = wall_depth - float(foreground_margin_mm)
    candidate = valid & (depth_mm < cutoff)

    if int(candidate.sum()) < min_pixels:
        cutoff = float(np.percentile(values, 65))
        candidate = valid & (depth_mm <= cutoff)

    if int(candidate.sum()) < min_pixels:
        cutoff = float(np.percentile(values, 80))
        candidate = valid & (depth_mm <= cutoff)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cleaned = cv2.morphologyEx(candidate.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    mask = select_body_component(cleaned > 0, roi_mask)
    mask = cv2.dilate(mask.astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1) > 0
    mask = select_body_component(mask, roi_mask)
    mask = mask & valid

    stats = {
        "wall_depth_mm": round(wall_depth, 2),
        "foreground_cutoff_mm": round(float(cutoff), 2),
        "mask_pixels": int(mask.sum()),
        "roi_valid_pixels": int(valid.sum()),
    }
    return mask, stats


def load_external_mask(path: str, shape: Tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to load mask: {path}")
    if mask.shape[:2] != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 127


def build_point_cloud(
    depth_mm: np.ndarray,
    texture_rgb: np.ndarray,
    mask: np.ndarray,
    intrinsics: Dict[str, float],
    stride: int,
):
    h, w = depth_mm.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    keep = mask[::stride, ::stride]
    xs = xs[keep].astype(np.float32)
    ys = ys[keep].astype(np.float32)
    z_mm = depth_mm[::stride, ::stride][keep].astype(np.float32)
    colors = texture_rgb[::stride, ::stride][keep].astype(np.uint8)

    z_m = z_mm / 1000.0
    x_m = (xs - float(intrinsics["cx"])) * z_m / float(intrinsics["fx"])
    y_m = (ys - float(intrinsics["cy"])) * z_m / float(intrinsics["fy"])
    vertices = np.stack([x_m, y_m, z_m], axis=1)
    return vertices, colors


def build_textured_mesh(
    depth_mm: np.ndarray,
    texture_rgb: np.ndarray,
    mask: np.ndarray,
    intrinsics: Dict[str, float],
    stride: int,
    max_edge_mm: float,
):
    h, w = depth_mm.shape
    idx_map = np.full((h, w), -1, dtype=np.int32)
    vertices = []
    colors = []
    uvs = []

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            if not mask[y, x]:
                continue
            z_mm = float(depth_mm[y, x])
            z_m = z_mm / 1000.0
            vx = (x - float(intrinsics["cx"])) * z_m / float(intrinsics["fx"])
            vy = (y - float(intrinsics["cy"])) * z_m / float(intrinsics["fy"])
            idx_map[y, x] = len(vertices)
            vertices.append((vx, vy, z_m))
            colors.append(tuple(int(v) for v in texture_rgb[y, x]))
            uvs.append((x / max(w - 1, 1), 1.0 - y / max(h - 1, 1)))

    faces = []
    for y in range(0, h - stride, stride):
        for x in range(0, w - stride, stride):
            quad = [(y, x), (y, x + stride), (y + stride, x), (y + stride, x + stride)]
            ids = [idx_map[qy, qx] for qy, qx in quad]
            if min(ids) < 0:
                continue

            depths = [float(depth_mm[qy, qx]) for qy, qx in quad]
            if max(depths) - min(depths) > max_edge_mm:
                continue

            a, b, c, d = ids
            faces.append((a, b, c))
            faces.append((b, d, c))

    return (
        np.asarray(vertices, dtype=np.float32),
        np.asarray(colors, dtype=np.uint8),
        np.asarray(uvs, dtype=np.float32),
        np.asarray(faces, dtype=np.int32),
    )


def write_point_cloud_ply(path: str, vertices: np.ndarray, colors: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(vertices, colors):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def write_textured_obj(
    obj_path: str,
    vertices: np.ndarray,
    uvs: np.ndarray,
    faces: np.ndarray,
    texture_name: str,
) -> None:
    mtl_name = os.path.splitext(os.path.basename(obj_path))[0] + ".mtl"
    mtl_path = os.path.join(os.path.dirname(obj_path), mtl_name)

    with open(mtl_path, "w", encoding="utf-8") as f:
        f.write("newmtl ir_texture\n")
        f.write("Ka 1.000 1.000 1.000\n")
        f.write("Kd 1.000 1.000 1.000\n")
        f.write("Ks 0.000 0.000 0.000\n")
        f.write(f"map_Kd {texture_name}\n")

    with open(obj_path, "w", encoding="utf-8") as f:
        f.write(f"mtllib {mtl_name}\n")
        f.write("usemtl ir_texture\n")
        for x, y, z in vertices:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for u, v in uvs:
            f.write(f"vt {u:.6f} {v:.6f}\n")
        for a, b, c in faces:
            a += 1
            b += 1
            c += 1
            f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")


def save_mask(path: str, mask: np.ndarray) -> None:
    cv2.imwrite(path, (mask.astype(np.uint8) * 255))


def make_depth_visualization(
    depth_mm: np.ndarray,
    mask: np.ndarray,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    low_mm: Optional[float] = None,
    high_mm: Optional[float] = None,
) -> np.ndarray:
    vis = np.zeros((depth_mm.shape[0], depth_mm.shape[1], 3), dtype=np.uint8)
    if not mask.any():
        return vis

    values = depth_mm[mask].astype(np.float32)
    lo = float(low_mm) if low_mm is not None else float(np.percentile(values, low_percentile))
    hi = float(high_mm) if high_mm is not None else float(np.percentile(values, high_percentile))
    if hi <= lo:
        hi = lo + 1.0

    norm = np.zeros_like(depth_mm, dtype=np.float32)
    norm[mask] = np.clip((depth_mm[mask].astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    depth_u8 = (norm * 255.0).astype(np.uint8)
    vis = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    vis[~mask] = 20
    return vis


def infer_posterior_focus_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0, 0, mask.shape[1] - 1, mask.shape[0] - 1)

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    width = x2 - x1 + 1
    height = y2 - y1 + 1

    fx1 = x1 + int(0.35 * width)
    fx2 = x1 + int(0.90 * width)
    fy1 = y1 + int(0.45 * height)
    fy2 = y2
    return (
        max(0, fx1),
        max(0, fy1),
        min(mask.shape[1] - 1, fx2),
        min(mask.shape[0] - 1, fy2),
    )


def infer_vulva_focus_box(texture_gray: Optional[np.ndarray], mask: np.ndarray) -> Tuple[Tuple[int, int, int, int], str]:
    if texture_gray is None or texture_gray.size == 0 or not mask.any():
        return infer_posterior_focus_box(mask), "fallback_posterior"

    body_mask = largest_component(mask)
    ys, xs = np.where(body_mask)
    if len(xs) == 0:
        return infer_posterior_focus_box(mask), "fallback_posterior"

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    body_w = x2 - x1 + 1
    body_h = y2 - y1 + 1

    rx1 = x1 + int(0.25 * body_w)
    rx2 = x1 + int(0.75 * body_w)
    ry1 = y1 + int(0.05 * body_h)
    ry2 = y1 + int(0.55 * body_h)

    candidate_roi = np.zeros_like(body_mask, dtype=bool)
    candidate_roi[ry1:ry2 + 1, rx1:rx2 + 1] = True
    candidate = body_mask & candidate_roi
    if int(candidate.sum()) < 200:
        return infer_posterior_focus_box(body_mask), "fallback_posterior"

    ir_f = texture_gray.astype(np.float32)
    local_small = cv2.GaussianBlur(ir_f, (0, 0), sigmaX=3.0)
    local_large = cv2.GaussianBlur(ir_f, (0, 0), sigmaX=17.0)
    local_darkness = local_large - local_small

    threshold = float(np.percentile(local_darkness[candidate], 95))
    dark_mask = candidate & (local_darkness >= threshold)
    dark_mask = cv2.morphologyEx(
        (dark_mask.astype(np.uint8) * 255),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ) > 0

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (dark_mask.astype(np.uint8) * 255),
        connectivity=8,
    )
    if num <= 1:
        return infer_posterior_focus_box(body_mask), "fallback_posterior"

    best_idx = -1
    best_area = -1
    for idx in range(1, num):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < 20:
            continue
        if area > best_area:
            best_idx = idx
            best_area = area

    if best_idx < 0:
        return infer_posterior_focus_box(body_mask), "fallback_posterior"

    comp_w = int(stats[best_idx, cv2.CC_STAT_WIDTH])
    comp_h = int(stats[best_idx, cv2.CC_STAT_HEIGHT])
    center_x, center_y = centroids[best_idx]

    box_w = max(int(0.28 * body_w), int(comp_w * 3.0))
    box_h = max(int(0.25 * body_h), int(comp_h * 2.6))

    bx1 = max(0, int(round(center_x - box_w / 2.0)))
    by1 = max(0, int(round(center_y - 0.35 * box_h)))
    bx2 = min(mask.shape[1] - 1, bx1 + box_w)
    by2 = min(mask.shape[0] - 1, by1 + box_h)
    return (bx1, by1, bx2, by2), "ir_local_dark"


def make_vulva_focus_panel(
    texture_gray: np.ndarray,
    depth_crop_bgr: np.ndarray,
    focus_box: Tuple[int, int, int, int],
) -> np.ndarray:
    overview = cv2.cvtColor(texture_gray, cv2.COLOR_GRAY2BGR)
    fx1, fy1, fx2, fy2 = focus_box
    cv2.rectangle(overview, (fx1, fy1), (fx2, fy2), (90, 220, 90), 2)
    cv2.putText(overview, "IR top view", (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    ir_crop = overview[fy1:fy2 + 1, fx1:fx2 + 1].copy()
    if ir_crop.size == 0:
        ir_crop = np.zeros((120, 120, 3), dtype=np.uint8)
    else:
        ir_crop = cv2.resize(ir_crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    cv2.putText(ir_crop, "IR crop", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    depth_crop = depth_crop_bgr.copy()
    cv2.putText(depth_crop, "Depth crop", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    target_w = max(ir_crop.shape[1], depth_crop.shape[1])
    target_h = max(ir_crop.shape[0], depth_crop.shape[0])
    ir_crop = cv2.resize(ir_crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    depth_crop = cv2.resize(depth_crop, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    pad = 14
    right_stack = np.vstack(
        [
            ir_crop,
            np.full((pad, target_w, 3), 18, dtype=np.uint8),
            depth_crop,
        ]
    )

    scale = min(right_stack.shape[0] / max(overview.shape[0], 1), right_stack.shape[1] / max(overview.shape[1], 1))
    scale = max(scale, 1.0)
    overview_resized = cv2.resize(
        overview,
        (int(round(overview.shape[1] * scale)), int(round(overview.shape[0] * scale))),
        interpolation=cv2.INTER_CUBIC,
    )

    canvas_h = max(overview_resized.shape[0], right_stack.shape[0]) + pad * 2
    canvas_w = overview_resized.shape[1] + right_stack.shape[1] + pad * 3
    canvas = np.full((canvas_h, canvas_w, 3), 18, dtype=np.uint8)
    canvas[pad:pad + overview_resized.shape[0], pad:pad + overview_resized.shape[1]] = overview_resized
    right_x = pad * 2 + overview_resized.shape[1]
    canvas[pad:pad + right_stack.shape[0], right_x:right_x + right_stack.shape[1]] = right_stack
    return canvas


def make_vulva_focus_visualization(
    depth_mm: np.ndarray,
    mask: np.ndarray,
    texture_gray: Optional[np.ndarray] = None,
    low_percentile: float = 2.0,
    high_percentile: float = 98.0,
    crop_scale: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    if not mask.any():
        empty_full = np.zeros((depth_mm.shape[0], depth_mm.shape[1], 3), dtype=np.uint8)
        empty_crop = np.zeros((256, 256, 3), dtype=np.uint8)
        empty_panel = np.zeros((256, 512, 3), dtype=np.uint8)
        return empty_full, empty_crop, empty_panel, {"focus_low_mm": 0.0, "focus_high_mm": 0.0}

    focus_mask = refine_render_mask(depth_mm, mask)
    if not focus_mask.any():
        focus_mask = mask

    (fx1, fy1, fx2, fy2), focus_method = infer_vulva_focus_box(texture_gray, focus_mask)
    roi_mask = np.zeros_like(focus_mask, dtype=bool)
    roi_mask[fy1:fy2 + 1, fx1:fx2 + 1] = True

    roi_values = depth_mm[focus_mask & roi_mask].astype(np.float32)
    if roi_values.size < 200:
        roi_values = depth_mm[focus_mask].astype(np.float32)

    lo = float(np.percentile(roi_values, low_percentile))
    hi = float(np.percentile(roi_values, high_percentile))
    if hi <= lo:
        hi = lo + 1.0

    full_vis = make_depth_visualization(depth_mm, focus_mask, low_mm=lo, high_mm=hi)
    cv2.rectangle(full_vis, (fx1, fy1), (fx2, fy2), (255, 255, 255), 1)
    cv2.putText(
        full_vis,
        f"focus {int(lo)}-{int(hi)} mm",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    crop = full_vis[fy1:fy2 + 1, fx1:fx2 + 1]
    if crop.size == 0:
        crop = np.zeros((256, 256, 3), dtype=np.uint8)
    else:
        crop = cv2.resize(crop, None, fx=crop_scale, fy=crop_scale, interpolation=cv2.INTER_NEAREST)
        cv2.putText(
            crop,
            f"{int(lo)}-{int(hi)} mm",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    panel = make_vulva_focus_panel(texture_gray if texture_gray is not None else np.zeros_like(depth_mm, dtype=np.uint8), crop, (fx1, fy1, fx2, fy2))

    return full_vis, crop, panel, {
        "focus_low_mm": round(lo, 2),
        "focus_high_mm": round(hi, 2),
        "focus_box": [int(fx1), int(fy1), int(fx2), int(fy2)],
        "focus_method": focus_method,
    }


def save_preview(
    path: str,
    texture_rgb: np.ndarray,
    depth_vis_rgb: np.ndarray,
    mask: np.ndarray,
    vertices: np.ndarray,
    colors: np.ndarray,
) -> None:
    if not HAS_PLT:
        masked_tex = texture_rgb.copy()
        masked_tex[~mask] = 0
        mask_vis = np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)
        canvas = np.hstack([texture_rgb, depth_vis_rgb, masked_tex, mask_vis])
        cv2.imwrite(path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        return

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")

    ax1.imshow(texture_rgb)
    ax1.set_title("Texture image")
    ax1.axis("off")

    ax2.imshow(depth_vis_rgb)
    ax2.set_title("Segmented depth")
    ax2.axis("off")

    masked_tex = texture_rgb.copy()
    masked_tex[~mask] = 0
    ax3.imshow(masked_tex)
    ax3.set_title("Masked texture")
    ax3.axis("off")

    if len(vertices) > 0:
        step = max(1, len(vertices) // 20000)
        pts = vertices[::step]
        cols = colors[::step] / 255.0
        ax4.scatter(pts[:, 0], -pts[:, 1], pts[:, 2], c=cols, s=0.8, linewidths=0)
    ax4.set_title("3D preview")
    ax4.set_xlabel("X (m)")
    ax4.set_ylabel("Y (m)")
    ax4.set_zlabel("Z (m)")

    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def mask_crop_bounds(mask: np.ndarray, pad: int = 8) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None

    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(mask.shape[1], int(xs.max()) + pad + 1)
    y2 = min(mask.shape[0], int(ys.max()) + pad + 1)
    return x1, y1, x2, y2


def build_masked_rgb_image(rgb_source_path: Optional[str], depth_shape: Tuple[int, int], mask: np.ndarray) -> Optional[np.ndarray]:
    if not rgb_source_path or not os.path.exists(rgb_source_path):
        return None

    rgb_img, _ = load_texture_image(rgb_source_path, depth_shape)
    masked_rgb = rgb_img.copy()
    masked_rgb[~mask] = 0
    return masked_rgb


def save_masked_rgb_image(path: str, rgb_source_path: Optional[str], depth_shape: Tuple[int, int], mask: np.ndarray) -> str:
    masked_rgb = build_masked_rgb_image(rgb_source_path, depth_shape, mask)
    if masked_rgb is None:
        return ""

    cv2.imwrite(path, cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2BGR))
    return path


def save_rgb_region_image(path: str, rgb_source_path: Optional[str], depth_shape: Tuple[int, int], mask: np.ndarray) -> str:
    masked_rgb = build_masked_rgb_image(rgb_source_path, depth_shape, mask)
    if masked_rgb is None:
        return ""

    bounds = mask_crop_bounds(mask)
    if bounds is None:
        return ""

    x1, y1, x2, y2 = bounds
    cropped = masked_rgb[y1:y2, x1:x2]
    if cropped.size == 0:
        return ""

    cv2.imwrite(path, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
    return path


def display_vertices(vertices: np.ndarray, flip_y: bool = True) -> np.ndarray:
    shown = vertices.astype(np.float32).copy()
    if flip_y:
        shown[:, 1] *= -1.0
    return shown


def rotation_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)

    rot_y = np.array(
        [
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ],
        dtype=np.float32,
    )
    rot_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cp, -sp],
            [0.0, sp, cp],
        ],
        dtype=np.float32,
    )
    return rot_x @ rot_y


def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float32)
    if len(vertices) == 0 or len(faces) == 0:
        return normals

    tris = vertices[faces]
    face_normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)
    valid = lengths[:, 0] > 1e-12
    face_normals[valid] /= lengths[valid]

    np.add.at(normals, faces[:, 0], face_normals)
    np.add.at(normals, faces[:, 1], face_normals)
    np.add.at(normals, faces[:, 2], face_normals)

    norm_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    valid_normals = norm_lengths[:, 0] > 1e-12
    normals[valid_normals] /= norm_lengths[valid_normals]
    normals[~valid_normals] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return normals


def project_for_render(vertices: np.ndarray, image_size: int, yaw_deg: float, pitch_deg: float):
    shown = display_vertices(vertices, flip_y=True)
    center = shown.mean(axis=0, keepdims=True)
    centered = shown - center
    radius = max(float(np.linalg.norm(centered, axis=1).max()), 1e-6)

    rot = rotation_matrix(yaw_deg, pitch_deg)
    rotated = centered @ rot.T
    depth = rotated[:, 2]
    cam_depth = depth - depth.min() + 2.7 * radius

    proj_x = rotated[:, 0] / cam_depth
    proj_y = rotated[:, 1] / cam_depth

    finite = np.isfinite(proj_x) & np.isfinite(proj_y)
    if not np.any(finite):
        raise ValueError("Projection failed: no finite vertices")

    x_span = max(float(np.quantile(np.abs(proj_x[finite]), 0.995)), 1e-6)
    y_span = max(float(np.quantile(np.abs(proj_y[finite]), 0.995)), 1e-6)
    scale = 0.45 * float(image_size) / max(x_span, y_span, 1e-6)

    sx = proj_x * scale + (image_size / 2.0)
    sy = -proj_y * scale + (image_size / 2.0)
    return shown, centered, rotated, cam_depth, sx, sy, radius, rot


def largest_component(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask.astype(np.uint8) * 255)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num <= 1:
        return mask
    best_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == best_idx


def refine_render_mask(depth_mm: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask

    values = depth_mm[mask].astype(np.float32)
    quantiles = [45, 50, 55, 60]
    selected = None
    min_pixels = max(4000, int(mask.sum() * 0.12))

    for quantile in quantiles:
        cutoff = float(np.percentile(values, quantile))
        candidate = mask & (depth_mm.astype(np.float32) <= cutoff)
        candidate = largest_component(candidate)
        if int(candidate.sum()) >= min_pixels:
            selected = candidate
            break

    if selected is None:
        selected = largest_component(mask)

    kernel_grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    refined = cv2.dilate((selected.astype(np.uint8) * 255), kernel_grow, iterations=1) > 0
    refined = cv2.morphologyEx((refined.astype(np.uint8) * 255), cv2.MORPH_CLOSE, kernel_close) > 0
    refined = refined & mask
    refined = largest_component(refined)
    return refined


def crop_to_square_canvas(gray: np.ndarray, mask: np.ndarray, image_size: int, fill_ratio: float = 0.88) -> np.ndarray:
    canvas = np.zeros((image_size, image_size), dtype=np.uint8)
    if not mask.any():
        return canvas

    ys, xs = np.where(mask)
    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())

    pad_y = int(0.12 * max(1, y2 - y1 + 1))
    pad_x = int(0.12 * max(1, x2 - x1 + 1))
    y1 = max(0, y1 - pad_y)
    y2 = min(gray.shape[0] - 1, y2 + pad_y)
    x1 = max(0, x1 - pad_x)
    x2 = min(gray.shape[1] - 1, x2 + pad_x)

    patch = gray[y1:y2 + 1, x1:x2 + 1]
    ph, pw = patch.shape[:2]
    if ph == 0 or pw == 0:
        return canvas

    scale = min((image_size * fill_ratio) / max(pw, 1), (image_size * fill_ratio) / max(ph, 1))
    new_w = max(1, int(round(pw * scale)))
    new_h = max(1, int(round(ph * scale)))
    resized = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    off_x = (image_size - new_w) // 2
    off_y = (image_size - new_h) // 2
    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized
    return canvas


def render_stylized_depthmap(
    depth_mm: np.ndarray,
    mask: np.ndarray,
    image_size: int,
) -> np.ndarray:
    canvas = np.zeros((image_size, image_size), dtype=np.uint8)
    if not mask.any():
        return canvas

    render_mask = refine_render_mask(depth_mm, mask)
    if not render_mask.any():
        return canvas

    depth_f = depth_mm.astype(np.float32)
    mask_f = render_mask.astype(np.float32)

    weighted = cv2.GaussianBlur(depth_f * mask_f, (0, 0), sigmaX=5.0)
    weights = cv2.GaussianBlur(mask_f, (0, 0), sigmaX=5.0)
    smooth = np.divide(weighted, np.maximum(weights, 1e-6))
    smooth[~render_mask] = 0.0

    values = smooth[render_mask]
    near_depth = float(np.percentile(values, 2))
    far_depth = float(np.percentile(values, 98))
    span = max(far_depth - near_depth, 1e-6)

    depth_term = 1.0 - np.clip((smooth - near_depth) / span, 0.0, 1.0)

    grad_y, grad_x = np.gradient(smooth)
    normal_x = -grad_x * 0.02
    normal_y = -grad_y * 0.02
    normal_z = np.ones_like(smooth, dtype=np.float32)
    mag = np.sqrt(normal_x * normal_x + normal_y * normal_y + normal_z * normal_z)
    normal_x /= np.maximum(mag, 1e-6)
    normal_y /= np.maximum(mag, 1e-6)
    normal_z /= np.maximum(mag, 1e-6)

    light = np.array([-0.20, -0.35, 0.92], dtype=np.float32)
    light /= max(float(np.linalg.norm(light)), 1e-6)
    lambert = np.clip(normal_x * light[0] + normal_y * light[1] + normal_z * light[2], 0.0, 1.0)

    intensity = 0.62 * depth_term + 0.38 * np.power(lambert, 0.85)
    intensity *= mask_f
    intensity = cv2.GaussianBlur(intensity, (0, 0), sigmaX=1.2)
    intensity = np.clip(np.power(intensity, 0.80), 0.0, 1.0)

    soft_mask = cv2.GaussianBlur(mask_f, (0, 0), sigmaX=1.0)
    portrait = np.clip(intensity * np.clip(soft_mask * 1.05, 0.0, 1.0), 0.0, 1.0)
    portrait_u8 = np.clip(portrait * 255.0, 0, 255).astype(np.uint8)

    return crop_to_square_canvas(portrait_u8, render_mask, image_size=image_size, fill_ratio=0.88)


def infer_texture_path(args, depth_raw: str) -> Optional[str]:
    if args.texture_img:
        return args.texture_img
    return infer_sibling_path(
        depth_raw,
        [
            ("ir_vis", "jpg"),
            ("ir", "raw"),
            ("depth_vis", "jpg"),
            ("rgb_aligned", "jpg"),
            ("rgb", "jpg"),
        ],
    )


def infer_rgb_image_path(depth_raw: str) -> Optional[str]:
    return infer_sibling_path(
        depth_raw,
        [
            ("rgb_aligned", "jpg"),
            ("rgb", "jpg"),
        ],
    )


def infer_output_dir(args, depth_raw: str) -> str:
    if args.output_dir:
        return args.output_dir
    stem = os.path.splitext(os.path.basename(depth_raw))[0].replace("_depth", "")
    return os.path.join(OUTPUT_DIR, f"{stem}_depthmap")


def summarize(vertices: np.ndarray, faces: np.ndarray, auto_stats: Dict[str, float], intrinsics: Dict[str, float], mask_source: str) -> Dict[str, object]:
    summary = {
        "mask_source": mask_source,
        "num_vertices": int(len(vertices)),
        "num_faces": int(len(faces)),
        "intrinsics": {
            "fx": intrinsics["fx"],
            "fy": intrinsics["fy"],
            "cx": intrinsics["cx"],
            "cy": intrinsics["cy"],
        },
        "calibration_path": intrinsics.get("calibration_path", ""),
    }
    summary.update(auto_stats)

    if len(vertices) > 0:
        mins = vertices.min(axis=0).tolist()
        maxs = vertices.max(axis=0).tolist()
        summary["bounds_m"] = {
            "x": [round(mins[0], 4), round(maxs[0], 4)],
            "y": [round(mins[1], 4), round(maxs[1], 4)],
            "z": [round(mins[2], 4), round(maxs[2], 4)],
        }
    return summary


def cleanup_full_outputs(out_dir: str) -> None:
    removable_files = [
        "mask.png",
        "segmented_depth.png",
        "segmented_depth_vulva.png",
        "vulva_focus_crop.png",
        "vulva_focus_panel.png",
        "mesh_texture.png",
        "pig_surface.obj",
        "pig_surface.mtl",
        "depthmap_bundle.npz",
        "preview.png",
        "depthmap_render.png",
        "raw_rgb_image.jpg",
    ]
    removable_dirs = [
        "vulva_manual",
    ]

    for name in removable_files:
        path = os.path.join(out_dir, name)
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass

    for name in removable_dirs:
        path = os.path.join(out_dir, name)
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
        except OSError:
            pass


def main(args):
    depth_mm = load_depth_raw(args.depth_raw)
    texture_path = infer_texture_path(args, args.depth_raw)
    if not texture_path:
        raise FileNotFoundError("Could not infer a texture image/raw file next to the depth frame")
    rgb_image_path = infer_rgb_image_path(args.depth_raw)

    intrinsics = load_calibration(args.calibration, os.path.dirname(args.depth_raw))
    texture_rgb, texture_gray = load_texture_image(texture_path, depth_mm.shape)
    roi_mask = make_roi_mask(depth_mm.shape, use_crop=not args.no_crop)
    valid_depth = (depth_mm >= args.min_depth) & (depth_mm <= args.max_depth)

    if args.mask:
        mask = load_external_mask(args.mask, depth_mm.shape) & roi_mask & valid_depth
        mask_source = "external_mask"
        auto_stats = {"mask_pixels": int(mask.sum())}
    elif args.keep_all_valid or args.paper_raw_acquisition:
        mask = roi_mask & valid_depth
        mask_source = "paper_raw_valid_depth" if args.paper_raw_acquisition else "all_valid_depth"
        auto_stats = {
            "mask_pixels": int(mask.sum()),
            "roi_valid_pixels": int((roi_mask & valid_depth).sum()),
        }
    else:
        mask, auto_stats = auto_segment_pig(
            depth_mm=depth_mm,
            roi_mask=roi_mask,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            foreground_margin_mm=args.foreground_margin_mm,
            min_pixels=args.min_pixels,
        )
        mask_source = "auto_pig_depth_segment"
        mask = mask & valid_depth

    if not mask.any():
        raise ValueError("Mask is empty after filtering")

    out_dir = infer_output_dir(args, args.depth_raw)
    os.makedirs(out_dir, exist_ok=True)
    if args.point_cloud_only:
        cleanup_full_outputs(out_dir)

    ply_path = os.path.join(out_dir, "pig_point_cloud.ply")
    raw_3d_color_path = os.path.join(out_dir, "raw_3d_color_image.png")
    rgb_region_path = os.path.join(out_dir, "point_cloud_rgb_region.jpg")
    summary_path = os.path.join(out_dir, "summary.json")

    pc_vertices, pc_colors = build_point_cloud(
        depth_mm=depth_mm,
        texture_rgb=texture_rgb,
        mask=mask,
        intrinsics=intrinsics,
        stride=max(1, args.point_stride),
    )
    write_point_cloud_ply(ply_path, pc_vertices, pc_colors)
    saved_raw_3d_color_path = save_masked_rgb_image(
        raw_3d_color_path,
        rgb_image_path,
        depth_mm.shape,
        mask,
    )
    saved_rgb_region_path = save_rgb_region_image(
        rgb_region_path,
        rgb_image_path,
        depth_mm.shape,
        mask,
    )

    if args.point_cloud_only:
        summary = summarize(
            pc_vertices,
            np.empty((0, 3), dtype=np.int32),
            auto_stats,
            intrinsics,
            mask_source,
        )
        output_mode = "paper_raw_acquisition" if args.paper_raw_acquisition else "point_cloud_only"
        summary.update(
            {
                "depth_raw": args.depth_raw,
                "texture_input": texture_path,
                "mask_input": args.mask or "",
                "output_dir": out_dir,
                "point_stride": args.point_stride,
                "output_mode": output_mode,
                "raw_3d_color_image": saved_raw_3d_color_path,
                "rgb_region_image": saved_rgb_region_path,
                "rgb_image_input": rgb_image_path or "",
            }
        )
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Depth raw:   {args.depth_raw}")
        print(f"Texture:     {texture_path}")
        print(f"Mask source: {mask_source}")
        print(f"Vertices:    {len(pc_vertices):,}")
        print(f"Output dir:  {out_dir}")
        print(f"  Point cloud: {ply_path}")
        if saved_raw_3d_color_path:
            print(f"  Raw 3D RGB:  {saved_raw_3d_color_path}")
        if saved_rgb_region_path:
            print(f"  RGB region:  {saved_rgb_region_path}")
        print(f"  Summary:     {summary_path}")
        return

    depth_vis_bgr = make_depth_visualization(depth_mm, mask)
    vulva_focus_bgr, vulva_focus_crop_bgr, vulva_focus_panel_bgr, vulva_focus_stats = make_vulva_focus_visualization(
        depth_mm,
        mask,
        texture_gray=texture_gray,
    )
    depth_vis_rgb = cv2.cvtColor(depth_vis_bgr, cv2.COLOR_BGR2RGB)

    mask_path = os.path.join(out_dir, "mask.png")
    depth_vis_path = os.path.join(out_dir, "segmented_depth.png")
    vulva_focus_path = os.path.join(out_dir, "segmented_depth_vulva.png")
    vulva_focus_crop_path = os.path.join(out_dir, "vulva_focus_crop.png")
    vulva_focus_panel_path = os.path.join(out_dir, "vulva_focus_panel.png")
    texture_path_out = os.path.join(out_dir, "mesh_texture.png")
    preview_path = os.path.join(out_dir, "preview.png")
    render_path = os.path.join(out_dir, "depthmap_render.png")
    obj_path = os.path.join(out_dir, "pig_surface.obj")
    bundle_path = os.path.join(out_dir, "depthmap_bundle.npz")

    save_mask(mask_path, mask)
    cv2.imwrite(depth_vis_path, depth_vis_bgr)
    cv2.imwrite(vulva_focus_path, vulva_focus_bgr)
    cv2.imwrite(vulva_focus_crop_path, vulva_focus_crop_bgr)
    cv2.imwrite(vulva_focus_panel_path, vulva_focus_panel_bgr)

    masked_texture = texture_rgb.copy()
    masked_texture[~mask] = 0
    cv2.imwrite(texture_path_out, cv2.cvtColor(masked_texture, cv2.COLOR_RGB2BGR))

    mesh_vertices, mesh_colors, mesh_uvs, mesh_faces = build_textured_mesh(
        depth_mm=depth_mm,
        texture_rgb=texture_rgb,
        mask=mask,
        intrinsics=intrinsics,
        stride=max(1, args.mesh_stride),
        max_edge_mm=args.max_edge_mm,
    )
    if len(mesh_vertices) > 0 and len(mesh_faces) > 0:
        write_textured_obj(obj_path, mesh_vertices, mesh_uvs, mesh_faces, os.path.basename(texture_path_out))

    np.savez_compressed(
        bundle_path,
        depth_mm=depth_mm,
        texture_gray=texture_gray,
        mask=mask.astype(np.uint8),
        point_cloud_xyz=pc_vertices,
        point_cloud_rgb=pc_colors,
        mesh_xyz=mesh_vertices,
        mesh_uv=mesh_uvs,
        mesh_faces=mesh_faces,
    )

    save_preview(preview_path, texture_rgb, depth_vis_rgb, mask, pc_vertices, pc_colors)
    render_gray = render_stylized_depthmap(
        depth_mm=depth_mm,
        mask=mask,
        image_size=max(256, args.render_size),
    )
    cv2.imwrite(render_path, render_gray)

    summary = summarize(pc_vertices, mesh_faces, auto_stats, intrinsics, mask_source)
    summary.update(
        {
            "depth_raw": args.depth_raw,
            "texture_input": texture_path,
            "mask_input": args.mask or "",
            "output_dir": out_dir,
            "point_stride": args.point_stride,
            "mesh_stride": args.mesh_stride,
            "max_edge_mm": args.max_edge_mm,
            "output_mode": "full_outputs",
            "raw_3d_color_image": saved_raw_3d_color_path,
            "rgb_region_image": saved_rgb_region_path,
            "rgb_image_input": rgb_image_path or "",
            "depthmap_render": render_path,
            "vulva_focus_visualization": vulva_focus_path,
            "vulva_focus_crop": vulva_focus_crop_path,
            "vulva_focus_panel": vulva_focus_panel_path,
        }
    )
    summary.update(vulva_focus_stats)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Depth raw:   {args.depth_raw}")
    print(f"Texture:     {texture_path}")
    print(f"Mask source: {mask_source}")
    print(f"Vertices:    {len(pc_vertices):,}")
    print(f"Mesh faces:  {len(mesh_faces):,}")
    print(f"Output dir:  {out_dir}")
    print(f"  Point cloud: {ply_path}")
    if saved_raw_3d_color_path:
        print(f"  Raw 3D RGB:  {saved_raw_3d_color_path}")
    if saved_rgb_region_path:
        print(f"  RGB region:  {saved_rgb_region_path}")
    if len(mesh_faces) > 0:
        print(f"  Mesh:        {obj_path}")
    print(f"  Bundle:      {bundle_path}")
    print(f"  Depth render:{render_path}")
    print(f"  Vulva focus: {vulva_focus_path}")
    print(f"  Focus crop:  {vulva_focus_crop_path}")
    print(f"  Focus panel: {vulva_focus_panel_path}")
    print(f"  Summary:     {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pig point cloud + textured mesh from OAK depth and IR")
    parser.add_argument("--depth-raw", required=True, help="Path to pigX_depth_TIMESTAMP.raw")
    parser.add_argument("--texture-img", "--ir-img", "--color-img", dest="texture_img", default=None,
                        help="Texture image/raw to paint onto the point cloud (IR image recommended)")
    parser.add_argument("--calibration", default=None, help="Optional path to calibration.json")
    parser.add_argument("--mask", default=None, help="Optional binary mask in depth/IR frame")
    parser.add_argument("--output-dir", default=None, help="Output folder (default: outputs/<frame>_depthmap)")
    parser.add_argument("--min-depth", type=int, default=200, help="Minimum valid depth in mm")
    parser.add_argument("--max-depth", type=int, default=5000, help="Maximum valid depth in mm")
    parser.add_argument("--foreground-margin-mm", type=int, default=140,
                        help="Foreground margin below the wall depth for auto pig segmentation")
    parser.add_argument("--min-pixels", type=int, default=4000, help="Minimum pig-mask size for auto segmentation")
    parser.add_argument("--point-stride", type=int, default=2, help="Pixel stride for the exported point cloud")
    parser.add_argument("--mesh-stride", type=int, default=3, help="Pixel stride for the exported mesh")
    parser.add_argument("--max-edge-mm", type=float, default=45.0,
                        help="Skip mesh triangles where local depth changes too sharply")
    parser.add_argument("--render-size", type=int, default=768,
                        help="Output size for the grayscale depth render")
    parser.add_argument("--no-crop", action="store_true", help="Disable the fixed stall ROI crop before segmentation")
    parser.add_argument("--keep-all-valid", action="store_true",
                        help="Skip pig auto-segmentation and keep all valid depth pixels in the ROI")
    parser.add_argument("--paper-raw-acquisition", action="store_true",
                        help="Paper-style rear-view acquisition: keep all valid depth pixels in the ROI instead of auto pig segmentation")
    parser.add_argument("--point-cloud-only", action="store_true",
                        help="Only save the rear-view point cloud plus summary.json")
    main(parser.parse_args())
