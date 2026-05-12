#!/usr/bin/env python3
"""Manual vulva depth-output workflow for a single depthmap frame.

This tool builds vulva-specific depth artifacts from a single frame:
1. Focus on the vulva region using IR + depth
2. Draw or auto-seed a vulva mask in that crop
3. Remove the vulva region from the depth surface
4. Smoothly fill the missing area to estimate a no-vulva baseline surface
5. Subtract baseline from original depth to obtain a vulva-only surface
6. Save the mask and derived depth visualizations under the frame's depthmap folder

Examples:
    python measure_vulva_manual.py depthmap\\20260211-09-06-52\\pig0_20260211-09-07-10
    python measure_vulva_manual.py depthmap\\20260211-09-06-52\\pig0_20260211-09-07-10 --auto-mask
"""

import argparse
import json
import os
from pathlib import Path
import sys


print(
    "The old manual vulva measurement workflow was removed. "
    "Use `python label_vulva_dataset.py` for the new IR point-click length/width workflow."
)
raise SystemExit(0)

import cv2
import numpy as np

import label_vulva as label_vulva_module
from label_vulva import PolygonDrawer, draw_polygon_overlay, make_mask, scale_for_display
from point_cloud import (
    infer_vulva_focus_box,
    largest_component,
    load_calibration,
    load_depth_raw,
    load_texture_image,
    make_depth_visualization,
)


WINDOW_NAME = "Manual Vulva Measurement"


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def resolve_repo_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return path
    return repo_root() / path


def load_frame_context(input_path: Path):
    if input_path.is_dir():
        frame_dir = input_path
        summary_path = frame_dir / "summary.json"
        bundle_path = frame_dir / "depthmap_bundle.npz"
        if not summary_path.exists() or not bundle_path.exists():
            raise FileNotFoundError(f"Missing summary.json or depthmap_bundle.npz in {frame_dir}")

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        depth_mm = np.load(bundle_path, allow_pickle=False)["depth_mm"]
        pig_mask = np.load(bundle_path, allow_pickle=False)["mask"].astype(bool)
        depth_raw_path = resolve_repo_path(summary.get("depth_raw"))
        texture_input_path = resolve_repo_path(summary.get("texture_input"))
        intr = summary.get("intrinsics", {})
        intrinsics = {
            "fx": float(intr["fx"]),
            "fy": float(intr["fy"]),
            "cx": float(intr["cx"]),
            "cy": float(intr["cy"]),
        }
        if texture_input_path is None or not texture_input_path.exists():
            raise FileNotFoundError(f"Texture image not found from summary: {summary.get('texture_input')}")
        texture_rgb, texture_gray = load_texture_image(str(texture_input_path), depth_mm.shape)
        return {
            "frame_dir": frame_dir,
            "summary": summary,
            "depth_mm": depth_mm,
            "pig_mask": pig_mask,
            "intrinsics": intrinsics,
            "texture_rgb": texture_rgb,
            "texture_gray": texture_gray,
            "depth_raw_path": depth_raw_path,
            "texture_path": texture_input_path,
        }

    if input_path.suffix.lower() == ".raw":
        depth_raw_path = input_path
        depth_mm = load_depth_raw(str(depth_raw_path))
        cal = load_calibration(None, str(depth_raw_path.parent))
        intrinsics = {
            "fx": float(cal["fx"]),
            "fy": float(cal["fy"]),
            "cx": float(cal["cx"]),
            "cy": float(cal["cy"]),
        }
        ir_path = depth_raw_path.with_name(depth_raw_path.name.replace("_depth_", "_ir_vis_").replace(".raw", ".jpg"))
        if not ir_path.exists():
            raise FileNotFoundError(f"Could not infer IR image next to depth raw: {ir_path}")
        texture_rgb, texture_gray = load_texture_image(str(ir_path), depth_mm.shape)
        pig_mask = (depth_mm >= 200) & (depth_mm <= 5000)
        frame_dir = depth_raw_path.parent
        return {
            "frame_dir": frame_dir,
            "summary": {},
            "depth_mm": depth_mm,
            "pig_mask": pig_mask,
            "intrinsics": intrinsics,
            "texture_rgb": texture_rgb,
            "texture_gray": texture_gray,
            "depth_raw_path": depth_raw_path,
            "texture_path": ir_path,
        }

    raise ValueError(f"Unsupported input: {input_path}")


def make_focus_crop(texture_gray: np.ndarray, focus_box: tuple[int, int, int, int], scale: float = 3.0):
    fx1, fy1, fx2, fy2 = focus_box
    crop = texture_gray[fy1:fy2 + 1, fx1:fx2 + 1]
    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    crop_bgr = cv2.resize(crop_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop_bgr


def auto_detect_vulva_mask(texture_gray: np.ndarray, body_mask: np.ndarray):
    focus_box, focus_method = infer_vulva_focus_box(texture_gray, body_mask)
    fx1, fy1, fx2, fy2 = focus_box

    candidate = np.zeros_like(body_mask, dtype=bool)
    candidate[fy1:fy2 + 1, fx1:fx2 + 1] = True
    candidate &= body_mask

    ir_f = texture_gray.astype(np.float32)
    small = cv2.GaussianBlur(ir_f, (0, 0), sigmaX=3.0)
    large = cv2.GaussianBlur(ir_f, (0, 0), sigmaX=17.0)
    local_darkness = large - small

    if int(candidate.sum()) < 50:
        raise ValueError("Auto vulva detection candidate area is too small")

    threshold = float(np.percentile(local_darkness[candidate], 95))
    vulva_mask = candidate & (local_darkness >= threshold)
    vulva_mask = cv2.morphologyEx(
        (vulva_mask.astype(np.uint8) * 255),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ) > 0
    vulva_mask = largest_component(vulva_mask)
    return vulva_mask, focus_box, focus_method


def draw_manual_vulva_mask(texture_gray: np.ndarray, focus_box: tuple[int, int, int, int], scale: float = 3.0):
    label_vulva_module.WINDOW_NAME = WINDOW_NAME
    focus_view = make_focus_crop(texture_gray, focus_box, scale=scale)
    overlay = focus_view.copy()
    cv2.putText(
        overlay,
        "Draw vulva polygon | Left=add Right=undo Enter=finish R=reset Q=cancel",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.imshow(WINDOW_NAME, overlay)

    drawer = PolygonDrawer(overlay)
    display_points = drawer.run()
    if display_points is None:
        raise RuntimeError("Manual vulva drawing cancelled")

    fx1, fy1, fx2, fy2 = focus_box
    full_points = []
    for x, y in display_points:
        full_x = int(round(fx1 + x / scale))
        full_y = int(round(fy1 + y / scale))
        full_points.append((full_x, full_y))

    mask = make_mask(full_points, texture_gray.shape)
    return mask > 0, focus_box, "manual_polygon"


def clean_vulva_mask(mask: np.ndarray):
    if not mask.any():
        return mask, mask

    base = largest_component(mask)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel_stretch = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))

    clean = cv2.morphologyEx((base.astype(np.uint8) * 255), cv2.MORPH_CLOSE, kernel_close) > 0
    clean = largest_component(clean)
    stretched = cv2.dilate((clean.astype(np.uint8) * 255), kernel_stretch, iterations=1) > 0
    return clean, stretched


def inpaint_no_vulva_surface(depth_mm: np.ndarray, domain_mask: np.ndarray, hole_mask: np.ndarray, iterations: int = 180):
    work = depth_mm.astype(np.float32).copy()
    fixed_mask = domain_mask & ~hole_mask
    if not np.any(fixed_mask):
        raise ValueError("No valid surrounding body surface available for inpainting")

    init_value = float(np.median(work[fixed_mask]))
    work[hole_mask] = init_value

    domain_f = domain_mask.astype(np.float32)
    fixed_values = work.copy()
    for _ in range(iterations):
        weighted = cv2.GaussianBlur(work * domain_f, (0, 0), sigmaX=3.0)
        weights = cv2.GaussianBlur(domain_f, (0, 0), sigmaX=3.0)
        filled = np.divide(weighted, np.maximum(weights, 1e-6))
        work[hole_mask] = filled[hole_mask]
        work[fixed_mask] = fixed_values[fixed_mask]

    return work


def make_height_visualization(height_mm: np.ndarray, mask: np.ndarray):
    vis = np.zeros((height_mm.shape[0], height_mm.shape[1], 3), dtype=np.uint8)
    if not mask.any():
        return vis

    values = height_mm[mask].astype(np.float32)
    hi = float(np.percentile(values, 99))
    hi = max(hi, 1.0)
    norm = np.zeros_like(height_mm, dtype=np.float32)
    norm[mask] = np.clip(height_mm[mask].astype(np.float32) / hi, 0.0, 1.0)
    vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    vis[~mask] = 20
    return vis


def compute_metric_points(mask: np.ndarray, depth_mm: np.ndarray, intrinsics: dict):
    ys, xs = np.where(mask)
    z_m = depth_mm[mask].astype(np.float32) / 1000.0
    x_m = (xs.astype(np.float32) - float(intrinsics["cx"])) * z_m / float(intrinsics["fx"])
    y_m = (ys.astype(np.float32) - float(intrinsics["cy"])) * z_m / float(intrinsics["fy"])
    return xs.astype(np.float32), ys.astype(np.float32), x_m, y_m


def backproject_points(xs: np.ndarray, ys: np.ndarray, depth_mm_values: np.ndarray, intrinsics: dict):
    z_m = depth_mm_values.astype(np.float32) / 1000.0
    x_m = (xs.astype(np.float32) - float(intrinsics["cx"])) * z_m / float(intrinsics["fx"])
    y_m = (ys.astype(np.float32) - float(intrinsics["cy"])) * z_m / float(intrinsics["fy"])
    return np.stack([x_m, y_m, z_m], axis=1)


def pca_axes(points_a: np.ndarray, points_b: np.ndarray):
    pts = np.stack([points_a, points_b], axis=1)
    center = pts.mean(axis=0)
    centered = pts - center
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    major = eigvecs[:, order[0]]
    minor = eigvecs[:, order[1]]
    return center, major, minor, centered


def measure_length_width(vulva_mask: np.ndarray, depth_mm: np.ndarray, intrinsics: dict):
    px_x, px_y, x_m, y_m = compute_metric_points(vulva_mask, depth_mm, intrinsics)
    metric_center, metric_major, metric_minor, metric_centered = pca_axes(x_m, y_m)
    major_proj = metric_centered @ metric_major
    minor_proj = metric_centered @ metric_minor
    length_m = float(major_proj.max() - major_proj.min())
    width_m = float(minor_proj.max() - minor_proj.min())

    pixel_center, pixel_major, pixel_minor, pixel_centered = pca_axes(px_x, px_y)
    major_px = pixel_centered @ pixel_major
    minor_px = pixel_centered @ pixel_minor

    major_line = (
        pixel_center + pixel_major * float(major_px.min()),
        pixel_center + pixel_major * float(major_px.max()),
    )
    minor_line = (
        pixel_center + pixel_minor * float(minor_px.min()),
        pixel_center + pixel_minor * float(minor_px.max()),
    )

    return {
        "length_m": length_m,
        "width_m": width_m,
        "length_mm": round(length_m * 1000.0, 2),
        "width_mm": round(width_m * 1000.0, 2),
        "major_line_px": major_line,
        "minor_line_px": minor_line,
    }


def expand_box(box: tuple[int, int, int, int], shape: tuple[int, int], margin: int = 24):
    x1, y1, x2, y2 = box
    height, width = shape
    return (
        max(0, x1 - margin),
        max(0, y1 - margin),
        min(width - 1, x2 + margin),
        min(height - 1, y2 + margin),
    )


def make_box_mask(shape: tuple[int, int], box: tuple[int, int, int, int]):
    x1, y1, x2, y2 = box
    mask = np.zeros(shape, dtype=bool)
    mask[y1:y2 + 1, x1:x2 + 1] = True
    return mask


def fit_reference_plane(depth_mm: np.ndarray, support_mask: np.ndarray, intrinsics: dict):
    ys, xs = np.where(support_mask)
    if len(xs) < 20:
        raise ValueError("Not enough support pixels to fit a local reference plane")

    xyz = backproject_points(xs.astype(np.float32), ys.astype(np.float32), depth_mm[support_mask], intrinsics)
    center = xyz.mean(axis=0)
    centered = xyz - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)

    axis_u = vh[0]
    axis_v = vh[1]
    normal = vh[2]
    if normal[2] < 0:
        normal = -normal
    if np.dot(np.cross(axis_u, axis_v), normal) < 0:
        axis_v = -axis_v
    return center, axis_u, axis_v, normal


def project_xyz_to_plane(xyz: np.ndarray, center: np.ndarray, axis_u: np.ndarray, axis_v: np.ndarray, normal: np.ndarray):
    centered = xyz - center
    u = centered @ axis_u
    v = centered @ axis_v
    w = centered @ normal
    return u, v, w


def largest_mask_contour(mask: np.ndarray):
    contours, _ = cv2.findContours((mask.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contour found for the vulva mask")
    contour = max(contours, key=cv2.contourArea)
    return contour.reshape(-1, 2).astype(np.float32)


def polygon_area(points: np.ndarray):
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def derive_ellipse_axes(ellipse):
    (center_u, center_v), (axis_a, axis_b), angle_deg = ellipse
    theta = np.deg2rad(float(angle_deg))
    if axis_a >= axis_b:
        major_len = float(axis_a)
        minor_len = float(axis_b)
        major_dir = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
        minor_dir = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float32)
    else:
        major_len = float(axis_b)
        minor_len = float(axis_a)
        theta += np.pi / 2.0
        major_dir = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
        minor_dir = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float32)
    center_uv = np.array([float(center_u), float(center_v)], dtype=np.float32)
    return center_uv, major_dir, minor_dir, major_len, minor_len, float(np.rad2deg(theta))


def make_plane_measurement_view(
    contour_uv: np.ndarray,
    ellipse_measurement: dict,
    summary_text: list[str],
    canvas_size: int = 320,
):
    canvas = np.full((canvas_size, canvas_size, 3), (90, 40, 120), dtype=np.uint8)
    if len(contour_uv) == 0:
        return canvas

    points = contour_uv.astype(np.float32)
    center = ellipse_measurement["ellipse_center_uv"]
    major_line = ellipse_measurement["major_line_uv"]
    minor_line = ellipse_measurement["minor_line_uv"]
    ellipse_size = ellipse_measurement["ellipse_size_uv"]
    ellipse_angle_deg = ellipse_measurement["ellipse_angle_deg"]

    extra = np.array(
        [
            major_line[0],
            major_line[1],
            minor_line[0],
            minor_line[1],
            center,
        ],
        dtype=np.float32,
    )
    all_points = np.vstack([points, extra])
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    scale = float((canvas_size * 0.74) / max(span[0], span[1], 1e-6))

    def to_canvas(uv_points: np.ndarray):
        out = np.empty_like(uv_points, dtype=np.float32)
        out[:, 0] = (uv_points[:, 0] - mins[0]) * scale + canvas_size * 0.13
        out[:, 1] = canvas_size - ((uv_points[:, 1] - mins[1]) * scale + canvas_size * 0.13)
        return out

    contour_px = np.round(to_canvas(points)).astype(np.int32)
    cv2.fillPoly(canvas, [contour_px.reshape(-1, 1, 2)], (0, 240, 255))

    ellipse_center_px = to_canvas(center.reshape(1, 2))[0]
    ellipse_size_px = (max(1, int(round(ellipse_size[0] * scale))), max(1, int(round(ellipse_size[1] * scale))))
    cv2.ellipse(
        canvas,
        (int(round(ellipse_center_px[0])), int(round(ellipse_center_px[1]))),
        (ellipse_size_px[0] // 2, ellipse_size_px[1] // 2),
        -ellipse_angle_deg,
        0,
        360,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    major_px = to_canvas(np.vstack([major_line[0], major_line[1]])).astype(np.int32)
    minor_px = to_canvas(np.vstack([minor_line[0], minor_line[1]])).astype(np.int32)
    cv2.arrowedLine(canvas, tuple(major_px[0]), tuple(major_px[1]), (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.04)
    cv2.arrowedLine(canvas, tuple(major_px[1]), tuple(major_px[0]), (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.04)
    cv2.arrowedLine(canvas, tuple(minor_px[0]), tuple(minor_px[1]), (255, 0, 0), 2, cv2.LINE_AA, tipLength=0.04)
    cv2.arrowedLine(canvas, tuple(minor_px[1]), tuple(minor_px[0]), (255, 0, 0), 2, cv2.LINE_AA, tipLength=0.04)

    cv2.putText(canvas, "Plane-aligned vulva region", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    text_y = canvas_size - 56
    for line in summary_text:
        cv2.putText(canvas, line, (12, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        text_y += 18
    return canvas


def measure_vulva_geometry_paper(
    vulva_mask: np.ndarray,
    baseline_depth: np.ndarray,
    original_depth: np.ndarray,
    intrinsics: dict,
    focus_box: tuple[int, int, int, int],
    pig_mask: np.ndarray,
):
    support_box = expand_box(focus_box, vulva_mask.shape, margin=max(24, int(0.3 * max(focus_box[2] - focus_box[0] + 1, focus_box[3] - focus_box[1] + 1))))
    support_mask = make_box_mask(vulva_mask.shape, support_box) & pig_mask
    plane_center, axis_u, axis_v, normal = fit_reference_plane(baseline_depth, support_mask, intrinsics)

    ys, xs = np.where(vulva_mask)
    original_xyz = backproject_points(xs.astype(np.float32), ys.astype(np.float32), original_depth[vulva_mask], intrinsics)
    baseline_xyz = backproject_points(xs.astype(np.float32), ys.astype(np.float32), baseline_depth[vulva_mask], intrinsics)
    u_all, v_all, w_orig = project_xyz_to_plane(original_xyz, plane_center, axis_u, axis_v, normal)
    _, _, w_base = project_xyz_to_plane(baseline_xyz, plane_center, axis_u, axis_v, normal)
    height_m = np.clip(w_base - w_orig, 0.0, None)

    contour_xy = largest_mask_contour(vulva_mask)
    contour_x = contour_xy[:, 0]
    contour_y = contour_xy[:, 1]
    contour_z = baseline_depth[contour_y.astype(np.int32), contour_x.astype(np.int32)]
    contour_xyz = backproject_points(contour_x, contour_y, contour_z, intrinsics)
    contour_u, contour_v, _ = project_xyz_to_plane(contour_xyz, plane_center, axis_u, axis_v, normal)
    contour_uv = np.stack([contour_u, contour_v], axis=1).astype(np.float32)

    if len(contour_uv) < 5:
        fallback = measure_length_width(vulva_mask, original_depth, intrinsics)
        peak_height_mm = round(float(height_m.max()) * 1000.0, 2) if len(height_m) else 0.0
        mean_height_mm = round(float(height_m.mean()) * 1000.0, 2) if len(height_m) else 0.0
        return {
            **fallback,
            "measurement_method": "fallback_metric_pca",
            "peak_height_mm": peak_height_mm,
            "mean_height_mm": mean_height_mm,
            "plane_normal": [round(float(v), 6) for v in normal],
            "plane_support_box": [int(v) for v in support_box],
            "base_area_mm2": 0.0,
            "ellipse_area_mm2": 0.0,
            "cubic_volume_mm3": round(fallback["length_mm"] * fallback["width_mm"] * peak_height_mm, 2),
            "contour_uv": contour_uv,
            "measurement_view": np.full((320, 320, 3), (90, 40, 120), dtype=np.uint8),
        }

    ellipse = cv2.fitEllipse(contour_uv.reshape(-1, 1, 2))
    ellipse_center_uv, major_dir_uv, minor_dir_uv, major_len_m, minor_len_m, major_angle_deg = derive_ellipse_axes(ellipse)
    major_line_uv = (
        ellipse_center_uv - major_dir_uv * (major_len_m / 2.0),
        ellipse_center_uv + major_dir_uv * (major_len_m / 2.0),
    )
    minor_line_uv = (
        ellipse_center_uv - minor_dir_uv * (minor_len_m / 2.0),
        ellipse_center_uv + minor_dir_uv * (minor_len_m / 2.0),
    )

    peak_height_mm = round(float(height_m.max()) * 1000.0, 2) if len(height_m) else 0.0
    mean_height_mm = round(float(height_m.mean()) * 1000.0, 2) if len(height_m) else 0.0
    base_area_mm2 = round(polygon_area(contour_uv) * 1_000_000.0, 2)
    ellipse_area_mm2 = round(np.pi * (major_len_m / 2.0) * (minor_len_m / 2.0) * 1_000_000.0, 2)
    cubic_volume_mm3 = round((major_len_m * 1000.0) * (minor_len_m * 1000.0) * peak_height_mm, 2)

    measurement_view = make_plane_measurement_view(
        contour_uv=contour_uv,
        ellipse_measurement={
            "ellipse_center_uv": ellipse_center_uv,
            "major_line_uv": major_line_uv,
            "minor_line_uv": minor_line_uv,
            "ellipse_size_uv": (major_len_m, minor_len_m),
            "ellipse_angle_deg": major_angle_deg,
        },
        summary_text=[
            f"L={major_len_m * 1000.0:.1f} mm  W={minor_len_m * 1000.0:.1f} mm",
            f"H={peak_height_mm:.1f} mm  BA={base_area_mm2:.1f} mm^2",
        ],
    )

    return {
        "measurement_method": "plane_aligned_ellipse_3d",
        "length_m": major_len_m,
        "width_m": minor_len_m,
        "length_mm": round(major_len_m * 1000.0, 2),
        "width_mm": round(minor_len_m * 1000.0, 2),
        "peak_height_mm": peak_height_mm,
        "mean_height_mm": mean_height_mm,
        "plane_normal": [round(float(v), 6) for v in normal],
        "plane_support_box": [int(v) for v in support_box],
        "base_area_mm2": base_area_mm2,
        "ellipse_area_mm2": ellipse_area_mm2,
        "cubic_volume_mm3": cubic_volume_mm3,
        "ellipse_center_uv": ellipse_center_uv,
        "major_line_uv": major_line_uv,
        "minor_line_uv": minor_line_uv,
        "ellipse_size_uv": (major_len_m, minor_len_m),
        "ellipse_angle_deg": major_angle_deg,
        "contour_uv": contour_uv,
        "measurement_view": measurement_view,
    }


def draw_axis_line(img: np.ndarray, line, color, label: str, value_text: str):
    p1 = tuple(int(round(v)) for v in line[0])
    p2 = tuple(int(round(v)) for v in line[1])
    cv2.arrowedLine(img, p1, p2, color, 2, cv2.LINE_AA, tipLength=0.04)
    cv2.arrowedLine(img, p2, p1, color, 2, cv2.LINE_AA, tipLength=0.04)
    mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
    cv2.putText(img, label, (mid[0] + 6, mid[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    cv2.putText(img, value_text, (mid[0] + 6, mid[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


def crop_with_box(image: np.ndarray, box: tuple[int, int, int, int], scale: float = 3.0):
    x1, y1, x2, y2 = box
    crop = image[y1:y2 + 1, x1:x2 + 1]
    if crop.size == 0:
        return np.zeros((200, 200, 3), dtype=np.uint8)
    return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


def make_measurement_overlay(texture_gray: np.ndarray, vulva_mask: np.ndarray, focus_box, measurement: dict):
    bg = np.full((texture_gray.shape[0], texture_gray.shape[1], 3), (90, 40, 120), dtype=np.uint8)
    bg[vulva_mask] = (0, 240, 255)

    if np.count_nonzero(vulva_mask) >= 5:
        pts = np.column_stack(np.where(vulva_mask)[::-1]).astype(np.int32)
        if len(pts) >= 5:
            ellipse = cv2.fitEllipse(pts)
            cv2.ellipse(bg, ellipse, (180, 180, 180), 1, cv2.LINE_AA)

    draw_axis_line(bg, measurement["major_line_px"], (0, 0, 255), "L", f'{measurement["length_mm"]:.1f} mm')
    draw_axis_line(bg, measurement["minor_line_px"], (255, 0, 0), "W", f'{measurement["width_mm"]:.1f} mm')
    cv2.putText(bg, "Vulva length/width", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return crop_with_box(bg, focus_box, scale=3.0)


def make_process_panel(texture_gray: np.ndarray, focus_box, ir_crop, depth_crop, baseline_crop, height_crop, measure_crop):
    full_ir = cv2.cvtColor(texture_gray, cv2.COLOR_GRAY2BGR)
    x1, y1, x2, y2 = focus_box
    cv2.rectangle(full_ir, (x1, y1), (x2, y2), (90, 220, 90), 2)
    cv2.putText(full_ir, "Raw IR Image", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    full_ir = cv2.resize(full_ir, (400, 260), interpolation=cv2.INTER_CUBIC)
    ir_crop = cv2.resize(ir_crop, (260, 260), interpolation=cv2.INTER_CUBIC)
    depth_crop = cv2.resize(depth_crop, (260, 260), interpolation=cv2.INTER_NEAREST)
    baseline_crop = cv2.resize(baseline_crop, (260, 260), interpolation=cv2.INTER_NEAREST)
    height_crop = cv2.resize(height_crop, (260, 260), interpolation=cv2.INTER_NEAREST)
    measure_crop = cv2.resize(measure_crop, (260, 260), interpolation=cv2.INTER_NEAREST)

    top_row = np.hstack([full_ir, ir_crop, depth_crop])
    bottom_row = np.hstack([np.full((260, 400, 3), 18, dtype=np.uint8), baseline_crop, measure_crop])
    bottom_row[:, :400] = cv2.resize(height_crop, (400, 260), interpolation=cv2.INTER_NEAREST)

    panel = np.vstack([top_row, bottom_row])
    return panel


def save_csv_row(csv_path: Path, row: dict):
    fields = [
        "frame_dir",
        "depth_raw",
        "texture_input",
        "focus_method",
        "focus_box",
        "mask_path",
        "stretched_mask_path",
        "no_vulva_surface_path",
        "vulva_only_surface_path",
        "panel_path",
    ]
    exists = csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def process_vulva_measurement(
    ctx: dict,
    raw_mask: np.ndarray,
    focus_box: tuple[int, int, int, int],
    focus_method: str,
    output_dir: Path,
    csv_path: Path | None = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_dir = ctx["frame_dir"]
    depth_mm = ctx["depth_mm"]
    pig_mask = largest_component(ctx["pig_mask"])
    texture_gray = ctx["texture_gray"]

    vulva_mask, stretched_mask = clean_vulva_mask(raw_mask & pig_mask)
    if not np.any(vulva_mask):
        raise ValueError("Vulva mask is empty after cleanup")

    baseline_depth = inpaint_no_vulva_surface(depth_mm, pig_mask, stretched_mask)
    vulva_height_mm = np.clip(baseline_depth - depth_mm.astype(np.float32), 0.0, None)
    vulva_height_mm[~vulva_mask] = 0.0

    focus_vals = depth_mm[vulva_mask]
    focus_depth_vis = make_depth_visualization(
        depth_mm,
        pig_mask,
        low_mm=float(np.percentile(focus_vals, 2)),
        high_mm=float(np.percentile(focus_vals, 98)),
    )
    baseline_vis = make_depth_visualization(
        baseline_depth.astype(np.uint16),
        pig_mask,
        low_mm=float(np.percentile(baseline_depth[pig_mask], 2)),
        high_mm=float(np.percentile(baseline_depth[pig_mask], 98)),
    )
    height_vis = make_height_visualization(vulva_height_mm, vulva_mask)

    ir_overlay = cv2.cvtColor(texture_gray, cv2.COLOR_GRAY2BGR)
    ir_overlay[vulva_mask] = cv2.addWeighted(
        ir_overlay[vulva_mask],
        0.3,
        np.full_like(ir_overlay[vulva_mask], (0, 220, 120)),
        0.7,
        0,
    )
    cv2.rectangle(ir_overlay, (focus_box[0], focus_box[1]), (focus_box[2], focus_box[3]), (90, 220, 90), 2)

    mask_vis = np.full((texture_gray.shape[0], texture_gray.shape[1], 3), 18, dtype=np.uint8)
    mask_vis[vulva_mask] = (0, 240, 255)
    cv2.putText(mask_vis, "Vulva mask", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    ir_crop = crop_with_box(ir_overlay, focus_box, scale=3.0)
    depth_crop = crop_with_box(focus_depth_vis, focus_box, scale=3.0)
    baseline_crop = crop_with_box(baseline_vis, focus_box, scale=3.0)
    height_crop = crop_with_box(height_vis, focus_box, scale=3.0)
    mask_crop = crop_with_box(mask_vis, focus_box, scale=3.0)
    process_panel = make_process_panel(texture_gray, focus_box, ir_crop, depth_crop, baseline_crop, height_crop, mask_crop)

    mask_path = output_dir / "vulva_mask.png"
    stretch_path = output_dir / "vulva_mask_stretched.png"
    baseline_path = output_dir / "no_vulva_surface.png"
    height_path = output_dir / "vulva_only_surface.png"
    panel_path = output_dir / "vulva_depth_panel.png"
    summary_path = output_dir / "vulva_depthmap_summary.json"

    cv2.imwrite(str(mask_path), (vulva_mask.astype(np.uint8) * 255))
    cv2.imwrite(str(stretch_path), (stretched_mask.astype(np.uint8) * 255))
    cv2.imwrite(str(baseline_path), baseline_vis)
    cv2.imwrite(str(height_path), height_vis)
    cv2.imwrite(str(panel_path), process_panel)

    summary = {
        "frame_dir": str(frame_dir),
        "depth_raw": str(ctx["depth_raw_path"]) if ctx["depth_raw_path"] else "",
        "texture_input": str(ctx["texture_path"]),
        "focus_method": focus_method,
        "focus_box": [int(v) for v in focus_box],
        "mask_path": str(mask_path),
        "stretched_mask_path": str(stretch_path),
        "no_vulva_surface_path": str(baseline_path),
        "vulva_only_surface_path": str(height_path),
        "panel_path": str(panel_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if csv_path is not None:
        save_csv_row(csv_path, summary)

    return summary


def main(args):
    input_path = Path(args.path)
    ctx = load_frame_context(input_path)
    frame_dir = ctx["frame_dir"]
    output_dir = Path(args.output_dir) if args.output_dir else frame_dir / "vulva_manual"
    output_dir.mkdir(parents=True, exist_ok=True)

    depth_mm = ctx["depth_mm"]
    pig_mask = largest_component(ctx["pig_mask"])
    texture_gray = ctx["texture_gray"]

    if args.auto_mask:
        raw_mask, focus_box, focus_method = auto_detect_vulva_mask(texture_gray, pig_mask)
    else:
        focus_box, _ = infer_vulva_focus_box(texture_gray, pig_mask)
        raw_mask, focus_box, focus_method = draw_manual_vulva_mask(texture_gray, focus_box)

    summary = process_vulva_measurement(
        ctx=ctx,
        raw_mask=raw_mask,
        focus_box=focus_box,
        focus_method=focus_method,
        output_dir=output_dir,
        csv_path=None,
    )

    print(f"Frame:        {frame_dir}")
    print(f"Focus method: {focus_method}")
    print(f"Focus box:    {focus_box}")
    print(f"Output dir:   {output_dir}")
    print(f"  Mask:       {summary['mask_path']}")
    print(f"  Surface:    {summary['vulva_only_surface_path']}")
    print(f"  Panel:      {summary['panel_path']}")
    print(f"  Summary:    {output_dir / 'vulva_depthmap_summary.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build manual vulva depth outputs on a single depthmap frame")
    parser.add_argument("path", help="Depthmap frame directory or a depth raw file")
    parser.add_argument("--output-dir", default="", help="Where to save vulva depth outputs (default: <frame>/vulva_manual)")
    parser.add_argument("--auto-mask", action="store_true", help="Use IR-based auto seed instead of manual polygon drawing")
    main(parser.parse_args())
