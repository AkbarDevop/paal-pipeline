#!/usr/bin/env python3
"""Build paper-style vulva surfaces from a manually cropped vulva rectangle.

Replicates Xu et al. (2023) "Detecting sow vulva size change around estrus
using machine vision technology" — Section 2.4, Figure 4.

Pipeline (matching the paper):
1. Load the manually cropped vulva-centered rectangle
2. Rotate the cropped 3D region to a horizontal plane (plane fit)
3. Rasterize to an Original Surface grid (auto-sized for coverage, or 300×300)
4. Detect the vulva as the largest round region (paper's regionprops)
5. Dilate the detected vulva region by 35% (paper's imdilate)
6. Interpolate background to estimate a No Vulva Surface
7. Vulva Only Surface = OS − No Vulva Surface
8. Fit ellipse → Width, Length, Height, SA, BA, HRA, VRA, V, CV

Uses IR images (not RGB) as primary texture — IR correlates with depth
and shows vulva detail that RGB misses.

Examples:
    python build_paper_vulva_surfaces.py depthmap_rect\\20260211-09-06-52\\pig0_20260211-09-07-10
    python build_paper_vulva_surfaces.py depthmap_rect\\20260211-09-06-52\\pig0_20260211-09-07-10\\summary.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_PLT = True
except ImportError:
    HAS_PLT = False

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False

from crop_vulva_pointcloud import (
    build_context_mask,
    build_rect_mask,
    derive_depth_raw_path,
    load_input_summary,
    load_inputs,
    resolve_input_path,
)


DEFAULT_GRID_SIZE = 0  # 0 = auto-size based on point count


def choose_grid_size(n_points: int, min_coverage: float = 0.40) -> int:
    """Pick a grid size so that at least *min_coverage* of cells have data.

    With sparse OAK-D data we get far fewer points than the paper's L515,
    so blindly using 300×300 leaves ~88% of the grid interpolated.  This
    picks the largest grid where coverage stays above *min_coverage*,
    clamped to [64, 300].
    """
    # Each point fills ~1 cell, so coverage ≈ n_points / grid^2.
    ideal = int(math.sqrt(n_points / min_coverage))
    return max(64, min(ideal, 300))


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-9:
        raise ValueError("Cannot normalize a near-zero vector")
    return vec / norm


def largest_component(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask.astype(np.uint8) * 255)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num <= 1:
        return mask
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == idx


def load_rect_input(input_path_str: str) -> tuple[Path, dict]:
    input_path = resolve_input_path(input_path_str)
    if input_path.is_file() and input_path.name.lower() == "summary.json":
        input_path = input_path.parent

    if not input_path.is_dir():
        raise ValueError("Input must be a depthmap_rect frame directory or its summary.json")

    summary = load_input_summary(input_path)
    if summary.get("output_mode") != "vulva_rect_crop":
        raise ValueError(f"Expected a vulva rectangle crop folder, got output_mode={summary.get('output_mode')!r}")

    return input_path, summary


def backproject_mask(
    depth_mm: np.ndarray,
    texture_rgb: np.ndarray,
    mask: np.ndarray,
    intrinsics: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("Crop mask does not contain any valid points")

    z_m = depth_mm[mask].astype(np.float32) / 1000.0
    x_m = (xs.astype(np.float32) - float(intrinsics["cx"])) * z_m / float(intrinsics["fx"])
    y_m = (ys.astype(np.float32) - float(intrinsics["cy"])) * z_m / float(intrinsics["fy"])
    xyz = np.stack([x_m, y_m, z_m], axis=1)
    colors = texture_rgb[mask].astype(np.float32)
    return xs.astype(np.float32), ys.astype(np.float32), xyz, colors


def fit_plane_svd(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    centered = points - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = normalize(vh[-1])
    return center, normal


def fit_rotating_plane(points: np.ndarray, iterations: int = 3) -> tuple[np.ndarray, np.ndarray, int]:
    support = points
    support_count = len(points)
    for _ in range(iterations):
        center, normal = fit_plane_svd(support)
        signed = (points - center) @ normal
        lo, hi = np.percentile(signed, [10, 80])
        next_support = points[(signed >= lo) & (signed <= hi)]
        if len(next_support) < 64:
            break
        support = next_support
        support_count = len(support)

    center, normal = fit_plane_svd(support)
    return center, normal, support_count


def project_vector_to_plane(vec: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return vec - normal * float(np.dot(vec, normal))


def build_plane_axes(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal = normalize(normal)

    axis_u = project_vector_to_plane(np.array([1.0, 0.0, 0.0], dtype=np.float32), normal)
    if float(np.linalg.norm(axis_u)) <= 1e-6:
        axis_u = project_vector_to_plane(np.array([0.0, 1.0, 0.0], dtype=np.float32), normal)
    axis_u = normalize(axis_u)

    axis_v = project_vector_to_plane(np.array([0.0, 1.0, 0.0], dtype=np.float32), normal)
    axis_v = axis_v - axis_u * float(np.dot(axis_v, axis_u))
    if float(np.linalg.norm(axis_v)) <= 1e-6:
        axis_v = np.cross(normal, axis_u)
    axis_v = normalize(axis_v)

    rebuilt_normal = normalize(np.cross(axis_u, axis_v))
    if float(np.dot(rebuilt_normal, normal)) < 0.0:
        axis_v = -axis_v
        rebuilt_normal = normalize(np.cross(axis_u, axis_v))
    return axis_u, axis_v, rebuilt_normal


def orient_normal_to_vulva_center(
    plane_center: np.ndarray,
    normal: np.ndarray,
    xyz: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    focus_box: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = focus_box
    inner_x1 = x1 + int(round(0.25 * (x2 - x1)))
    inner_x2 = x2 - int(round(0.25 * (x2 - x1)))
    inner_y1 = y1 + int(round(0.25 * (y2 - y1)))
    inner_y2 = y2 - int(round(0.25 * (y2 - y1)))

    center_sel = (xs >= inner_x1) & (xs <= inner_x2) & (ys >= inner_y1) & (ys <= inner_y2)
    outer_sel = ~center_sel

    signed = (xyz - plane_center) @ normal
    if int(center_sel.sum()) >= 20 and int(outer_sel.sum()) >= 20:
        center_median = float(np.median(signed[center_sel]))
        outer_median = float(np.median(signed[outer_sel]))
        if center_median < outer_median:
            normal = -normal

    axis_u, axis_v, normal = build_plane_axes(normal)
    return axis_u, axis_v, normal


def project_xyz_to_plane(
    xyz: np.ndarray,
    plane_center: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
    normal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = xyz - plane_center
    u = centered @ axis_u
    v = centered @ axis_v
    w = centered @ normal
    return u.astype(np.float32), v.astype(np.float32), w.astype(np.float32)


def safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    if float(np.std(a)) <= 1e-9 or float(np.std(b)) <= 1e-9:
        return 0.0
    return float(np.corrcoef(a.astype(np.float32), b.astype(np.float32))[0, 1])


def align_plane_axes_to_image(
    plane_center: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
    normal: np.ndarray,
    xyz: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    u, v, _ = project_xyz_to_plane(xyz, plane_center, axis_u, axis_v, normal)
    corr_u_x_before = safe_corrcoef(u, xs)
    corr_v_y_before = safe_corrcoef(v, ys)

    if corr_u_x_before < 0.0:
        axis_u = -axis_u
        u = -u
    if corr_v_y_before < 0.0:
        axis_v = -axis_v
        v = -v

    corr_u_x_after = safe_corrcoef(u, xs)
    corr_v_y_after = safe_corrcoef(v, ys)
    return axis_u, axis_v, {
        "corr_u_x_before": round(corr_u_x_before, 6),
        "corr_v_y_before": round(corr_v_y_before, 6),
        "corr_u_x_after": round(corr_u_x_after, 6),
        "corr_v_y_after": round(corr_v_y_after, 6),
    }


def diffuse_fill(channel: np.ndarray, known_mask: np.ndarray, iterations: int = 180, sigma: float = 2.0) -> np.ndarray:
    if not np.any(known_mask):
        raise ValueError("Known mask is empty; cannot interpolate a surface")

    work = channel.astype(np.float32).copy()
    fixed_values = work.copy()
    fill_mask = ~known_mask
    init_value = float(np.median(work[known_mask]))
    work[fill_mask] = init_value

    for _ in range(iterations):
        blurred = cv2.GaussianBlur(work, (0, 0), sigmaX=sigma)
        work[fill_mask] = blurred[fill_mask]
        work[known_mask] = fixed_values[known_mask]
    return work


def segment_vulva_frequency(
    original_surface_mm: np.ndarray,
    sigma_body: float = 15.0,
) -> dict:
    """Separate vulva (high-freq protrusion) from body surface (low-freq) via Gaussian bandpass.

    The body surface (butt, tail contours) varies slowly and is captured by a
    large-sigma Gaussian blur.  Subtracting this low-pass estimate from the
    Original Surface isolates the vulva as the positive residual.

    Args:
        original_surface_mm: 2-D array of the rasterized OS height in mm.
        sigma_body: Gaussian sigma (in pixels) for the low-pass body estimate.
            Larger values smooth over bigger features. 15 px works well for a
            300x300 grid covering ~120 mm x 180 mm.

    Returns:
        dict with keys: vulva_mask, body_surface_mm, residual_mm,
        vulva_only_mm, sigma_body_px, threshold_mm, mask_pixels.
    """
    os_f = original_surface_mm.astype(np.float32)

    # Low-pass: body surface estimate
    ksize = int(np.ceil(sigma_body * 6)) | 1  # ensure odd kernel
    body_surface_mm = cv2.GaussianBlur(os_f, (ksize, ksize), sigmaX=sigma_body, sigmaY=sigma_body)

    # High-pass residual = original - body = vulva protrusion
    residual_mm = os_f - body_surface_mm

    # Adaptive threshold via Otsu on positive residual
    pos_res = np.clip(residual_mm, 0.0, None)
    peak = float(pos_res.max())
    if peak > 0.01:
        norm8 = (pos_res / peak * 255.0).astype(np.uint8)
        otsu_val, _ = cv2.threshold(norm8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold_mm = float(otsu_val) / 255.0 * peak
    else:
        threshold_mm = 0.5

    threshold_mm = max(threshold_mm, 0.3)  # floor to reject noise

    vulva_mask = residual_mm > threshold_mm

    # Morphological cleanup: open removes noise, close fills holes
    vulva_mask = cv2.morphologyEx(
        (vulva_mask.astype(np.uint8) * 255),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ) > 0
    vulva_mask = cv2.morphologyEx(
        (vulva_mask.astype(np.uint8) * 255),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    ) > 0

    vulva_mask = largest_component(vulva_mask)

    # Vulva-only surface: positive residual inside mask, zero outside
    vulva_only_mm = np.clip(residual_mm, 0.0, None).copy()
    vulva_only_mm[~vulva_mask] = 0.0

    return {
        "vulva_mask": vulva_mask,
        "body_surface_mm": body_surface_mm,
        "residual_mm": residual_mm,
        "vulva_only_mm": vulva_only_mm,
        "sigma_body_px": sigma_body,
        "threshold_mm": round(float(threshold_mm), 4),
        "mask_pixels": int(vulva_mask.sum()),
    }


def _cluster_circularity(pts_u: np.ndarray, pts_v: np.ndarray) -> tuple[float, float]:
    """Compute circularity and aspect ratio of a 2D point set.

    Circularity = 4π·area / perimeter² (1.0 = perfect circle).
    Aspect ratio = minor_axis / major_axis (1.0 = circle, 0 = line).
    Uses the convex hull perimeter and area for robustness.
    """
    if len(pts_u) < 5:
        return 0.0, 0.0
    pts_2d = np.stack([pts_u, pts_v], axis=1).astype(np.float32)
    hull = cv2.convexHull(pts_2d)
    area = float(cv2.contourArea(hull))
    perimeter = float(cv2.arcLength(hull, closed=True))
    if perimeter < 1e-9:
        return 0.0, 0.0
    circularity = 4.0 * math.pi * area / (perimeter * perimeter)
    # Fit ellipse for aspect ratio (needs ≥ 5 points)
    if len(pts_2d) >= 5:
        _, (minor, major), _ = cv2.fitEllipse(pts_2d)
        aspect = minor / max(major, 1e-9)
    else:
        aspect = 0.0
    return float(circularity), float(aspect)


def detect_vulva_3d(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    grid_size: int,
    u_bounds: tuple[float, float],
    v_bounds: tuple[float, float],
) -> tuple[np.ndarray, dict]:
    """Detect the vulva as a circular/oval protruding 3D cluster using Open3D DBSCAN.

    The vulva is always circular or oval shaped.  This method:
      1. Select points in the top 25% of height (strong protrusion)
      2. DBSCAN cluster the protruding points in (u, v, w) space
      3. Score clusters by circularity, centrality, height, and size
         — circularity is weighted most heavily since vulva is always round/oval
      4. Map the best cluster back to a grid mask for measurement

    Args:
        u, v, w: Per-point plane coordinates (metres).
        grid_size: Size of the rasterization grid.
        u_bounds, v_bounds: (min, max) of the u/v coordinates.

    Returns:
        (grid_mask, stats_dict) where grid_mask is a bool array on the
        rasterization grid marking vulva pixels.
    """
    if not HAS_O3D:
        raise ImportError("Open3D is required for 3D vulva detection. pip install open3d")

    w_mm = w * 1000.0  # work in mm
    n_total = len(w_mm)
    median_h = float(np.median(w_mm))
    std_h = float(np.std(w_mm))
    u_range = float(u.max() - u.min())
    v_range = float(v.max() - v.min())
    u_center = float(u.mean())
    v_center = float(v.mean())
    max_dist = max(float(np.hypot(u_range, v_range)) / 2.0, 0.001)
    avg_spacing = math.sqrt(u_range * v_range / max(n_total, 1))

    u_min, u_max = u_bounds
    v_min, v_max = v_bounds
    if abs(u_max - u_min) < 1e-9:
        u_max = u_min + 1e-6
    if abs(v_max - v_min) < 1e-9:
        v_max = v_min + 1e-6

    # Escalating threshold search: start at 75th pct, raise if the
    # result covers too much of the grid (>40%) — this handles crops
    # where the whole surface protrudes uniformly.
    max_grid_coverage = 0.40  # ellipse shouldn't cover > 40% of grid

    for attempt, pct in enumerate([75, 80, 85, 90]):
        protrusion_thr = float(np.percentile(w_mm, pct))
        protrusion_thr = max(protrusion_thr, median_h + max(1.0 * std_h, 2.0))
        protruding = w_mm > protrusion_thr

        n_protruding = int(protruding.sum())
        if n_protruding < 20:
            if attempt == 0:
                protrusion_thr = float(np.percentile(w_mm, 65))
                protruding = w_mm > protrusion_thr
                n_protruding = int(protruding.sum())
            else:
                continue

        if n_protruding < 10:
            continue

        # DBSCAN clustering
        proto_uvw = np.stack([u[protruding], v[protruding], w[protruding]], axis=1).astype(np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(proto_uvw)

        eps = max(avg_spacing * 5.0, 0.004)
        min_points = max(5, min(n_protruding // 50, 15))

        labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
        if len(labels) == 0 or labels.max() < 0:
            continue

        # Score clusters
        best_label = -1
        best_score = -1e9
        best_circ = 0.0
        best_aspect = 0.0
        cluster_ids = set(labels[labels >= 0].tolist())

        proto_u = u[protruding]
        proto_v = v[protruding]
        proto_w_mm = w_mm[protruding]

        for cid in cluster_ids:
            cmask = labels == cid
            count = int(cmask.sum())
            if count < 10:
                continue

            cu = float(proto_u[cmask].mean())
            cv_ = float(proto_v[cmask].mean())
            mean_h = float(proto_w_mm[cmask].mean())

            circ, aspect = _cluster_circularity(proto_u[cmask], proto_v[cmask])
            centrality = 1.0 - min(float(np.hypot(cu - u_center, cv_ - v_center)) / max_dist, 1.0)

            size_frac = count / max(n_total, 1)
            size_penalty = max(0.0, size_frac - 0.30) * 5.0

            score = (
                circ * 5.0
                + aspect * 2.0
                + centrality * 2.0
                + (mean_h / max(std_h, 1.0)) * 1.0
                - size_penalty
            )
            if score > best_score:
                best_score = score
                best_label = cid
                best_circ = circ
                best_aspect = aspect

        if best_label < 0:
            continue

        # Fit ellipse
        cluster_mask_pts = labels == best_label
        proto_indices = np.where(protruding)[0]
        vulva_u = u[proto_indices[cluster_mask_pts]]
        vulva_v = v[proto_indices[cluster_mask_pts]]

        cols_f = (vulva_u - u_min) / (u_max - u_min) * (grid_size - 1)
        rows_f = (vulva_v - v_min) / (v_max - v_min) * (grid_size - 1)

        grid_mask = np.zeros((grid_size, grid_size), dtype=np.uint8)
        if len(vulva_u) >= 5:
            pts_2d = np.stack([cols_f, rows_f], axis=1).astype(np.float32)
            ellipse = cv2.fitEllipse(pts_2d)
            cv2.ellipse(grid_mask, ellipse, 255, thickness=-1)
        else:
            cols_i = np.clip(np.round(cols_f), 0, grid_size - 1).astype(np.int32)
            rows_i = np.clip(np.round(rows_f), 0, grid_size - 1).astype(np.int32)
            grid_mask[rows_i, cols_i] = 255

        grid_mask = grid_mask > 0
        coverage = float(grid_mask.sum()) / (grid_size * grid_size)

        if coverage <= max_grid_coverage:
            # Good — ellipse is a reasonable size
            break
        # Too large — try a stricter threshold next iteration

    else:
        # All attempts exhausted — use the last result
        if best_label < 0:
            raise ValueError("No suitable circular cluster found for vulva")

    stats = {
        "seed_detection_method": "open3d_dbscan",
        "protrusion_threshold_mm": round(float(protrusion_thr), 2),
        "protruding_points": n_protruding,
        "dbscan_eps_m": round(float(eps), 6),
        "dbscan_min_points": min_points,
        "num_clusters": len(cluster_ids),
        "vulva_cluster_points": int(cluster_mask_pts.sum()),
        "vulva_cluster_label": int(best_label),
        "vulva_circularity": round(best_circ, 4),
        "vulva_aspect_ratio": round(best_aspect, 4),
        "seed_pixels": int(grid_mask.sum()),
        "seed_area_pixels": int(grid_mask.sum()),
        "seed_circularity": round(best_circ, 4),
        "seed_aspect": round(best_aspect, 4),
        "seed_bbox": [],
    }
    return grid_mask, stats


def rasterize_to_original_surface(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    colors: np.ndarray,
    grid_size: int,
) -> dict:
    u_min = float(u.min())
    u_max = float(u.max())
    v_min = float(v.min())
    v_max = float(v.max())

    if abs(u_max - u_min) < 1e-9:
        u_max = u_min + 1e-6
    if abs(v_max - v_min) < 1e-9:
        v_max = v_min + 1e-6

    cols = np.clip(np.round((u - u_min) / (u_max - u_min) * (grid_size - 1)), 0, grid_size - 1).astype(np.int32)
    rows = np.clip(np.round((v - v_min) / (v_max - v_min) * (grid_size - 1)), 0, grid_size - 1).astype(np.int32)

    count_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    w_sum = np.zeros((grid_size, grid_size), dtype=np.float32)
    rgb_sum = np.zeros((grid_size, grid_size, 3), dtype=np.float32)

    np.add.at(count_grid, (rows, cols), 1.0)
    np.add.at(w_sum, (rows, cols), w.astype(np.float32))
    for channel_idx in range(3):
        np.add.at(rgb_sum[..., channel_idx], (rows, cols), colors[:, channel_idx].astype(np.float32))

    valid_grid = count_grid > 0
    w_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    rgb_grid = np.zeros((grid_size, grid_size, 3), dtype=np.float32)
    w_grid[valid_grid] = w_sum[valid_grid] / count_grid[valid_grid]
    for channel_idx in range(3):
        rgb_grid[..., channel_idx][valid_grid] = rgb_sum[..., channel_idx][valid_grid] / count_grid[valid_grid]

    w_grid_filled = diffuse_fill(w_grid, valid_grid, iterations=220, sigma=1.8)
    rgb_filled = np.zeros_like(rgb_grid, dtype=np.float32)
    for channel_idx in range(3):
        rgb_filled[..., channel_idx] = diffuse_fill(rgb_grid[..., channel_idx], valid_grid, iterations=120, sigma=1.6)
    rgb_filled = np.clip(rgb_filled, 0.0, 255.0).astype(np.uint8)

    u_coords = np.linspace(u_min, u_max, grid_size, dtype=np.float32)
    v_coords = np.linspace(v_min, v_max, grid_size, dtype=np.float32)
    uu, vv = np.meshgrid(u_coords, v_coords)

    return {
        "u_coords": u_coords,
        "v_coords": v_coords,
        "u_grid": uu,
        "v_grid": vv,
        "w_grid": w_grid_filled,
        "rgb_grid": rgb_filled,
        "valid_grid": valid_grid,
        "coverage_ratio": float(valid_grid.mean()),
        "u_bounds": [u_min, u_max],
        "v_bounds": [v_min, v_max],
    }


def make_depth_colormap(
    surface_mm: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Paper-style depth view: linear depth → JET colormap (Figure 3.d, 4.b).

    Matches MATLAB's depth-to-RGB conversion used in the paper.
    Blue = low height, green = mid, red = high.  Black = background / zero.
    """
    h, w = surface_mm.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    data = surface_mm.astype(np.float32).copy()
    if mask is not None:
        data[~mask] = 0.0

    vmin = float(data.min())
    vmax = float(data.max())
    if abs(vmax - vmin) < 0.01:
        return vis

    # Linear normalisation min→max, then JET colormap
    norm = np.clip((data - vmin) / (vmax - vmin), 0.0, 1.0)
    vis = cv2.applyColorMap((norm * 255.0).astype(np.uint8), cv2.COLORMAP_JET)

    # Black out background where height ≤ 0 (for vulva-only surfaces)
    vis[data <= 0.0] = 0

    return vis


def component_score(
    component: np.ndarray,
    signal: np.ndarray,
    prominence: np.ndarray,
    image_shape: tuple[int, int],
) -> float:
    area = float(component.sum())
    if area <= 0:
        return -1e9

    mask_u8 = (component.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return -1e9
    contour = max(contours, key=cv2.contourArea)
    perimeter = max(float(cv2.arcLength(contour, True)), 1.0)
    circularity = float(4.0 * math.pi * area / (perimeter * perimeter))

    x, y, w, h = cv2.boundingRect(contour)
    aspect = float(min(w, h) / max(w, h, 1))
    moments = cv2.moments(contour)
    if abs(moments["m00"]) > 1e-9:
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
    else:
        cy, cx = np.array(np.where(component)).mean(axis=1)

    center_x = 0.5 * (image_shape[1] - 1)
    center_y = 0.5 * (image_shape[0] - 1)
    max_dist = max(float(np.hypot(center_x, center_y)), 1.0)
    centrality = 1.0 - min(float(np.hypot(cx - center_x, cy - center_y)) / max_dist, 1.0)

    signal_scale = max(float(np.std(signal)), 1.0)
    prominence_scale = max(float(np.std(prominence)), 1.0)
    mean_signal = float(signal[component].mean()) / signal_scale
    mean_prominence = float(prominence[component].mean()) / prominence_scale
    area_frac = area / max(float(image_shape[0] * image_shape[1]), 1.0)
    area_pref = math.exp(-((area_frac - 0.025) ** 2) / (2.0 * (0.02 ** 2)))
    border_penalty = 1.0 if (x > 3 and y > 3 and (x + w) < image_shape[1] - 3 and (y + h) < image_shape[0] - 3) else 0.45

    return (
        2.4 * circularity
        + 1.4 * aspect
        + 3.8 * centrality
        + 1.4 * area_pref
        + 0.9 * mean_signal
        + 0.5 * mean_prominence
        + 1.2 * border_penalty
    )


def detect_vulva_seed_region(original_w_mm: np.ndarray) -> tuple[np.ndarray, dict]:
    """Approximate the paper's 'largest round region' rule on the OS depth map."""
    signal = original_w_mm.astype(np.float32)
    valid_mask = np.ones(signal.shape, dtype=bool)
    image_h, image_w = signal.shape

    small = cv2.GaussianBlur(signal, (0, 0), sigmaX=2.0)
    large = cv2.GaussianBlur(signal, (0, 0), sigmaX=12.0)
    prominence = small - large

    min_area = max(120, int(round(signal.size * 0.0015)))
    max_area = int(round(signal.size * 0.20))
    roundness_threshold = 0.35
    aspect_threshold = 0.45

    best_mask = None
    best_meta = None

    def consider_candidates(source_name: str, arr: np.ndarray, percentiles: list[int]):
        nonlocal best_mask, best_meta
        for pct in percentiles:
            thr = float(np.percentile(arr[valid_mask], pct))
            candidate = valid_mask & (arr >= thr)
            candidate = cv2.morphologyEx(
                (candidate.astype(np.uint8) * 255),
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            ) > 0
            candidate = cv2.morphologyEx(
                (candidate.astype(np.uint8) * 255),
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            ) > 0

            num, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), connectivity=8)
            for idx in range(1, num):
                area = int(stats[idx, cv2.CC_STAT_AREA])
                if area < min_area or area > max_area:
                    continue

                component = labels == idx
                contours, _ = cv2.findContours((component.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if not contours:
                    continue
                contour = max(contours, key=cv2.contourArea)
                perimeter = max(float(cv2.arcLength(contour, True)), 1.0)
                circularity = float(4.0 * math.pi * area / (perimeter * perimeter))

                x, y, w, h = cv2.boundingRect(contour)
                aspect = float(min(w, h) / max(w, h, 1))
                touches_border = x <= 3 or y <= 3 or (x + w) >= image_w - 3 or (y + h) >= image_h - 3

                if circularity < roundness_threshold or aspect < aspect_threshold or touches_border:
                    continue

                candidate_key = (
                    area,
                    circularity,
                    aspect,
                    -abs(pct - 92),
                )
                if best_meta is None or candidate_key > best_meta["key"]:
                    best_mask = component
                    best_meta = {
                        "key": candidate_key,
                        "source_name": source_name,
                        "percentile": pct,
                        "area": area,
                        "circularity": circularity,
                        "aspect": aspect,
                        "bbox": [int(x), int(y), int(x + w - 1), int(y + h - 1)],
                    }

    consider_candidates("largest_round_region_os_height", signal, [88, 90, 92, 94, 96])
    if best_mask is None:
        consider_candidates("largest_round_region_os_prominence", prominence, [88, 90, 92, 94, 96, 97, 98])

    if best_mask is None:
        fallback_thr = float(np.percentile(prominence[valid_mask], 94))
        best_mask = largest_component(prominence >= fallback_thr)
        best_meta = {
            "source_name": "fallback_prominence_component",
            "percentile": 94,
            "area": int(best_mask.sum()),
            "circularity": 0.0,
            "aspect": 0.0,
            "bbox": [],
            "key": (int(best_mask.sum()), 0.0, 0.0, 0),
        }

    best_mask = largest_component(best_mask)
    return best_mask, {
        "seed_detection_method": best_meta["source_name"],
        "seed_percentile": int(best_meta["percentile"]),
        "seed_pixels": int(best_mask.sum()),
        "seed_area_pixels": int(best_meta["area"]),
        "seed_circularity": round(float(best_meta["circularity"]), 4),
        "seed_aspect": round(float(best_meta["aspect"]), 4),
        "seed_bbox": best_meta["bbox"],
    }


def grow_mask_by_percent(mask: np.ndarray, growth_fraction: float) -> np.ndarray:
    mask = largest_component(mask)
    target_pixels = int(math.ceil(mask.sum() * (1.0 + growth_fraction)))
    grown = mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    for _ in range(64):
        if int(grown.sum()) >= target_pixels:
            break
        next_mask = cv2.dilate((grown.astype(np.uint8) * 255), kernel, iterations=1) > 0
        next_mask = largest_component(next_mask)
        if int(next_mask.sum()) <= int(grown.sum()):
            break
        grown = next_mask
    return grown


def contour_from_mask(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours((mask.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contour found in mask")
    contour = max(contours, key=cv2.contourArea)
    return contour.reshape(-1, 2).astype(np.float32)


def derive_ellipse_axes(ellipse) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
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


def pixel_points_to_uv(points_xy: np.ndarray, u_coords: np.ndarray, v_coords: np.ndarray) -> np.ndarray:
    xs = np.clip(np.round(points_xy[:, 0]).astype(np.int32), 0, len(u_coords) - 1)
    ys = np.clip(np.round(points_xy[:, 1]).astype(np.int32), 0, len(v_coords) - 1)
    return np.stack([u_coords[xs], v_coords[ys]], axis=1).astype(np.float32)


def compute_surface_area_mm2(height_mm: np.ndarray, mask: np.ndarray, du_m: float, dv_m: float) -> float:
    height_m = height_mm.astype(np.float32) / 1000.0
    dv_grid, du_grid = np.gradient(height_m, dv_m, du_m)
    local_area = np.sqrt(1.0 + du_grid * du_grid + dv_grid * dv_grid) * (du_m * dv_m)
    return round(float(local_area[mask].sum()) * 1_000_000.0, 2)


def compute_measurements(
    vulva_only_mm: np.ndarray,
    measurement_mask: np.ndarray,
    u_coords: np.ndarray,
    v_coords: np.ndarray,
) -> dict:
    if not np.any(measurement_mask):
        raise ValueError("Measurement mask is empty")

    contour_xy = contour_from_mask(measurement_mask)
    contour_uv = pixel_points_to_uv(contour_xy, u_coords, v_coords)
    if len(contour_uv) < 5:
        raise ValueError("Not enough contour points to fit an ellipse")

    ellipse = cv2.fitEllipse(contour_uv.reshape(-1, 1, 2))
    ellipse_center_uv, major_dir_uv, minor_dir_uv, major_len_m, minor_len_m, ellipse_angle_deg = derive_ellipse_axes(ellipse)
    major_line_uv = (
        ellipse_center_uv - major_dir_uv * (major_len_m / 2.0),
        ellipse_center_uv + major_dir_uv * (major_len_m / 2.0),
    )
    minor_line_uv = (
        ellipse_center_uv - minor_dir_uv * (minor_len_m / 2.0),
        ellipse_center_uv + minor_dir_uv * (minor_len_m / 2.0),
    )

    positive_values = vulva_only_mm[measurement_mask]
    peak_height_mm = round(float(positive_values.max()), 2)
    mean_height_mm = round(float(positive_values.mean()), 2)

    du_m = float(abs(u_coords[-1] - u_coords[0]) / max(len(u_coords) - 1, 1))
    dv_m = float(abs(v_coords[-1] - v_coords[0]) / max(len(v_coords) - 1, 1))
    cell_area_m2 = du_m * dv_m

    base_area_mm2 = round(float(measurement_mask.sum()) * cell_area_m2 * 1_000_000.0, 2)
    surface_area_mm2 = compute_surface_area_mm2(vulva_only_mm, measurement_mask, du_m, dv_m)
    volume_mm3 = round(float((vulva_only_mm[measurement_mask] / 1000.0).sum() * cell_area_m2) * 1_000_000_000.0, 2)

    length_mm = round(major_len_m * 1000.0, 2)
    width_mm = round(minor_len_m * 1000.0, 2)
    hra_mm2 = round(length_mm * width_mm, 2)
    vra_mm2 = round(width_mm * peak_height_mm, 2)
    cubic_volume_mm3 = round(length_mm * width_mm * peak_height_mm, 2)

    return {
        "measurement_method": "paper_rotated_os_ellipse",
        "length_mm": length_mm,
        "width_mm": width_mm,
        "peak_height_mm": peak_height_mm,
        "mean_height_mm": mean_height_mm,
        "surface_area_mm2": surface_area_mm2,
        "base_area_mm2": base_area_mm2,
        "horizontal_rect_area_mm2": hra_mm2,
        "vertical_rect_area_mm2": vra_mm2,
        "volume_mm3": volume_mm3,
        "cubic_volume_mm3": cubic_volume_mm3,
        "grid_spacing_u_mm": round(du_m * 1000.0, 4),
        "grid_spacing_v_mm": round(dv_m * 1000.0, 4),
        "ellipse_center_uv": [round(float(v), 6) for v in ellipse_center_uv],
        "major_line_uv": [[round(float(v), 6) for v in major_line_uv[0]], [round(float(v), 6) for v in major_line_uv[1]]],
        "minor_line_uv": [[round(float(v), 6) for v in minor_line_uv[0]], [round(float(v), 6) for v in minor_line_uv[1]]],
        "ellipse_size_uv_m": [round(float(major_len_m), 6), round(float(minor_len_m), 6)],
        "ellipse_angle_deg": round(float(ellipse_angle_deg), 4),
    }


def save_mask(path: Path, mask: np.ndarray) -> None:
    cv2.imwrite(str(path), (mask.astype(np.uint8) * 255))


def label_tile(image_bgr: np.ndarray, title: str) -> np.ndarray:
    tile = image_bgr.copy()
    cv2.putText(tile, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return tile


def make_overlay(base_rgb: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int], title: str) -> np.ndarray:
    overlay = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)
    tint = np.zeros_like(overlay)
    tint[:, :] = color_bgr
    overlay[mask] = cv2.addWeighted(overlay[mask], 0.35, tint[mask], 0.65, 0)
    contours, _ = cv2.findContours((mask.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(overlay, contours, -1, color_bgr, 2, cv2.LINE_AA)
    return label_tile(overlay, title)


def uv_to_pixel(uv_points: np.ndarray, u_coords: np.ndarray, v_coords: np.ndarray) -> np.ndarray:
    x = np.interp(uv_points[:, 0], [float(u_coords[0]), float(u_coords[-1])], [0.0, len(u_coords) - 1])
    y = np.interp(uv_points[:, 1], [float(v_coords[0]), float(v_coords[-1])], [0.0, len(v_coords) - 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def make_measurement_overlay(
    vulva_only_mm: np.ndarray,
    measurement_mask: np.ndarray,
    measurements: dict,
    u_coords: np.ndarray,
    v_coords: np.ndarray,
) -> np.ndarray:
    vis = make_depth_colormap(vulva_only_mm)
    vis = label_tile(vis, "Vulva Only Surface")

    major_line_uv = np.asarray(measurements["major_line_uv"], dtype=np.float32)
    minor_line_uv = np.asarray(measurements["minor_line_uv"], dtype=np.float32)
    major_line_px = uv_to_pixel(major_line_uv, u_coords, v_coords)
    minor_line_px = uv_to_pixel(minor_line_uv, u_coords, v_coords)

    def draw_line(line_px: np.ndarray, color: tuple[int, int, int], label: str, value_text: str):
        p1 = tuple(int(round(v)) for v in line_px[0])
        p2 = tuple(int(round(v)) for v in line_px[1])
        cv2.arrowedLine(vis, p1, p2, color, 2, cv2.LINE_AA, tipLength=0.04)
        cv2.arrowedLine(vis, p2, p1, color, 2, cv2.LINE_AA, tipLength=0.04)
        mx = int(round((p1[0] + p2[0]) / 2.0))
        my = int(round((p1[1] + p2[1]) / 2.0))
        cv2.putText(vis, label, (mx + 6, my - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        cv2.putText(vis, value_text, (mx + 6, my + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)

    draw_line(major_line_px, (0, 0, 255), "L", f'{measurements["length_mm"]:.1f} mm')
    draw_line(minor_line_px, (255, 0, 0), "W", f'{measurements["width_mm"]:.1f} mm')

    contours, _ = cv2.findContours((measurement_mask.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(vis, contours, -1, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(
        vis,
        f'H={measurements["peak_height_mm"]:.1f} mm  BA={measurements["base_area_mm2"]:.1f} mm^2',
        (10, vis.shape[0] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return vis


def make_frequency_panel(
    original_surface_mm: np.ndarray,
    body_surface_mm: np.ndarray,
    residual_mm: np.ndarray,
    vulva_mask: np.ndarray,
    vulva_only_mm: np.ndarray,
    measurement_mask: np.ndarray,
    measurements: dict,
    u_coords: np.ndarray,
    v_coords: np.ndarray,
    original_surface_rgb: np.ndarray,
) -> np.ndarray:
    """Paper-style 3x3 panel showing frequency decomposition of the vulva surface.

    Layout (3 columns x 3 rows):
        Row 1: Rotated Top View     | Depth View (OS)     | Body Surface (Low-pass)
        Row 2: Vulva Mask on RGB    | Vulva Mask on Depth | Vulva Only Surface
        Row 3: No-Vulva Surface     | Vulva Depth View    | Measurements

    The 'Depth View (OS)' tile uses the positive frequency residual with JET
    colormap (blue=flat body, yellow/red=vulva peak) to match the paper's
    Figure 3.d.
    """
    # Row 1: IR top | OS depth (3-level) | Body Surface (3-level)
    ir_top = label_tile(cv2.cvtColor(original_surface_rgb, cv2.COLOR_RGB2BGR), "IR Top View")
    depth_view = label_tile(make_depth_colormap(original_surface_mm), "Original Surface (OS)")
    body_vis = label_tile(make_depth_colormap(body_surface_mm), "Body Surface (Low-pass)")

    # Row 2: Vulva mask on IR | Vulva mask on depth | Vulva-Only surface
    mask_overlay_ir = make_overlay(original_surface_rgb, vulva_mask, (0, 240, 255), "Vulva Mask (IR)")
    os_depth_rgb = cv2.cvtColor(make_depth_colormap(original_surface_mm), cv2.COLOR_BGR2RGB)
    mask_overlay_depth = make_overlay(os_depth_rgb, vulva_mask, (0, 240, 255), "Vulva Mask (Depth)")
    vulva_only_vis = label_tile(make_depth_colormap(vulva_only_mm), "Vulva Only Surface")

    # Row 3: No-Vulva surface | Vulva peak region | Measurements
    no_vulva_vis = label_tile(make_depth_colormap(body_surface_mm), "No-Vulva Surface")
    vulva_peak = label_tile(make_depth_colormap(vulva_only_mm, measurement_mask), "Vulva Peak Region")
    meas_overlay = make_measurement_overlay(
        vulva_only_mm, measurement_mask, measurements, u_coords, v_coords
    )

    return build_panel([
        ir_top, depth_view, body_vis,
        mask_overlay_ir, mask_overlay_depth, vulva_only_vis,
        no_vulva_vis, vulva_peak, meas_overlay,
    ])


def build_panel(tiles: list[np.ndarray]) -> np.ndarray:
    if not tiles:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    tile_h = max(tile.shape[0] for tile in tiles)
    tile_w = max(tile.shape[1] for tile in tiles)
    prepared = []
    for tile in tiles:
        if tile.shape[:2] != (tile_h, tile_w):
            tile = cv2.resize(tile, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)
        prepared.append(tile)

    cols = 3
    rows = []
    for start in range(0, len(prepared), cols):
        row_tiles = prepared[start:start + cols]
        while len(row_tiles) < cols:
            row_tiles.append(np.full((tile_h, tile_w, 3), 18, dtype=np.uint8))
        rows.append(np.hstack(row_tiles))
    return np.vstack(rows)


def render_surface_views(
    path: Path,
    title: str,
    rgb_top: np.ndarray,
    depth_mm: np.ndarray,
    u_coords: np.ndarray,
    v_coords: np.ndarray,
) -> bool:
    if not HAS_PLT:
        return False

    uu_mm, vv_mm = np.meshgrid(u_coords * 1000.0, v_coords * 1000.0)
    facecolors = np.clip(rgb_top.astype(np.float32) / 255.0, 0.0, 1.0)
    stride = 2 if depth_mm.shape[0] >= 220 else 1

    fig = plt.figure(figsize=(12, 9))
    ax_top = fig.add_subplot(221)
    ax_top.imshow(rgb_top)
    ax_top.set_title("Top View")
    ax_top.axis("off")

    ax_front = fig.add_subplot(222, projection="3d")
    ax_front.plot_surface(
        uu_mm[::stride, ::stride],
        vv_mm[::stride, ::stride],
        depth_mm[::stride, ::stride],
        facecolors=facecolors[::stride, ::stride],
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    ax_front.view_init(elev=12, azim=-90)
    ax_front.set_title("Front View")
    ax_front.set_xlabel("U (mm)")
    ax_front.set_ylabel("V (mm)")
    ax_front.set_zlabel("Height (mm)")

    ax_depth = fig.add_subplot(223)
    ax_depth.imshow(make_depth_colormap(depth_mm))
    ax_depth.set_title("Depth View (OS)")
    ax_depth.axis("off")

    ax_side = fig.add_subplot(224, projection="3d")
    ax_side.plot_surface(
        uu_mm[::stride, ::stride],
        vv_mm[::stride, ::stride],
        depth_mm[::stride, ::stride],
        facecolors=facecolors[::stride, ::stride],
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    ax_side.view_init(elev=12, azim=0)
    ax_side.set_title("Side View")
    ax_side.set_xlabel("U (mm)")
    ax_side.set_ylabel("V (mm)")
    ax_side.set_zlabel("Height (mm)")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def render_paper_depth_view(
    path: Path,
    positive_residual_mm: np.ndarray,
    u_coords: np.ndarray,
    v_coords: np.ndarray,
    focus_mask: np.ndarray | None = None,
    title: str = "Depth View (OS)",
) -> bool:
    """Render a paper-style depth view (Figure 3.d) with axes and colorbar.

    Shows the positive frequency residual — vulva protrusion as a bright
    peak against a flat (zero) background — with proper axis labels in mm
    and a Z-height colorbar.  Auto-crops to *focus_mask* bounding box if
    provided.
    """
    if not HAS_PLT:
        return False

    data = positive_residual_mm.copy()
    u_mm = u_coords * 1000.0
    v_mm = v_coords * 1000.0

    # Auto-crop to vulva region
    if focus_mask is not None and np.any(focus_mask):
        ys, xs = np.where(focus_mask)
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        margin = max(int(max(y2 - y1, x2 - x1) * 0.5), 15)
        y1 = max(0, y1 - margin)
        y2 = min(data.shape[0], y2 + margin)
        x1 = max(0, x1 - margin)
        x2 = min(data.shape[1], x2 + margin)
        data = data[y1:y2, x1:x2]
        u_mm = u_mm[x1:x2] if x2 <= len(u_mm) else u_mm
        v_mm = v_mm[y1:y2] if y2 <= len(v_mm) else v_mm

    # Noise floor
    peak = float(data.max())
    if peak > 0.01:
        data[data < peak * 0.05] = 0.0

    extent = [float(u_mm[0]), float(u_mm[-1]), float(v_mm[-1]), float(v_mm[0])]

    fig, ax = plt.subplots(figsize=(5, 7))
    from matplotlib.colors import PowerNorm
    im = ax.imshow(
        data, extent=extent, cmap="jet", aspect="equal",
        norm=PowerNorm(gamma=0.5, vmin=0.0, vmax=float(data.max())),
    )
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Z (mm)", shrink=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def infer_metadata(depth_raw_path: Path) -> dict:
    stem_parts = depth_raw_path.stem.split("_")
    pig_id = stem_parts[0] if stem_parts else ""
    frame_timestamp = stem_parts[-1] if stem_parts else ""
    return {
        "timeslot_name": depth_raw_path.parent.name,
        "pig_id": pig_id,
        "frame_timestamp": frame_timestamp,
    }


def main(args):
    rect_dir, rect_summary = load_rect_input(args.input_path)
    depth_raw_path = derive_depth_raw_path(rect_dir)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else rect_dir / "paper_vulva"
    output_dir.mkdir(parents=True, exist_ok=True)

    depth_mm, intrinsics, texture_path, texture_gray, color_rgb, color_input, rgb_input = load_inputs(args, depth_raw_path)

    # Use IR image as primary texture — it correlates with depth and shows
    # vulva detail that RGB misses.  RGB is only used if --rgb-img is
    # explicitly given.
    ir_rgb = cv2.cvtColor(texture_gray, cv2.COLOR_GRAY2RGB)
    primary_texture = ir_rgb

    context_mask, mask_source, auto_stats = build_context_mask(args, depth_mm, rect_summary)
    focus_box = tuple(int(v) for v in rect_summary["focus_box"])
    rect_mask = build_rect_mask(depth_mm.shape, focus_box)
    crop_mask = context_mask & rect_mask
    if not np.any(crop_mask):
        raise ValueError("Rectangular crop does not contain any valid points")

    xs, ys, xyz, colors = backproject_mask(depth_mm, primary_texture, crop_mask, intrinsics)

    # Auto-size grid if requested (default) — pick size for decent coverage
    if args.grid_size <= 0:
        args.grid_size = choose_grid_size(len(xyz))
        print(f"Auto grid size: {args.grid_size} (from {len(xyz)} points)")

    plane_center, raw_normal, support_count = fit_rotating_plane(xyz)
    axis_u, axis_v, normal = orient_normal_to_vulva_center(plane_center, raw_normal, xyz, xs, ys, focus_box)
    axis_u, axis_v, orientation_stats = align_plane_axes_to_image(
        plane_center,
        axis_u,
        axis_v,
        normal,
        xyz,
        xs,
        ys,
    )
    u, v, w = project_xyz_to_plane(xyz, plane_center, axis_u, axis_v, normal)

    os_grid = rasterize_to_original_surface(u, v, w, colors, grid_size=args.grid_size)
    original_surface_mm = os_grid["w_grid"] * 1000.0
    original_surface_rgb = os_grid["rgb_grid"]

    # ---- Vulva segmentation ----
    freq = None

    if args.segmentation_method == "frequency":
        freq = segment_vulva_frequency(original_surface_mm, sigma_body=args.sigma_body)
        seed_mask = freq["vulva_mask"]
        dilated_mask = freq["vulva_mask"]
        no_vulva_surface_mm = freq["body_surface_mm"]
        vulva_only_surface_mm = freq["vulva_only_mm"]
        positive_eps = freq["threshold_mm"]
        seed_stats = {
            "seed_detection_method": "frequency_bandpass",
            "sigma_body_px": freq["sigma_body_px"],
            "threshold_mm": freq["threshold_mm"],
            "seed_pixels": freq["mask_pixels"],
            "seed_area_pixels": freq["mask_pixels"],
            "seed_circularity": 0.0,
            "seed_aspect": 0.0,
            "seed_bbox": [],
        }
        measurement_mask = freq["vulva_mask"].copy()
        measurement_mask = largest_component(measurement_mask)
        if not np.any(measurement_mask):
            measurement_mask = vulva_only_surface_mm > 0.3
            measurement_mask = largest_component(measurement_mask)

    elif args.segmentation_method == "open3d":
        # ---- Open3D 3D DBSCAN vulva detection ----
        seed_mask, seed_stats = detect_vulva_3d(
            u, v, w,
            grid_size=args.grid_size,
            u_bounds=(float(u.min()), float(u.max())),
            v_bounds=(float(v.min()), float(v.max())),
        )
        dilated_mask = grow_mask_by_percent(seed_mask, growth_fraction=args.mask_growth_fraction)

        no_vulva_surface_mm = diffuse_fill(
            original_surface_mm,
            known_mask=~dilated_mask,
            iterations=args.inpaint_iterations,
            sigma=2.1,
        )
        vulva_only_surface_mm = np.clip(original_surface_mm - no_vulva_surface_mm, 0.0, None)

        positive_eps = max(0.25, 0.03 * float(vulva_only_surface_mm.max()))
        measurement_mask = (vulva_only_surface_mm > positive_eps) & dilated_mask
        measurement_mask = cv2.morphologyEx(
            (measurement_mask.astype(np.uint8) * 255),
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        ) > 0
        measurement_mask = largest_component(measurement_mask)
        if not np.any(measurement_mask):
            measurement_mask = seed_mask.copy()

    else:
        # ---- Original seed-based (OpenCV regionprops) ----
        seed_mask, seed_stats = detect_vulva_seed_region(original_surface_mm)
        dilated_mask = grow_mask_by_percent(seed_mask, growth_fraction=args.mask_growth_fraction)

        no_vulva_surface_mm = diffuse_fill(
            original_surface_mm,
            known_mask=~dilated_mask,
            iterations=args.inpaint_iterations,
            sigma=2.1,
        )
        vulva_only_surface_mm = np.clip(original_surface_mm - no_vulva_surface_mm, 0.0, None)

        positive_eps = max(0.25, 0.03 * float(vulva_only_surface_mm.max()))
        measurement_mask = (vulva_only_surface_mm > positive_eps) & dilated_mask
        measurement_mask = cv2.morphologyEx(
            (measurement_mask.astype(np.uint8) * 255),
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        ) > 0
        measurement_mask = largest_component(measurement_mask)
        if not np.any(measurement_mask):
            measurement_mask = seed_mask.copy()

    measurements = compute_measurements(
        vulva_only_mm=vulva_only_surface_mm,
        measurement_mask=measurement_mask,
        u_coords=os_grid["u_coords"],
        v_coords=os_grid["v_coords"],
    )

    # ---- Visualization (clean 3-level colormaps) ----
    all_mask = np.ones(original_surface_mm.shape, dtype=bool)
    original_depth_vis = label_tile(
        make_depth_colormap(original_surface_mm),
        "Original Surface (OS)",
    )
    no_vulva_vis = label_tile(
        make_depth_colormap(no_vulva_surface_mm),
        "No Vulva Surface",
    )
    vulva_only_vis = label_tile(
        make_depth_colormap(vulva_only_surface_mm),
        "Vulva Only Surface",
    )

    original_ir_top = label_tile(cv2.cvtColor(original_surface_rgb, cv2.COLOR_RGB2BGR), "IR Top View")
    os_depth_rgb = cv2.cvtColor(original_depth_vis, cv2.COLOR_BGR2RGB)
    seed_overlay_ir = make_overlay(original_surface_rgb, seed_mask, (0, 240, 255), "Detected Vulva (IR)")
    dilated_overlay_ir = make_overlay(original_surface_rgb, dilated_mask, (90, 220, 90), "Mask +35% (IR)")
    measurement_overlay = make_measurement_overlay(
        vulva_only_surface_mm,
        measurement_mask,
        measurements,
        os_grid["u_coords"],
        os_grid["v_coords"],
    )

    # 3x3 panel: IR view, depth views, masks, measurements
    panel = build_panel(
        [
            original_ir_top,
            original_depth_vis,
            seed_overlay_ir,
            dilated_overlay_ir,
            no_vulva_vis,
            vulva_only_vis,
            measurement_overlay,
            label_tile(make_depth_colormap(vulva_only_surface_mm, measurement_mask), "Vulva Peak Region"),
            make_overlay(os_depth_rgb, measurement_mask, (0, 200, 0), "Measurement Mask (OS)"),
        ]
    )

    metadata = infer_metadata(depth_raw_path)

    original_surface_xyz = (
        plane_center.reshape(1, 1, 3)
        + os_grid["u_grid"][..., None] * axis_u.reshape(1, 1, 3)
        + os_grid["v_grid"][..., None] * axis_v.reshape(1, 1, 3)
        + os_grid["w_grid"][..., None] * normal.reshape(1, 1, 3)
    )
    no_vulva_surface_xyz = (
        plane_center.reshape(1, 1, 3)
        + os_grid["u_grid"][..., None] * axis_u.reshape(1, 1, 3)
        + os_grid["v_grid"][..., None] * axis_v.reshape(1, 1, 3)
        + (no_vulva_surface_mm / 1000.0)[..., None] * normal.reshape(1, 1, 3)
    )

    original_surface_rgb_path = output_dir / "original_surface_top_view.png"
    original_depth_path = output_dir / "original_surface_depth_os.png"
    seed_mask_path = output_dir / "vulva_seed_region.png"
    dilated_mask_path = output_dir / "vulva_mask_scaled_35pct.png"
    seed_overlay_rgb_path = output_dir / "vulva_seed_region_rgb_overlay.png"
    dilated_overlay_rgb_path = output_dir / "vulva_mask_scaled_35pct_rgb_overlay.png"
    seed_overlay_os_path = output_dir / "vulva_seed_region_os_overlay.png"
    dilated_overlay_os_path = output_dir / "vulva_mask_scaled_35pct_os_overlay.png"
    no_vulva_path = output_dir / "no_vulva_surface.png"
    vulva_only_path = output_dir / "vulva_only_surface.png"
    measurement_path = output_dir / "vulva_measurements.png"
    panel_path = output_dir / "paper_vulva_process_panel.png"
    bundle_path = output_dir / "paper_vulva_surface_bundle.npz"
    summary_path = output_dir / "paper_vulva_summary.json"
    views_path = output_dir / "paper_rotated_surface_views.png"

    cv2.imwrite(str(original_surface_rgb_path), cv2.cvtColor(original_surface_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(original_depth_path), original_depth_vis)
    save_mask(seed_mask_path, seed_mask)
    save_mask(dilated_mask_path, dilated_mask)
    cv2.imwrite(str(seed_overlay_rgb_path), seed_overlay_ir)
    cv2.imwrite(str(dilated_overlay_rgb_path), dilated_overlay_ir)
    # OS overlays — reuse the depth view as base
    seed_overlay_os = make_overlay(os_depth_rgb, seed_mask, (0, 240, 255), "Detected Vulva (OS)")
    dilated_overlay_os = make_overlay(os_depth_rgb, dilated_mask, (90, 220, 90), "Mask +35% (OS)")
    cv2.imwrite(str(seed_overlay_os_path), seed_overlay_os)
    cv2.imwrite(str(dilated_overlay_os_path), dilated_overlay_os)
    cv2.imwrite(str(no_vulva_path), no_vulva_vis)
    cv2.imwrite(str(vulva_only_path), vulva_only_vis)
    cv2.imwrite(str(measurement_path), measurement_overlay)
    cv2.imwrite(str(panel_path), panel)

    # Save frequency-specific outputs when using bandpass segmentation
    freq_residual_path = ""
    freq_body_path = ""
    if freq is not None:
        freq_residual_path = output_dir / "frequency_residual.png"
        freq_body_path = output_dir / "frequency_body_surface.png"
        cv2.imwrite(
            str(freq_residual_path),
            make_depth_colormap(freq["residual_mm"]),
        )
        cv2.imwrite(
            str(freq_body_path),
            make_depth_colormap(freq["body_surface_mm"]),
        )

    np.savez_compressed(
        bundle_path,
        original_surface_xyz=np.moveaxis(original_surface_xyz.astype(np.float32), 2, 0),
        no_vulva_surface_xyz=np.moveaxis(no_vulva_surface_xyz.astype(np.float32), 2, 0),
        original_surface_mm=original_surface_mm.astype(np.float32),
        no_vulva_surface_mm=no_vulva_surface_mm.astype(np.float32),
        vulva_only_surface_mm=vulva_only_surface_mm.astype(np.float32),
        original_surface_rgb=original_surface_rgb.astype(np.uint8),
        seed_mask=seed_mask.astype(np.uint8),
        dilated_mask=dilated_mask.astype(np.uint8),
        measurement_mask=measurement_mask.astype(np.uint8),
        u_coords=os_grid["u_coords"].astype(np.float32),
        v_coords=os_grid["v_coords"].astype(np.float32),
    )

    rendered_views = render_surface_views(
        views_path,
        title=f"{metadata['pig_id']} | Rotated Surface Views",
        rgb_top=original_surface_rgb,
        depth_mm=original_surface_mm,
        u_coords=os_grid["u_coords"],
        v_coords=os_grid["v_coords"],
    )

    # Standalone clean depth views
    depth_view_path = output_dir / "depth_view_os.png"
    cv2.imwrite(
        str(depth_view_path),
        make_depth_colormap(vulva_only_surface_mm, measurement_mask),
    )

    # Paper-style depth view with axes + colorbar (if matplotlib available)
    paper_depth_path = output_dir / "paper_depth_view.png"
    rendered_depth_view = render_paper_depth_view(
        paper_depth_path,
        vulva_only_surface_mm,
        os_grid["u_coords"],
        os_grid["v_coords"],
        focus_mask=measurement_mask,
        title=f"{metadata['pig_id']} | Vulva Only Surface",
    )

    summary = {
        **metadata,
        "input_rect_dir": str(rect_dir),
        "depth_raw": str(depth_raw_path),
        "texture_input": texture_path,
        "primary_texture": "ir",
        "rgb_image_input": color_input,
        "requested_rgb_input": rgb_input,
        "output_dir": str(output_dir),
        "output_mode": "paper_vulva_surfaces",
        "segmentation_method": args.segmentation_method,
        "grid_size": args.grid_size,
        "focus_box": [int(v) for v in focus_box],
        "context_mask_mode": mask_source,
        "crop_mask_pixels": int(crop_mask.sum()),
        "plane_support_points": support_count,
        "plane_center_m": [round(float(v), 6) for v in plane_center],
        "plane_normal": [round(float(v), 6) for v in normal],
        "plane_axis_u": [round(float(v), 6) for v in axis_u],
        "plane_axis_v": [round(float(v), 6) for v in axis_v],
        "image_orientation_alignment": orientation_stats,
        "u_bounds_m": [round(float(v), 6) for v in os_grid["u_bounds"]],
        "v_bounds_m": [round(float(v), 6) for v in os_grid["v_bounds"]],
        "os_grid_coverage_ratio": round(float(os_grid["coverage_ratio"]), 6),
        "mask_growth_fraction": args.mask_growth_fraction,
        "inpaint_iterations": args.inpaint_iterations,
        "seed_detection": seed_stats,
        "measurement_mask_pixels": int(measurement_mask.sum()),
        "measurement_positive_threshold_mm": round(float(positive_eps), 4),
        "rendered_surface_views": rendered_views,
        "original_surface_top_view": str(original_surface_rgb_path),
        "original_surface_depth_os": str(original_depth_path),
        "vulva_seed_region": str(seed_mask_path),
        "vulva_mask_scaled_35pct": str(dilated_mask_path),
        "vulva_seed_region_rgb_overlay": str(seed_overlay_rgb_path),
        "vulva_mask_scaled_35pct_rgb_overlay": str(dilated_overlay_rgb_path),
        "vulva_seed_region_os_overlay": str(seed_overlay_os_path),
        "vulva_mask_scaled_35pct_os_overlay": str(dilated_overlay_os_path),
        "no_vulva_surface": str(no_vulva_path),
        "vulva_only_surface": str(vulva_only_path),
        "vulva_measurements_image": str(measurement_path),
        "paper_process_panel": str(panel_path),
        "surface_views_image": str(views_path) if rendered_views else "",
        "surface_bundle": str(bundle_path),
        "paper_depth_view": str(paper_depth_path) if rendered_depth_view else "",
        "depth_view_os": str(output_dir / "depth_view_os.png"),
        "frequency_residual": str(freq_residual_path) if freq_residual_path else "",
        "frequency_body_surface": str(freq_body_path) if freq_body_path else "",
        "formulas": {
            "base_area_mm2": "count(vulva_only_surface > 0) * du * dv",
            "surface_area_mm2": "sum(sqrt(1 + (df/du)^2 + (df/dv)^2) * du * dv)",
            "horizontal_rect_area_mm2": "width_mm * length_mm",
            "vertical_rect_area_mm2": "width_mm * peak_height_mm",
            "volume_mm3": "sum(vulva_only_surface_height * du * dv)",
            "cubic_volume_mm3": "width_mm * length_mm * peak_height_mm",
        },
        "measurements": measurements,
        "auto_segmentation_stats": auto_stats,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Input rect:   {rect_dir}")
    print(f"Depth raw:    {depth_raw_path}")
    print(f"Output dir:   {output_dir}")
    print(f"  OS top:     {original_surface_rgb_path}")
    print(f"  OS depth:   {original_depth_path}")
    print(f"  Seed mask:  {seed_mask_path}")
    print(f"  No-vulva:   {no_vulva_path}")
    print(f"  Vulva-only: {vulva_only_path}")
    print(f"  Measure:    {measurement_path}")
    print(f"  Summary:    {summary_path}")
    print(
        "Measurements:"
        f" L={measurements['length_mm']:.2f} mm"
        f" W={measurements['width_mm']:.2f} mm"
        f" H={measurements['peak_height_mm']:.2f} mm"
        f" BA={measurements['base_area_mm2']:.2f} mm^2"
        f" V={measurements['volume_mm3']:.2f} mm^3"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build paper-style Original / No Vulva / Vulva Only surfaces")
    parser.add_argument("input_path", help="A depthmap_rect frame directory or its summary.json")
    parser.add_argument("--output-dir", default="", help="Where to save paper-style vulva outputs")
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE,
                        help="Grid size for the Original Surface (0 = auto-size for good coverage, "
                             "300 = paper's fixed size)")
    parser.add_argument("--mask-growth-fraction", type=float, default=0.35,
                        help="Grow the detected vulva region by this fraction before interpolation")
    parser.add_argument("--inpaint-iterations", type=int, default=220,
                        help="Iterations used to interpolate the No Vulva Surface (seed method)")
    parser.add_argument("--segmentation-method", choices=["open3d", "seed", "frequency"], default="open3d",
                        help="Vulva segmentation method: 'open3d' (3D DBSCAN clustering, default), "
                             "'seed' (2D regionprops + inpainting), "
                             "or 'frequency' (Gaussian bandpass — experimental)")
    parser.add_argument("--sigma-body", type=float, default=15.0,
                        help="Gaussian sigma (pixels) for the low-pass body surface estimate "
                             "(frequency method only, default=15)")
    parser.add_argument("--texture-img", "--ir-img", dest="texture_img", default=None,
                        help="Optional IR/texture image override")
    parser.add_argument("--rgb-img", default=None, help="Optional RGB image override")
    parser.add_argument("--calibration", default=None, help="Optional calibration.json path")
    parser.add_argument("--mask", default=None, help="Optional external context mask")
    parser.add_argument("--min-depth", type=int, default=200, help="Minimum valid depth in mm")
    parser.add_argument("--max-depth", type=int, default=5000, help="Maximum valid depth in mm")
    parser.add_argument("--foreground-margin-mm", type=int, default=140,
                        help="Foreground margin for the old auto pig segmentation path")
    parser.add_argument("--min-pixels", type=int, default=4000, help="Minimum mask size for auto pig segmentation")
    parser.add_argument("--auto-pig-mask", action="store_true",
                        help="Use the old heuristic pig mask before rebuilding the rectangle crop")
    parser.add_argument("--no-crop", action="store_true", help="Disable the fixed stall ROI crop")
    parser.add_argument("--keep-all-valid", action="store_true",
                        help="Keep all valid depth pixels in the ROI before applying the rectangle crop")
    main(parser.parse_args())
