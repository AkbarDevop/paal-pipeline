"""3D vulva measurements from a mask on the IR image and the matching ToF depth.

IR and depth come from the same OAK-D ToF sensor, so a mask drawn or predicted
on the 640x480 IR image indexes the depth image directly.

Steps for one frame:
  1. Skin plane: robust plane fit to the depth points in a ring just outside the
     mask (the sow's rump around the vulva).
  2. Height: distance of each mask point from that plane, towards the camera.
  3. Width and length: mask points projected onto the plane and measured along
     the in-plane principal axes (2nd-98th percentile extent). Measuring in the
     plane, not in image pixels, removes the foreshortening from a tilted view.
  4. Volume: sum of height x pixel footprint on the plane; plus the W x L x H
     proxy used by Xu et al. (2023), and Lucas's caliper index W x L / 200.

Every frame also gets quality numbers (points used, plane fit error, view angle)
so the daily aggregation can drop bad frames.
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np

TOF_W, TOF_H = 640, 480
DEFAULT_INTRINSICS = {"fx": 471.583, "fy": 471.430, "cx": 323.905, "cy": 246.816}
MIN_DEPTH_MM, MAX_DEPTH_MM = 300, 3000

MEASURE_FIELDS = [
    "width_mm", "length_mm", "peak_height_mm", "mean_height_mm",
    "area_mm2", "volume_mm3", "wlh_mm3", "vulva_index",
    "n_points", "depth_coverage", "plane_rmse_mm", "view_angle_deg", "distance_mm",
    "measure_status",
]


def load_depth_raw(path):
    """OAK-D ToF depth .raw (uint16 mm, optional 8-byte header) -> (480, 640) array or None."""
    if not path or not os.path.exists(path):
        return None
    size = os.path.getsize(path)
    expected = TOF_W * TOF_H * 2
    if size == expected + 8:
        raw = np.fromfile(path, dtype=np.uint16, offset=8)
    elif size == expected:
        raw = np.fromfile(path, dtype=np.uint16)
    else:
        return None
    return raw.reshape((TOF_H, TOF_W))


def load_intrinsics(path=None):
    """calibration.json with tof_intrinsics {fx, fy, cx, cy}, else the OAK-D defaults."""
    if path and os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path) as f:
            tof = json.load(f).get("tof_intrinsics", {})
        if all(k in tof for k in ("fx", "fy", "cx", "cy")):
            return {k: float(tof[k]) for k in ("fx", "fy", "cx", "cy")}
    return dict(DEFAULT_INTRINSICS)


def backproject(depth_mm, ys, xs, K):
    z = depth_mm[ys, xs].astype(np.float64)
    x = (xs - K["cx"]) * z / K["fx"]
    y = (ys - K["cy"]) * z / K["fy"]
    return np.stack([x, y, z], axis=1)


def fit_plane(points, iters=200, tol_mm=4.0, seed=0):
    """RANSAC plane, refined by least squares on the inliers.
    Returns (normal pointing to the camera, d, rmse) with normal.p + d = 0."""
    rng = np.random.default_rng(seed)
    best, best_n = None, -1
    n = len(points)
    for _ in range(iters if n >= 3 else 0):
        p = points[rng.choice(n, 3, replace=False)]
        nrm = np.cross(p[1] - p[0], p[2] - p[0])
        norm = np.linalg.norm(nrm)
        if norm < 1e-9:
            continue
        nrm /= norm
        d = -nrm @ p[0]
        inl = np.abs(points @ nrm + d) < tol_mm
        if inl.sum() > best_n:
            best, best_n = inl, inl.sum()
    inl_pts = points[best] if best is not None and best_n >= 3 else points
    c = inl_pts.mean(axis=0)
    _, _, vt = np.linalg.svd(inl_pts - c, full_matrices=False)
    nrm = vt[2]
    if nrm[2] > 0:          # camera looks down +z; make the normal face the camera
        nrm = -nrm
    d = -nrm @ c
    rmse = float(np.sqrt(np.mean((inl_pts @ nrm + d) ** 2)))
    return nrm, d, rmse


def _empty(status):
    out = {k: "" for k in MEASURE_FIELDS}
    out["measure_status"] = status
    return out


def measure(depth_mm, mask, K=None, erode_px=2, ring=(3, 12), min_points=40):
    """Measure one vulva. mask: bool/uint8 array the size of depth_mm."""
    K = K or DEFAULT_INTRINSICS
    if depth_mm is None:
        return _empty("no_depth")
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return _empty("empty_mask")
    valid = (depth_mm > MIN_DEPTH_MM) & (depth_mm < MAX_DEPTH_MM)
    m8 = mask.astype(np.uint8)
    inner = cv2.erode(m8, np.ones((3, 3), np.uint8), iterations=erode_px).astype(bool) if erode_px else mask
    if inner.sum() < min_points // 2:
        inner = mask
    outer = cv2.dilate(m8, np.ones((3, 3), np.uint8), iterations=ring[1]).astype(bool)
    near = cv2.dilate(m8, np.ones((3, 3), np.uint8), iterations=ring[0]).astype(bool)
    ring_mask = outer & ~near & valid
    inner_valid = inner & valid
    coverage = float(inner_valid.sum() / max(inner.sum(), 1))
    if inner_valid.sum() < min_points or ring_mask.sum() < min_points:
        out = _empty("too_few_points")
        out["n_points"] = int(inner_valid.sum())
        out["depth_coverage"] = round(coverage, 3)
        return out

    ys, xs = np.nonzero(ring_mask)
    ring_pts = backproject(depth_mm, ys, xs, K)
    nrm, d, rmse = fit_plane(ring_pts)

    # width / length / area from the mask shape mapped onto the skin plane:
    # each mask pixel's ray is intersected with the plane, so edge pixels with
    # bad ("flying") depth cannot distort the outline, and tilt is undone.
    ys_m, xs_m = np.nonzero(mask)
    rays = np.stack([(xs_m - K["cx"]) / K["fx"], (ys_m - K["cy"]) / K["fy"], np.ones(len(xs_m))], axis=1)
    denom = rays @ nrm
    ok = np.abs(denom) > 1e-6
    rays, ys_m, xs_m, denom = rays[ok], ys_m[ok], xs_m[ok], denom[ok]
    s = -d / denom                             # z of the plane along each ray
    on_plane = rays * s[:, None]
    unit = rays / np.linalg.norm(rays, axis=1, keepdims=True)
    cos = np.clip(np.abs(unit @ nrm), 0.2, 1.0)
    foot = (s / K["fx"]) * (s / K["fy"]) / cos     # mm^2 of skin each pixel covers
    area = float(foot.sum())

    c = on_plane.mean(axis=0)
    u_ref = np.array([1.0, 0.0, 0.0]) - nrm[0] * nrm
    u_ref /= np.linalg.norm(u_ref)
    v_ref = np.cross(nrm, u_ref)
    uv = np.stack([(on_plane - c) @ u_ref, (on_plane - c) @ v_ref], axis=1)
    cov = np.cov((uv * np.sqrt(foot / foot.mean())[:, None]).T)
    evals, evecs = np.linalg.eigh(cov)
    major, minor = float(4 * np.sqrt(max(evals[1], 0))), float(4 * np.sqrt(max(evals[0], 0)))   # filled ellipse: axis = 4 sigma
    ey = np.array([u_ref[1], v_ref[1]])        # image-down direction, in plane coordinates
    ey /= max(np.linalg.norm(ey), 1e-9)
    vert = abs(evecs[:, 1] @ ey)
    # the vulva is taller than wide; "length" is the axis closer to image-vertical
    length, width = (major, minor) if vert >= 0.5 else (minor, major)

    # height from the depth itself: eroded mask for the peak, full mask for volume
    vm = mask & valid
    ys_v, xs_v = np.nonzero(vm)
    pts_v = backproject(depth_mm, ys_v, xs_v, K)
    h_v = np.clip(pts_v @ nrm + d, 0, None)
    ys, xs = np.nonzero(inner_valid)
    h_in = np.clip(backproject(depth_mm, ys, xs, K) @ nrm + d, 0, None)
    lut = np.zeros(depth_mm.shape, np.float64)
    lut[ys_m, xs_m] = foot
    volume = float((h_v * lut[ys_v, xs_v]).sum() * mask.sum() / max(vm.sum(), 1))
    z_mean = float(pts_v[:, 2].mean())
    view_angle = float(np.degrees(np.arccos(np.clip(abs(nrm @ np.array([0, 0, -1.0])), 0, 1))))
    peak = float(np.percentile(h_in, 95))
    return {
        "width_mm": round(width, 2),
        "length_mm": round(length, 2),
        "peak_height_mm": round(peak, 2),
        "mean_height_mm": round(float(h_in.mean()), 2),
        "area_mm2": round(area, 1),
        "volume_mm3": round(volume, 1),
        "wlh_mm3": round(width * length * peak, 1),
        "vulva_index": round(width * length / 200.0, 3),
        "n_points": int(inner_valid.sum()),
        "depth_coverage": round(coverage, 3),
        "plane_rmse_mm": round(rmse, 2),
        "view_angle_deg": round(view_angle, 1),
        "distance_mm": round(z_mean, 1),
        "measure_status": "ok",
    }
