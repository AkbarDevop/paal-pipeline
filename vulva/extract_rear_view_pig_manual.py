#!/usr/bin/env python3
"""Manual rear-view pig-only extraction for paper-style raw 3D color image data.

This tool replaces the heuristic pig masking stage with a manual silhouette
selection that is closer to the paper's manual processing workflow.

Outputs:
  - pig_point_cloud.ply
  - raw_3d_color_image.png
  - pig_mask_manual.png
  - manual_polygon_overlay.png
  - summary.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import label_vulva as label_vulva_module
from label_vulva import PolygonDrawer, make_mask, scale_for_display
from point_cloud import (
    build_point_cloud,
    infer_rgb_image_path,
    infer_texture_path,
    load_calibration,
    load_depth_raw,
    load_external_mask,
    load_texture_image,
    summarize,
    write_point_cloud_ply,
)


WINDOW_NAME = "Rear View Pig Extraction"


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def resolve_existing_path(path_str: str | None) -> Optional[Path]:
    if not path_str:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == repo_root().name:
        candidate = repo_root().parent / path
        if candidate.exists():
            return candidate.resolve()

    candidates = [
        Path.cwd() / path,
        repo_root() / path,
        repo_root().parent / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return (repo_root() / path).resolve()


def resolve_input_path(input_path_str: str) -> Path:
    resolved = resolve_existing_path(input_path_str)
    if resolved is None:
        raise FileNotFoundError(f"Input path not found: {input_path_str}")
    return resolved


def resolve_output_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == repo_root().name:
        return (repo_root().parent / path).resolve()
    return (repo_root() / path).resolve()


def derive_depth_raw_path(input_path: Path) -> Path:
    if input_path.is_dir():
        summary_path = input_path / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"summary.json not found in {input_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        depth_raw = resolve_existing_path(summary.get("depth_raw"))
        if depth_raw is None or not depth_raw.exists():
            raise FileNotFoundError(f"Depth raw referenced in summary does not exist: {summary.get('depth_raw')}")
        return depth_raw

    if input_path.suffix.lower() == ".raw":
        if not input_path.exists():
            raise FileNotFoundError(f"Depth raw not found: {input_path}")
        return input_path

    raise ValueError(f"Unsupported input: {input_path}")


def infer_default_output_dir(depth_raw_path: Path) -> Path:
    timestamp_dir = depth_raw_path.parent.name
    stem = depth_raw_path.stem.replace("_depth", "")
    return repo_root() / "rear_view_pig_manual" / timestamp_dir / stem


def load_inputs(args, depth_raw_path: Path):
    depth_mm = load_depth_raw(str(depth_raw_path))
    calibration_path = str(resolve_existing_path(args.calibration)) if args.calibration else None
    intrinsics = load_calibration(calibration_path, str(depth_raw_path.parent))

    texture_override = str(resolve_existing_path(args.texture_img)) if args.texture_img else None
    texture_path = infer_texture_path(argparse.Namespace(texture_img=texture_override), str(depth_raw_path))
    if not texture_path:
        raise FileNotFoundError("Could not infer IR/texture image next to the depth frame")
    texture_rgb, texture_gray = load_texture_image(texture_path, depth_mm.shape)

    rgb_override = str(resolve_existing_path(args.rgb_img)) if args.rgb_img else None
    rgb_path = rgb_override or infer_rgb_image_path(str(depth_raw_path))
    if rgb_path and os.path.exists(rgb_path):
        color_rgb, _ = load_texture_image(rgb_path, depth_mm.shape)
        color_input = rgb_path
    else:
        color_rgb = texture_rgb
        color_input = texture_path

    return depth_mm, intrinsics, texture_path, texture_gray, color_rgb, color_input


def draw_header(img_bgr: np.ndarray) -> np.ndarray:
    vis = img_bgr.copy()
    cv2.putText(vis, "Paper-style step: manually outline the rear-view pig region", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, "Left-click=add point  Right-click=undo  Enter=finish  R=reset  Q=cancel", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def run_manual_polygon(color_rgb: np.ndarray):
    label_vulva_module.WINDOW_NAME = WINDOW_NAME
    base_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
    display_bgr, scale = scale_for_display(draw_header(base_bgr), max_h=760, max_w=1100)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.imshow(WINDOW_NAME, display_bgr)
    drawer = PolygonDrawer(display_bgr)
    points = drawer.run()
    cv2.destroyWindow(WINDOW_NAME)
    if points is None:
        return None, None

    original_points = []
    for x, y in points:
        ox = int(round(x / max(scale, 1e-6)))
        oy = int(round(y / max(scale, 1e-6)))
        ox = min(max(ox, 0), color_rgb.shape[1] - 1)
        oy = min(max(oy, 0), color_rgb.shape[0] - 1)
        original_points.append((ox, oy))
    return original_points, scale


def masked_rgb(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = image_rgb.copy()
    out[~mask] = 0
    return out


def save_rgb_image(path: Path, image_rgb: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def save_overlay(path: Path, image_rgb: np.ndarray, polygon_points: list[tuple[int, int]]) -> None:
    overlay = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if len(polygon_points) >= 3:
        pts = np.array(polygon_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts], isClosed=True, color=(80, 220, 80), thickness=2)
    save_rgb_image(path, cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))


def main(args):
    input_path = resolve_input_path(args.input_path)
    depth_raw_path = derive_depth_raw_path(input_path)
    output_dir = resolve_output_path(args.output_dir) if args.output_dir else infer_default_output_dir(depth_raw_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    depth_mm, intrinsics, texture_path, texture_gray, color_rgb, color_input = load_inputs(args, depth_raw_path)
    valid_depth = (depth_mm >= args.min_depth) & (depth_mm <= args.max_depth)

    if args.mask:
        mask_path = resolve_existing_path(args.mask)
        polygon_mask = load_external_mask(str(mask_path), depth_mm.shape)
        polygon_points = []
        mask_source = "external_mask"
    else:
        polygon_points, _ = run_manual_polygon(color_rgb)
        if polygon_points is None:
            raise SystemExit("Rear-view pig extraction cancelled")
        polygon_mask = make_mask(polygon_points, depth_mm.shape) > 0
        mask_source = "manual_polygon"

    pig_mask = polygon_mask & valid_depth
    if not pig_mask.any():
        raise ValueError("Manual pig mask contains no valid depth pixels")

    pig_rgb = masked_rgb(color_rgb, pig_mask)
    vertices, colors = build_point_cloud(
        depth_mm=depth_mm,
        texture_rgb=color_rgb,
        mask=pig_mask,
        intrinsics=intrinsics,
        stride=max(1, args.point_stride),
    )

    ply_path = output_dir / "pig_point_cloud.ply"
    raw_rgb_path = output_dir / "raw_3d_color_image.png"
    mask_path_out = output_dir / "pig_mask_manual.png"
    overlay_path = output_dir / "manual_polygon_overlay.png"
    summary_path = output_dir / "summary.json"

    write_point_cloud_ply(str(ply_path), vertices, colors)
    save_rgb_image(raw_rgb_path, pig_rgb)
    cv2.imwrite(str(mask_path_out), (pig_mask.astype(np.uint8) * 255))
    if polygon_points:
        save_overlay(overlay_path, color_rgb, polygon_points)

    summary = summarize(
        vertices,
        np.empty((0, 3), dtype=np.int32),
        {
            "mask_pixels": int(pig_mask.sum()),
            "roi_valid_pixels": int(valid_depth.sum()),
        },
        intrinsics,
        mask_source,
    )
    summary.update(
        {
            "input_path": str(input_path),
            "depth_raw": str(depth_raw_path),
            "texture_input": texture_path,
            "rgb_image_input": color_input,
            "output_dir": str(output_dir),
            "output_mode": "rear_view_pig_manual",
            "point_stride": args.point_stride,
            "pig_mask_manual": str(mask_path_out),
            "raw_3d_color_image": str(raw_rgb_path),
            "manual_polygon_overlay": str(overlay_path) if polygon_points else "",
            "manual_polygon_points": polygon_points,
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Input:       {input_path}")
    print(f"Depth raw:   {depth_raw_path}")
    print(f"Mask source: {mask_source}")
    print(f"Vertices:    {len(vertices):,}")
    print(f"Output dir:  {output_dir}")
    print(f"  Point cloud: {ply_path}")
    print(f"  Raw 3D RGB:  {raw_rgb_path}")
    print(f"  Pig mask:    {mask_path_out}")
    if polygon_points:
        print(f"  Overlay:     {overlay_path}")
    print(f"  Summary:     {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manually extract the rear-view pig-only point cloud")
    parser.add_argument("input_path", help="Depth raw path or a frame directory with summary.json")
    parser.add_argument("--output-dir", default=None, help="Output folder for the manual rear-view pig extraction")
    parser.add_argument("--texture-img", "--ir-img", dest="texture_img", default=None,
                        help="Optional IR/texture image override")
    parser.add_argument("--rgb-img", default=None, help="Optional RGB image override")
    parser.add_argument("--calibration", default=None, help="Optional path to calibration.json")
    parser.add_argument("--mask", default=None, help="Optional pre-made binary mask to bypass the manual polygon step")
    parser.add_argument("--min-depth", type=int, default=200, help="Minimum valid depth in mm")
    parser.add_argument("--max-depth", type=int, default=5000, help="Maximum valid depth in mm")
    parser.add_argument("--point-stride", type=int, default=2, help="Pixel stride for the exported point cloud")
    main(parser.parse_args())
