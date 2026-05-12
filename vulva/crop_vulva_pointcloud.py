#!/usr/bin/env python3
"""Crop a vulva-centered rectangular point cloud region from a rear-view frame.

This is the next paper-aligned step after the raw rear-vulva-area point cloud:
1. Keep the rear-view acquisition as the base surface
2. Manually choose a vulva-centered rectangular crop in image coordinates
3. Export only the point-cloud points that fall inside that rectangle
4. Save matching RGB/IR rectangular crops for inspection

Examples:
    python crop_vulva_pointcloud.py images\\20260211-09-06-52\\pig0_depth_20260211-09-07-10.raw
    python crop_vulva_pointcloud.py depthmap\\20260211-09-06-52\\pig0_20260211-09-07-10
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from point_cloud import (
    auto_segment_pig,
    build_point_cloud,
    infer_rgb_image_path,
    infer_texture_path,
    infer_vulva_focus_box,
    load_calibration,
    load_depth_raw,
    load_external_mask,
    load_texture_image,
    make_roi_mask,
    summarize,
    write_point_cloud_ply,
)

WINDOW_NAME = "Vulva Rectangle Crop"


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


def load_input_summary(input_path: Path) -> dict:
    if not input_path.is_dir():
        return {}
    summary_path = input_path / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def infer_default_output_dir(input_path: Path, depth_raw_path: Path) -> Path:
    if input_path.is_dir():
        return input_path / "vulva_rect"

    timestamp_dir = depth_raw_path.parent.name
    stem = depth_raw_path.stem.replace("_depth", "")
    return repo_root() / "depthmap_rect" / timestamp_dir / stem


def resolve_output_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == repo_root().name:
        return (repo_root().parent / path).resolve()
    return (repo_root() / path).resolve()


def expand_box(
    box: Tuple[int, int, int, int],
    shape: Tuple[int, int],
    pad_px: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    h, w = shape
    x1 = max(0, x1 - pad_px)
    y1 = max(0, y1 - pad_px)
    x2 = min(w - 1, x2 + pad_px)
    y2 = min(h - 1, y2 + pad_px)
    return x1, y1, x2, y2


def masked_rgb(texture_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = texture_rgb.copy()
    out[~mask] = 0
    return out


def crop_box_image(image_rgb: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    crop = image_rgb[y1:y2 + 1, x1:x2 + 1]
    if crop.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return crop


def save_overlay_image(
    path: Path,
    image_rgb: np.ndarray,
    box: Tuple[int, int, int, int],
    label: str,
) -> None:
    overlay = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    x1, y1, x2, y2 = box
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (80, 220, 80), 2)
    cv2.putText(
        overlay,
        label,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(path), overlay)


def save_rgb_image(path: Path, image_rgb: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def build_rect_mask(shape: Tuple[int, int], box: Tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    x1, y1, x2, y2 = box
    mask[y1:y2 + 1, x1:x2 + 1] = True
    return mask


def normalize_box(start: Tuple[int, int], end: Tuple[int, int], shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    h, w = shape
    x1 = max(0, min(start[0], end[0]))
    y1 = max(0, min(start[1], end[1]))
    x2 = min(w - 1, max(start[0], end[0]))
    y2 = min(h - 1, max(start[1], end[1]))
    return x1, y1, x2, y2


def draw_box_overlay(
    image_rgb: np.ndarray,
    selected_box: Optional[Tuple[int, int, int, int]],
    draft_box: Optional[Tuple[int, int, int, int]],
    suggested_method: str,
    footer: str = "",
) -> np.ndarray:
    vis = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if selected_box is not None:
        x1, y1, x2, y2 = selected_box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (80, 220, 80), 2)
        cv2.putText(vis, "Selected box", (x1 + 6, max(22, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 220, 80), 2, cv2.LINE_AA)
    if draft_box is not None:
        x1, y1, x2, y2 = draft_box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 255), 1)

    cv2.putText(vis, "Paper-style step: manually choose the vulva-centered rectangle on the raw rear-view acquisition", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, "Drag left mouse to draw or replace the rectangle", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, f"Enter=accept  R=reset  Q=cancel  Auto suggestion source={suggested_method}", (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    if footer:
        cv2.putText(vis, footer, (12, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 180, 255), 2, cv2.LINE_AA)
    return vis


class RectangleSelector:
    def __init__(
        self,
        image_rgb: np.ndarray,
        initial_box: Optional[Tuple[int, int, int, int]],
        suggested_method: str,
    ):
        self.image_rgb = image_rgb
        self.selected_box = initial_box
        self.suggested_method = suggested_method
        self.drag_start: Optional[Tuple[int, int]] = None
        self.drag_current: Optional[Tuple[int, int]] = None
        self.cancelled = False

    def current_draft_box(self) -> Optional[Tuple[int, int, int, int]]:
        if self.drag_start is None or self.drag_current is None:
            return None
        return normalize_box(self.drag_start, self.drag_current, self.image_rgb.shape[:2])

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.drag_current = (x, y)
            self.show()
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            self.drag_current = (x, y)
            self.show()
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            self.drag_current = (x, y)
            box = self.current_draft_box()
            self.drag_start = None
            self.drag_current = None
            if box is not None and (box[2] - box[0] >= 10) and (box[3] - box[1] >= 10):
                self.selected_box = box
            self.show()

    def show(self, footer: str = ""):
        vis = draw_box_overlay(
            self.image_rgb,
            self.selected_box,
            self.current_draft_box(),
            self.suggested_method,
            footer=footer,
        )
        cv2.imshow(WINDOW_NAME, vis)

    def run(self) -> Optional[Tuple[int, int, int, int]]:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback)
        self.show()
        while True:
            key = cv2.waitKey(30) & 0xFF
            if key == ord("r"):
                self.selected_box = None
                self.drag_start = None
                self.drag_current = None
                self.show()
            elif key == ord("q"):
                self.cancelled = True
                break
            elif key == 13:
                if self.selected_box is not None:
                    break
                self.show("Draw a rectangle before pressing Enter")

        cv2.setMouseCallback(WINDOW_NAME, lambda *args: None)
        cv2.destroyWindow(WINDOW_NAME)
        if self.cancelled:
            return None
        return self.selected_box


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

    return depth_mm, intrinsics, texture_path, texture_gray, color_rgb, color_input, rgb_path or ""


def build_context_mask(args, depth_mm: np.ndarray, input_summary: dict) -> Tuple[np.ndarray, str, dict]:
    roi_mask = make_roi_mask(depth_mm.shape, use_crop=not args.no_crop)
    valid_depth = (depth_mm >= args.min_depth) & (depth_mm <= args.max_depth)

    if args.mask:
        mask_path = resolve_existing_path(args.mask)
        mask = load_external_mask(str(mask_path), depth_mm.shape) & roi_mask & valid_depth
        return mask, "external_mask", {"mask_pixels": int(mask.sum())}

    summary_mask_path = resolve_existing_path(input_summary.get("pig_mask_manual"))
    if summary_mask_path and summary_mask_path.exists():
        mask = load_external_mask(str(summary_mask_path), depth_mm.shape) & valid_depth
        return mask, "manual_pig_mask_from_summary", {"mask_pixels": int(mask.sum())}

    if args.keep_all_valid or input_summary.get("output_mode") == "paper_raw_acquisition" or not args.auto_pig_mask:
        mask = roi_mask & valid_depth
        return mask, "paper_raw_valid_depth", {
            "mask_pixels": int(mask.sum()),
            "roi_valid_pixels": int((roi_mask & valid_depth).sum()),
        }

    mask, auto_stats = auto_segment_pig(
        depth_mm=depth_mm,
        roi_mask=roi_mask,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        foreground_margin_mm=args.foreground_margin_mm,
        min_pixels=args.min_pixels,
    )
    mask = mask & valid_depth
    return mask, "auto_pig_depth_segment", auto_stats


def main(args):
    input_path = resolve_input_path(args.input_path)
    input_summary = load_input_summary(input_path)
    depth_raw_path = derive_depth_raw_path(input_path)
    output_dir = resolve_output_path(args.output_dir) if args.output_dir else infer_default_output_dir(input_path, depth_raw_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    depth_mm, intrinsics, texture_path, texture_gray, color_rgb, color_input, rgb_input = load_inputs(args, depth_raw_path)
    context_mask, mask_source, auto_stats = build_context_mask(args, depth_mm, input_summary)
    if not context_mask.any():
        raise ValueError("Rear-view acquisition mask is empty after filtering")

    pig_rgb_masked = masked_rgb(color_rgb, context_mask)
    ir_rgb = cv2.cvtColor(texture_gray, cv2.COLOR_GRAY2RGB)
    pig_ir_masked = masked_rgb(ir_rgb, context_mask)

    # Use IR for rectangle selection — IR and depth are from the same sensor
    # so pixel coordinates are perfectly aligned.  RGB is from a separate
    # camera with ~1.73 cm baseline and no extrinsic calibration available,
    # so drawing on RGB would give misaligned depth coordinates.
    selector_image = pig_ir_masked

    suggested_box, suggested_method = infer_vulva_focus_box(texture_gray, context_mask)
    suggested_box = expand_box(suggested_box, depth_mm.shape, args.box_pad_px)
    if args.auto_box:
        focus_box = suggested_box
        focus_method = suggested_method
        box_selection_mode = "auto_box"
    else:
        selector = RectangleSelector(
            image_rgb=selector_image,
            initial_box=suggested_box,
            suggested_method=suggested_method,
        )
        selected_box = selector.run()
        if selected_box is None:
            raise SystemExit("Rectangle selection cancelled")
        focus_box = selected_box
        focus_method = "manual_rectangle"
        box_selection_mode = "manual_box"

    rect_mask = build_rect_mask(depth_mm.shape, focus_box)
    crop_mask = context_mask & rect_mask
    if not crop_mask.any():
        raise ValueError("Rectangular vulva crop does not contain any valid rear-view points")

    # Use IR for point cloud texture — IR is pixel-aligned with depth
    crop_vertices, crop_colors = build_point_cloud(
        depth_mm=depth_mm,
        texture_rgb=ir_rgb,
        mask=crop_mask,
        intrinsics=intrinsics,
        stride=max(1, args.point_stride),
    )

    crop_rgb_rect = crop_box_image(pig_rgb_masked, focus_box)
    crop_ir_rect = crop_box_image(pig_ir_masked, focus_box)

    ply_path = output_dir / "cropped_vulva_point_cloud.ply"
    raw_overlay_path = output_dir / "raw_3d_color_with_rect.png"
    rgb_crop_path = output_dir / "cropped_vulva_rgb_rect.png"
    ir_crop_path = output_dir / "cropped_vulva_ir_rect.png"
    summary_path = output_dir / "summary.json"

    write_point_cloud_ply(str(ply_path), crop_vertices, crop_colors)
    save_overlay_image(raw_overlay_path, pig_ir_masked, focus_box, f"Vulva rectangle ({focus_method})")
    save_rgb_image(rgb_crop_path, crop_rgb_rect)
    save_rgb_image(ir_crop_path, crop_ir_rect)

    summary = summarize(
        crop_vertices,
        np.empty((0, 3), dtype=np.int32),
        auto_stats,
        intrinsics,
        mask_source,
    )
    summary.update(
        {
            "input_path": str(input_path),
            "depth_raw": str(depth_raw_path),
            "texture_input": texture_path,
            "rgb_image_input": color_input,
            "requested_rgb_input": rgb_input,
            "output_dir": str(output_dir),
            "output_mode": "vulva_rect_crop",
            "point_stride": args.point_stride,
            "context_mask_mode": mask_source,
            "box_selection_mode": box_selection_mode,
            "focus_box": [int(v) for v in focus_box],
            "focus_method": focus_method,
            "suggested_focus_box": [int(v) for v in suggested_box],
            "suggested_focus_method": suggested_method,
            "box_pad_px": args.box_pad_px,
            "crop_pixels": int(crop_mask.sum()),
            "cropped_vulva_point_cloud": str(ply_path),
            "raw_3d_color_with_rect": str(raw_overlay_path),
            "cropped_vulva_rgb_rect": str(rgb_crop_path),
            "cropped_vulva_ir_rect": str(ir_crop_path),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Input:       {input_path}")
    print(f"Depth raw:   {depth_raw_path}")
    print(f"Mask source: {mask_source}")
    print(f"Focus box:   {focus_box} ({focus_method})")
    print(f"Vertices:    {len(crop_vertices):,}")
    print(f"Output dir:  {output_dir}")
    print(f"  Point cloud: {ply_path}")
    print(f"  Raw RGB:     {raw_overlay_path}")
    print(f"  RGB crop:    {rgb_crop_path}")
    print(f"  IR crop:     {ir_crop_path}")
    print(f"  Summary:     {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop a vulva-centered rectangular point cloud region")
    parser.add_argument("input_path", help="Depth raw path or an existing depthmap frame directory with summary.json")
    parser.add_argument("--output-dir", default=None, help="Output folder for the cropped vulva rectangle assets")
    parser.add_argument("--texture-img", "--ir-img", dest="texture_img", default=None,
                        help="Optional IR/texture image override for focus-box inference")
    parser.add_argument("--rgb-img", default=None, help="Optional RGB image override for color outputs")
    parser.add_argument("--calibration", default=None, help="Optional path to calibration.json")
    parser.add_argument("--mask", default=None, help="Optional binary pig mask in depth-frame coordinates")
    parser.add_argument("--min-depth", type=int, default=200, help="Minimum valid depth in mm")
    parser.add_argument("--max-depth", type=int, default=5000, help="Maximum valid depth in mm")
    parser.add_argument("--foreground-margin-mm", type=int, default=140,
                        help="Foreground margin below wall depth for auto pig segmentation")
    parser.add_argument("--min-pixels", type=int, default=4000, help="Minimum pig-mask size for auto segmentation")
    parser.add_argument("--point-stride", type=int, default=2, help="Pixel stride for the cropped point cloud")
    parser.add_argument("--box-pad-px", type=int, default=0, help="Extra padding to add around the inferred rectangle")
    parser.add_argument("--auto-box", action="store_true",
                        help="Use the heuristic IR-based rectangle directly instead of paper-style manual selection")
    parser.add_argument("--auto-pig-mask", action="store_true",
                        help="Use the old heuristic pig-only masking before rectangle selection")
    parser.add_argument("--no-crop", action="store_true", help="Disable the fixed stall ROI crop before segmentation")
    parser.add_argument("--keep-all-valid", action="store_true",
                        help="Skip pig auto-segmentation and keep all valid depth pixels in the ROI")
    main(parser.parse_args())
