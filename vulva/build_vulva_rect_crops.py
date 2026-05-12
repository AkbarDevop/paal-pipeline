#!/usr/bin/env python3
"""Batch-generate vulva-centered rectangular crops for all OAK depth frames.

For each `*_depth_*.raw` file under images/, this script calls
crop_vulva_pointcloud.py and writes outputs into:

    depthmap_rect/<timestamp_folder>/<frame_stem>/

It also writes:
  - depthmap_rect/index.csv
  - depthmap_rect/index.json
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


EXPECTED_DEPTH_RAW_SIZES = {640 * 480 * 2, 640 * 480 * 2 + 8}


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def resolve_path_arg(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return script_dir() / path


def find_depth_frames(images_root: Path):
    return sorted(images_root.rglob("*_depth_*.raw"))


def output_dir_for(depth_raw: Path, images_root: Path, output_root: Path) -> Path:
    rel_parent = depth_raw.parent.relative_to(images_root)
    stem = depth_raw.stem.replace("_depth", "")
    return output_root / rel_parent / stem


def summary_path_for(out_dir: Path) -> Path:
    return out_dir / "summary.json"


def validate_depth_source(depth_raw: Path) -> str:
    if not depth_raw.exists():
        return "missing source file"
    size = depth_raw.stat().st_size
    if size not in EXPECTED_DEPTH_RAW_SIZES:
        return f"invalid depth raw size: {size} bytes"
    return ""


def check_runtime_dependencies() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import cv2, numpy"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(script_dir()),
    )
    if result.returncode == 0:
        return

    error_text = (result.stderr or result.stdout).strip()
    if "No module named 'cv2'" in error_text:
        raise SystemExit(
            "OpenCV is missing for this Python interpreter.\n"
            f"Interpreter: {sys.executable}\n"
            "Please rerun build_vulva_rect_crops.py with the Python that has cv2 installed."
        )

    raise SystemExit(
        "Python dependency check failed before batch processing.\n"
        f"Interpreter: {sys.executable}\n"
        f"{error_text}"
    )


def summary_has_required_outputs(data: dict) -> bool:
    required = [
        data.get("cropped_vulva_point_cloud", ""),
        data.get("raw_3d_color_with_rect", ""),
        data.get("cropped_vulva_rgb_rect", ""),
        data.get("cropped_vulva_ir_rect", ""),
    ]
    return all(path and Path(path).exists() for path in required)


def build_command(depth_raw: Path, out_dir: Path, args) -> list[str]:
    cmd = [
        sys.executable,
        str(script_dir() / "crop_vulva_pointcloud.py"),
        str(depth_raw),
        "--output-dir",
        str(out_dir),
        "--min-depth",
        str(args.min_depth),
        "--max-depth",
        str(args.max_depth),
        "--foreground-margin-mm",
        str(args.foreground_margin_mm),
        "--min-pixels",
        str(args.min_pixels),
        "--point-stride",
        str(args.point_stride),
        "--box-pad-px",
        str(args.box_pad_px),
    ]
    if args.no_crop:
        cmd.append("--no-crop")
    if args.keep_all_valid:
        cmd.append("--keep-all-valid")
    if args.auto_box:
        cmd.append("--auto-box")
    if args.auto_pig_mask:
        cmd.append("--auto-pig-mask")
    return cmd


def run_one(depth_raw: Path, out_dir: Path, args) -> dict:
    summary_path = summary_path_for(out_dir)
    error_path = out_dir / "error.txt"
    source_error = validate_depth_source(depth_raw)
    if source_error:
        return {
            "depth_raw": str(depth_raw),
            "output_dir": str(out_dir),
            "status": "invalid_source",
            "num_vertices": "",
            "focus_method": "",
            "crop_pixels": "",
            "seconds": 0.0,
            "error": source_error,
        }

    if summary_path.exists() and not args.force:
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        requested_box_mode = "auto_box" if args.auto_box else "manual_box"
        requested_context_mode = "auto_pig_depth_segment" if args.auto_pig_mask else "paper_raw_valid_depth"
        if (
            data.get("output_mode") == "vulva_rect_crop"
            and data.get("box_selection_mode") == requested_box_mode
            and data.get("context_mask_mode") == requested_context_mode
            and summary_has_required_outputs(data)
        ):
            return {
                "depth_raw": str(depth_raw),
                "output_dir": str(out_dir),
                "status": "skipped_existing",
                "num_vertices": data.get("num_vertices", ""),
                "focus_method": data.get("focus_method", ""),
                "crop_pixels": data.get("crop_pixels", ""),
                "seconds": 0.0,
                "error": "",
            }

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_command(depth_raw, out_dir, args)

    t0 = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(script_dir()),
    )
    dt = round(time.time() - t0, 2)

    if result.returncode != 0:
        error_text = (result.stderr or result.stdout).strip()[:2000]
        error_path.write_text(error_text, encoding="utf-8")
        return {
            "depth_raw": str(depth_raw),
            "output_dir": str(out_dir),
            "status": "failed",
            "num_vertices": "",
            "focus_method": "",
            "crop_pixels": "",
            "seconds": dt,
            "error": error_text,
        }

    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        error_text = f"summary read failed: {exc}"
        error_path.write_text(error_text, encoding="utf-8")
        return {
            "depth_raw": str(depth_raw),
            "output_dir": str(out_dir),
            "status": "failed_summary",
            "num_vertices": "",
            "focus_method": "",
            "crop_pixels": "",
            "seconds": dt,
            "error": error_text,
        }

    if error_path.exists():
        error_path.unlink()

    return {
        "depth_raw": str(depth_raw),
        "output_dir": str(out_dir),
        "status": "generated",
        "num_vertices": data.get("num_vertices", ""),
        "focus_method": data.get("focus_method", ""),
        "crop_pixels": data.get("crop_pixels", ""),
        "seconds": dt,
        "error": "",
    }


def write_index(output_root: Path, rows: list[dict]) -> None:
    csv_path = output_root / "index.csv"
    json_path = output_root / "index.json"

    fields = [
        "depth_raw",
        "output_dir",
        "status",
        "num_vertices",
        "focus_method",
        "crop_pixels",
        "seconds",
        "error",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "total": len(rows),
        "generated": sum(1 for row in rows if row["status"] == "generated"),
        "skipped_existing": sum(1 for row in rows if row["status"] == "skipped_existing"),
        "invalid_source": sum(1 for row in rows if row["status"] == "invalid_source"),
        "failed": sum(1 for row in rows if row["status"] not in {"generated", "skipped_existing", "invalid_source"}),
        "rows": rows,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build vulva-centered rectangular point-cloud crops for all pigs in images/")
    parser.add_argument("--images-root", default="images", help="Source root with timestamp folders")
    parser.add_argument("--output-root", default="depthmap_rect", help="Where to store rectangular vulva-crop outputs")
    parser.add_argument("--force", action="store_true", help="Rebuild outputs even if summary.json already exists")
    parser.add_argument("--min-depth", type=int, default=200, help="Minimum valid depth in mm")
    parser.add_argument("--max-depth", type=int, default=5000, help="Maximum valid depth in mm")
    parser.add_argument("--foreground-margin-mm", type=int, default=140,
                        help="Foreground margin below wall depth for auto pig segmentation")
    parser.add_argument("--min-pixels", type=int, default=4000, help="Minimum pig-mask size for auto segmentation")
    parser.add_argument("--point-stride", type=int, default=2, help="Pixel stride for the cropped point cloud")
    parser.add_argument("--box-pad-px", type=int, default=0, help="Extra padding to add around the inferred rectangle")
    parser.add_argument("--auto-box", action="store_true",
                        help="Use heuristic boxes instead of the paper-style manual rectangle selection")
    parser.add_argument("--auto-pig-mask", action="store_true",
                        help="Use the old heuristic pig-only masking before rectangle selection")
    parser.add_argument("--no-crop", action="store_true", help="Disable the fixed stall ROI crop before segmentation")
    parser.add_argument("--keep-all-valid", action="store_true",
                        help="Skip pig auto-segmentation and keep all valid depth pixels in the ROI")
    args = parser.parse_args()

    images_root = resolve_path_arg(args.images_root)
    output_root = resolve_path_arg(args.output_root)
    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found: {images_root}")

    check_runtime_dependencies()
    depth_files = find_depth_frames(images_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Python:       {sys.executable}")
    print(f"Images root:  {images_root}")
    print(f"Output root:  {output_root}")
    print(f"Box mode:     {'auto_box' if args.auto_box else 'manual_box'}")
    print(f"Context mode: {'auto_pig_depth_segment' if args.auto_pig_mask else 'paper_raw_valid_depth'}")
    print(f"Depth frames: {len(depth_files)}")

    rows = []
    for idx, depth_raw in enumerate(depth_files, 1):
        out_dir = output_dir_for(depth_raw, images_root, output_root)
        rel_depth = depth_raw.relative_to(images_root)
        print(f"[{idx}/{len(depth_files)}] {rel_depth}")
        row = run_one(depth_raw, out_dir, args)
        rows.append(row)
        if row["status"] == "generated":
            print(
                f"  generated | vertices={row['num_vertices']} "
                f"focus={row['focus_method']} crop_pixels={row['crop_pixels']} | {row['seconds']}s"
            )
        elif row["status"] == "skipped_existing":
            print("  skipped_existing")
        elif row["status"] == "invalid_source":
            print(f"  invalid_source | {row['error']}")
        else:
            print(f"  failed | {row['error'][:160]}")

    write_index(output_root, rows)

    generated = sum(1 for row in rows if row["status"] == "generated")
    skipped = sum(1 for row in rows if row["status"] == "skipped_existing")
    invalid = sum(1 for row in rows if row["status"] == "invalid_source")
    failed = sum(1 for row in rows if row["status"] not in {"generated", "skipped_existing", "invalid_source"})
    print()
    print(f"Done. generated={generated} skipped_existing={skipped} invalid_source={invalid} failed={failed}")
    print(f"Index: {output_root / 'index.csv'}")


if __name__ == "__main__":
    main()
