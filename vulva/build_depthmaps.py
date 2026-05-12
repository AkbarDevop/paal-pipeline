#!/usr/bin/env python3
"""Batch-generate pig depth maps for all OAK frames under images/.

For each `*_depth_*.raw` file, this script calls point_cloud.py and writes
outputs into:

    depthmap/<timestamp_folder>/<frame_stem>/

It also writes:
  - depthmap/index.csv
  - depthmap/index.json

Example:
    python build_depthmaps.py
    python build_depthmaps.py --force
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
            "Please rerun build_depthmaps.py with the Python that has cv2 installed."
        )

    raise SystemExit(
        "Python dependency check failed before batch processing.\n"
        f"Interpreter: {sys.executable}\n"
        f"{error_text}"
    )


def run_one(depth_raw: Path, out_dir: Path, force: bool, full_outputs: bool, auto_pig_segmentation: bool) -> dict:
    summary_path = summary_path_for(out_dir)
    error_path = out_dir / "error.txt"
    if full_outputs:
        requested_output_mode = "full_outputs"
    elif auto_pig_segmentation:
        requested_output_mode = "point_cloud_only"
    else:
        requested_output_mode = "paper_raw_acquisition"
    source_error = validate_depth_source(depth_raw)
    if source_error:
        return {
            "depth_raw": str(depth_raw),
            "output_dir": str(out_dir),
            "status": "invalid_source",
            "num_vertices": "",
            "num_faces": "",
            "mask_source": "",
            "seconds": 0.0,
            "error": source_error,
        }

    if summary_path.exists() and not force:
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        existing_output_mode = data.get("output_mode", "full_outputs")
        if existing_output_mode == requested_output_mode:
            if requested_output_mode == "point_cloud_only":
                raw_3d_color_image = data.get("raw_3d_color_image", "")
                if not raw_3d_color_image or not Path(raw_3d_color_image).exists():
                    data = {}
                else:
                    rgb_region_image = data.get("rgb_region_image", "")
                    if rgb_region_image and not Path(rgb_region_image).exists():
                        data = {}
            if data:
                return {
                    "depth_raw": str(depth_raw),
                    "output_dir": str(out_dir),
                    "status": "skipped_existing",
                    "num_vertices": data.get("num_vertices", ""),
                    "num_faces": data.get("num_faces", ""),
                    "mask_source": data.get("mask_source", ""),
                    "seconds": 0.0,
                    "error": "",
                }

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(script_dir() / "point_cloud.py"),
        "--depth-raw",
        str(depth_raw),
        "--output-dir",
        str(out_dir),
    ]
    if not full_outputs:
        cmd.append("--point-cloud-only")
        if not auto_pig_segmentation:
            cmd.append("--paper-raw-acquisition")

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
            "num_faces": "",
            "mask_source": "",
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
            "num_faces": "",
            "mask_source": "",
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
        "num_faces": data.get("num_faces", ""),
        "mask_source": data.get("mask_source", ""),
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
        "num_faces",
        "mask_source",
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
    parser = argparse.ArgumentParser(description="Build depth maps for all pigs in images/")
    parser.add_argument("--images-root", default="images", help="Source root with timestamp folders")
    parser.add_argument("--output-root", default="depthmap", help="Where to store batch depth-map outputs")
    parser.add_argument("--force", action="store_true", help="Rebuild outputs even if summary.json already exists")
    parser.add_argument("--full-outputs", action="store_true",
                        help="Keep the old full point-cloud/mesh/render outputs instead of point-cloud-only mode")
    parser.add_argument("--auto-pig-segmentation", action="store_true",
                        help="Use the old heuristic pig-only masking instead of the paper-style raw rear-view acquisition")
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
    if args.full_outputs:
        mode_name = "full_outputs"
    elif args.auto_pig_segmentation:
        mode_name = "point_cloud_only"
    else:
        mode_name = "paper_raw_acquisition"
    print(f"Mode:         {mode_name}")
    print(f"Depth frames: {len(depth_files)}")

    rows = []
    for idx, depth_raw in enumerate(depth_files, 1):
        out_dir = output_dir_for(depth_raw, images_root, output_root)
        rel_depth = depth_raw.relative_to(images_root)
        print(f"[{idx}/{len(depth_files)}] {rel_depth}")
        row = run_one(
            depth_raw,
            out_dir,
            force=args.force,
            full_outputs=args.full_outputs,
            auto_pig_segmentation=args.auto_pig_segmentation,
        )
        rows.append(row)
        if row["status"] == "generated":
            print(f"  generated | vertices={row['num_vertices']} faces={row['num_faces']} | {row['seconds']}s")
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
