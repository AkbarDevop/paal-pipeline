#!/usr/bin/env python3
"""Detect which physical pig is missing in folders with < 20 pigs.

Strategy:
  1. Find folders with != 20 pigs
  2. For each, find the nearest normal (20-pig) folder
  3. Compare depth fingerprints (median depth per pig) between them
  4. If abnormal pig N's depth matches normal pig N+1, physical pig N was skipped
  5. Generate a visual side-by-side comparison grid for manual verification

Usage:
    python detect_missing_pigs.py "C:\\PAAL_data\\Fed_pig\\Pictures_OAk"
    python detect_missing_pigs.py /path/to/data --visual   # also save comparison images
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict

import cv2
import numpy as np

from config import CROP_TOF, OUTPUT_DIR

# ── Constants ────────────────────────────────────────────────────────────────

TOF_W, TOF_H = 640, 480
BAR_MARGIN = 50
FOLDER_RE = re.compile(r"^\d{8}-\d{2}-\d{2}-\d{2}$")
FILENAME_RE = re.compile(
    r"^pig(\d+)_(depth_vis|depth|ir_vis|ir|rgb_aligned|rgb)_"
    r"(\d{8}-\d{2}-\d{2}-\d{2})(_cropped)?\.(jpg|raw)$"
)


# ── Depth fingerprinting ────────────────────────────────────────────────────

def load_depth_raw(path):
    """Load a raw 16-bit depth file."""
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
    if raw.size != TOF_W * TOF_H:
        return None
    return raw.reshape((TOF_H, TOF_W))


def depth_fingerprint(depth_map):
    """Compute a depth fingerprint from the cropped stall region.

    Returns (median_depth, std_depth) of valid pixels.
    """
    if depth_map is None:
        return (0.0, 0.0)

    x1, y1, x2, y2 = CROP_TOF
    crop = depth_map[y1:y2, x1:x2].copy()
    crop[:, :BAR_MARGIN] = 0
    crop[:, -BAR_MARGIN:] = 0

    valid = crop[(crop > 200) & (crop < 5000)]
    if len(valid) == 0:
        return (0.0, 0.0)
    return (float(np.median(valid)), float(np.std(valid)))


# ── Folder scanning ─────────────────────────────────────────────────────────

def scan_folder_pigs(folder_path):
    """Scan a folder and return {pig_id: {modality: filepath}} mapping."""
    pigs = defaultdict(dict)
    for fname in os.listdir(folder_path):
        m = FILENAME_RE.match(fname)
        if not m or m.group(4) is not None:  # skip cropped
            continue
        pid = int(m.group(1))
        modality = m.group(2)
        ext = m.group(5)
        key = f"{modality}_{ext}"
        pigs[pid][key] = os.path.join(folder_path, fname)
    return dict(pigs)


def get_all_folders(data_dir):
    """Return sorted list of timestamp folder names."""
    return sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and FOLDER_RE.match(d)
    )


def classify_folders(data_dir, folders):
    """Classify folders into normal (20 pigs), under, and over."""
    normal = []
    under = []
    over = []

    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        pigs = scan_folder_pigs(folder_path)
        pig_ids = sorted(pigs.keys())
        standard_ids = [p for p in pig_ids if p < 20]

        if len(standard_ids) == 20:
            normal.append((folder, pig_ids))
        elif len(standard_ids) < 20:
            under.append((folder, pig_ids))
        if any(p >= 20 for p in pig_ids):
            over.append((folder, pig_ids))

    return normal, under, over


def find_nearest_normal(target_folder, normal_folders):
    """Find the normal folder closest in time (by name sorting)."""
    if not normal_folders:
        return None
    # Folders are named by timestamp, so string comparison = time comparison
    best = min(normal_folders, key=lambda nf: abs(ord(nf[0][0]) - ord(target_folder[0])) if nf[0] != target_folder else float('inf'))
    # Actually just find closest by string distance
    best = None
    best_dist = float('inf')
    for nf_name, nf_pigs in normal_folders:
        # Simple: compare folder name strings (they're timestamp-based)
        dist = abs(hash(nf_name) - hash(target_folder))
        # Better: just find the nearest by sorted position
        if nf_name < target_folder:
            d = len(target_folder)  # placeholder
        else:
            d = len(target_folder)

    # Simplest approach: binary search by name
    names = [nf[0] for nf in normal_folders]
    import bisect
    idx = bisect.bisect_left(names, target_folder)

    candidates = []
    if idx > 0:
        candidates.append(normal_folders[idx - 1])
    if idx < len(normal_folders):
        candidates.append(normal_folders[idx])

    if not candidates:
        return normal_folders[0]

    return min(candidates, key=lambda nf: abs(
        int(nf[0].replace("-", "")) - int(target_folder.replace("-", ""))
    ))


# ── Comparison logic ─────────────────────────────────────────────────────────

def compare_folders(data_dir, abnormal_folder, abnormal_pigs, normal_folder, normal_pigs):
    """Compare depth fingerprints between abnormal and normal folders.

    Returns a list of findings.
    """
    abn_path = os.path.join(data_dir, abnormal_folder)
    norm_path = os.path.join(data_dir, normal_folder)

    abn_data = scan_folder_pigs(abn_path)
    norm_data = scan_folder_pigs(norm_path)

    # Get depth fingerprints for each pig
    abn_prints = {}
    for pid in sorted(abn_data.keys()):
        if pid >= 20:
            continue
        depth_path = abn_data[pid].get("depth_raw")
        abn_prints[pid] = depth_fingerprint(load_depth_raw(depth_path))

    norm_prints = {}
    for pid in sorted(norm_data.keys()):
        if pid >= 20:
            continue
        depth_path = norm_data[pid].get("depth_raw")
        norm_prints[pid] = depth_fingerprint(load_depth_raw(depth_path))

    # Find which pig is missing by comparing depth profiles
    # If abnormal has 19 pigs (0-18), one physical pig was skipped
    # Compare: does abn[0] match norm[0] or norm[1]?
    abn_ids = sorted(p for p in abn_data.keys() if p < 20)
    norm_ids = sorted(p for p in norm_data.keys() if p < 20)

    num_missing = len(norm_ids) - len(abn_ids)
    if num_missing <= 0:
        return []

    # Try all possible missing positions and find best match
    best_offset_score = float('inf')
    best_missing_pigs = []

    # For 1 missing pig: try each possible position 0-19
    if num_missing == 1:
        for missing_pos in range(20):
            # Build mapping: cam_id -> physical_id
            score = 0
            count = 0
            for cam_id in abn_ids:
                physical_id = cam_id if cam_id < missing_pos else cam_id + 1
                if cam_id in abn_prints and physical_id in norm_prints:
                    abn_med, _ = abn_prints[cam_id]
                    norm_med, _ = norm_prints[physical_id]
                    if abn_med > 0 and norm_med > 0:
                        score += abs(abn_med - norm_med)
                        count += 1

            if count > 0:
                avg_score = score / count
                if avg_score < best_offset_score:
                    best_offset_score = avg_score
                    best_missing_pigs = [missing_pos]

    # For 2+ missing: try combinations (limited)
    elif num_missing == 2:
        for m1 in range(20):
            for m2 in range(m1 + 1, 20):
                score = 0
                count = 0
                skip_set = {m1, m2}
                physical_order = [p for p in range(20) if p not in skip_set]
                for i, cam_id in enumerate(abn_ids):
                    if i < len(physical_order):
                        physical_id = physical_order[i]
                        if cam_id in abn_prints and physical_id in norm_prints:
                            abn_med, _ = abn_prints[cam_id]
                            norm_med, _ = norm_prints[physical_id]
                            if abn_med > 0 and norm_med > 0:
                                score += abs(abn_med - norm_med)
                                count += 1
                if count > 0:
                    avg_score = score / count
                    if avg_score < best_offset_score:
                        best_offset_score = avg_score
                        best_missing_pigs = [m1, m2]

    return {
        "abnormal_folder": abnormal_folder,
        "normal_ref": normal_folder,
        "num_pigs_found": len(abn_ids),
        "missing_physical_pigs": best_missing_pigs,
        "confidence_score": round(best_offset_score, 1),
        "abn_depth_prints": {k: round(v[0], 1) for k, v in abn_prints.items()},
        "norm_depth_prints": {k: round(v[0], 1) for k, v in norm_prints.items()},
    }


# ── Visual comparison ────────────────────────────────────────────────────────

def generate_comparison_grid(data_dir, abnormal_folder, normal_folder, out_dir):
    """Generate side-by-side IR image grid for visual verification."""
    abn_path = os.path.join(data_dir, abnormal_folder)
    norm_path = os.path.join(data_dir, normal_folder)

    abn_data = scan_folder_pigs(abn_path)
    norm_data = scan_folder_pigs(norm_path)

    thumb_size = (160, 120)
    all_pids = sorted(set(list(abn_data.keys()) + list(norm_data.keys())))
    all_pids = [p for p in all_pids if p < 25]  # include some overflow

    rows = []
    for pid in all_pids:
        row_imgs = []

        # Normal folder image
        if pid in norm_data:
            ir_path = norm_data[pid].get("ir_vis_jpg") or norm_data[pid].get("ir_jpg", "")
            if ir_path and os.path.exists(ir_path):
                img = cv2.imread(ir_path)
                if img is not None:
                    img = cv2.resize(img, thumb_size)
                    cv2.putText(img, f"norm pig{pid}", (5, 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    row_imgs.append(img)
                else:
                    row_imgs.append(np.zeros((*thumb_size[::-1], 3), dtype=np.uint8))
            else:
                row_imgs.append(np.zeros((*thumb_size[::-1], 3), dtype=np.uint8))
        else:
            blank = np.zeros((*thumb_size[::-1], 3), dtype=np.uint8)
            cv2.putText(blank, f"norm pig{pid} N/A", (5, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            row_imgs.append(blank)

        # Abnormal folder image
        if pid in abn_data:
            ir_path = abn_data[pid].get("ir_vis_jpg") or abn_data[pid].get("ir_jpg", "")
            if ir_path and os.path.exists(ir_path):
                img = cv2.imread(ir_path)
                if img is not None:
                    img = cv2.resize(img, thumb_size)
                    cv2.putText(img, f"abn pig{pid}", (5, 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    row_imgs.append(img)
                else:
                    row_imgs.append(np.zeros((*thumb_size[::-1], 3), dtype=np.uint8))
            else:
                row_imgs.append(np.zeros((*thumb_size[::-1], 3), dtype=np.uint8))
        else:
            blank = np.zeros((*thumb_size[::-1], 3), dtype=np.uint8)
            cv2.putText(blank, f"abn pig{pid} MISSING", (5, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            row_imgs.append(blank)

        rows.append(np.hstack(row_imgs))

    if rows:
        grid = np.vstack(rows)
        out_path = os.path.join(out_dir, f"compare_{abnormal_folder}.png")
        cv2.imwrite(out_path, grid)
        return out_path
    return None


# ── Overflow analysis ────────────────────────────────────────────────────────

def analyze_overflow(data_dir, folder, pig_ids):
    """For overflow folders, find which pig was double-counted by comparing depths."""
    folder_path = os.path.join(data_dir, folder)
    pig_data = scan_folder_pigs(folder_path)

    overflow_ids = sorted(p for p in pig_ids if p >= 20)
    normal_ids = sorted(p for p in pig_ids if p < 20)

    results = []
    for ov_pid in overflow_ids:
        ov_depth_path = pig_data[ov_pid].get("depth_raw")
        ov_fp = depth_fingerprint(load_depth_raw(ov_depth_path))

        # Compare with all normal pigs to find closest match
        best_match = -1
        best_diff = float('inf')
        for n_pid in normal_ids:
            n_depth_path = pig_data[n_pid].get("depth_raw")
            n_fp = depth_fingerprint(load_depth_raw(n_depth_path))
            if ov_fp[0] > 0 and n_fp[0] > 0:
                diff = abs(ov_fp[0] - n_fp[0])
                if diff < best_diff:
                    best_diff = diff
                    best_match = n_pid

        results.append({
            "folder": folder,
            "overflow_pig": ov_pid,
            "likely_duplicate_of": best_match,
            "depth_diff": round(best_diff, 1),
        })

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Detect missing/shifted pig IDs")
    parser.add_argument("data_dir", help="Path to OAK-D data folder")
    parser.add_argument("--visual", action="store_true", help="Generate comparison grid images")
    parser.add_argument("--output", default=None, help="Output CSV path")
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = args.output or os.path.join(OUTPUT_DIR, "pig_audit")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Scanning folders in {data_dir}...")
    folders = get_all_folders(data_dir)
    print(f"Found {len(folders)} timestamp folders")

    print("Classifying folders...")
    normal, under, over = classify_folders(data_dir, folders)
    print(f"  Normal (20 pigs): {len(normal)}")
    print(f"  Under  (< 20):    {len(under)}")
    print(f"  Overflow (> 20):  {len(over)}")

    # ── Analyze under-20 folders ─────────────────────────────────────────
    if under:
        print(f"\n{'='*60}")
        print("MISSING PIG DETECTION (depth fingerprint comparison)")
        print(f"{'='*60}")

        csv_rows = []
        for folder, pig_ids in under:
            nearest = find_nearest_normal(folder, normal)
            if not nearest:
                print(f"  {folder}: no normal reference folder found, skipping")
                continue

            result = compare_folders(data_dir, folder, pig_ids, nearest[0], nearest[1])
            if not result:
                continue

            missing = result["missing_physical_pigs"]
            score = result["confidence_score"]
            print(f"\n  {folder} ({result['num_pigs_found']} pigs)")
            print(f"    Reference: {result['normal_ref']}")
            print(f"    Missing physical pig(s): {missing}")
            print(f"    Avg depth diff score: {score} (lower = more confident)")
            print(f"    Abnormal depths: {result['abn_depth_prints']}")
            print(f"    Normal depths:   {result['norm_depth_prints']}")

            csv_rows.append({
                "timestamp_folder": folder,
                "num_pigs": result["num_pigs_found"],
                "reference_folder": result["normal_ref"],
                "missing_physical_pigs": str(missing),
                "depth_diff_score": score,
            })

            if args.visual:
                img_path = generate_comparison_grid(data_dir, folder, nearest[0], out_dir)
                if img_path:
                    print(f"    Visual: {img_path}")

        # Save CSV
        if csv_rows:
            csv_path = os.path.join(out_dir, "missing_pig_report.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
                writer.writeheader()
                writer.writerows(csv_rows)
            print(f"\nSaved: {csv_path}")

    # ── Analyze overflow folders ─────────────────────────────────────────
    if over:
        print(f"\n{'='*60}")
        print("OVERFLOW DETECTION (duplicate pig analysis)")
        print(f"{'='*60}")

        ov_rows = []
        for folder, pig_ids in over[:20]:  # limit output
            results = analyze_overflow(data_dir, folder, pig_ids)
            for r in results:
                print(f"  {r['folder']}: pig{r['overflow_pig']} is likely duplicate of pig{r['likely_duplicate_of']} (depth diff: {r['depth_diff']})")
                ov_rows.append(r)

        if len(over) > 20:
            print(f"  ... and {len(over) - 20} more overflow folders")

        if ov_rows:
            csv_path = os.path.join(out_dir, "overflow_pig_report.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=ov_rows[0].keys())
                writer.writeheader()
                writer.writerows(ov_rows)
            print(f"\nSaved: {csv_path}")

    print(f"\nDone. Results in {out_dir}/")


if __name__ == "__main__":
    main()
