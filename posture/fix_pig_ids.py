#!/usr/bin/env python3
"""Fix pig ID labeling errors in specific timestamp folder ranges.

Applies manual corrections based on visual inspection of heatmap data
(rules in id_corrections.py). Dry run by default.

You normally do NOT need this: run_inference.py applies the same rules in
memory without touching the raw data (lab rule: never modify raw folders).
Use --apply only on a copy of the data.

Usage:
    python fix_pig_ids.py "C:\\PAAL_data\\Fed_pig\\Pictures_OAk" --dry-run    # preview changes
    python fix_pig_ids.py "D:\\copy_of_data" --apply                              # apply to a COPY
"""

import argparse
import os
import re

FOLDER_RE = re.compile(r"^\d{8}-\d{2}-\d{2}-\d{2}$")
PIG_FILE_RE = re.compile(r"^pig(\d+)_(.+)$")

# ── Correction rules ────────────────────────────────────────────────────────
# Shared with run_inference.py / fix_predictions_csv.py (id_corrections.py).
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
from id_corrections import CORRECTIONS  # noqa: E402


def get_folders_in_range(data_dir, start, end):
    """Get all timestamp folders between start and end (inclusive)."""
    folders = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and FOLDER_RE.match(d)
    )
    return [f for f in folders if start <= f <= end]


def apply_correction(data_dir, rule, dry_run=True):
    """Apply a correction rule to all folders in its range."""
    folders = get_folders_in_range(data_dir, rule["start"], rule["end"])
    print(f"\n{'='*60}")
    print(f"Rule: {rule['name']}")
    print(f"Range: {rule['start']} → {rule['end']} ({len(folders)} folders)")
    print(f"Mode: {'DRY RUN (preview only)' if dry_run else 'APPLYING CHANGES'}")
    print(f"{'='*60}")

    total_deleted = 0
    total_renamed = 0

    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        files = os.listdir(folder_path)
        deleted = []
        renamed = []

        # Plan every file first, then delete before renaming, so a rename like
        # pig20 -> pig19 never collides with the pig19 duplicate it replaces
        # (v1.0 renamed in os.listdir order and could skip the real pig 19).
        to_delete, to_rename = [], []
        for fname in files:
            m = PIG_FILE_RE.match(fname)
            if not m:
                continue
            pid = int(m.group(1))
            rest = m.group(2)
            if pid in rule["delete_pids"] or pid >= rule["delete_gte"]:
                to_delete.append(fname)
            elif pid in rule["rename"]:
                to_rename.append((fname, f"pig{rule['rename'][pid]}_{rest}"))

        for fname in to_delete:
            if not dry_run:
                os.remove(os.path.join(folder_path, fname))
            deleted.append(fname)
        for fname, new_fname in to_rename:
            src = os.path.join(folder_path, fname)
            dst = os.path.join(folder_path, new_fname)
            if not dry_run:
                if os.path.exists(dst):
                    print(f"  WARNING: {dst} already exists, skipping rename of {fname}")
                    continue
                os.rename(src, dst)
            renamed.append((fname, new_fname))

        if deleted or renamed:
            total_deleted += len(deleted)
            total_renamed += len(renamed)
            if len(folders) <= 40 or deleted or renamed:
                print(f"\n  {folder}:")
                for f in deleted[:5]:
                    print(f"    DELETE: {f}")
                if len(deleted) > 5:
                    print(f"    ... and {len(deleted) - 5} more deletions")
                for old, new in renamed[:5]:
                    print(f"    RENAME: {old} → {new}")
                if len(renamed) > 5:
                    print(f"    ... and {len(renamed) - 5} more renames")

    print(f"\n  Summary: {total_deleted} files deleted, {total_renamed} files renamed")
    return total_deleted, total_renamed


def main():
    parser = argparse.ArgumentParser(description="Fix pig ID labeling errors")
    parser.add_argument("data_dir", help="Path to OAK-D data folder")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes without applying them (default)")
    parser.add_argument("--apply", action="store_true",
                        help="Actually delete/rename files. Only on a COPY of the data: lab rule is "
                             "never to modify the raw data folders. run_inference.py already applies "
                             "these corrections in memory, so this script is rarely needed.")
    args = parser.parse_args()
    args.dry_run = not args.apply

    if not os.path.isdir(args.data_dir):
        print(f"Error: {args.data_dir} is not a directory")
        return

    total_d = 0
    total_r = 0
    for rule in CORRECTIONS:
        d, r = apply_correction(args.data_dir, rule, dry_run=args.dry_run)
        total_d += d
        total_r += r

    print(f"\n{'='*60}")
    if args.dry_run:
        print(f"DRY RUN COMPLETE: would delete {total_d} files, rename {total_r} files")
        print(f"Add --apply to change files (only on a copy of the raw data)")
    else:
        print(f"DONE: deleted {total_d} files, renamed {total_r} files")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
