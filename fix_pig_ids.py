#!/usr/bin/env python3
"""Fix pig ID labeling errors in specific timestamp folder ranges.

Applies manual corrections based on visual inspection of heatmap data.
Currently handles the 20260213-13-41-49 to 20260213-19-01-34 range
where a camera magnet issue caused pig ID errors.

Usage:
    python fix_pig_ids.py "C:\\PAAL_data\\Fed_pig\\Pictures_OAk" --dry-run    # preview changes
    python fix_pig_ids.py "C:\\PAAL_data\\Fed_pig\\Pictures_OAk"              # apply changes
"""

import argparse
import os
import re

FOLDER_RE = re.compile(r"^\d{8}-\d{2}-\d{2}-\d{2}$")
PIG_FILE_RE = re.compile(r"^pig(\d+)_(.+)$")

# ── Correction rules ────────────────────────────────────────────────────────
# Each rule: (start_folder, end_folder, list of actions)
# Actions: ("delete", pig_id) or ("rename", from_pig_id, to_pig_id)

CORRECTIONS = [
    {
        "name": "Camera magnet issue — pig19 duplication",
        "start": "20260213-13-41-49",
        "end": "20260213-19-01-34",
        "actions": [
            ("delete", 19),       # pig19 files are not real pig19
            ("rename", 20, 19),   # pig20 files are actually pig19
            ("delete_gte", 21),   # pig21+ are duplicates of pig19
        ],
    },
]


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

        for fname in files:
            m = PIG_FILE_RE.match(fname)
            if not m:
                continue
            pid = int(m.group(1))
            rest = m.group(2)

            for action in rule["actions"]:
                if action[0] == "delete" and pid == action[1]:
                    fpath = os.path.join(folder_path, fname)
                    if dry_run:
                        deleted.append(fname)
                    else:
                        os.remove(fpath)
                        deleted.append(fname)

                elif action[0] == "rename" and pid == action[1]:
                    new_pid = action[2]
                    new_fname = f"pig{new_pid}_{rest}"
                    src = os.path.join(folder_path, fname)
                    dst = os.path.join(folder_path, new_fname)
                    if dry_run:
                        renamed.append((fname, new_fname))
                    else:
                        # Check for conflict
                        if os.path.exists(dst):
                            # Target exists — skip to avoid overwrite
                            print(f"  WARNING: {dst} already exists, skipping rename of {fname}")
                            continue
                        os.rename(src, dst)
                        renamed.append((fname, new_fname))

                elif action[0] == "delete_gte" and pid >= action[1]:
                    fpath = os.path.join(folder_path, fname)
                    if dry_run:
                        deleted.append(fname)
                    else:
                        os.remove(fpath)
                        deleted.append(fname)

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
                        help="Preview changes without applying them")
    args = parser.parse_args()

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
        print(f"Run without --dry-run to apply changes")
    else:
        print(f"DONE: deleted {total_d} files, renamed {total_r} files")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
