"""Check which labelled frames can be found under the data roots, before training.

  python posture/check_label_paths.py --data-root "\\\\umad.umsystem.edu\\rde\\zhoujianf-lab\\Pig project\\Sow Data\\Pictures_OAK"
  python posture/check_label_paths.py --data-root <share> --data-root E:\\Pig_data\\Pictures_OAK

For every row of the labels CSV it reports whether the image was found, whether
its timestamp folder exists but the file is missing, or whether the folder is
missing altogether, broken down by day and by pig. For missing files it lists
what that pig's files in the folder are actually called. Read-only.
"""

import argparse
import csv
import os
import sys
import pathlib
from collections import Counter, defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import LABELS_POSTURE3_CSV  # noqa: E402
from data_loader import timestamp_folders  # noqa: E402

STATES = ["found", "file_missing", "folder_missing"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", action="append", required=True)
    ap.add_argument("--labels-csv", default=LABELS_POSTURE3_CSV)
    ap.add_argument("--column", default="ir_jpg", help="Image column to check (ir_jpg, rgb_jpg, depth_jpg)")
    ap.add_argument("--examples", type=int, default=6)
    args = ap.parse_args()

    folders = {}
    for root in args.data_root:
        found = timestamp_folders(root)
        print(f"  {root}: {len(found)} timestamp folders")
        for k, v in found.items():
            folders.setdefault(k, []).extend(v)

    with open(args.labels_csv, encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("label") or "").strip()]

    total = Counter()
    by_day, by_pig = defaultdict(Counter), defaultdict(Counter)
    by_label = defaultdict(Counter)
    examples = []
    for r in rows:
        parts = [x for x in r[args.column].replace("\\", "/").split("/") if x]
        folder, fname = parts[-2], parts[-1]
        dirs = folders.get(folder, [])
        if any(os.path.exists(os.path.join(d, fname)) for d in dirs):
            st = "found"
        elif dirs:
            st = "file_missing"
            if len(examples) < args.examples:
                prefix = f"pig{r['pig_id']}_"
                try:
                    same_pig = sorted(n for n in os.listdir(dirs[0]) if n.startswith(prefix))
                except OSError:
                    same_pig = ["(cannot list folder)"]
                examples.append((dirs[0], fname, same_pig))
        else:
            st = "folder_missing"
        total[st] += 1
        by_day[folder[:8]][st] += 1
        by_pig[int(r["pig_id"])][st] += 1
        by_label[r["label"]][st] += 1

    print(f"\nLabelled frames: {len(rows)} | " + " | ".join(f"{s}: {total[s]}" for s in STATES))
    print("\nBy day:")
    print(f"  {'day':<10}" + "".join(f"{s:>16}" for s in STATES))
    for day in sorted(by_day):
        print(f"  {day:<10}" + "".join(f"{by_day[day][s]:>16}" for s in STATES))
    print("\nBy pig:")
    print("  " + ", ".join(f"{p}: {by_pig[p]['found']}/{sum(by_pig[p].values())}" for p in sorted(by_pig)))
    print("By label (0 standing, 1 sitting, 2 lying): " +
          ", ".join(f"{k}: {v['found']}/{sum(v.values())}" for k, v in sorted(by_label.items())))
    if examples:
        print("\nMissing files, with what that pig's files in the folder are called:")
        for d, fname, same in examples:
            print(f"  {d}\n    expected: {fname}\n    present:  {', '.join(same[:6]) or '(no files for this pig)'}")


if __name__ == "__main__":
    main()
