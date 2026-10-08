"""Camera pig-ID corrections and ID policy (single source of truth).

The OAK-D rig numbers pigs sequentially as it walks the stalls. When a stall
magnet misfires it double-counts or skips, which produces IDs >= 20 and
duplicate captures inside one timestamp folder. Known bad ranges were found
by visual inspection (see CLAUDE.md "Camera Magnet Corrections").

Before v1.1 these rules lived in two files that had drifted apart
(fix_pig_ids.py had 2 rules, fix_predictions_csv.py had 8), and the inference
pipeline mapped every leftover ID with `pid % 20`, which silently merged
overflow captures into pigs 0-3. Everything now goes through this module:

    apply_corrections(folder, pid) -> corrected pid, or None to drop
    assign_ids(frames, policy="strict") -> annotates each frame dict
    folder_quality(frames) -> per-folder data-quality rows

Corrections are applied in memory only; raw data folders are never touched
(lab rule from Dr. Zhou / Fengkai, Apr 20 2026).
"""

from collections import defaultdict

NUM_STALLS = 20

# Rules are matched on the timestamp folder name (lexicographic == chronological).
# delete_pids: camera IDs that are duplicates inside the range -> drop
# rename:      camera ID -> real stall ID
# delete_gte:  any camera ID >= this value inside the range -> drop
CORRECTIONS = [
    {
        "name": "Camera magnet issue - pig19 duplication (02/13)",
        "start": "20260213-13-41-49",
        "end": "20260213-19-22-53",
        "delete_pids": [19],
        "rename": {20: 19},
        "delete_gte": 21,
    },
    {
        "name": "Camera counter overflow (02/17)",
        "start": "20260217-06-01-04",
        "end": "20260217-06-01-04",
        "delete_pids": [19],
        "rename": {20: 19},
        "delete_gte": 21,
    },
    {
        "name": "Camera counter overflow - morning (02/18)",
        "start": "20260218-10-54-39",
        "end": "20260218-11-15-52",
        "delete_pids": [19, 20],
        "rename": {21: 19},
        "delete_gte": 22,
    },
    {
        "name": "Camera counter overflow - afternoon (02/18)",
        "start": "20260218-15-33-04",
        "end": "20260218-19-00-39",
        "delete_pids": [19],
        "rename": {20: 19},
        "delete_gte": 21,
    },
    {
        "name": "Camera counter overflow - pig19=pig17, pig20=pig18 (02/19)",
        "start": "20260219-15-09-15",
        "end": "20260219-17-58-36",
        "delete_pids": [19, 20],
        "rename": {21: 19},
        "delete_gte": 22,
    },
    {
        "name": "Camera counter overflow (02/20)",
        "start": "20260220-16-36-37",
        "end": "20260220-16-36-37",
        "delete_pids": [],
        "rename": {},
        "delete_gte": 20,
    },
    {
        "name": "Camera counter overflow (03/08)",
        "start": "20260308-05-39-19",
        "end": "20260308-05-39-19",
        "delete_pids": [],
        "rename": {},
        "delete_gte": 20,
    },
    {
        "name": "Camera counter overflow (03/10-03/11)",
        "start": "20260310-17-43-30",
        "end": "20260311-14-11-47",
        "delete_pids": [],
        "rename": {},
        "delete_gte": 20,
    },
]


def apply_corrections(folder, pid, rules=CORRECTIONS):
    """Return (new_pid, rule_name). new_pid is None when the capture must be dropped."""
    for rule in rules:
        if folder < rule["start"] or folder > rule["end"]:
            continue
        if pid in rule["delete_pids"]:
            return None, rule["name"]
        if pid >= rule["delete_gte"]:
            return None, rule["name"]
        if pid in rule["rename"]:
            return rule["rename"][pid], rule["name"]
        return pid, None
    return pid, None


def assign_ids(frames, policy="strict", num_stalls=NUM_STALLS, rules=CORRECTIONS):
    """Annotate frames in place with pig_id_raw, pig_id, id_status and include.

    Each frame needs timestamp_folder, pig_id (camera id) and pig_timestamp.

    id_status values:
      ok          id used as-is
      corrected   renamed by a known correction rule
      dropped     removed by a correction rule (known duplicate)
      overflow    id >= num_stalls with no rule (strict policy excludes it)
      modulo      legacy behaviour (policy="modulo"): id mapped with % num_stalls
      duplicate   a second capture of the same stall in the same folder

    Only frames with include=True should feed per-pig outputs.
    """
    if policy not in ("strict", "modulo"):
        raise ValueError("policy must be 'strict' or 'modulo'")

    for f in frames:
        raw = int(f.get("pig_id_raw", f["pig_id"]))
        f["pig_id_raw"] = raw
        new_pid, rule = apply_corrections(f["timestamp_folder"], raw, rules)
        if new_pid is None:
            f.update(pig_id=raw, id_status="dropped", include=False, id_rule=rule)
            continue
        status = "corrected" if new_pid != raw else "ok"
        if new_pid >= num_stalls:
            if policy == "modulo":
                new_pid, status = new_pid % num_stalls, "modulo"
            else:
                f.update(pig_id=new_pid, id_status="overflow", include=False, id_rule=rule)
                continue
        f.update(pig_id=new_pid, id_status=status, include=True, id_rule=rule)

    # One capture per stall per timestamp folder: keep the earliest.
    groups = defaultdict(list)
    for f in frames:
        if f["include"]:
            groups[(f["timestamp_folder"], f["pig_id"])].append(f)
    for group in groups.values():
        if len(group) < 2:
            continue
        group.sort(key=lambda g: (g.get("pig_timestamp", ""), g["pig_id_raw"]))
        for dup in group[1:]:
            dup["id_status"] = "duplicate"
            dup["include"] = False
    return frames


def folder_quality(frames, num_stalls=NUM_STALLS):
    """Per timestamp-folder data quality rows (computed after assign_ids)."""
    by_folder = defaultdict(list)
    for f in frames:
        by_folder[f["timestamp_folder"]].append(f)

    rows = []
    for folder in sorted(by_folder):
        fs = by_folder[folder]
        raw_ids = sorted({f["pig_id_raw"] for f in fs})
        kept_ids = sorted({f["pig_id"] for f in fs if f["include"]})
        missing = [i for i in range(num_stalls) if i not in kept_ids]
        rows.append({
            "timestamp_folder": folder,
            "n_captures": len(fs),
            "n_raw_ids": len(raw_ids),
            "n_stalls_kept": len(kept_ids),
            "missing_stalls": " ".join(map(str, missing)),
            "overflow_ids": " ".join(str(f["pig_id_raw"]) for f in fs if f["id_status"] == "overflow"),
            "dropped_by_rule": " ".join(str(f["pig_id_raw"]) for f in fs if f["id_status"] == "dropped"),
            "renamed": " ".join(f"{f['pig_id_raw']}->{f['pig_id']}" for f in fs if f["id_status"] == "corrected"),
            "duplicates": " ".join(str(f["pig_id"]) for f in fs if f["id_status"] == "duplicate"),
            "needs_review": int(len(raw_ids) != num_stalls or any(
                f["id_status"] in ("overflow", "duplicate") for f in fs)),
        })
    return rows
