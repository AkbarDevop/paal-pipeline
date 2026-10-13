"""Fast checks for the estrus sow-day table and evaluation (no lab data needed).

Run:  python tests/test_estrus_baseline.py      (or: pytest tests/)
"""

import csv
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "estrus"))

from evaluate import auc, loso  # noqa: E402
from sow_days import build, night_key, posture_features  # noqa: E402

PY = sys.executable


def test_night_belongs_to_the_morning_date():
    assert str(night_key(datetime(2026, 2, 9, 20, 0))) == "2026-02-10"
    assert str(night_key(datetime(2026, 2, 10, 3, 0))) == "2026-02-10"
    assert night_key(datetime(2026, 2, 10, 12, 0)) is None


def test_transitions_counted_within_night_only_and_gaps_skipped():
    t0 = datetime(2026, 2, 9, 19, 0)
    seq = ["lying", "standing", "lying", "lying", "sitting"]
    frames = [(t0 + timedelta(minutes=10 * i), p) for i, p in enumerate(seq)]
    frames.append((t0 + timedelta(hours=5), "standing"))          # after a 4-hour gap: not a transition
    frames.append((datetime(2026, 2, 10, 12, 0), "standing"))     # daytime
    f = posture_features(frames)
    night = f[datetime(2026, 2, 10).date()]
    assert night["night_transitions"] == 3 and night["night_frames"] == 6
    assert f[datetime(2026, 2, 10).date()]["day_standing_frac"] == 1.0


def _synthetic(n_pigs=8, seed=0):
    """Sows moved in on Feb 6; heat on days 4-5; restless nights during heat."""
    import random
    rnd = random.Random(seed)
    preds, truth = [], []
    for pid in range(n_pigs):
        onset = 4 + (pid % 2)
        for day in range(0, 9):
            d = datetime(2026, 2, 6) + timedelta(days=day)
            heat = int(onset <= day <= onset + 1)
            truth.append({"pig_id": pid, "date": d.date().isoformat(), "day": day, "heat": heat, "action": ""})
            # the night ending this morning: 19:00 yesterday -> 08:00 today, a frame every 10 min
            start = d - timedelta(hours=5)
            p_change = 0.35 if heat else 0.08
            cur = "lying"
            for k in range(13 * 6):
                if rnd.random() < p_change:
                    cur = rnd.choice([x for x in ("standing", "sitting", "lying") if x != cur])
                ts = (start + timedelta(minutes=10 * k)).strftime("%Y%m%d-%H-%M-%S")
                preds.append({"timestamp_folder": ts, "pig_id": pid, "pig_timestamp": ts,
                              "prediction_name": cur, "id_status": "ok"})
    return preds, truth


def test_posture_signal_and_loso_on_synthetic_herd():
    preds, truth = _synthetic()
    by = {}
    for r in preds:
        by.setdefault(r["pig_id"], []).append((datetime.strptime(r["pig_timestamp"], "%Y%m%d-%H-%M-%S"),
                                               r["prediction_name"]))
    rows = build(by, truth, baseline_days=2)
    heat = [r["night_transitions_delta"] for r in rows if r["heat"] and r["night_transitions_delta"] is not None]
    other = [r["night_transitions_delta"] for r in rows if not r["heat"] and r["night_transitions_delta"] is not None]
    assert auc(heat, other) > 0.9
    res = loso(rows, ["night_transitions_delta"])
    assert res["sows"] == 8 and res["auc"] > 0.9 and res["sensitivity"] > 0.7
    assert rows[0]["onset_dfw"] in (4, 5)


def test_cli_end_to_end():
    preds, truth = _synthetic(n_pigs=6, seed=1)
    tmp = tempfile.mkdtemp()
    pp, tp = os.path.join(tmp, "predictions.csv"), os.path.join(tmp, "truth.csv")
    for path, rows in ((pp, preds), (tp, truth)):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    feats = os.path.join(tmp, "feat.csv")
    out = subprocess.run([PY, os.path.join(ROOT, "estrus", "sow_days.py"), pp, tp, "--out", feats],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    out = subprocess.run([PY, os.path.join(ROOT, "estrus", "evaluate.py"), feats,
                          "--out", os.path.join(tmp, "eval.json")], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "days_from_move + posture" in out.stdout and os.path.exists(os.path.join(tmp, "eval.json"))


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)):
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{'All' if not fails else fails} {'checks passed' if not fails else 'failed'}")
    sys.exit(1 if fails else 0)
