"""Collect the harness's slices.csv of every baseline into the tables NOTES.md quotes.

    python -m baselines.tables --eval /workspace/data/e2/eval --out /workspace/data/e2/baselines/tables

overview.csv   one row per method × set × K: the harness's 'all' row (runs, median / p90 of
               b and b%, drift% median, fractions within budget)
slices.csv     the same for the slices NOTES.md discusses (sun elevation, altitude, ground
               speed, wind speed, swell, gimbal pitch, sea_rate, flight pattern)
Every number is copied from e2.evaluate's slices.csv; nothing is recomputed here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

METHODS = ["b0-phasecorr", "b1-searaft", "b2-shell-dispersive", "b2-shell-static", "b3-mlp", "b3-mlp-full",
           "b3-mlp-radial", "b3-const", "b2-shell-dispersive-kmax0475", "b2-shell-static-kmax0475"]
SETS = ["val", "test-iid", "test-sea", "test-alt", "test-rate", "e1-core"]
COLS = ["runs", "b_median", "b_p90", "b_pct_median", "b_pct_p90", "rms_median", "drift_pct_median",
        "frac_b_le_017", "frac_bpct_le_028", "frac_bpct_le_05"]
SLICES = ["sun_elevation", "altitude", "ground_speed", "wind_speed", "swell", "gimbal_pitch", "sea_rate", "pattern"]


def collect(eval_root: Path) -> pd.DataFrame:
    rows = []
    for m in METHODS:
        for s in SETS:
            f = eval_root / m / s / "slices.csv"
            if f.exists():
                d = pd.read_csv(f)
                d.insert(0, "method", m)
                rows.append(d)
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="python -m baselines.tables")
    ap.add_argument("--eval", default="/workspace/data/e2/eval")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sl = collect(Path(a.eval))
    ov = sl[sl.slice == "all"][["method", "set", "K"] + COLS]
    ov.to_csv(out / "overview.csv", index=False)
    sl[sl.slice.isin(SLICES)][["method", "set", "K", "slice", "bin"] + COLS].to_csv(out / "slices.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(ov.round(3).to_string(index=False))
