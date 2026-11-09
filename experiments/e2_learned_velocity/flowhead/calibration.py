"""σ calibration of a method's prediction CSVs: how the per-window error compares with the
predicted σ (per tile axis), per set.

    python -m flowhead.calibration --preds /workspace/data/e2/preds/flowhead --out calibration.csv

Per set: windows; RMS of z = (v_est − v_true) / σ over both axes (1 if calibrated); the
fraction of components with |z| ≤ 1 and ≤ 2 (0.683 and 0.954 for a Gaussian error); median σ
and median |error| per component; the Spearman rank correlation between σ and |error| (does a
larger σ mark a worse window?); and the same z-RMS for run-mean errors against the run-mean σ
divided by √(windows / 8) — a rough check whether σ says anything about the bias, which it is
not trained for (windows of a run share their error, so this is only indicative).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def calib(df: pd.DataFrame) -> dict:
    e = np.concatenate([df.v_est_x - df.v_true_x, df.v_est_y - df.v_true_y])
    s = np.concatenate([df.sigma_x, df.sigma_y])
    z = e / s
    g = df.assign(ex=df.v_est_x - df.v_true_x, ey=df.v_est_y - df.v_true_y).groupby("run_id")
    m = g[["ex", "ey", "sigma_x", "sigma_y"]].mean()
    nw = g.size()
    zr = np.concatenate([m.ex / (m.sigma_x / np.sqrt(nw / 8)), m.ey / (m.sigma_y / np.sqrt(nw / 8))])
    return {"windows": len(df), "z_rms": float(np.sqrt(np.mean(z ** 2))),
            "frac_z_le_1": float(np.mean(np.abs(z) <= 1)), "frac_z_le_2": float(np.mean(np.abs(z) <= 2)),
            "sigma_median": float(np.median(s)), "abs_err_median": float(np.median(np.abs(e))),
            "spearman_sigma_abs_err": float(spearmanr(s, np.abs(e)).statistic),
            "run_z_rms": float(np.sqrt(np.mean(zr ** 2)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="directory of <set>.csv prediction files")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = []
    for f in sorted(Path(args.preds).glob("*.csv")):
        if f.name.endswith(".diag.csv"):
            continue
        df = pd.read_csv(f)
        if df.sigma_x.isna().all():
            continue
        if f.stem == "test-rate":
            rate = df.run_id.str.extract(r"-rate(\d+(?:p\d+)?)$", expand=False).str.replace("p", ".")
            for r, g in df.groupby(rate):
                rows.append({"set": f"test-rate seaRate {r}", **calib(g)})
        rows.append({"set": f.stem, **calib(df)})
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
