"""σ calibration of a spectral prediction CSV, per window and per axis (tile x, y pooled).

    python -m spectral.calibration --pred /workspace/data/e2/preds/spectral/*.csv --out calibration.csv

For a calibrated Gaussian σ, 68.3 % of |z| = |v_est − v_true| / σ lie within 1 and 95.4 %
within 2. Rows: one 'all' row per set, then the windows split into σ deciles (pooled over
sets in the file) with the mean σ against the RMS error in each (reliability).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def calib(df: pd.DataFrame) -> list[dict]:
    e = np.concatenate([df.v_est_x - df.v_true_x, df.v_est_y - df.v_true_y])
    s = np.concatenate([df.sigma_x, df.sigma_y])
    z = np.abs(e) / s
    rows = [{"bin": "all", "n": len(e), "sigma_mean": s.mean(), "sigma_median": np.median(s),
             "rms_err": np.sqrt(np.mean(e ** 2)), "frac_z_le_1": (z <= 1).mean(), "frac_z_le_2": (z <= 2).mean(),
             "median_abs_z": np.median(z)}]
    q = np.quantile(s, np.linspace(0, 1, 11))
    for i in range(10):
        m = (s >= q[i]) & ((s < q[i + 1]) if i < 9 else (s <= q[i + 1]))
        rows.append({"bin": f"sigma decile {i + 1}", "n": int(m.sum()), "sigma_mean": s[m].mean(),
                     "sigma_median": np.median(s[m]), "rms_err": np.sqrt(np.mean(e[m] ** 2)),
                     "frac_z_le_1": (z[m] <= 1).mean(), "frac_z_le_2": (z[m] <= 2).mean(), "median_abs_z": np.median(z[m])})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m spectral.calibration")
    ap.add_argument("--pred", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows = []
    for p in a.pred:
        df = pd.read_csv(p)
        for r in calib(df):
            rows.append({"set": df["set"].iloc[0], "K": int(df.K.iloc[0]), **r})
        if "test-rate" in p:  # the frozen and the live seas apart
            runs = pd.read_csv(Path("/workspace/data/e2/sets/test-rate/runs.csv"))[["run_id", "sea_rate"]]
            d = df.merge(runs, on="run_id")
            for rate, g in d.groupby("sea_rate"):
                r = calib(g)[0]
                rows.append({"set": f"test-rate seaRate {rate:g}", "K": int(df.K.iloc[0]), **r})
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    print(out[out.bin == "all"].to_string(index=False))


if __name__ == "__main__":
    main()
