"""σ calibration and per-iteration bias of model-A predictions.

    python -m e2e.calib --preds /workspace/data/e2/preds/e2e --out results/e2e

Writes, from every <set>.csv (+ .npz) in --preds:
- calibration.csv: per set, the fraction of window errors within 1σ and 2σ per axis (a
  Gaussian puts 68.3 % and 95.4 % there), the same for the 2-D error against the χ²(2)
  radii (39.3 % within 1σ, 86.5 % within 2σ, with σ = the per-axis σ) and the median z.
  σ is a per-window uncertainty; overlapping windows share frames and a run's error is
  mostly a bias, so no per-run coverage is claimed from it.
- iterations.csv: per set, the median and p90 of the per-run bias b (m/s, as e2.evaluate
  computes it) for the phase-correlation start (iteration 0 = 'init') and every
  read-out iteration, all windows of the set.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def run_bias(run_id, err):
    d = pd.DataFrame({"r": run_id, "x": err[:, 0], "y": err[:, 1]}).groupby("r")[["x", "y"]].mean()
    return np.hypot(d.x, d.y)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m e2e.calib")
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cal, its = [], []
    for csv in sorted(Path(a.preds).glob("*.csv")):
        df = pd.read_csv(csv)
        e = df[["v_est_x", "v_est_y"]].to_numpy() - df[["v_true_x", "v_true_y"]].to_numpy()
        s = df[["sigma_x", "sigma_y"]].to_numpy()
        z = np.abs(e) / s
        r2 = ((e / s) ** 2).sum(1)
        cal.append({"set": csv.stem, "windows": len(df), "cov1_axis": float((z <= 1).mean()), "cov2_axis": float((z <= 2).mean()),
                    "cov1_2d": float((r2 <= 1).mean()), "cov2_2d": float((r2 <= 4).mean()), "z_median": float(np.median(z)),
                    "sigma_median": float(np.median(np.hypot(s[:, 0], s[:, 1]))),
                    "err_median": float(np.median(np.hypot(e[:, 0], e[:, 1])))})
        npz = csv.with_suffix(".npz")
        if npz.exists():
            p = np.load(npz, allow_pickle=True)
            row = {"set": csv.stem}
            for name, v in [("init", p["v0"])] + [(f"it{i + 1}", p["vs"][:, i]) for i in range(p["vs"].shape[1])]:
                b = run_bias(p["run_id"], v - p["v_true"])
                row[f"b_median_{name}"], row[f"b_p90_{name}"] = float(b.median()), float(b.quantile(0.9))
            its.append(row)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cal).to_csv(out / "calibration.csv", index=False)
    if its:
        pd.DataFrame(its).to_csv(out / "iterations.csv", index=False)
    print(pd.DataFrame(cal).to_string(index=False))
    if its:
        print(pd.DataFrame(its).to_string(index=False))


if __name__ == "__main__":
    main()
