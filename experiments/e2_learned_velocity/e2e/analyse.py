"""Does the network measure the waves' motion or subtract an expected one? On test-rate the
same 100 conditions are rendered with the sea's clock at 0 (frozen), 0.5, 1 and 2 × its
rate; a network trained only at rate 1 that measured the waves would stay unbiased at every
rate, one that subtracts a learned expectation would be biased along the wind sea in
opposite senses at rate 0 and rate 2.

    python -m e2e.analyse --eval /workspace/data/e2/eval --methods e2e e2e-K4 e2e-K2 phasecorr \\
        --out results/e2e/sea_rate.csv

Per method and sea rate, from the harness's test-rate/per_run.csv: runs, median b, and the
median and mean of the signed bias along the wind sea (b_along_wind, m/s; positive = the
estimate is too fast toward where the wind sea travels) and across it."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m e2e.analyse")
    ap.add_argument("--eval", default="/workspace/data/e2/eval")
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows = []
    for m in a.methods:
        f = Path(a.eval, m, "test-rate", "per_run.csv")
        if not f.exists():
            continue
        pr = pd.read_csv(f)
        for r, g in pr.groupby("sea_rate"):
            rows.append({"method": m, "sea_rate": r, "runs": len(g), "b_median": g.b.median(),
                         "frac_b_le_017": (g.b <= 0.17).mean(),
                         "b_along_wind_median": g.b_along_wind.median(), "b_along_wind_mean": g.b_along_wind.mean(),
                         "b_cross_wind_median": g.b_cross_wind.median()})
    t = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(a.out, index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
