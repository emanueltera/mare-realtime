"""Summarise flow_bias.py output: per (run, model, g) medians of the apparent pattern velocity, and
per (model, g) the median and p95 of |a| over pairs (for the frozen-sea control).

Usage: python flow_summary.py <flow_bias csv> [--out summary.csv]
"""

import argparse
import sys

import numpy as np
import pandas as pd

from flow_bias import is_failed


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df.flow_failed == 0]
    keys = ["set", "run_id", "model", "g", "sea", "light", "sea_rate", "altitude", "airspeed", "heading", "pitch"]
    agg = ok.groupby(keys, dropna=False).agg(
        pairs=("a_E", "size"),
        a_E=("a_E", "median"), a_N=("a_N", "median"),
        a_along=("a_along", "median"), a_cross=("a_cross", "median"), a_along_wave=("a_along_wave", "median"),
        a_abs_med=("a_abs", "median"), a_abs_p95=("a_abs", lambda x: np.percentile(x, 95)),
        ground_speed=("ground_speed", "median"), valid_frac=("valid_frac", "median"),
        static_res_px=("static_res_px", "median"), flow_px_med=("flow_px_med", "median"),
        ms_per_pair=("model_ms_per_pair", "median"),
    ).reset_index()
    agg["abs_of_median"] = np.hypot(agg.a_E, agg.a_N)
    agg["pct_of_speed"] = 100 * agg.abs_of_median / agg.ground_speed
    failed = df.groupby(["run_id", "model", "g"]).flow_failed.sum().rename("failed_pairs").reset_index()
    agg = agg.merge(failed, on=["run_id", "model", "g"], how="right")
    return agg


def pooled(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df.flow_failed == 0]
    return ok.groupby(["model", "g"]).agg(
        pairs=("a_abs", "size"),
        a_abs_med=("a_abs", "median"), a_abs_p95=("a_abs", lambda x: np.percentile(x, 95)),
        a_along_med=("a_along", "median"), a_cross_med=("a_cross", "median"),
        pct_med=("a_pct_of_speed", "median"), pct_p95=("a_pct_of_speed", lambda x: np.percentile(x, 95)),
    ).reset_index().merge(df.groupby(["model", "g"]).flow_failed.sum().rename("failed_pairs").reset_index())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    df["flow_failed"] = is_failed(df.a_E, df.a_N, df.cam_vE, df.cam_vN, df.spread_E, df.spread_N).astype(int)  # current rule
    s = summarise(df)
    p = pooled(df)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    cols = ["run_id", "model", "g", "sea", "altitude", "airspeed", "heading", "pairs", "failed_pairs", "a_E", "a_N",
            "a_along", "a_cross", "a_along_wave", "abs_of_median", "pct_of_speed", "a_abs_p95", "valid_frac"]
    print(s[cols].round(3).to_string(index=False))
    print()
    print(p.round(4).to_string(index=False))
    if a.out:
        s.to_csv(a.out, index=False, float_format="%.5g")
        p.to_csv(a.out.replace(".csv", "-pooled.csv"), index=False, float_format="%.5g")


if __name__ == "__main__":
    sys.exit(main())
