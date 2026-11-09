"""Collect the harness outputs of the flowhead methods into small tables for NOTES.md.

    python -m flowhead.summarize --eval /workspace/data/e2/eval --preds /workspace/data/e2/preds --out results/flowhead

Writes
  summary_all.csv  one row per (method, set): the harness's 'all' slice from <eval>/<method>/<set>/slices.csv
  reference_all.csv the same rows for other agents' baselines (context only; theirs, not flowhead's)
  raw_vs_head.csv  per (method, set): median per-run b of the head's estimate v and of its
                   un-corrected weighted flow average v_raw (from <preds>/<method>/<set>.diag.csv),
                   same b definition as the harness (derived here, not by the harness)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEEP = ["runs", "b_median", "b_p90", "b_pct_median", "b_pct_p90", "rms_median", "drift_pct_median",
        "frac_b_le_017", "frac_bpct_le_028", "frac_bpct_le_05"]


def per_run_b(df, ex, ey):
    m = df.assign(ex=ex, ey=ey).groupby("run_id")[["ex", "ey"]].mean()
    return np.hypot(m.ex, m.ey)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="/workspace/data/e2/eval")
    ap.add_argument("--preds", default="/workspace/data/e2/preds")
    ap.add_argument("--methods", default="flowhead,flowhead-gap1,flowhead-onthefly,flowhead-ft")
    ap.add_argument("--reference", default="b0-phasecorr,b1-searaft",
                    help="other agents' methods, copied to reference_all.csv for context only")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows, raw = [], []
    for m in args.methods.split(","):
        for d in sorted((Path(args.eval) / m).glob("*/slices.csv")):
            s = pd.read_csv(d)
            a = s[s.slice == "all"]
            for _, r in a.iterrows():
                rows.append({"method": m, "set": d.parent.name, "K": r["K"], **{k: r[k] for k in KEEP}})
        for f in sorted((Path(args.preds) / m).glob("*.diag.csv")):
            dg = pd.read_csv(f)
            pr = pd.read_csv(str(f).replace(".diag.csv", ".csv"))
            b = per_run_b(pr, pr.v_est_x - pr.v_true_x, pr.v_est_y - pr.v_true_y)
            br = per_run_b(dg, dg.v_raw_x - dg.v_true_x, dg.v_raw_y - dg.v_true_y)
            raw.append({"method": m, "set": f.name.removesuffix(".diag.csv"), "runs": len(b),
                        "b_median_head": float(b.median()), "b_median_raw_flow": float(br.median()),
                        "b_p90_head": float(b.quantile(0.9)), "b_p90_raw_flow": float(br.quantile(0.9))})
    out = Path(args.out)
    ref = []
    for m in args.reference.split(","):
        for d in sorted((Path(args.eval) / m).glob("*/slices.csv")):
            s = pd.read_csv(d)
            for _, r in s[s.slice == "all"].iterrows():
                ref.append({"method": m, "set": d.parent.name, "K": r["K"], **{k: r[k] for k in KEEP}})
    pd.DataFrame(ref).to_csv(out / "reference_all.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "summary_all.csv", index=False)
    pd.DataFrame(raw).to_csv(out / "raw_vs_head.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(raw).to_string(index=False))


if __name__ == "__main__":
    main()
