"""Gather the harness's per-set results of model-A methods into one table.

    python -m e2e.collect --eval /workspace/data/e2/eval --methods e2e e2e-K2 e2e-K4 --out results/e2e/summary_by_set.csv

Each row is the harness's `slice == "all"` row of <eval>/<method>/<set>/slices.csv (copied,
not recomputed), with the method name added; plus, for e1-core, the medians of
e1_compare.csv's columns per K (this method's b and E1's estimators on the same runs)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m e2e.collect")
    ap.add_argument("--eval", default="/workspace/data/e2/eval")
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows, cmp = [], []
    for m in a.methods:
        for d in sorted(Path(a.eval, m).glob("*/slices.csv")):
            sl = pd.read_csv(d)
            r = sl[sl.slice == "all"].copy()
            r.insert(0, "method", m)
            rows.append(r)
            e1 = d.parent / "e1_compare.csv"
            if e1.exists():
                for K, c in pd.read_csv(e1).groupby("K"):
                    cols = ["b"] + [x for x in c if x.startswith("e1_")]
                    cmp.append({"method": m, "K": K, "runs": len(c), **{f"median_{x}": c[x].median() for x in cols},
                                **{f"frac_le_017_{x}": (c[x] <= 0.17).mean() for x in cols}})
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t = pd.concat(rows, ignore_index=True)
    t.to_csv(out, index=False)
    print(t[["method", "set", "K", "runs", "b_median", "b_p90", "b_pct_median", "drift_pct_median", "frac_b_le_017",
             "frac_bpct_le_028", "frac_bpct_le_05"]].to_string(index=False))
    if cmp:
        c = pd.DataFrame(cmp)
        c.to_csv(out.with_name("e1_core_compare_medians.csv"), index=False)
        print(c.T.to_string())


if __name__ == "__main__":
    main()
