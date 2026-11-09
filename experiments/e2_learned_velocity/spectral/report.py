"""Collect the harness outputs of the spectral runs into the tables NOTES.md quotes.

    python -m spectral.report --eval /workspace/data/e2/eval --train /workspace/data/e2/spectral \
        --out results/spectral

Writes (every number derived from the harness's slices.csv / per_run.csv / e1_compare.csv,
or the training curve.csv):
- summary_all.csv   method × set × K: runs, b median / p90, b% median, the budget fractions, drift%
- slices_main.csv   the main method's slices (sun, altitude, ground speed, wind, swell, pitch, seaRate)
- e1_compare_by_sun.csv  e1-core per-run b of each method against E1's shell fit (dispersive T = 3.2 /
                         6.4) and flow, median by sun (midday / low sun) and altitude. E1's per-run value
                         was a median over its windows, E2's b is |mean| over windows (not identical).
- curves.csv, fig_curves.png  validation curves of the three trainings
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

METHODS = ["spectral", "spectral-k16", "spectral-noreg", "spectral-tta"]
COLS = ["runs", "b_median", "b_p90", "b_pct_median", "b_pct_p90", "rms_median", "drift_pct_median",
        "frac_b_le_017", "frac_bpct_le_028", "frac_bpct_le_05"]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m spectral.report")
    ap.add_argument("--eval", default="/workspace/data/e2/eval")
    ap.add_argument("--train", default="/workspace/data/e2/spectral")
    ap.add_argument("--out", required=True)
    ap.add_argument("--main", default="spectral")
    a = ap.parse_args(argv)
    ev, out = Path(a.eval), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, main_sl = [], []
    for m in METHODS + ["phasecorr"]:
        for p in sorted((ev / m).glob("*/slices.csv")):
            sl = pd.read_csv(p)
            sl.insert(0, "method", m)
            sl.insert(1, "set_dir", p.parent.name)
            rows.append(sl[sl.slice == "all"])
            if m == a.main:
                main_sl.append(sl[sl.slice != "all"])
    allr = pd.concat(rows, ignore_index=True)
    allr[["method", "set_dir", "set", "K", *COLS]].to_csv(out / "summary_all.csv", index=False)
    if main_sl:
        pd.concat(main_sl, ignore_index=True).to_csv(out / "slices_main.csv", index=False)
    # e1-core against E1
    cmp_rows = []
    for m in METHODS + ["phasecorr"]:
        p = ev / m / "e1-core" / "e1_compare.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        d["sun"] = d.sun_elevation.map(lambda s: "midday" if s >= 35 else "low sun")
        for (sun, alt), g in [*(((s, "all"), g) for s, g in d.groupby("sun")),
                              *(((s, al), g) for (s, al), g in d.groupby(["sun", "altitude"]))]:
            cmp_rows.append({"method": m, "sun": sun, "altitude": alt, "runs": len(g), "b_median": g.b.median(),
                             "b_p90": g.b.quantile(0.9), "frac_b_le_017": (g.b <= 0.17).mean(),
                             "e1_shell_T3.2_median": g["e1_shell_dispersive_T3.2"].median(),
                             "e1_shell_T6.4_median": g["e1_shell_dispersive_T6.4"].median(),
                             "e1_flow_searaft_median": g["e1_flow_searaft_g1"].median()})
    if cmp_rows:
        pd.DataFrame(cmp_rows).to_csv(out / "e1_compare_by_sun.csv", index=False)
    # training curves
    curves = []
    for name in ["k32", "k16", "k32-noreg"]:
        p = Path(a.train) / name / "curve.csv"
        if p.exists():
            c = pd.read_csv(p)
            c.insert(0, "run", name)
            curves.append(c)
    if curves:
        cv = pd.concat(curves, ignore_index=True)
        cv.to_csv(out / "curves.csv", index=False)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
        for name, g in cv.groupby("run"):
            ax[0].plot(g.minutes, g.val_b_median, marker="o", ms=3, label=name)
            ax[1].plot(g.minutes, g.val_win_err_median, marker="o", ms=3, label=name)
            ax[2].plot(g.minutes, g.train_loss, marker="o", ms=3, label=name)
        ax[0].axhline(0.17, color="k", lw=0.8, ls="--")
        ax[0].set(xlabel="minutes", ylabel="val median per-run b (m/s), val subset", yscale="log")
        ax[1].set(xlabel="minutes", ylabel="val median window |error| (m/s)", yscale="log")
        ax[2].set(xlabel="minutes", ylabel="train loss (epoch mean)")
        ax[0].legend()
        fig.tight_layout()
        fig.savefig(out / "fig_curves.png", dpi=120)
    print(allr[["method", "set_dir", "K", "runs", "b_median", "b_p90", "b_pct_median", "frac_b_le_017",
                "frac_bpct_le_028"]].to_string(index=False))


if __name__ == "__main__":
    main()
