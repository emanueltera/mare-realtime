"""Re-check verification findings on RESULTS.md. Writes results/analysis/verify_findings.csv.

    uv run --no-project --with numpy --with pandas --with tabulate --with matplotlib \
        python analysis/verify_findings.py

Run from experiments/e3_relative_pose. Runs are the independent unit (bootstrap over runs or
test-rate conditions, 2000 draws, fixed seed).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from e3.data import Run, list_runs  # noqa: E402

SCRATCH = Path("/private/tmp/claude-501/-Users-emanuel-repos-mayfly-demo/02eac86e-e939-4778-9996-84cb2607a222/scratchpad/e3data")
RES = HERE.parent / "results"
OUT = RES / "analysis"
RNG = np.random.default_rng(20261031)
rows = []


def add(finding, check, value, lo=np.nan, hi=np.nan, n=np.nan, note=""):
    rows.append(dict(finding=finding, check=check, value=value, lo=lo, hi=hi, n=n, note=note))
    print(f"[{finding}] {check}: {value:.4g}" + (f" [{lo:.4g}, {hi:.4g}]" if np.isfinite(lo) else "") + (f"  n={n}" if np.isfinite(n) else ""))


def boot(fn, n, B=2000):
    v = np.array([fn(RNG.integers(0, n, n)) for _ in range(B)])
    return np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)


# 1. Table 2 'runs within 10 % of own true drift': one-sided vs two-sided
for X in ("0", "0.3", "0.5", "1", "2", "3", "5"):
    pr = pd.read_csv(SCRATCH / "eval_yaw" / X / "per_run.csv")
    pr = pr[pr.method == "net"]
    t = pr[pr.att == "true"].set_index("run").drift_pct
    a = pr[pr.att == "ahrs"].set_index("run").drift_pct.reindex(t.index)
    r = (a / t).to_numpy()
    add("1 yaw table", f"sigma {X}d: ahrs <= 1.1 x true (%)", (r <= 1.1).mean() * 100, n=len(r))
    add("1 yaw table", f"sigma {X}d: |ahrs/true - 1| <= 0.1 (%)", (np.abs(r - 1) <= 0.1).mean() * 100, n=len(r))

# 2. altitude bins on test-long (net, true attitude) by mean altitude
pr = pd.read_csv(RES / "eval" / "test-long" / "per_run.csv")
d = pr[(pr.method == "net") & (pr.att == "true")]
edges = [30, 50, 80, 120, 200]
cnt = pd.cut(d.altitude, edges, include_lowest=True).value_counts().sort_index()
for iv, c in cnt.items():
    add("2 altitude bins", f"test-long runs at {iv}", c, note="mean altitude from per_run.csv")
pv = pd.read_csv(RES / "eval" / "val" / "per_run.csv")
dv = pv[(pv.method == "net") & (pv.att == "true")]
for iv, c in pd.cut(dv.altitude, edges, include_lowest=True).value_counts().sort_index().items():
    add("2 altitude bins", f"val runs at {iv}", c)
# altitude from the run tables, to check the per_run column is the mean
runs = {r.id: r for r in (Run.load(p) for p in list_runs(SCRATCH / "sets" / "test-long"))}
alt_tab = np.array([runs[k].p[:, 2].mean() for k in d.run])
add("2 altitude bins", "max |per_run altitude - mean trajectory height| (m)", float(np.max(np.abs(alt_tab - d.altitude.to_numpy()))),
    note="checks the binning variable; large value means per_run altitude is not the trajectory mean")

# 3. sea rate: ratio of median b vs median of per-condition ratios
src = pd.read_csv(OUT / "sea_rate_per_condition.csv")
w = src.pivot_table(index="cond", columns="sea_rate", values="b_ms")
n = len(w)
for r in (0.0, 2.0):
    x, y = w[r].to_numpy(), w[1.0].to_numpy()
    rom = np.median(x) / np.median(y)
    lo, hi = boot(lambda i: np.median(x[i]) / np.median(y[i]), n)
    add("3 sea rate", f"ratio of median b rate{int(r)}/rate1", rom, lo, hi, n)
    q = x / y
    lo, hi = boot(lambda i: np.median(q[i]), n)
    add("3 sea rate", f"median of per-condition b ratios rate{int(r)}/rate1", np.median(q), lo, hi, n)
for r in (0.0, 1.0, 2.0):
    add("3 sea rate", f"median b rate{int(r)} (m/s)", float(w[r].median()), n=n)

# 5. random-walk comparison: which autocorrelation goes with which scatter
s = pd.read_csv(OUT / "drift_per_run_structure.csv")
s = s[s.method == "net"]
ratio_all = (s.end_err_m / s.rw_all_m).to_numpy()
lo, hi = boot(lambda i: np.median(ratio_all[i]), len(s))
add("5 random walk", "|E| / (sqrt(N) sigma, raw step error about run mean)", np.median(ratio_all), lo, hi, len(s))
ratio_res = (s.end_err_m / s.rw_expect_m).to_numpy()
lo, hi = boot(lambda i: np.median(ratio_res[i]), len(s))
add("5 random walk", "|E| / (sqrt(N) sigma, affine-fit residual)", np.median(ratio_res), lo, hi, len(s))
add("5 random walk", "median acf lag1 of affine-fit residual (csv acf1)", s.acf1.median(), n=len(s))
# raw per-step error, camera axes, mean over axes (same estimator as drift_structure.acf)
z = np.load(SCRATCH / "preds" / "net-test-long.npz")
ids = [str(k) for k in z["runs"]]
ac_raw = []
for ri, rid in enumerate(ids):
    run = runs[rid]
    m = z["run"] == ri
    o = np.argsort(z["i"][m])
    dp = z["dp_pred"][m][o].astype(np.float64)
    dp_true, _ = run.labels()
    e = dp - dp_true
    ac_raw.append(np.mean([np.corrcoef(e[:-1, a], e[1:, a])[0, 1] for a in range(3)]))
ac_raw = np.array(ac_raw)
lo, hi = boot(lambda i: np.median(ac_raw[i]), len(ac_raw))
add("5 random walk", "median acf lag1 of raw per-step error (camera axes)", np.median(ac_raw), lo, hi, len(ac_raw))

# 4. training log
log = pd.read_csv(RES / "net_log.csv")
for _, r in log[log.epoch >= 12].iterrows():
    add("4 training", f"val_t_err_m epoch {int(r.epoch)}", r.val_t_err_m)

# 6. conditions missing from §3.4
dc = pd.read_csv(OUT / "drift_vs_conditions.csv").set_index("condition")
for c in ("bank_abs_deg", "bank_abs_deg | partial on disp_over_path", "airspeed", "altitude"):
    add("6 conditions", f"spearman {c}", dc.loc[c, "spearman_rho"], dc.loc[c, "rho_lo"], dc.loc[c, "rho_hi"], 120,
        note=f"perm p {dc.loc[c, 'perm_p']}")

pd.DataFrame(rows).to_csv(OUT / "verify_findings.csv", index=False)
