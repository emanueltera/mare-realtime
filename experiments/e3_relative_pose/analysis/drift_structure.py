"""Structure of the drift on test-long, and its sensitivity to heading error.

    uv run --no-project --with numpy --with pandas --with tabulate --with matplotlib \
        python analysis/drift_structure.py [--sets <sets dir>] [--preds <preds dir>] [--yaw <eval_yaw dir>]

Run from experiments/e3_relative_pose. Reads the run tables and the per-pair predictions,
re-chains the steps exactly as e3/evaluate.py does, and writes to results/analysis/:

  error_growth.csv           median / RMS horizontal error vs time (net and blind, true attitude)
  drift_per_run_structure.csv per run: growth exponent, bias/random split, along/cross, conditions
  drift_vs_conditions.csv    Spearman rho (bootstrap 95% CI) of net drift vs each condition, tail stats
  drift_by_pattern.csv       net drift per pattern with bootstrap CIs
  heading_sensitivity.csv    net drift vs AHRS yaw-bias sigma (from e3.evaluate runs in --yaw)
  error_growth.png           median error curves

Runs are the independent unit: every CI resamples runs (2000 bootstrap draws, fixed seed).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from e3.data import DT, FX, NOISE, Run, list_runs, run_seed, simulate  # noqa: E402
from e3.evaluate import chain  # noqa: E402

SCRATCH = Path("/private/tmp/claude-501/-Users-emanuel-repos-mayfly-demo/02eac86e-e939-4778-9996-84cb2607a222/scratchpad/e3data")
OUT = HERE.parent / "results" / "analysis"
B = 2000
RNG = np.random.default_rng(20261031)


def boot(fn, n, B=B):
    """Percentile 95% CI of fn(idx) over bootstrap resamples of n runs."""
    vals = np.array([fn(RNG.integers(0, n, n)) for _ in range(B)])
    return np.nanpercentile(vals, 2.5, axis=0), np.nanpercentile(vals, 97.5, axis=0)


def rank(x):
    return pd.Series(x).rank().to_numpy()


def spearman(x, y):
    rx, ry = rank(x), rank(y)
    return np.corrcoef(rx, ry)[0, 1]


def loglog_slope(t, e):
    m = (e > 0) & np.isfinite(e)
    return np.polyfit(np.log(t[m]), np.log(e[m]), 1)[0]


def horiz(v):
    return v[..., :2]


def load_preds(path, runs):
    z = np.load(path)
    assert [str(r) for r in z["runs"]] == [r.id for r in runs]
    out = []
    for ri, run in enumerate(runs):
        m = z["run"] == ri
        o = np.argsort(z["i"][m])
        assert (z["i"][m][o] == np.arange(run.n - 1)).all()
        out.append(z["dp_pred"][m][o].astype(np.float64))
    return out


def analyse_run(run, dp, s):
    """Everything per run for one method's translation, chained with the TRUE attitude."""
    dp_true, _ = run.labels()
    n = run.n - 1
    R = run.R[:-1]
    t = np.arange(run.n) * DT
    p_hat = chain(run.p[0], R, dp)
    err_vec = horiz(p_hat - run.p)                     # (N, 2) world horizontal error per frame
    errs = np.linalg.norm(err_vec, axis=1)
    steps_true = np.diff(run.p[:, :2], axis=0)
    path_t = np.concatenate([[0], np.cumsum(np.linalg.norm(steps_true, axis=1))])
    E = err_vec[-1]

    # --- systematic vs random, in camera axes (the camera is hard-mounted) ---
    d = dp - dp_true                                    # (n, 3) per-step error, camera axes of frame i
    # model A: constant camera-frame offset
    b = d.mean(0)
    E_offs = horiz(np.einsum("nij,j->i", R, b))
    # model B: affine in the true step, d ≈ A dp_true + c (12 parameters over n steps)
    X = np.concatenate([dp_true, np.ones((n, 1))], 1)
    coef, *_ = np.linalg.lstsq(X, d, rcond=None)
    d_sys = X @ coef
    r = d - d_sys
    E_sys = horiz(np.einsum("nij,nj->i", R, d_sys))
    E_res = horiz(np.einsum("nij,nj->i", R, r))
    w_res = horiz(np.einsum("nij,nj->ni", R, r))       # world-horizontal residual per step
    sig2 = (w_res ** 2).sum(1).mean()                   # mean squared horizontal residual per step
    rw_expect = np.sqrt(n * sig2)                       # RMS end error if residual steps were independent
    w_all = horiz(np.einsum("nij,nj->ni", R, d))
    rw_all = np.sqrt(n * ((w_all - w_all.mean(0)) ** 2).sum(1).mean())
    # lag autocorrelation of the camera-frame residual (mean over axes)
    def acf(k):
        return np.mean([np.corrcoef(r[:-k, a], r[k:, a])[0, 1] for a in range(3)])

    # --- along / cross track ---
    sl = np.linalg.norm(steps_true, axis=1)
    u = steps_true / np.maximum(sl, 1e-9)[:, None]
    nrm = np.stack([-u[:, 1], u[:, 0]], 1)              # left of track
    w = horiz(np.einsum("nij,nj->ni", R, d))            # = world step error (R dp_true is the true step)
    along_sum, cross_sum = (w * u).sum(), (w * nrm).sum()
    k = int(round(1 / DT))
    u_end = run.p[-1, :2] - run.p[-1 - k, :2]
    u_end /= np.linalg.norm(u_end)
    n_end = np.array([-u_end[1], u_end[0]])

    # growth exponent over 1-60 s and normalised drift at several windows
    sel = t >= 1.0
    out = dict(
        end_err_m=np.linalg.norm(E), path_m=path_t[-1], drift_pct=np.linalg.norm(E) / path_t[-1] * 100,
        b_ms=np.linalg.norm(E) / (n * DT),
        growth_exp=loglog_slope(t[sel], errs[sel]),
        growth_exp_5_60=loglog_slope(t[t >= 5], errs[t >= 5]),
        E_offset_m=np.linalg.norm(E_offs), E_sys_m=np.linalg.norm(E_sys), E_res_m=np.linalg.norm(E_res),
        rw_expect_m=rw_expect, rw_all_m=rw_all, step_sigma_h_m=np.sqrt(sig2),
        b_rw_ms=rw_all / (n * DT),
        acf1=acf(1), acf10=acf(10), acf50=acf(50),
        **split_half(R, dp_true, d),
        along_sum_m=along_sum, cross_sum_m=cross_sum,
        along_pct=along_sum / path_t[-1] * 100, cross_pct=cross_sum / path_t[-1] * 100,
        end_along_m=E @ u_end, end_cross_m=E @ n_end,
        disp_over_path=np.linalg.norm(run.p[-1, :2] - run.p[0, :2]) / path_t[-1],
    )
    for T in (5, 10, 12, 20, 30, 60):
        j = int(round(T / DT))
        out[f"drift_pct_{T}s"] = errs[j] / path_t[j] * 100
    return out, errs, path_t


def split_half(R, dp_true, d):
    """Out of sample: fit the systematic error on the first half of the steps, predict the
    world-horizontal error accumulated over the second half. (In-sample fits with an intercept
    force the residual's camera-frame sum to zero, so they overstate the systematic share.)"""
    n = len(d)
    h1, h2 = slice(0, n // 2), slice(n // 2, n)
    E2 = horiz(np.einsum("nij,nj->i", R[h2], d[h2]))
    X = np.concatenate([dp_true, np.ones((n, 1))], 1)
    coef, *_ = np.linalg.lstsq(X[h1], d[h1], rcond=None)
    E2_aff = horiz(np.einsum("nij,nj->i", R[h2], X[h2] @ coef))
    E2_off = horiz(np.einsum("nij,j->i", R[h2], d[h1].mean(0)))
    sl = np.linalg.norm(dp_true, axis=1)
    fwd = dp_true / sl[:, None]
    scale = lambda h: ((d[h] * fwd[h]).sum(1)).sum() / sl[h].sum() * 100  # noqa: E731
    return dict(E2_m=np.linalg.norm(E2), E2_miss_aff_m=np.linalg.norm(E2 - E2_aff), E2_miss_off_m=np.linalg.norm(E2 - E2_off),
                along_scale_h1_pct=scale(h1), along_scale_h2_pct=scale(h2))


def conditions(run, s):
    tr = run.meta["trajectory"]
    trj = pd.read_csv(run.dir / "trajectory.csv")
    h = run.p[:, 2].mean()
    gs = np.linalg.norm(np.diff(run.p[:, :2], axis=0), axis=1).sum() / ((run.n - 1) * DT)
    crab = np.degrees(np.angle(np.exp(1j * np.radians(trj["heading"] - trj["course"]))))
    return dict(
        altitude=h, ground_speed=gs, airspeed=run.airspeed.mean(),
        wind_sea=run.meta["sea"]["windSpeed"], wind_aloft=np.hypot(trj["windE"], trj["windN"]).mean(),
        crab_abs_deg=np.abs(crab).mean(), sun_elev=run.meta["light"]["sunElevation"],
        mount_pitch=tr["gimbal"]["pitch"], turbulence=tr.get("turbulence", np.nan),
        bank_abs_deg=trj["bank"].abs().mean(),
        overcast=run.meta["light"]["overcast"], haze=run.meta["light"]["haze"],
        swell_height=run.meta["sea"]["swellHeight"],
        flow_px_nadir=gs * DT / h * FX,                 # image motion per frame at nadir, 288-px-wide frames (approx.)
        baro_bias_rel_pct=(s.baro - run.p[:, 2]).mean() / h * 100,
        abs_baro_bias_rel_pct=abs((s.baro - run.p[:, 2]).mean() / h * 100),
        air_scale_pct=(s.airspeed / run.airspeed - 1).mean() * 100,
        pattern=tr["pattern"]["type"],
    )


def heading_table(yaw_dir: Path):
    xs = sorted((float(p.name), p) for p in yaw_dir.iterdir() if p.is_dir() and (p / "per_run.csv").exists())
    if not xs:
        return None
    rows, curves = [], {}
    for x, p in xs:
        pr = pd.read_csv(p / "per_run.csv")
        for method in ("net", "blind", "airspeed"):
            for att in ("true", "ahrs", "chain"):
                d = pr[(pr.method == method) & (pr.att == att)].sort_values("run")
                v = d.drift_pct.to_numpy()
                curves[(x, method, att)] = v
                row = dict(yaw_bias_deg=x, method=method, att=att, runs=len(v), drift_med=np.median(v),
                           drift_p90=np.percentile(v, 90), under_0p5_pct=(v <= 0.5).mean() * 100,
                           max_drift_med=d.max_drift_pct.median(), b_med=d.b_ms.median(),
                           yaw_err_abs_med=d.yaw_err_deg.abs().median())
                if method == "net":
                    lo, hi = boot(lambda i: np.median(v[i]), len(v))
                    row.update(drift_med_lo=lo, drift_med_hi=hi)
                    lo, hi = boot(lambda i: np.percentile(v[i], 90), len(v))
                    row.update(drift_p90_lo=lo, drift_p90_hi=hi)
                rows.append(row)
    tab = pd.DataFrame(rows)
    # ratio to true-attitude drift, per run and at the median
    X = np.array([x for x, _ in xs])
    vt = curves[(X[0], "net", "true")]
    for att in ("ahrs", "chain"):
        for x in X:
            v = curves[(x, "net", att)]
            m = (tab.yaw_bias_deg == x) & (tab.method == "net") & (tab.att == att)
            tab.loc[m, "ratio_med_to_true"] = np.median(v) / np.median(vt)
            tab.loc[m, "runs_within_10pct_of_true_pct"] = (v <= 1.1 * vt).mean() * 100
            tab.loc[m, "per_run_ratio_med"] = np.median(v / vt)

    # heading-bias sigma at which median(ahrs) = 1.1 * median(true), with bootstrap CI
    A = np.stack([curves[(x, "net", "ahrs")] for x in X])     # (nx, runs)

    def crossing(idx):
        r = np.median(A[:, idx], 1) / np.median(vt[idx])
        above = np.nonzero(r > 1.1)[0]
        if len(above) == 0:
            return np.inf
        j = above[0]
        if j == 0:
            return 0.0
        return X[j - 1] + (1.1 - r[j - 1]) * (X[j] - X[j - 1]) / (r[j] - r[j - 1])

    def crossing_runs(idx, frac=0.5):
        # sigma at which the median per-run ratio ahrs/true reaches 1.1
        r = np.median(A[:, idx] / vt[idx], 1)
        above = np.nonzero(r > 1.1)[0]
        if len(above) == 0:
            return np.inf
        j = above[0]
        if j == 0:
            return 0.0
        return X[j - 1] + (1.1 - r[j - 1]) * (X[j] - X[j - 1]) / (r[j] - r[j - 1])

    all_idx = np.arange(len(vt))
    thr = dict(crossing_median=crossing(all_idx), crossing_per_run_median=crossing_runs(all_idx))
    thr["crossing_median_lo"], thr["crossing_median_hi"] = boot(crossing, len(vt))
    thr["crossing_per_run_lo"], thr["crossing_per_run_hi"] = boot(crossing_runs, len(vt))
    return tab, thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default=str(SCRATCH / "sets"))
    ap.add_argument("--preds", default=str(SCRATCH / "preds"))
    ap.add_argument("--yaw", default=str(SCRATCH / "eval_yaw"))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    runs = [Run.load(d) for d in list_runs(Path(a.sets) / "test-long")]
    preds = {m: load_preds(Path(a.preds) / f"{m}-test-long.npz", runs) for m in ("net", "blind")}
    rows, curves, paths = [], {"net": [], "blind": []}, []
    for ri, run in enumerate(runs):
        s = simulate(run, np.random.default_rng(run_seed(run)), NOISE)
        cond = conditions(run, s)
        for m in ("net", "blind"):
            o, errs, path_t = analyse_run(run, preds[m][ri], s)
            curves[m].append(errs)
            rows.append(dict(run=run.id, method=m, **o, **cond))
        paths.append(path_t)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "drift_per_run_structure.csv", index=False)
    lines = []
    P = lambda *x: lines.append(" ".join(str(v) for v in x))  # noqa: E731

    # ---------------- 1. error growth ----------------
    t = np.arange(len(curves["net"][0])) * DT
    paths = np.stack(paths)
    g_rows = []
    for m in ("net", "blind"):
        C = np.stack(curves[m])                                         # (runs, N)
        med = np.median(C, 0)
        rms = np.sqrt((C ** 2).mean(0))
        dr = np.median(C[:, 1:] / paths[:, 1:], 0) * 100
        for j in range(0, len(t), 5):
            g_rows.append(dict(method=m, t_s=t[j], err_med_m=med[j], err_rms_m=rms[j],
                               err_p90_m=np.percentile(C[:, j], 90), drift_pct_med=dr[j - 1] if j else np.nan))
        sel = t >= 1
        sel5 = t >= 5
        d = df[df.method == m]
        ge = d.growth_exp.to_numpy()
        P(f"\n## 1. growth ({m}, true attitude)")
        P(f"per-run exponent (1-60 s): median {np.median(ge):.3f} CI {boot(lambda i: np.median(ge[i]), len(ge))}; IQR {np.percentile(ge, 25):.2f}-{np.percentile(ge, 75):.2f}")
        ge5 = d.growth_exp_5_60.to_numpy()
        P(f"per-run exponent (5-60 s): median {np.median(ge5):.3f} CI {boot(lambda i: np.median(ge5[i]), len(ge5))}")
        P(f"median-curve exponent 1-60 s: {loglog_slope(t[sel], med[sel]):.3f} CI {boot(lambda i: loglog_slope(t[sel], np.median(C[i][:, sel], 0)), len(C))}")
        P(f"median-curve exponent 5-60 s: {loglog_slope(t[sel5], med[sel5]):.3f}")
        P(f"RMS-curve exponent 1-60 s: {loglog_slope(t[sel], rms[sel]):.3f} CI {boot(lambda i: loglog_slope(t[sel], np.sqrt((C[i][:, sel] ** 2).mean(0))), len(C))}")
        for T in (5, 10, 12, 20, 30, 60):
            v = d[f"drift_pct_{T}s"].to_numpy()
            P(f"  drift at {T:>2d} s: median {np.median(v):.3f}% CI {boot(lambda i: np.median(v[i]), len(v))}  p90 {np.percentile(v, 90):.2f}%")
        # systematic vs random
        r_end = d.end_err_m.to_numpy()
        P(f"end error median {np.median(r_end):.2f} m; b_ms median {d.b_ms.median():.3f}; random-walk b (sqrt(N)*step sigma / T) median {d.b_rw_ms.median():.4f}")
        ratio = (d.end_err_m / d.rw_all_m).to_numpy()
        P(f"|E| / (sqrt(N)*sigma_step) median {np.median(ratio):.2f} CI {boot(lambda i: np.median(ratio[i]), len(ratio))}; IQR {np.percentile(ratio, 25):.2f}-{np.percentile(ratio, 75):.2f}")
        fs = (d.E_sys_m / d.end_err_m).to_numpy()
        fr = (d.E_res_m / d.end_err_m).to_numpy()
        fo = (d.E_offset_m / d.end_err_m).to_numpy()
        P(f"|E_offset|/|E| median {np.median(fo):.2f}; |E_affine-sys|/|E| median {np.median(fs):.2f}; |E_res|/|E| median {np.median(fr):.2f} CI {boot(lambda i: np.median(fr[i]), len(fr))}")
        e2, ma, mo = d.E2_m.to_numpy(), d.E2_miss_aff_m.to_numpy(), d.E2_miss_off_m.to_numpy()
        for lab, miss in (("affine", ma), ("offset", mo)):
            ex = lambda i: 1 - (miss[i] ** 2).sum() / (e2[i] ** 2).sum()  # noqa: E731
            fr2 = miss / e2
            P(f"split-half ({lab} fit on 0-30 s, predict 30-60 s): pooled explained share of 2nd-half error energy {ex(np.arange(len(e2))):.3f} CI {boot(ex, len(e2))}; median |miss|/|E2| {np.median(fr2):.2f}")
        h1, h2 = d.along_scale_h1_pct.to_numpy(), d.along_scale_h2_pct.to_numpy()
        P(f"along-track scale error, half 1 vs half 2: spearman {spearman(h1, h2):+.3f} CI {boot(lambda i: spearman(h1[i], h2[i]), len(h1))}; pearson {np.corrcoef(h1, h2)[0, 1]:+.3f}")
        for pt in sorted(d.pattern.unique()):
            q = d[d.pattern == pt]
            P(f"  {pt:10s} n={len(q):2d} growth exp median {q.growth_exp.median():.2f}  |E|/(sqrt(N)sigma) median {(q.end_err_m / q.rw_all_m).median():.1f}  drift60/drift12 median {(q.drift_pct_60s / q.drift_pct_12s).median():.2f}")
        pooled = lambda i: (d.E_res_m.to_numpy()[i] ** 2).sum() / (d.end_err_m.to_numpy()[i] ** 2).sum()  # noqa: E731
        P(f"pooled sum|E_res|^2 / sum|E|^2 = {pooled(np.arange(len(d))):.3f} CI {boot(pooled, len(d))}; runs with |E_res|<|E_sys|: {(d.E_res_m < d.E_sys_m).mean() * 100:.1f}%")
        rr = (d.E_res_m / d.rw_expect_m).to_numpy()
        P(f"|E_res| / sqrt(N sigma_res^2) median {np.median(rr):.2f} CI {boot(lambda i: np.median(rr[i]), len(rr))} (1 = independent residual steps); acf lag1 {d.acf1.median():.3f} lag10 {d.acf10.median():.3f} lag50 {d.acf50.median():.3f}")
    gr = pd.DataFrame(g_rows)
    gr.to_csv(OUT / "error_growth.csv", index=False)

    # ---------------- 2. along / cross ----------------
    d = df[df.method == "net"]
    n = len(d)
    P("\n## 2. along/cross (net, true attitude)")
    for c in ("along_pct", "cross_pct"):
        v = d[c].to_numpy()
        av = np.abs(v)
        P(f"{c}: median {np.median(v):+.3f} CI {boot(lambda i: np.median(v[i]), n)}; median |.| {np.median(av):.3f} CI {boot(lambda i: np.median(av[i]), n)}; mean {v.mean():+.3f}")
    a_, c_ = d.along_sum_m.to_numpy(), d.cross_sum_m.to_numpy()
    sh = lambda i: (a_[i] ** 2).sum() / ((a_[i] ** 2).sum() + (c_[i] ** 2).sum())  # noqa: E731
    P(f"share of path-integrated error energy that is along-track: {sh(np.arange(n)):.3f} CI {boot(sh, n)}; runs |along|>|cross| {(np.abs(a_) > np.abs(c_)).mean() * 100:.1f}%")
    ea, ec = d.end_along_m.to_numpy(), d.end_cross_m.to_numpy()
    sh2 = lambda i: (ea[i] ** 2).sum() / ((ea[i] ** 2).sum() + (ec[i] ** 2).sum())  # noqa: E731
    P(f"end error in final-track frame: median |along| {np.median(np.abs(ea)):.2f} m, |cross| {np.median(np.abs(ec)):.2f} m; along share {sh2(np.arange(n)):.3f} CI {boot(sh2, n)}")
    # does the along-track scale error follow the baro bias / airspeed scale error?
    for c in ("baro_bias_rel_pct", "air_scale_pct"):
        x, y = d[c].to_numpy(), d.along_pct.to_numpy()
        sl = lambda i: np.polyfit(x[i], y[i], 1)[0]  # noqa: E731
        P(f"along_pct vs {c}: OLS slope {sl(np.arange(n)):+.3f} CI {boot(sl, n)}; spearman {spearman(x, y):+.3f} CI {boot(lambda i: spearman(x[i], y[i]), n)}")
    x, y = d.baro_bias_rel_pct.to_numpy(), d.along_pct.to_numpy()
    P("(full inheritance of the baro bias would give slope +1: dp is predicted in units of h~ = h + b)")
    x, y = d.baro_bias_rel_pct.to_numpy(), d.along_pct.to_numpy()
    r2 = lambda i: np.corrcoef(x[i], y[i])[0, 1] ** 2  # noqa: E731
    k = np.polyfit(x, y, 1)
    res = y - np.polyval(k, x)
    P(f"R^2 of along_pct on baro_bias_rel_pct: {r2(np.arange(n)):.3f} CI {boot(r2, n)}; median |along_pct| {np.median(np.abs(y)):.3f} -> after removing the baro-bias fit {np.median(np.abs(res)):.3f}")
    # same with true-attitude drift: |along - slope*baro| as a rough no-baro-bias counterfactual is NOT computed (needs new predictions)

    # ---------------- 3. conditions ----------------
    conds = ["altitude", "ground_speed", "airspeed", "wind_sea", "wind_aloft", "crab_abs_deg", "sun_elev", "mount_pitch",
             "flow_px_nadir", "turbulence", "bank_abs_deg", "overcast", "haze", "swell_height", "baro_bias_rel_pct", "abs_baro_bias_rel_pct", "air_scale_pct"]
    named = {"altitude", "ground_speed", "wind_sea", "wind_aloft", "sun_elev", "mount_pitch"}
    crow = []
    for att_name, y in (("true", d.drift_pct.to_numpy()),):
        p90 = np.percentile(y, 90)
        tail = y > p90
        for c in conds:
            x = d[c].to_numpy().astype(float)
            if c == "mount_pitch":
                x = np.abs(x)   # degrees below horizontal; 90 = nadir
            rho = spearman(x, y)
            lo, hi = boot(lambda i: spearman(x[i], y[i]), n)
            # permutation p for rho
            perm = np.array([spearman(x, RNG.permutation(y)) for _ in range(2000)])
            p = (np.abs(perm) >= abs(rho)).mean()
            # tail: condition medians, and tail rate per tercile
            terc = np.quantile(x, [1 / 3, 2 / 3])
            ti = np.digitize(x, terc)
            rates = [tail[ti == k].mean() * 100 for k in range(3)]
            # rank-sum (tail vs rest) permutation p
            rx = rank(x)
            obs = rx[tail].mean() - rx[~tail].mean()
            pp = np.array([(lambda m: rx[m].mean() - rx[~m].mean())(RNG.permutation(tail)) for _ in range(2000)])
            p_tail = (np.abs(pp) >= abs(obs)).mean()
            crow.append(dict(condition=c + (" (abs, deg below horizon)" if c == "mount_pitch" else ""), asked=c in named,
                             att=att_name, spearman_rho=rho, rho_lo=lo, rho_hi=hi, perm_p=p,
                             median_all=np.median(x), median_tail=np.median(x[tail]), median_rest=np.median(x[~tail]),
                             tail_vs_rest_perm_p=p_tail, tail_rate_low_tercile_pct=rates[0], tail_rate_mid_tercile_pct=rates[1],
                             tail_rate_high_tercile_pct=rates[2], tercile_edges=f"{terc[0]:.3g}|{terc[1]:.3g}"))
    # ahrs attitude too (same runs, default noise), from the evaluate output if present
    cv = pd.DataFrame(crow)
    def partial(x, y, z):
        rx, ry, rz = rank(x), rank(y), rank(z)
        ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
        ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
        return np.corrcoef(ex, ey)[0, 1]
    y = d.drift_pct.to_numpy()
    for c, z in (("altitude", "abs_baro_bias_rel_pct"), ("abs_baro_bias_rel_pct", "altitude"), ("ground_speed", "abs_baro_bias_rel_pct"),
                 ("bank_abs_deg", "disp_over_path"), ("disp_over_path", "bank_abs_deg")):
        x, zz = d[c].to_numpy(), d[z].to_numpy()
        pr_ = partial(x, y, zz)
        lo, hi = boot(lambda i: partial(x[i], y[i], zz[i]), n)
        P(f"partial spearman drift~{c} | {z}: {pr_:+.3f} CI ({lo:+.3f}, {hi:+.3f})")
        crow.append(dict(condition=f"{c} | partial on {z}", asked=False, att="true", spearman_rho=pr_, rho_lo=lo, rho_hi=hi))
    x = d.disp_over_path.to_numpy()
    crow.append(dict(condition="disp_over_path", asked=False, att="true", spearman_rho=spearman(x, y),
                     rho_lo=boot(lambda i: spearman(x[i], y[i]), n)[0], rho_hi=boot(lambda i: spearman(x[i], y[i]), n)[1]))
    cv = pd.DataFrame(crow)
    # pattern as a categorical
    pat = d.pattern.to_numpy()
    y = d.drift_pct.to_numpy()
    p90 = np.percentile(y, 90)
    prow = []
    for pt in sorted(set(pat)):
        v = y[pat == pt]
        lo, hi = boot(lambda i: np.median(v[i]), len(v))
        prow.append(dict(pattern=pt, runs=len(v), drift_med=np.median(v), drift_med_lo=lo, drift_med_hi=hi,
                         drift_p90=np.percentile(v, 90), runs_in_tail=int((v > p90).sum()), tail_rate_pct=(v > p90).mean() * 100,
                         along_abs_med=np.median(np.abs(d.along_pct.to_numpy()[pat == pt])),
                         cross_abs_med=np.median(np.abs(d.cross_pct.to_numpy()[pat == pt])),
                         disp_over_path_med=np.median(d.disp_over_path.to_numpy()[pat == pt])))
    pdf = pd.DataFrame(prow)
    # Kruskal-Wallis H by permutation
    def kw(lbl):
        r = rank(y)
        return sum(len(r[lbl == g]) * (r[lbl == g].mean() - r.mean()) ** 2 for g in set(lbl))
    h0 = kw(pat)
    hp = np.array([kw(RNG.permutation(pat)) for _ in range(2000)])
    pdf.to_csv(OUT / "drift_by_pattern.csv", index=False)
    cv = pd.concat([cv, pd.DataFrame([dict(condition="pattern (categorical, Kruskal-Wallis by permutation)", asked=True, att="true",
                                           perm_p=(hp >= h0).mean())])], ignore_index=True)
    cv.to_csv(OUT / "drift_vs_conditions.csv", index=False)
    P("\n## 3. conditions (net, true attitude drift)")
    P(cv.to_markdown(index=False, floatfmt=".3f"))
    P(pdf.to_markdown(index=False, floatfmt=".3f"))
    P("tail runs (drift > p90):")
    P(d[d.drift_pct > p90][["run", "drift_pct", "along_pct", "cross_pct", "growth_exp", "altitude", "ground_speed", "flow_px_nadir",
                            "wind_sea", "sun_elev", "mount_pitch", "pattern", "baro_bias_rel_pct"]].sort_values("drift_pct").to_markdown(index=False, floatfmt=".2f"))

    # ---------------- 4. heading ----------------
    ht = heading_table(Path(a.yaw))
    if ht is not None:
        tab, thr = ht
        tab.to_csv(OUT / "heading_sensitivity.csv", index=False)
        P("\n## 4. heading sensitivity (net)")
        P(tab[tab.method == "net"].drop(columns=["method"]).to_markdown(index=False, floatfmt=".3f"))
        P(tab[(tab.method == "airspeed") & (tab.att != "true")][["yaw_bias_deg", "att", "drift_med", "yaw_err_abs_med"]].to_markdown(index=False, floatfmt=".3f"))
        P("threshold (median ahrs drift = 1.1 x median true):", {k: np.round(v, 3) for k, v in thr.items()})

    # ---------------- figure ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for m, col in (("net", "C0"), ("blind", "C1")):
            g = gr[(gr.method == m) & (gr.t_s > 0)]
            ax[0].loglog(g.t_s, g.err_med_m, col, label=f"{m} median")
            ax[0].loglog(g.t_s, g.err_rms_m, col + "--", label=f"{m} RMS")
            ax[1].plot(g.t_s, g.drift_pct_med, col, label=m)
        tt = np.array([1, 60.0])
        g = gr[(gr.method == "net") & (gr.t_s >= 1)]
        e1 = g.err_med_m.iloc[0]
        ax[0].loglog(tt, e1 * tt, "k:", lw=0.8, label="slope 1")
        ax[0].loglog(tt, e1 * np.sqrt(tt), "k-.", lw=0.8, label="slope 1/2")
        ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("horizontal error (m)"); ax[0].legend(fontsize=7)
        ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("median drift so far (%)"); ax[1].set_yscale("log"); ax[1].legend()
        fig.suptitle("E3 test-long, true attitude (simulated)")
        fig.tight_layout()
        fig.savefig(OUT / "error_growth.png", dpi=120)
    except Exception as e:  # pragma: no cover
        P("figure skipped:", e)

    print("\n".join(lines))


if __name__ == "__main__":
    main()
