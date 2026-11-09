"""E3 sea-rate analysis: does PoseNet measure motion, or lean on the sim's wave dispersion (E2)?

    uv run --no-project --with numpy --with pandas --with scipy python analysis/sea_rate.py

Reads results/eval/{test-rate,test-long}/per_run.csv, the per-pair predictions (npz) and the run
tables (no frames). Writes results/analysis/sea_rate.csv (long format: one statistic per row) and
results/analysis/sea_rate_per_condition.csv, sea_rate_horizons.csv.

Conventions
- Errors are estimate minus truth. Per step, the camera-axes error dp_pred - dp_true is rotated to
  the world with the TRUE attitude (as evaluate.py's 'true' chaining) and its horizontal part kept.
- Wave direction u = (sin d, cos d) in (E, N), d = run.json sea.windDirection: compass degrees the
  wind sea travels TOWARD (src/ocean/scene.js, src/conditions/presets.js). 'along_wave' is the
  error projected on u, in m/s (mean per-step error / DT). The same with swellDirection is kept too.
- A network that tracks the water's pattern and adds a learned rate-1 wave correction would err by
  (1 - r) * c along u: + at r = 0, - at r = 2, the same size. One that measures the ground's motion
  would keep its rate-1 error at every rate.
- Conditions (cNNN in the run id) are the independent unit in test-rate: rates share trajectory
  seed, sea seed and therefore the sensor draw. Runs are the unit in test-long.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from e3.data import DT, Run, list_runs  # noqa: E402

SCRATCH = Path("/private/tmp/claude-501/-Users-emanuel-repos-mayfly-demo/02eac86e-e939-4778-9996-84cb2607a222/scratchpad/e3data")
EVAL = HERE / "results/eval"
OUT = HERE / "results/analysis"
B = 10000
RNG = np.random.default_rng(20261031)
E2 = {  # E2 RESULTS.md §2.4, rate_diagnostic.csv: median b (m/s) and signed along-wind median at r = 0 / 0.5 / 1 / 2
    "A e2e K=8": dict(b=(1.242, 1.063, 0.182, 2.123), along=(-0.097, 1.009, -0.008, -2.093)),
    "A e2e K=2": dict(b=(0.605, 0.720, 0.233, 1.087), along=(-0.038, 0.646, -0.029, -1.061)),
    "C spectral K=32": dict(b=(0.356, 0.503, 0.317, 1.647), along=(0.005, 0.169, -0.097, -1.584)),
    "B0 phase correlation": dict(b=(0.085, 0.654, 1.324, 2.342), along=(0.006, -0.631, -1.308, -2.300)),
}

rows: list[dict] = []


def add(section, quantity, stat, value, lo=np.nan, hi=np.nan, n=np.nan, p=np.nan, set_="test-rate", rate=np.nan, note=""):
    rows.append(dict(section=section, set=set_, quantity=quantity, rate=rate, stat=stat, value=value, ci_lo=lo, ci_hi=hi, n=n, p_value=p, note=note))


def boot(x, f, b=B):
    """Percentile 95 % CI of f over resamples of the first axis of x (independent units)."""
    x = np.asarray(x)
    idx = RNG.integers(0, len(x), (b, len(x)))
    vals = np.array([f(x[i]) for i in idx])
    return np.percentile(vals, [2.5, 97.5])


def gmean(x):
    return float(np.exp(np.mean(np.log(x))))


def describe(section, quantity, x, rate=np.nan, set_="test-rate", note="", stats_=("median", "mean")):
    x = np.asarray(x, float)
    for s in stats_:
        f = {"median": np.median, "mean": np.mean, "gmean": gmean}[s]
        lo, hi = boot(x, f)
        add(section, quantity, s, float(f(x)), lo, hi, len(x), set_=set_, rate=rate, note=note)


# ---- per-pair errors from the predictions ---------------------------------------------------


def step_table(set_name: str) -> tuple[pd.DataFrame, dict]:
    """Per run: step errors (true attitude, world horizontal) and the chained error at every frame."""
    runs = [Run.load(d) for d in list_runs(SCRATCH / "sets" / set_name)]
    z = np.load(SCRATCH / "preds" / f"net-{set_name}.npz")
    assert list(z["runs"]) == [r.id for r in runs]
    out, chains = [], {}
    for ri, run in enumerate(runs):
        m = z["run"] == ri
        o = np.argsort(z["i"][m])
        dp = z["dp_pred"][m][o]
        dp_true, _ = run.labels()
        n = run.n - 1
        assert len(dp) == n
        e_cam = dp - dp_true
        e_w = np.einsum("nij,nj->ni", run.R[:-1], e_cam)[:, :2]          # world E, N error per step
        d_w = np.diff(run.p[:, :2], axis=0)                               # true horizontal step
        step_len = np.linalg.norm(d_w, axis=1)
        fwd = d_w / step_len[:, None]
        left = np.stack([-fwd[:, 1], fwd[:, 0]], 1)
        sea = run.meta["sea"]
        wd, sd = np.radians(sea["windDirection"]), np.radians(sea["swellDirection"])
        u = np.array([np.sin(wd), np.cos(wd)])
        us = np.array([np.sin(sd), np.cos(sd)])
        uc = np.array([-u[1], u[0]])
        cum = np.concatenate([np.zeros((1, 2)), np.cumsum(e_w, 0)])     # chained horizontal error (true attitude)
        hpath = np.concatenate([[0.0], np.cumsum(step_len)])
        chains[run.id] = (cum, hpath)
        sl3 = np.linalg.norm(dp_true, axis=1)
        mean_w = e_w.mean(0)
        resid = e_w - mean_w
        meta = json.loads((run.dir / "meta.json").read_text())
        cond = re.search(r"-c(\d+)-", run.id)
        out.append(dict(
            run=run.id, cond=int(cond.group(1)) if cond else -1, sea_rate=run.sea_rate, steps=n,
            pattern=run.meta["trajectory"]["pattern"]["type"], altitude=float(run.p[0, 2]), airspeed=float(run.airspeed.mean()),
            ground_speed=hpath[-1] / (n * DT), wind=sea["windSpeed"], wind_dir=sea["windDirection"], swell_dir=sea["swellDirection"],
            peak_band=meta["seaStats"]["peakBand"], peak_period=meta["seaStats"]["peakPeriod"],
            peak_phase_speed=9.81 * meta["seaStats"]["peakPeriod"] / (2 * np.pi), hs=meta["seaStats"]["significantHeight"],
            step_err_m=float(np.linalg.norm(e_cam, axis=1).mean()),                            # as evaluate.py (3-D, camera axes)
            along_bias_pct_eval=float(np.mean((e_cam * (dp_true / sl3[:, None])).sum(1)) / sl3.mean() * 100),
            along_track_ms=float((e_w * fwd).sum(1).mean() / DT),                              # world horizontal, along the true step
            cross_track_ms=float((e_w * left).sum(1).mean() / DT),
            along_wave_ms=float(mean_w @ u / DT), cross_wave_ms=float(mean_w @ uc / DT),
            along_swell_ms=float(mean_w @ us / DT),
            bias_E_ms=float(mean_w[0] / DT), bias_N_ms=float(mean_w[1] / DT),
            bias_ms=float(np.linalg.norm(mean_w) / DT),                                        # |mean horizontal step error| / DT
            step_scatter_m=float(np.sqrt((resid**2).sum(1).mean())),                           # RMS of the step error about its run mean
            end_err_m=float(np.linalg.norm(cum[-1])), end_err_bias_m=float(np.linalg.norm(mean_w) * n),
            end_err_rw_m=float(np.sqrt((resid**2).sum())),                                     # what i.i.d. scatter alone would give
            max_err_m=float(np.linalg.norm(cum, axis=1).max()), path_m=float(hpath[-1]),
        ))
    df = pd.DataFrame(out)
    df["drift_pct"] = df.end_err_m / df.path_m * 100
    df["b_ms"] = df.end_err_m / (df.steps * DT)
    return df, chains


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tr, _ = step_table("test-rate")

    # Cross-check against evaluate.py's per_run.csv (net, true attitude).
    pr = pd.read_csv(EVAL / "test-rate/per_run.csv")
    net = pr[(pr.method == "net")].copy()
    chk = net[net.att == "true"].set_index("run").loc[tr.run]
    dmax = float(np.abs(chk.drift_pct.values - tr.drift_pct.values).max())
    add("check", "max |drift_pct recomputed - per_run.csv|", "max", dmax, n=len(tr), note="net, true attitude")
    assert dmax < 1e-6, dmax

    # ---- 1. paired drift and b per condition --------------------------------------------------
    net["cond"] = net.run.str.extract(r"-c(\d+)-")[0].astype(int)
    for att in ("true", "ahrs", "chain"):
        w = net[net.att == att].pivot(index="cond", columns="sea_rate", values=["drift_pct", "b_ms", "path_m", "end_err_m"])
        assert np.allclose(w["path_m"][0.0], w["path_m"][1.0]) and np.allclose(w["path_m"][2.0], w["path_m"][1.0])
        for q in ("drift_pct", "b_ms"):
            for r in (0.0, 1.0, 2.0):
                describe("1 paired", f"{q} ({att})", w[q][r].values, rate=r, stats_=("median", "mean") if att == "true" else ("median",))
        # Within a condition path and duration are equal across rates, so drift, b and end-error ratios coincide.
        for r in (0.0, 2.0):
            ratio = (w["drift_pct"][r] / w["drift_pct"][1.0]).values
            x = np.stack([w["drift_pct"][r].values, w["drift_pct"][1.0].values], 1)
            note = f"rate{int(r)}/rate1, = b and end-error ratio (same path and duration per condition)"
            lo, hi = boot(x, lambda a: np.median(a[:, 0]) / np.median(a[:, 1]))
            add("1 ratio", f"drift ratio ({att})", "ratio of medians", float(np.median(x[:, 0]) / np.median(x[:, 1])), lo, hi, 30, rate=r, note=note)
            lo, hi = boot(ratio, gmean)
            wil = stats.wilcoxon(x[:, 0], x[:, 1], alternative="two-sided")
            add("1 ratio", f"drift ratio ({att})", "geometric mean of per-condition ratios", gmean(ratio), lo, hi, 30, p=wil.pvalue, rate=r,
                note=note + "; p = Wilcoxon signed-rank on paired drift")
            if att == "true":
                lo, hi = boot(ratio, np.median)
                add("1 ratio", f"drift ratio ({att})", "median of per-condition ratios", float(np.median(ratio)), lo, hi, 30, rate=r, note=note)
                add("1 ratio", f"drift ratio ({att})", "fraction of conditions worse than rate 1", float((ratio > 1).mean()), n=30, rate=r)
                d = (w["b_ms"][r] - w["b_ms"][1.0]).values
                lo, hi = boot(d, np.median)
                add("1 ratio", f"b_ms change ({att})", "median paired difference (m/s)", float(np.median(d)), lo, hi, 30,
                    p=stats.wilcoxon(d).pvalue, rate=r, note=f"rate{int(r)} - rate1")
        if att == "true":
            x = np.stack([w["drift_pct"][0.0].values, w["drift_pct"][2.0].values], 1)
            add("1 ratio", "drift rate0 vs rate2 (true)", "geometric mean ratio rate0/rate2", gmean(x[:, 0] / x[:, 1]),
                *boot(x[:, 0] / x[:, 1], gmean), 30, p=stats.wilcoxon(x[:, 0], x[:, 1]).pvalue)

    # Baselines are identical across rates (they do not see images): a check that the pairing is clean.
    for m in ("airspeed", "blind"):
        bw = pr[(pr.method == m) & (pr.att == "true")].assign(cond=lambda d: d.run.str.extract(r"-c(\d+)-")[0].astype(int))
        bw = bw.pivot(index="cond", columns="sea_rate", values="drift_pct")
        add("check", f"{m}: max |drift(rate r) - drift(rate 1)|", "max", float(np.abs(bw.values - bw[[1.0]].values).max()), n=30)

    # ---- 2. per step: error size and its direction --------------------------------------------
    tr.to_csv(OUT / "sea_rate_per_condition.csv", index=False)
    wide = tr.pivot(index="cond", columns="sea_rate")
    for q, label in [("step_err_m", "mean step error |dp_pred - dp_true| (m, as evaluate.py)"),
                     ("along_bias_pct_eval", "along-track bias % (as evaluate.py)"),
                     ("along_track_ms", "along-track mean velocity error (m/s, world horizontal)"),
                     ("cross_track_ms", "cross-track mean velocity error (m/s, + left)"),
                     ("along_wave_ms", "along-wave mean velocity error (m/s, + toward windDirection)"),
                     ("cross_wave_ms", "cross-wave mean velocity error (m/s)"),
                     ("along_swell_ms", "along-swell mean velocity error (m/s, + toward swellDirection)"),
                     ("bias_ms", "|mean horizontal step error| / DT (m/s)"),
                     ("step_scatter_m", "RMS step error about the run mean (m)")]:
        for r in (0.0, 1.0, 2.0):
            describe("2 per step", q, wide[q][r].values, rate=r, note=label)
        for r in (0.0, 2.0):
            d = (wide[q][r] - wide[q][1.0]).values
            lo, hi = boot(d, np.median)
            add("2 per step paired", q, "median paired difference", float(np.median(d)), lo, hi, 30, p=stats.wilcoxon(d).pvalue, rate=r,
                note=f"rate{int(r)} - rate1; {label}")
            lo, hi = boot(d, np.mean)
            add("2 per step paired", q, "mean paired difference", float(np.mean(d)), lo, hi, 30, rate=r, note=f"rate{int(r)} - rate1")
            if q in ("along_wave_ms", "along_track_ms", "along_swell_ms"):
                add("2 per step paired", q, "fraction of conditions with positive difference", float((d > 0).mean()), n=30, rate=r)

    # Direction of the rate-induced change: Δ = mean step-error vector at rate r minus rate 1 (same condition).
    for r in (0.0, 2.0):
        dE = (wide["bias_E_ms"][r] - wide["bias_E_ms"][1.0]).values
        dN = (wide["bias_N_ms"][r] - wide["bias_N_ms"][1.0]).values
        wd = np.radians(wide["wind_dir"][1.0].values)
        along = dE * np.sin(wd) + dN * np.cos(wd)
        cross = -dE * np.cos(wd) + dN * np.sin(wd)
        ang = np.arctan2(cross, along)                                     # angle of Δ relative to the wave direction
        R = np.abs(np.mean(np.exp(1j * ang)))
        mean_ang = np.degrees(np.angle(np.mean(np.exp(1j * ang))))
        nn = len(ang)
        p_ray = np.exp(np.sqrt(1 + 4 * nn + 4 * (nn**2 - (nn * R) ** 2)) - (1 + 2 * nn))   # Rayleigh test (Zar 27.4)
        add("2 direction", "angle of rate-induced change Δ relative to wave travel", "mean resultant length R", float(R),
            *boot(ang, lambda a: np.abs(np.mean(np.exp(1j * a)))), nn, p=p_ray, rate=r, note=f"mean angle {mean_ang:.1f} deg (0 = toward windDirection); p = Rayleigh")
        mag = np.hypot(along, cross)
        describe("2 direction", "|Δ| rate-induced change of mean step-error vector (m/s)", mag, rate=r)
        describe("2 direction", "Δ along wave (m/s)", along, rate=r)
        describe("2 direction", "|Δ cross wave| (m/s)", np.abs(cross), rate=r, stats_=("median",))
        add("2 direction", "median |Δ along| / median |Δ cross|", "ratio", float(np.median(np.abs(along)) / np.median(np.abs(cross))), n=nn, rate=r)
        # Does the change scale with the waves' speed (a dispersion prior would)?
        rho = stats.spearmanr(np.abs(along), wide["peak_phase_speed"][1.0].values)
        add("2 direction", "Spearman |Δ along| vs peak phase speed gT/2π", "rho", float(rho.statistic), n=nn, p=rho.pvalue, rate=r)
        rho = stats.spearmanr(mag, wide["altitude"][1.0].values)
        add("2 direction", "Spearman |Δ| vs altitude", "rho", float(rho.statistic), n=nn, p=rho.pvalue, rate=r)
        rho = stats.spearmanr(mag, wide["ground_speed"][1.0].values)
        add("2 direction", "Spearman |Δ| vs ground speed", "rho", float(rho.statistic), n=nn, p=rho.pvalue, rate=r)
    # A fixed rate-1 prior predicts Δ_along(r=2) = -Δ_along(r=0) per condition.
    a0 = (wide["along_wave_ms"][0.0] - wide["along_wave_ms"][1.0]).values
    a2 = (wide["along_wave_ms"][2.0] - wide["along_wave_ms"][1.0]).values
    rho = stats.spearmanr(a0, a2)
    add("2 direction", "Spearman Δ_along(rate0) vs Δ_along(rate2)", "rho", float(rho.statistic), n=30, p=rho.pvalue,
        note="a fixed rate-1 wave prior predicts a strong negative relation (Δ2 = -Δ0)")
    add("2 direction", "fraction with Δ_along(rate0) > 0 and Δ_along(rate2) < 0", "fraction", float(((a0 > 0) & (a2 < 0)).mean()), n=30)

    # Bias or noise: end error from the run-mean bias alone vs i.i.d. scatter alone.
    for r in (0.0, 1.0, 2.0):
        x = wide["end_err_bias_m"][r].values / wide["end_err_rw_m"][r].values
        describe("2 bias vs noise", "end error from run-mean bias / from i.i.d. scatter alone", x, rate=r, stats_=("median",))

    # ---- 3. against E2 (relative degradation only) -------------------------------------------
    wt = net[net.att == "true"].pivot(index="cond", columns="sea_rate", values="b_ms")
    e3_b = [float(np.median(wt[r])) for r in (0.0, 1.0, 2.0)]
    for r_i, r in ((0, 0.0), (2, 2.0)):
        x = np.stack([wt[r].values, wt[1.0].values], 1)
        lo, hi = boot(x, lambda a: np.median(a[:, 0]) / np.median(a[:, 1]))
        add("3 vs E2", "E3 net median b ratio", f"rate{int(r)}/rate1", e3_b[r_i] / e3_b[1], lo, hi, 30, rate=r, note="chained drift, true attitude, 20 s")
        aw = wide["along_wave_ms"]
        add("3 vs E2", "E3 net median signed along-wave error (m/s)", "median", float(np.median(aw[r])), *boot(aw[r].values, np.median), 30, rate=r)
        for name, v in E2.items():
            k = {0.0: 0, 2.0: 3}[r]
            add("3 vs E2", f"E2 {name} median b ratio", f"rate{int(r)}/rate1", v["b"][k] / v["b"][2], rate=r, note="from E2 RESULTS.md §2.4 (derived)")
            add("3 vs E2", f"E2 {name} median b increase (m/s)", f"rate{int(r)} - rate1", v["b"][k] - v["b"][2], rate=r, note="from E2 RESULTS.md §2.4 (derived)")
        add("3 vs E2", "E3 net median b increase (m/s)", f"rate{int(r)} - rate1 (of medians)", e3_b[r_i] - e3_b[1], rate=r)
    add("3 vs E2", "E3 net median along-wave error at rate 1 (m/s)", "median", float(np.median(wide["along_wave_ms"][1.0])),
        *boot(wide["along_wave_ms"][1.0].values, np.median), 30, rate=1.0)

    # ---- 4. test-rate rate 1 (20 s) against test-long (60 s) ---------------------------------
    tl, chains = step_table("test-long")
    plr = pd.read_csv(EVAL / "test-long/per_run.csv")
    chk = plr[(plr.method == "net") & (plr.att == "true")].set_index("run").loc[tl.run]
    dmax = float(np.abs(chk.drift_pct.values - tl.drift_pct.values).max())
    add("check", "max |drift_pct recomputed - per_run.csv| (test-long)", "max", dmax, n=len(tl), set_="test-long")
    assert dmax < 1e-6, dmax
    hz = []
    for rid, (cum, hpath) in chains.items():
        for k in (50, 100, 120, 200, 300, 400, 500, 600):
            e = np.linalg.norm(cum[: k + 1], axis=1)
            hz.append(dict(run=rid, horizon_s=k * DT, end_err_m=e[k], max_err_m=e.max(), path_m=hpath[k],
                           drift_pct=e[k] / hpath[k] * 100, max_drift_pct=e.max() / hpath[k] * 100))
    hz = pd.DataFrame(hz).merge(tl[["run", "pattern", "altitude", "ground_speed", "wind"]], on="run")
    hz.to_csv(OUT / "sea_rate_horizons.csv", index=False)
    r1 = tr[tr.sea_rate == 1.0]
    describe("4 length", "drift_pct test-rate rate1 (20 s)", r1.drift_pct.values, rate=1.0, stats_=("median",))
    for h in (5.0, 10.0, 12.0, 20.0, 30.0, 40.0, 50.0, 60.0):
        x = hz[hz.horizon_s == h]
        describe("4 length", f"drift_pct test-long truncated at {h:g} s", x.drift_pct.values, set_="test-long", stats_=("median",))
        describe("4 length", f"end_err_m test-long truncated at {h:g} s", x.end_err_m.values, set_="test-long", stats_=("median",))
    x20 = hz[hz.horizon_s == 20.0].drift_pct.values
    ra, rb = r1.drift_pct.values, x20
    bs = [np.median(RNG.choice(ra, len(ra))) / np.median(RNG.choice(rb, len(rb))) for _ in range(B)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    add("4 length", "median drift: test-rate rate1 / test-long first 20 s", "ratio", float(np.median(r1.drift_pct) / np.median(x20)), lo, hi,
        note="independent sets; bootstrap resamples each set's runs")
    mw = stats.mannwhitneyu(r1.drift_pct.values, x20)
    add("4 length", "test-rate rate1 vs test-long first 20 s drift", "Mann-Whitney U statistic", float(mw.statistic), n=len(x20), p=mw.pvalue)
    # Growth of the end error with time: err ∝ T^alpha. alpha = 1 for a constant bias, 0.5 for a random walk.
    e = hz.pivot(index="run", columns="horizon_s", values="end_err_m")
    alpha = np.log(e[60.0] / e[20.0]) / np.log(3)
    describe("4 length", "growth exponent alpha, end error 20 s -> 60 s (per run)", alpha.values, set_="test-long", stats_=("median",))
    add("4 length", "pooled exponent, median end error 20 s -> 60 s", "alpha", float(np.log(np.median(e[60.0]) / np.median(e[20.0])) / np.log(3)),
        *boot(e[[20.0, 60.0]].values, lambda a: np.log(np.median(a[:, 1]) / np.median(a[:, 0])) / np.log(3)), len(e), set_="test-long")
    m = hz.pivot(index="run", columns="horizon_s", values="max_err_m")
    alpha_m = np.log(m[60.0] / m[20.0]) / np.log(3)
    describe("4 length", "growth exponent of max error 20 s -> 60 s (per run)", alpha_m.values, set_="test-long", stats_=("median",))
    for pat, g in tl.groupby("pattern"):
        a = alpha.loc[g.run]
        add("4 length", f"alpha end error 20->60 s, pattern {pat}", "median", float(np.median(a)), *boot(a.values, np.median), len(a), set_="test-long")
    x = tl.end_err_bias_m / tl.end_err_rw_m
    describe("4 length", "end error from run-mean bias / from i.i.d. scatter alone (60 s)", x.values, set_="test-long", stats_=("median",))
    describe("4 length", "|mean step error| / DT over 60 s (m/s)", tl.bias_ms.values, set_="test-long", stats_=("median",))
    describe("4 length", "end error / steps*DT over 60 s = b (m/s)", tl.b_ms.values, set_="test-long", stats_=("median",))
    # Why closed patterns grow sublinearly: a bias fixed in the track frame (scale, side-slip) turns with
    # the aircraft and cancels round a loop; a bias fixed in the world (e.g. along the waves) does not.
    tl["track_frame_bias_ms"] = np.hypot(tl.along_track_ms, tl.cross_track_ms)
    for pat, g in tl.groupby("pattern"):
        add("4 length", f"median |track-frame mean error| vs |world-frame mean error| (m/s), pattern {pat}", "track / world",
            float(np.median(g.track_frame_bias_ms)), n=len(g), set_="test-long", note=f"world {np.median(g.bias_ms):.3f}")
    describe("4 length", "track-frame mean error / world-frame mean error (60 s, per run)", (tl.track_frame_bias_ms / tl.bias_ms).values,
             set_="test-long", stats_=("median",))
    add("2 per step", "peak band counts (rate 1)", "counts", np.nan, n=30, note=json.dumps(tr[tr.sea_rate == 1.0].peak_band.value_counts().to_dict()))
    # The two sets' conditions.
    for col in ("altitude", "ground_speed", "wind"):
        add("4 length", f"{col} median", "test-rate rate1 / test-long", float(np.median(r1[col])), note=f"test-long {np.median(tl[col]):.2f}")
    add("4 length", "pattern mix", "counts", np.nan, note="test-rate " + json.dumps(r1.pattern.value_counts().to_dict()) + "; test-long " + json.dumps(tl.pattern.value_counts().to_dict()))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sea_rate.csv", index=False)
    with pd.option_context("display.max_rows", 500, "display.width", 250, "display.max_colwidth", 70):
        print(df.drop(columns=["note"]).to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
