"""Where does the network's translation scale come from, and what limits its drift?

    uv run --no-project --with numpy --with pandas --with tabulate --with matplotlib \
        python analysis/scale_source.py [--data <e3data dir>] [--boot 4000]

For every run of val, test-long and test-rate, and for net, blind and the airspeed baseline:

  per-run translation errors, world frame with the TRUE attitude (horizontal plane):
    scale_h   = sum(dp_w,h . u_h) / sum|dp_true,w,h| - 1      (u_h = unit true horizontal step)
    cross_h   = sum(u_h x dp_w,h) / sum|dp_true,w,h|          (signed; ~ a mean heading offset, rad)
    cross_abs = sum|u_h x dp_w,h| / sum|dp_true,w,h|
    scale_3d  = sum(dp . u) / sum|dp_true| - 1                (camera axes, the task's definition)
  sensors, re-simulated with run_seed and the default noise (the draw the predictions used):
    baro_rel   = mean(baro - true camera height) / mean(true height)
    baro_scale = step-length weighted mean of (max(baro_i, 5) / h_i - 1): the scale error a
                 method gets if it outputs exact tenths of the baro height (full inheritance)
    air_scale  = sum(airspeed measured) / sum(airspeed true) - 1
  drift (true attitude, horizontal end error / horizontal path, %):
    drift          as e3.evaluate (checked against results/eval/<set>/per_run.csv)
    drift_noscale  the method's steps divided by (1 + scale_h): its own per-run scale removed
    drift_nobaro   the method's steps times h_i / max(baro_i, 5): the baro error undone step by step
    drift_noscale_nocross  also rotated by -cross_h about the vertical
    drift_baro_only        the TRUE steps times max(baro_i, 5) / h_i: the drift full inheritance
                           of the baro error would cause by itself
    scale_pred_drift = |scale_h| |p_end - p_0|_h / path: the drift the scale error alone implies

Writes results/analysis/scale_source.csv (one row per set, run, method) and
results/analysis/scale_source_summary.csv (regressions, drift decomposition, altitude bins),
and prints the summary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from e3.data import DT, NOISE, Run, list_runs, run_seed, simulate  # noqa: E402

DATA = Path("/private/tmp/claude-501/-Users-emanuel-repos-mayfly-demo/02eac86e-e939-4778-9996-84cb2607a222/scratchpad/e3data")
SETS = ("val", "test-long", "test-rate")
ALT_BINS = [0, 50, 80, 120, 1000]
ALT_LABELS = ["30-50", "50-80", "80-120", "120-200"]


def chain_drift(p, R, dp):
    """Horizontal end error / horizontal path (%), and max error / path (%), true attitude."""
    steps = np.einsum("nij,nj->ni", R, dp)
    p_hat = np.concatenate([p[:1], p[:1] + np.cumsum(steps, 0)])
    errs = np.linalg.norm(p_hat[:, :2] - p[:, :2], axis=1)
    path = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).sum()
    return errs[-1] / path * 100, errs.max() / path * 100


def rotz(dp_w, ang):
    c, s = np.cos(ang), np.sin(ang)
    out = dp_w.copy()
    out[:, 0] = c * dp_w[:, 0] - s * dp_w[:, 1]
    out[:, 1] = s * dp_w[:, 0] + c * dp_w[:, 1]
    return out


def per_run(set_name: str, data: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = [Run.load(d) for d in list_runs(data / "sets" / set_name)]
    preds = {m: np.load(data / "preds" / f"{m}-{set_name}.npz") for m in ("net", "blind")}
    for m, z in preds.items():
        assert list(z["runs"]) == [r.id for r in runs], (set_name, m)
    rows, steprows = [], []
    for ri, run in enumerate(runs):
        s = simulate(run, np.random.default_rng(run_seed(run)), NOISE)
        dp_true, _ = run.labels()
        n = run.n - 1
        h_true = run.p[:-1, 2]
        h_baro = np.maximum(s.baro[:-1], 5.0)
        k_baro = h_baro / h_true                      # factor full inheritance would apply per step
        R = run.R[:-1]
        dwt = np.einsum("nij,nj->ni", R, dp_true)
        lt = np.linalg.norm(dwt[:, :2], axis=1)
        u = dwt[:, :2] / lt[:, None]
        l3 = np.linalg.norm(dp_true, axis=1)
        u3 = dp_true / l3[:, None]
        disp = np.linalg.norm(run.p[-1, :2] - run.p[0, :2])
        path = lt.sum()
        baro_rel = np.mean(s.baro - run.p[:, 2]) / np.mean(run.p[:, 2])
        baro_scale = np.sum(lt * (k_baro - 1)) / path
        air_scale = s.airspeed.sum() / run.airspeed.sum() - 1
        air_white = s.airspeed[:-1] - run.airspeed[:-1] * (1 + air_scale)
        d_bo, _ = chain_drift(run.p, R, dp_true * k_baro[:, None])
        methods = {"airspeed": run.nose[None] * s.airspeed[:-1, None] * DT}
        for m, z in preds.items():
            sel = z["run"] == ri
            order = np.argsort(z["i"][sel])
            assert len(order) == n and (z["i"][sel][order] == np.arange(n)).all()
            methods[m] = z["dp_pred"][sel][order].astype(np.float64)
        for m, dp in methods.items():
            dw = np.einsum("nij,nj->ni", R, dp)
            along = (dw[:, :2] * u).sum(1)
            cross = u[:, 0] * dw[:, 1] - u[:, 1] * dw[:, 0]
            scale_h = along.sum() / path - 1
            cross_h = cross.sum() / path
            scale_3d = (dp * u3).sum() / l3.sum() - 1
            vert = (dw[:, 2] - dwt[:, 2]).sum() / path
            d, dmax = chain_drift(run.p, R, dp)
            d_ns, _ = chain_drift(run.p, R, dp / (1 + scale_h))
            d_nb, _ = chain_drift(run.p, R, dp / k_baro[:, None])
            # remove scale and the mean cross-track angle: rotate the world steps about z
            dw_c = rotz(dw / (1 + scale_h), -np.arctan2(cross_h, 1 + scale_h))
            I = np.broadcast_to(np.eye(3), R.shape)
            d_nsc, _ = chain_drift(run.p, I, dw_c)
            rows.append(dict(set=set_name, run=run.id, method=m, sea_rate=run.sea_rate,
                             pattern=run.meta["trajectory"]["pattern"]["type"], altitude=run.p[0, 2],
                             alt_mean=run.p[:, 2].mean(), airspeed=run.airspeed.mean(), ground_speed=path / (n * DT),
                             wind=run.meta["sea"]["windSpeed"], path_m=path, disp_m=disp, duration=n * DT,
                             baro_bias_m=np.mean(s.baro - run.p[:, 2]), baro_rel=baro_rel, baro_scale=baro_scale,
                             air_scale=air_scale, scale_h=scale_h, scale_3d=scale_3d, cross_h=cross_h,
                             cross_abs=np.abs(cross).sum() / path, vert_rel=vert,
                             drift=d, max_drift=dmax, drift_noscale=d_ns, drift_nobaro=d_nb,
                             drift_noscale_nocross=d_nsc, drift_baro_only=d_bo,
                             scale_pred_drift=abs(scale_h) * disp / path * 100))
            # per-step: does the output follow the baro's white noise within a run?
            steprows.append(pd.DataFrame(dict(set=set_name, run=run.id, method=m, sea_rate=run.sea_rate,
                                              lr=np.log(np.maximum(along, 1e-6) / lt), lk=np.log(k_baro),
                                              aw=air_white / np.maximum(run.airspeed[:-1], 1.0))))
    return pd.DataFrame(rows), pd.concat(steprows)


def ols(x, y):
    X = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


def boot_ols(df, xcols, ycol, rng, B):
    """Coefficients (const, *x) with bootstrap 95% CI over runs."""
    X = df[xcols].to_numpy()
    y = df[ycol].to_numpy()
    b = ols(X, y)
    yhat = np.column_stack([np.ones(len(y)), X]) @ b
    r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    bs = np.empty((B, len(b)))
    n = len(y)
    for k in range(B):
        i = rng.integers(0, n, n)
        bs[k] = ols(X[i], y[i])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return b, lo, hi, r2


def boot_stat(x, f, rng, B):
    x = np.asarray(x)
    n = len(x)
    v = np.array([f(x[rng.integers(0, n, n)]) for _ in range(B)])
    return f(x), *np.percentile(v, [2.5, 97.5])


def spearman(a, b):
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return np.corrcoef(ra, rb)[0, 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--boot", type=int, default=4000)
    a = ap.parse_args()
    rng = np.random.default_rng(20261031)
    frames, steps = zip(*(per_run(s, a.data) for s in SETS))
    df = pd.concat(frames, ignore_index=True)
    st = pd.concat(steps, ignore_index=True)
    out = ROOT / "results" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "scale_source.csv", index=False)

    # check: our drift reproduces e3.evaluate's true-attitude drift
    for s in SETS:
        ev = pd.read_csv(ROOT / "results" / "eval" / s / "per_run.csv")
        ev = ev[ev.att == "true"].set_index(["run", "method"])["drift_pct"]
        mine = df[df.set == s].set_index(["run", "method"])["drift"]
        print(f"check {s}: max |drift - evaluate| = {np.abs(mine - ev.reindex(mine.index)).max():.2e} %")

    summ = []

    def add(**k):
        summ.append(k)

    B = a.boot
    groups = [("val", 1.0), ("test-long", 1.0), ("test-rate", 0.0), ("test-rate", 1.0), ("test-rate", 2.0)]
    for s, rate in groups:
        for m in ("net", "blind", "airspeed"):
            g = df[(df.set == s) & (df.method == m) & (df.sea_rate == rate)].reset_index(drop=True)
            tag = dict(set=s, sea_rate=rate, method=m, runs=len(g))
            # 3. scale error vs the baro-implied scale error (and with airspeed's)
            for ycol in ("scale_h", "scale_3d"):
                b, lo, hi, r2 = boot_ols(g, ["baro_scale"], ycol, rng, B)
                add(**tag, analysis=f"{ycol} ~ baro_scale", term="slope", value=b[1], lo=lo[1], hi=hi[1], r2=r2)
                add(**tag, analysis=f"{ycol} ~ baro_scale", term="intercept", value=b[0], lo=lo[0], hi=hi[0], r2=r2)
            b, lo, hi, r2 = boot_ols(g, ["baro_scale", "air_scale"], "scale_h", rng, B)
            for j, t in enumerate(("intercept", "baro_slope", "air_slope")):
                add(**tag, analysis="scale_h ~ baro_scale + air_scale", term=t, value=b[j], lo=lo[j], hi=hi[j], r2=r2)
            b, lo, hi, r2 = boot_ols(g, ["baro_rel"], "scale_h", rng, B)
            add(**tag, analysis="scale_h ~ baro_rel", term="slope", value=b[1], lo=lo[1], hi=hi[1], r2=r2)
            # spread of the scale errors
            for col in ("scale_h", "cross_h", "baro_scale", "air_scale"):
                v, lo, hi = boot_stat(g[col].abs(), np.median, rng, B)
                add(**tag, analysis=f"median |{col}|", term="", value=v, lo=lo, hi=hi)
            v, lo, hi = boot_stat(g["scale_h"], np.mean, rng, B)
            add(**tag, analysis="mean scale_h", term="", value=v, lo=lo, hi=hi)
            v, lo, hi = boot_stat(g["scale_h"], np.std, rng, B)
            add(**tag, analysis="sd scale_h", term="", value=v, lo=lo, hi=hi)
            # 4. drift decomposition
            for col in ("drift", "drift_noscale", "drift_nobaro", "drift_noscale_nocross", "drift_baro_only", "scale_pred_drift"):
                v, lo, hi = boot_stat(g[col], np.median, rng, B)
                add(**tag, analysis=f"median {col}", term="", value=v, lo=lo, hi=hi)
            idx = np.arange(len(g))
            for col in ("drift_noscale", "drift_nobaro", "drift_noscale_nocross"):
                f = lambda i, c=col: np.median(g[c].to_numpy()[i] / g["drift"].to_numpy()[i])
                v = f(idx)
                bs = [f(rng.integers(0, len(g), len(g))) for _ in range(B)]
                add(**tag, analysis=f"median {col}/drift (paired)", term="", value=v, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5))
            for xcol in ("baro_scale", "scale_h", "air_scale"):
                f = lambda i, c=xcol: spearman(g[c].abs().to_numpy()[i], g["drift"].to_numpy()[i])
                v = f(idx)
                bs = [f(rng.integers(0, len(g), len(g))) for _ in range(B)]
                add(**tag, analysis=f"spearman drift vs |{xcol}|", term="", value=v, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5))
            # 5. altitude bins
            g["alt_bin"] = pd.cut(g.alt_mean, ALT_BINS, labels=ALT_LABELS)
            for lab, gb in g.groupby("alt_bin", observed=True):
                for col in ("drift", "drift_noscale", "drift_nobaro", "drift_baro_only"):
                    v, lo, hi = boot_stat(gb[col], np.median, rng, B)
                    add(**tag, analysis=f"alt {lab}: median {col}", term=f"n={len(gb)}", value=v, lo=lo, hi=hi)
                for col in ("scale_h", "baro_scale"):
                    v, lo, hi = boot_stat(gb[col].abs(), np.median, rng, B)
                    add(**tag, analysis=f"alt {lab}: median |{col}|", term=f"n={len(gb)}", value=v, lo=lo, hi=hi)
            # per step, within run: does the output follow the baro's white noise?
            sg = st[(st.set == s) & (st.method == m) & (st.sea_rate == rate)].copy()
            for c in ("lr", "lk", "aw"):
                sg[c] = sg[c] - sg.groupby("run")[c].transform("mean")
            runs_ = sg.run.unique()
            by = {r: x for r, x in sg.groupby("run")}

            def slope(rs):
                x = pd.concat([by[r] for r in rs])
                X = np.column_stack([x.lk, x.aw])
                bb, *_ = np.linalg.lstsq(X, x.lr.to_numpy(), rcond=None)
                return bb
            b0 = slope(runs_)
            bs = np.array([slope(runs_[rng.integers(0, len(runs_), len(runs_))]) for _ in range(min(B, 1000))])
            for j, t in enumerate(("baro_white_slope", "air_white_slope")):
                add(**tag, analysis="within-run log(step along/true) ~ baro + airspeed white noise", term=t, value=b0[j],
                    lo=np.percentile(bs[:, j], 2.5), hi=np.percentile(bs[:, j], 97.5))

    sm = pd.DataFrame(summ)
    sm.to_csv(out / "scale_source_summary.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 2000, "display.max_colwidth", 70)
    print(sm.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
