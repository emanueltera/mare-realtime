"""Adversarial checks of RESULTS.md claims. Writes results/analysis/verify_claims.csv."""
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
A = ROOT / "results" / "analysis"
rng = np.random.default_rng(7)
B = 2000
rows = []
def add(check, value, lo=np.nan, hi=np.nan, n=np.nan, note=""):
    rows.append(dict(check=check, value=value, lo=lo, hi=hi, n=n, note=note)); print(f"{check}: {value:.4g} [{lo:.4g}, {hi:.4g}] n={n} {note}")
def ols_slope(x, y):
    X = np.column_stack([np.ones(len(x)), x]); return np.linalg.lstsq(X, y, rcond=None)[0]
def boot(f, n):
    v = [f(rng.integers(0, n, n)) for _ in range(B)]; return np.nanpercentile(v, [2.5, 97.5])
def spear(a, b):
    return np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1]

ss = pd.read_csv(A / "scale_source.csv")
# 1. scale slope by altitude bin
for s, rate in (("test-long", 1.0), ("val", 1.0)):
    g = ss[(ss.set == s) & (ss.method == "net") & (ss.sea_rate == rate)].reset_index(drop=True)
    for lo_, hi_ in ((0, 50), (50, 80), (80, 120), (120, 1000), (0, 80), (80, 1000)):
        q = g[(g.alt_mean > lo_) & (g.alt_mean <= hi_)].reset_index(drop=True)
        x, y = q.baro_scale.to_numpy(), q.scale_h.to_numpy()
        b = ols_slope(x, y)[1]; ci = boot(lambda i: ols_slope(x[i], y[i])[1], len(q))
        add(f"{s} net slope scale_h~baro_scale alt {lo_}-{hi_}", b, *ci, len(q))
    # interaction with log altitude
    X = np.column_stack([np.ones(len(g)), g.baro_scale, g.baro_scale * np.log(g.alt_mean / 80)])
    f = lambda i: np.linalg.lstsq(X[i], g.scale_h.to_numpy()[i], rcond=None)[0][2]
    add(f"{s} net interaction slope baro_scale*log(alt/80)", f(np.arange(len(g))), *boot(f, len(g)), len(g))
    # weighted share: fraction of sum(x-xbar)^2 from runs below 50 m
    x = g.baro_scale.to_numpy(); w = (x - x.mean()) ** 2
    add(f"{s} share of baro_scale leverage from runs <=50 m", w[g.alt_mean <= 50].sum() / w.sum(), n=len(g))
    # runs whose baro reads below the 30 m training floor
    add(f"{s} runs with min baro_rel*alt below 30 m (approx alt+bias<30)", float(((g.alt_mean + g.baro_bias_m) < 30).sum()), n=len(g))
    # air slope vs shortfall 1-baro slope
    X2 = np.column_stack([np.ones(len(g)), g.baro_scale, g.air_scale]); y = g.scale_h.to_numpy()
    f2 = lambda i: (np.linalg.lstsq(X2[i], y[i], rcond=None)[0])
    bb = f2(np.arange(len(g))); bs = np.array([f2(rng.integers(0, len(g), len(g))) for _ in range(B)])
    d = bs[:, 2] - (1 - bs[:, 1])
    add(f"{s} air_slope - (1 - baro_slope)", bb[2] - (1 - bb[1]), *np.percentile(d, [2.5, 97.5]), len(g), "0 inside => airspeed could account for the non-inherited share")
    add(f"{s} SD air_scale / SD baro_scale", g.air_scale.std() / g.baro_scale.std(), n=len(g))

# 2. oracle per-run correction by route type (test-long, net)
g = ss[(ss.set == "test-long") & (ss.method == "net")].reset_index(drop=True)
for lab, q in (("straight-ish disp/path>0.9", g[g.disp_m / g.path_m > 0.9]), ("closed disp/path<0.5", g[g.disp_m / g.path_m < 0.5])):
    r = (q.drift_noscale_nocross / q.drift).to_numpy()
    add(f"test-long net oracle scale+cross ratio, {lab}", np.median(r), *boot(lambda i: np.median(r[i]), len(r)), len(r))
v = ss[(ss.set == "val") & (ss.method == "net")]
add("val median disp/path", float(np.median(v.disp_m / v.path_m)), n=len(v))

# 3. blind scale error vs airspeed-baseline scale error (wind proxy)
for s in ("test-long", "val"):
    bl = ss[(ss.set == s) & (ss.method == "blind")].set_index("run"); ar = ss[(ss.set == s) & (ss.method == "airspeed")].set_index("run")
    a, b = bl.scale_h.to_numpy(), ar.scale_h.reindex(bl.index).to_numpy()
    add(f"{s} pearson blind scale_h vs airspeed-baseline scale_h", np.corrcoef(a, b)[0, 1], *boot(lambda i: np.corrcoef(a[i], b[i])[0, 1], len(a)), len(a))

# 4. quantile ratios baseline/net per attitude, test-long
pr = pd.read_csv(ROOT / "results/eval/test-long/per_run.csv")
qs = np.linspace(0.02, 0.98, 49)
for att in ("true", "ahrs", "chain"):
    n_ = pr[(pr.method == "net") & (pr.att == att)].drift_pct
    for m in ("blind", "airspeed"):
        r = np.quantile(pr[(pr.method == m) & (pr.att == att)].drift_pct, qs) / np.quantile(n_, qs)
        add(f"test-long {att} min over quantiles 2-98% of {m}/net drift", r.min(), n=120)
    add(f"test-long {att} net max drift", n_.max(), n=120); add(f"test-long {att} blind min drift", pr[(pr.method == 'blind') & (pr.att == att)].drift_pct.min(), n=120)

# 5. baselines vs wind / ground speed (fig claim)
for m in ("airspeed", "blind", "net"):
    q = pr[(pr.method == m) & (pr.att == "true")].reset_index(drop=True)
    for c in ("wind", "ground_speed", "altitude"):
        x, y = q[c].to_numpy(), q.drift_pct.to_numpy()
        add(f"test-long {m} spearman drift vs {c}", spear(x, y), *boot(lambda i: spear(x[i], y[i]), len(q)), len(q))

# 6. altitude partial: difference rho(alt) - rho(alt | |baro_rel|)
ds = pd.read_csv(A / "drift_per_run_structure.csv"); ds = ds[ds.method == "net"].reset_index(drop=True)
def partial(x, y, z):
    rx, ry, rz = (pd.Series(v).rank().to_numpy() for v in (x, y, z))
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz); ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return np.corrcoef(ex, ey)[0, 1]
x, y, z = ds.altitude.to_numpy(), ds.drift_pct.to_numpy(), ds.abs_baro_bias_rel_pct.to_numpy()
f = lambda i: spear(x[i], y[i]) - partial(x[i], y[i], z[i])
add("test-long rho(alt,drift) - partial rho(alt,drift | |baro_rel|)", f(np.arange(len(ds))), *boot(f, len(ds)), len(ds))
add("test-long spearman altitude vs |baro_rel|", spear(x, z), n=len(ds))

# 7. random-noise share under AR(1) with the observed lag-1 autocorrelation
ratio = (ds.end_err_m / ds.rw_all_m).to_numpy(); rho1 = ds.acf1.median()
infl = (1 + rho1) / (1 - rho1)
add("median |E|/rw_all", np.median(ratio), n=len(ds)); add("median acf1, acf10, acf50", rho1, ds.acf10.median(), ds.acf50.median(), len(ds))
add("AR(1)-adjusted random share of end-error energy (infl/ratio^2)", infl / np.median(ratio) ** 2, n=len(ds), note=f"variance inflation {(infl):.2f}")

# 8. bad tail composition
thr = np.percentile(ds.drift_pct, 90); tail = ds[ds.drift_pct > thr]
print("tail patterns:", tail.pattern.value_counts().to_dict(), "all:", ds.pattern.value_counts().to_dict())

# 9. test-rate: wind vs swell direction, and delta angle vs swell
pc = pd.read_csv(A / "sea_rate_per_condition.csv")
w = pc.pivot(index="cond", columns="sea_rate")
dd = np.radians(w["wind_dir"][1.0] - w["swell_dir"][1.0])
add("test-rate mean |angle wind-swell| deg", float(np.degrees(np.abs(np.angle(np.exp(1j * dd)))).mean()), n=30)
add("test-rate median |angle wind-swell| deg", float(np.median(np.degrees(np.abs(np.angle(np.exp(1j * dd)))))), n=30)
for r in (0.0, 2.0):
    dE = (w["bias_E_ms"][r] - w["bias_E_ms"][1.0]).to_numpy(); dN = (w["bias_N_ms"][r] - w["bias_N_ms"][1.0]).to_numpy()
    for nm, col in (("wind", "wind_dir"), ("swell", "swell_dir")):
        d = np.radians(w[col][1.0].to_numpy()); al = dE * np.sin(d) + dN * np.cos(d)
        add(f"rate {r:g}: median paired change along {nm} dir (m/s)", np.median(al), *boot(lambda i: np.median(al[i]), 30), 30)
    for band in ("wind", "swell"):
        m = (w["peak_band"][1.0] == band).to_numpy()
        al = (dE * np.sin(np.radians(w["wind_dir"][1.0])) + dN * np.cos(np.radians(w["wind_dir"][1.0]))).to_numpy()[m]
        add(f"rate {r:g}: median change along wind dir, peak band={band}", np.median(al), *boot(lambda i: np.median(al[i]), m.sum()), int(m.sum()))
    x = w["peak_phase_speed"][1.0].to_numpy(); al = dE * np.sin(np.radians(w["wind_dir"][1.0])) + dN * np.cos(np.radians(w["wind_dir"][1.0]))
    al = np.asarray(al); sgn = 1 if r == 0 else -1
    add(f"rate {r:g}: spearman signed change*sgn vs peak phase speed", spear(sgn * al, x), *boot(lambda i: spear(sgn * al[i], x[i]), 30), 30)
# absolute E3 per-condition paired b change
for r in (0.0, 2.0):
    d = (w["b_ms"][r] - w["b_ms"][1.0]).to_numpy()
    add(f"test-rate paired b change rate {r:g} - 1 (median, m/s)", np.median(d), *boot(lambda i: np.median(d[i]), 30), 30)
pd.DataFrame(rows).to_csv(A / "verify_claims.csv", index=False)
