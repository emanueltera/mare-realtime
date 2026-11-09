"""Third verification round on RESULTS.md (scale by altitude, altitude channels, sea-rate mechanism).
Writes results/analysis/verify_round3.csv.

    uv run --no-project --with numpy --with pandas --with scipy --with tabulate --with matplotlib \
        python analysis/verify_round3.py

Run from experiments/e3_relative_pose. Bootstrap over runs (or test-rate conditions), 2000 draws,
fixed seed. Reads scale_source.csv, sea_rate_per_condition.csv and the test-rate run tables.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from e3.data import Run, list_runs  # noqa: E402

SCRATCH = Path("/private/tmp/claude-501/-Users-emanuel-repos-mayfly-demo/02eac86e-e939-4778-9996-84cb2607a222/scratchpad/e3data")
A = HERE.parent / "results" / "analysis"
RNG = np.random.default_rng(20261031)
BINS = [(0, 50), (50, 80), (80, 120), (120, 1000)]
rows = []


def add(finding, check, value, lo=np.nan, hi=np.nan, n=np.nan, p=np.nan, note=""):
    rows.append(dict(finding=finding, check=check, value=value, lo=lo, hi=hi, n=n, p=p, note=note))
    print(f"[{finding}] {check}: {value:.4g}" + (f" [{lo:.4g}, {hi:.4g}]" if np.isfinite(lo) else "")
          + (f" n={n}" if np.isfinite(n) else "") + (f" p={p:.3g}" if np.isfinite(p) else "") + (f"  {note}" if note else ""))


def boot(fn, n, B=2000):
    v = np.array([fn(RNG.integers(0, n, n)) for _ in range(B)])
    return np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)


def ols(x, y):
    return np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]


def r2(x, y):
    b = ols(x, y)
    e = y - (b[0] + b[1] * x)
    return 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()


ss = pd.read_csv(A / "scale_source.csv")
for s in ("test-long", "val"):
    g = ss[(ss.set == s) & (ss.method == "net")].reset_index(drop=True)
    x, y, alt = g.baro_scale.to_numpy(), g.scale_h.to_numpy(), g.alt_mean.to_numpy()
    # F1: is the bias independent of altitude; slope, R2 and baro_scale spread per bin
    r = stats.spearmanr(g.baro_bias_m, alt)
    add("F1", f"{s} spearman baro_bias_m vs alt_mean", r.statistic, n=len(g), p=r.pvalue)
    for lo_, hi_ in BINS:
        m = (alt > lo_) & (alt <= hi_)
        add("F1", f"{s} slope scale_h~baro_scale alt {lo_}-{hi_}", ols(x[m], y[m])[1], n=int(m.sum()),
            note=f"R2 {r2(x[m], y[m]):.2f}; SD baro_scale {100 * x[m].std(ddof=1):.2f} %")
    # F2: drop runs whose baro height sits near or outside the 30-200 m training prior
    hb = g.alt_mean + g.baro_bias_m
    for lo_e, hi_e in ((30, 200), (33, 180), (35, 175)):
        keep = ((hb > lo_e) & (hb < hi_e)).to_numpy()
        add("F2", f"{s} slope with baro height in ({lo_e}, {hi_e}) only", ols(x[keep], y[keep])[1],
            n=int(keep.sum()), note=f"dropped {int((~keep).sum())}")
    # F3: altitude channels besides the baro
    b = ols(x, y)
    res = y - (b[0] + b[1] * x)
    for nm, v in (("|scale residual after baro fit|", np.abs(res)), ("signed scale residual", res),
                  ("cross_abs", g.cross_abs.to_numpy()), ("|cross_h|", np.abs(g.cross_h.to_numpy())),
                  ("drift_noscale", g.drift_noscale.to_numpy())):
        r = stats.spearmanr(v, alt)
        add("F3", f"{s} spearman {nm} vs alt_mean", r.statistic, *boot(lambda i: stats.spearmanr(v[i], alt[i]).statistic, len(g)),
            n=len(g), p=r.pvalue)

# F4: does the rate-induced along-wave change scale with wave speed; wind sea against swell
pc = pd.read_csv(A / "sea_rate_per_condition.csv")
w = pc.pivot(index="cond", columns="sea_rate")
c = w["peak_phase_speed"][1.0].to_numpy()
add("F4", "median peak phase speed (m/s)", np.median(c), n=len(c), note=f"range {c.min():.1f} to {c.max():.1f}")
wd, sd = np.radians(w["wind_dir"][1.0].to_numpy()), np.radians(w["swell_dir"][1.0].to_numpy())
diff = np.degrees(np.abs(np.angle(np.exp(1j * (wd - sd)))))
big = diff > 60
for r_ in (0.0, 2.0):
    dE = (w["bias_E_ms"][r_] - w["bias_E_ms"][1.0]).to_numpy()
    dN = (w["bias_N_ms"][r_] - w["bias_N_ms"][1.0]).to_numpy()
    aw = dE * np.sin(wd) + dN * np.cos(wd)
    asw = dE * np.sin(sd) + dN * np.cos(sd)
    rr = stats.spearmanr(aw, c)
    add("F4", f"rate {r_:g}: spearman signed Δ along wind vs peak phase speed", rr.statistic, n=30, p=rr.pvalue)
    rr = stats.spearmanr(np.abs(aw), c)
    add("F4", f"rate {r_:g}: spearman |Δ along wind| vs peak phase speed", rr.statistic, n=30, p=rr.pvalue)
    add("F4", f"rate {r_:g}: median |Δ along wind| / median phase speed", np.median(np.abs(aw)) / np.median(c), n=30)
    add("F4", f"rate {r_:g}: median Δ along wind, wind-swell > 60°", np.median(aw[big]), n=int(big.sum()),
        note=f"min {aw[big].min():.2f} max {aw[big].max():.2f}")
    add("F4", f"rate {r_:g}: median Δ along swell, wind-swell > 60°", np.median(asw[big]), n=int(big.sum()),
        note=f"min {asw[big].min():.2f} max {asw[big].max():.2f}")

# F5: are flight headings related to the wave direction (would mix along-wave and along-track)?
cosv = []
for d in list_runs(SCRATCH / "sets" / "test-rate"):
    run = Run.load(d)
    if run.sea_rate != 1.0:
        continue
    st = np.diff(run.p[:, :2], axis=0)
    st = st / np.linalg.norm(st, axis=1)[:, None]
    a = np.radians(run.meta["sea"]["windDirection"])
    cosv.append(float((st @ np.array([np.sin(a), np.cos(a)])).mean()))
cosv = np.array(cosv)
add("F5", "mean over conditions of mean cos(track, wind-sea direction)", cosv.mean(), *boot(lambda i: cosv[i].mean(), len(cosv)), n=len(cosv),
    note=f"median |cos| {np.median(np.abs(cosv)):.2f}")

pd.DataFrame(rows).to_csv(A / "verify_round3.csv", index=False)
