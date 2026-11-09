"""E1 analysis: per-run velocity bias for every estimator, a summary by slice, and plots.

Inputs (results/, pulled from the GPU box):
  flow-{core,frozen,oblique}.csv    flow_bias.py run   (one row per model × frame pair)
  shell-{core,frozen,oblique}.csv   shell_fit.py       (one row per model × window)

Conventions
- Velocity bias b = v_est − v_true (m/s, ENU). For the flow models v_est = v_true − a (a static-sea
  estimator), so b = −a, where a is the apparent pattern velocity over the sea. For the shell fit
  b = V_est − V_true directly ('dispersive': waves modelled; 'static': ω' = −k·V, a spectral
  pattern tracker).
- Per run, b is the component-wise MEDIAN over pairs (flow; pairs flagged by
  flow_bias.is_failed are excluded and counted) or over windows (shell). |b| is the norm of that
  median vector; along/cross-track, along-wave (toward the WIND SEA's travel direction, 60° in
  every E1 preset) and along-swell (toward the swell's travel direction: 75° moderate-swell, 140°
  cross-swell; NaN for seas without swell) components are its projections. Sea directions are
  compass degrees the waves travel toward (src/ocean/scene.js).
  % of speed = 100·|b| / ground speed.
- Budget lines: 0.28 % of speed (the RSS share for water motion, 0.17 m/s at 60 m/s) and 0.5 %
  (the whole drift budget).

Derived comparison (not measured): deep-water phase speed c(λ) = sqrt(g·λ / 2π) of the shortest
wavelengths the camera resolves, λ = n px × GSD with GSD = cam_u / f at nadir (f = 618.04 px),
and foam drift 0.035·U10 (src/ocean/engine/waves.js). From a measured |a| the equivalent
wavelength is λ_eq = 2π·|a|² / g, given here in ground pixels (λ_eq / GSD).

Usage:  python analyze.py [--results results]
Outputs: results/per_run.csv, results/summary_by_slice.csv, results/phase_speed.csv,
         results/altitude_exponent.csv, results/decomp_by_layer.csv, results/audit_checks.csv
         (the numbers RESULTS.md's audit corrections quote), results/fig_*.png
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from flow_bias import is_failed  # noqa: E402

G = 9.81
F_PX = 618.0408  # (1280/2)/tan(46°), checked against frames.csv
WAVE_DIR = 60.0  # wind-sea travel direction ('toward') of every E1 preset (run.json windDirection)
# Swell travel direction ('toward', run.json swellDirection) of the presets that have a swell. Used
# when a CSV predates the wind_dir / swell_dir columns that flow_bias.py and shell_fit.py now write.
SWELL_DIR = {"moderate-swell": 75.0, "cross-swell": 140.0}
FOAM_K = 0.035  # foam drift = FOAM_K · U10 (waves.js)
GATE_SPEED = 1.98  # whitecap break-gate scroll, m/s toward 59° (engine.js)
BUDGET_RSS = 0.28  # % of speed
BUDGET_ALL = 0.5  # % of speed
BUDGET_MS = 0.17  # m/s (RSS share at 60 m/s)

SETS = ["core", "frozen", "oblique", "decomp-shaded", "decomp-normals", "decomp-foam"]
LAYERS = ["shaded", "normals", "foam"]
SEAS = ["calm", "moderate", "fresh", "strong", "moderate-swell", "cross-swell"]
HEAD_NAME = {60: "down-wave (60)", 150: "cross (150)", 240: "up-wave (240)"}
FLOW_MODELS = ["searaft", "searaft_half", "ufm", "ufmrefine980"]
FLOW_LABEL = {"searaft": "SEA-RAFT-M full res", "searaft_half": "SEA-RAFT-M half res",
              "ufm": "UFM-Base (560x420)", "ufmrefine980": "UFM-Refine-980"}
# fixed categorical order (dataviz reference palette, light mode)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def unit(bearing):
    r = math.radians(bearing)
    return np.array([math.sin(r), math.cos(r)])


def project(bE, bN, heading, wave_dir=WAVE_DIR, swell_dir=np.nan):
    """Along-track, cross-track (+ right of track), along the wind sea and along the swell
    (direction arrays or scalars, compass degrees toward; NaN swell_dir gives NaN)."""
    h = np.radians(np.asarray(heading, dtype=float))
    al = bE * np.sin(h) + bN * np.cos(h)
    cr = bE * np.sin(h + np.pi / 2) + bN * np.cos(h + np.pi / 2)  # + = right of track
    w = np.radians(np.asarray(wave_dir, dtype=float))
    sw = np.radians(np.asarray(swell_dir, dtype=float))
    return al, cr, bE * np.sin(w) + bN * np.cos(w), bE * np.sin(sw) + bN * np.cos(sw)


META = ["sea", "light", "altitude", "cam_u", "airspeed", "heading", "pitch", "wind_speed", "hs", "sea_rate",
        "wind_dir", "swell_dir"]


def load_flow(res: Path, s: str) -> pd.DataFrame | None:
    p = res / f"flow-{s}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d["failed"] = is_failed(d.a_E, d.a_N, d.cam_vE, d.cam_vN, d.spread_E, d.spread_N)
    d["param"] = "g=" + d.g.astype(str)
    d["method"] = "flow:" + d.model
    ok = d[~d.failed]
    keys = ["run_id", "method", "model", "param", "g"]
    agg = ok.groupby(keys).agg(a_E=("a_E", "median"), a_N=("a_N", "median"), n=("a_E", "size"),
                               ground_speed=("ground_speed", "median"), valid_frac=("valid_frac", "median"),
                               spread=("spread_E", "median"),
                               a_abs_pair_med=("a_abs", "median"))
    meta = d.groupby("run_id")[[c for c in META if c in d.columns]].first()
    nf = d.groupby(keys).failed.sum().rename("n_failed")
    ntot = d.groupby(keys).size().rename("n_pairs")
    out = pd.concat([ntot, nf], axis=1).join(agg).reset_index().join(meta, on="run_id")
    out["n"] = out.n.fillna(0).astype(int)
    out["b_E"], out["b_N"] = -out.a_E, -out.a_N
    out["estimator"] = "flow"
    out["T"] = np.nan
    out["set"] = s
    return out


SHELL_WEIGHTS = ["energy", "sqrt"]  # optional reruns beside the default 'whiten' (shell-<set>-<weight>.csv)


def load_shell(res: Path, s: str, weight: str | None = None) -> pd.DataFrame | None:
    """Per-run shell values. weight=None reads the default ('whiten') file shell-<set>.csv, as
    method shell:<model>; a weight reads shell-<set>-<weight>.csv, as method shell:<model>/<weight>."""
    p = res / (f"shell-{s}.csv" if weight is None else f"shell-{s}-{weight}.csv")
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d["method"] = "shell:" + d.model + ("" if weight is None else f"/{weight}")
    d["param"] = "T=" + d["T"].map(lambda t: f"{t:g}")
    keys = ["run_id", "method", "model", "param", "T"]
    agg = d.groupby(keys).agg(b_E=("err_e", "median"), b_N=("err_n", "median"), n=("err_e", "size"),
                              b_E_mean=("err_e", "mean"), b_N_mean=("err_n", "mean"),
                              rms_window=("err_abs", lambda x: float(np.sqrt(np.mean(np.square(x))))),
                              ground_speed=("speed", "median"), lambda_c=("lambda_c", "median"),
                              at_edge=("at_edge", "mean"), shell_frac=("shell_frac", "median"),
                              n_pairs=("err_e", "size"))
    meta = d.groupby("run_id")[[c for c in META if c in d.columns]].first()
    out = agg.reset_index().join(meta, on="run_id")
    out["n_failed"] = 0
    out["estimator"] = "shell"
    out["g"] = np.nan
    out["set"] = s
    return out


def per_run(res: Path) -> pd.DataFrame:
    parts = []
    for s in SETS:
        for f in (load_flow, load_shell):
            x = f(res, s)
            if x is not None:
                parts.append(x)
        for w in SHELL_WEIGHTS:
            x = load_shell(res, s, w)
            if x is not None:
                parts.append(x)
    df = pd.concat(parts, ignore_index=True)
    df["b_abs"] = np.hypot(df.b_E, df.b_N)
    df["wind_dir"] = df["wind_dir"].fillna(WAVE_DIR) if "wind_dir" in df else WAVE_DIR
    sw_preset = df.sea.map(SWELL_DIR).astype(float)
    df["swell_dir"] = df["swell_dir"].fillna(sw_preset) if "swell_dir" in df else sw_preset
    df["b_along"], df["b_cross"], df["b_along_wave"], df["b_along_swell"] = project(
        df.b_E.values, df.b_N.values, df.heading.values, df.wind_dir.values, df.swell_dir.values)
    df["b_pct"] = 100 * df.b_abs / df.ground_speed
    df["b_along_pct"] = 100 * df.b_along / df.ground_speed
    # bearing of the apparent pattern velocity a = −b (deg, from north, clockwise)
    df["a_bearing"] = np.degrees(np.arctan2(-df.b_E, -df.b_N)) % 360
    df["heading_to_waves"] = df.heading.map(HEAD_NAME)
    df["gsd_m"] = df.cam_u / F_PX
    df["foam_drift"] = FOAM_K * df.wind_speed
    return df


SLICES = [
    ("all", []), ("sea", ["sea"]), ("altitude", ["altitude"]), ("airspeed", ["airspeed"]),
    ("heading", ["heading"]), ("light", ["light"]), ("sea x altitude", ["sea", "altitude"]),
    ("altitude x airspeed", ["altitude", "airspeed"]), ("sea x heading", ["sea", "heading"]),
    ("altitude x heading", ["altitude", "heading"]),
]


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (s, method, param), grp in df.groupby(["set", "method", "param"]):
        for sname, keys in SLICES:
            groups = [((), grp)] if not keys else grp.groupby(keys)
            for kv, g_all in groups:
                kv = kv if isinstance(kv, tuple) else (kv,)
                g = g_all[np.isfinite(g_all.b_abs)]
                if not len(g):
                    continue
                r = {"set": s, "method": method, "param": param, "slice": sname,
                     "value": " | ".join(f"{k}={v}" for k, v in zip(keys, kv)) or "all",
                     "n_runs": len(g), "runs_excluded_all_flagged": len(g_all) - len(g),
                     "pairs_or_windows_used": int(g.n.sum()), "flagged_pairs": int(g_all.n_failed.sum()),
                     "b_abs_median": g.b_abs.median(), "b_abs_p90": g.b_abs.quantile(0.9), "b_abs_max": g.b_abs.max(),
                     "b_along_median": g.b_along.median(), "b_cross_median": g.b_cross.median(),
                     "b_along_wave_median": g.b_along_wave.median(),
                     "b_along_swell_median": g.b_along_swell.median(),
                     "pct_median": g.b_pct.median(), "pct_p90": g.b_pct.quantile(0.9), "pct_max": g.b_pct.max(),
                     "frac_runs_le_0.17ms": float((g.b_abs <= BUDGET_MS).mean()),
                     "frac_runs_le_0.28pct": float((g.b_pct <= BUDGET_RSS).mean()),
                     "frac_runs_le_0.5pct": float((g.b_pct <= BUDGET_ALL).mean())}
                rows.append(r)
    return pd.DataFrame(rows)


def phase_speed_table(df: pd.DataFrame) -> pd.DataFrame:
    """Measured flow |a| (median over runs, by sea × altitude, nadir core, g=1) beside the derived
    phase speeds and foam drift."""
    f = df[(df.set == "core") & (df.estimator == "flow") & (df.g == 1)]
    rows = []
    for (model, sea, alt), g in f.groupby(["model", "sea", "altitude"]):
        cam_u = g.cam_u.median()
        gsd = cam_u / F_PX
        a = g.b_abs.median()
        lam_eq = 2 * math.pi * a * a / G
        r = {"model": model, "sea": sea, "altitude": alt, "cam_u": cam_u, "gsd_m": gsd, "n_runs": len(g),
             "a_abs_median": a, "a_along_wave_median": -g.b_along_wave.median(),
             "a_along_swell_median": -g.b_along_swell.median(),
             "a_bearing_median": g.a_bearing.median(),
             "lambda_eq_m": lam_eq, "lambda_eq_px": lam_eq / gsd,
             "foam_drift_0.035U10": FOAM_K * g.wind_speed.median(), "gate_scroll": GATE_SPEED}
        for n in (3, 4, 8, 16):
            r[f"c_{n}px"] = math.sqrt(G * n * gsd / (2 * math.pi))
        rows.append(r)
    t = pd.DataFrame(rows)
    return t


def altitude_exponent(df: pd.DataFrame) -> pd.DataFrame:
    """Least-squares slope of log|a| on log cam_u per model and sea (nadir core, g=1, per run).
    Phase speed at a fixed pixel scale gives 0.5; an altitude-independent drift gives 0."""
    f = df[(df.set == "core") & (df.estimator == "flow") & (df.g == 1) & (df.b_abs > 0)]
    rows = []
    for (model, sea), g in f.groupby(["model", "sea"]):
        p = np.polyfit(np.log(g.cam_u), np.log(g.b_abs), 1)
        rows.append({"model": model, "sea": sea, "slope_log_a_vs_log_h": p[0], "n_runs": len(g)})
    for model, g in f.groupby("model"):
        p = np.polyfit(np.log(g.cam_u), np.log(g.b_abs), 1)
        rows.append({"model": model, "sea": "all", "slope_log_a_vs_log_h": p[0], "n_runs": len(g)})
    return pd.DataFrame(rows)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval for a binomial share k/n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def audit_checks(df: pd.DataFrame, res: Path) -> pd.DataFrame:
    """Numbers RESULTS.md's audit corrections quote, in long form (check, slice, value, n)."""
    rows = []

    def add(check, slc, value, n=None):
        rows.append({"check": check, "slice": slc, "value": value, "n": n})

    core = df[df.set == "core"]
    # (1) SEA-RAFT full g=1 at midday vs low sun: all runs, and on matched cells only.
    sr = core[(core.method == "flow:searaft") & (core.param == "g=1") & np.isfinite(core.b_abs)]
    for light, g in sr.groupby("light"):
        add("searaft_g1_median_b_abs_by_light", f"light={light}", g.b_abs.median(), len(g))
    key = ["sea", "altitude", "airspeed", "heading"]
    piv = sr.pivot_table(index=key, columns="light", values="b_abs").dropna()
    add("searaft_g1_median_b_abs_matched_cells", "light=midday", piv["midday"].median(), len(piv))
    add("searaft_g1_median_b_abs_matched_cells", "light=low-sun", piv["low-sun"].median(), len(piv))
    add("searaft_g1_median_diff_lowsun_minus_midday_matched", "all", (piv["low-sun"] - piv["midday"]).median(),
        len(piv))
    # (2) SEA-RAFT full: flagged / collapsed pairs by light (pair level, from flow-core.csv).
    p = res / "flow-core.csv"
    if p.exists():
        d = pd.read_csv(p)
        d = d[d.model == "searaft"].copy()
        gs = np.hypot(d.cam_vE, d.cam_vN)
        d["collapsed"] = np.hypot(d.a_E - d.cam_vE, d.a_N - d.cam_vN) < 0.25 * gs
        d["failed"] = is_failed(d.a_E, d.a_N, d.cam_vE, d.cam_vN, d.spread_E, d.spread_N)
        d["flow_px_expected"] = gs * d.dt * F_PX / d.cam_u  # derived: camera motion in pixels
        for (g_, alt, spd, light), x in d.groupby(["g", "altitude", "airspeed", "light"]):
            slc = f"g={g_} | altitude={alt} | airspeed={spd} | light={light}"
            add("searaft_pairs_flagged_share", slc, float(x.failed.mean()), len(x))
            add("searaft_pairs_collapsed_share", slc, float(x.collapsed.mean()), len(x))
            add("searaft_flow_px_expected_median", slc, float(x.flow_px_expected.median()), len(x))
            add("searaft_flow_px_measured_median", slc, float(x.flow_px_med.median()), len(x))
            add("searaft_runs_with_unflagged_pair", slc, int(x[~x.failed].run_id.nunique()), int(x.run_id.nunique()))
    # (3) Shares of runs <= 0.28 % per altitude x airspeed cell (shell dispersive, midday), with a
    # 95 % Wilson interval: the 'best cells' were selected from 12 cells x 4 windows.
    sh = core[(core.method == "shell:dispersive") & (core.light == "midday")]
    for (alt, spd, par), x in sh.groupby(["altitude", "airspeed", "param"]):
        k, n = int((x.b_pct <= BUDGET_RSS).sum()), len(x)
        lo, hi = wilson(k, n)
        slc = f"altitude={alt} | airspeed={spd} | {par}"
        add("shell_disp_midday_runs_le_0.28pct", slc, k, n)
        add("shell_disp_midday_runs_le_0.28pct_wilson95_lo", slc, lo, n)
        add("shell_disp_midday_runs_le_0.28pct_wilson95_hi", slc, hi, n)
    # (4) Shell weight comparison (whiten = default; energy, sqrt = reruns), per set x light.
    for (st, m, par, light), x in df[(df.estimator == "shell")].groupby(["set", "method", "param", "light"]):
        x = x[np.isfinite(x.b_abs)]
        slc = f"set={st} | {m} | {par} | light={light}"
        add("shell_b_abs_median", slc, x.b_abs.median(), len(x))
        add("shell_b_abs_p90", slc, x.b_abs.quantile(0.9), len(x))
        add("shell_b_abs_max", slc, x.b_abs.max(), len(x))
        add("shell_runs_le_0.17ms", slc, int((x.b_abs <= BUDGET_MS).sum()), len(x))
    # (5) Is a large shell error the objective's own maximum (J_est >= J_truth) or a search failure?
    for st in ("core", "frozen", "decomp-normals"):
        p = res / f"shell-{st}.csv"
        if not p.exists():
            continue
        w = pd.read_csv(p)
        thr = 0.15 if st == "frozen" else 0.5
        for (m, light), x in w.groupby(["model", "light"]):
            big = x[x.err_abs > thr]
            if len(big):
                add("shell_windows_large_err_with_J_est_ge_J_truth",
                    f"set={st} | model={m} | light={light} | err_abs>{thr}", float((big.J_est >= big.J_truth).mean()),
                    len(big))
    # (6) Flow |a| over foam drift 0.035*U10 for seas with foam (calm has none): per sea x altitude median.
    r = sr[sr.sea != "calm"].copy()
    r["ratio"] = r.b_abs / (FOAM_K * r.wind_speed)
    cell = r.groupby(["sea", "altitude"]).ratio.median()
    add("searaft_g1_abs_a_over_foam_drift_cell_median_min", "seas with foam", cell.min(), len(cell))
    add("searaft_g1_abs_a_over_foam_drift_cell_median_max", "seas with foam", cell.max(), len(cell))
    # (7) Pilot failure rules: new (collapse or scatter) vs old (|a| >= 0.25 ground speed).
    for fn in ("flow-pilot.csv", "flow-pilot-frozen.csv"):
        p = res / fn
        if p.exists():
            d = pd.read_csv(p)
            new = is_failed(d.a_E, d.a_N, d.cam_vE, d.cam_vN, d.spread_E, d.spread_N)
            old = (d.a_abs >= 0.25 * d.ground_speed).values
            add("pilot_pairs_flagged_new_rule", fn, int(new.sum()), len(d))
            add("pilot_pairs_flagged_old_rule", fn, int(old.sum()), len(d))
            add("pilot_pairs_flagged_new_not_old", fn, int((new & ~old).sum()), len(d))
    # (8) Direction of a = -b by sea (SEA-RAFT g=1, core): cross-swell's swell travels toward 140°.
    for sea, x in sr.groupby("sea"):
        add("searaft_g1_a_bearing_median", f"sea={sea}", x.a_bearing.median(), len(x))
        add("searaft_g1_a_along_wind_sea_median", f"sea={sea}", -x.b_along_wave.median(), len(x))
        if x.b_along_swell.notna().any():
            add("searaft_g1_a_along_swell_median", f"sea={sea}", -x.b_along_swell.median(), len(x))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Plots


def _style(ax):
    ax.grid(True, color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#8a8984")
    ax.tick_params(colors="#52514e", labelsize=8)


def series_list(df):
    """(method, param, label) of the series shown in most plots, in fixed colour order."""
    out = [(f"flow:{m}", "g=1", f"{FLOW_LABEL[m]}, g=1") for m in FLOW_MODELS
           if ((df.method == f"flow:{m}") & (df.param == "g=1")).any()]
    out += [("shell:dispersive", "T=6.4", "shell fit, dispersive, T=6.4 s"),
            ("shell:static", "T=3.2", "shell fit, static (pattern), T=3.2 s")]
    return [s for s in out if ((df.method == s[0]) & (df.param == s[1])).any()]


def plots(df: pd.DataFrame, res: Path, phase: pd.DataFrame):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                         "text.color": "#0b0b0b", "axes.labelcolor": "#52514e"})
    core = df[df.set == "core"]
    ser = series_list(core)
    alts = sorted(core.altitude.dropna().unique())

    # 1. |b| vs altitude, one panel per sea state
    fig, axs = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    for ax, sea in zip(axs.flat, SEAS):
        for k, (m, p, lab) in enumerate(ser):
            g = core[(core.method == m) & (core.param == p) & (core.sea == sea)].groupby("altitude").b_abs
            med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
            if med.empty:
                continue
            ax.plot(med.index, med.values, color=COLORS[k], marker=MARKERS[k], ms=5, lw=1.6, label=lab)
            ax.fill_between(med.index, q1.values, q3.values, color=COLORS[k], alpha=0.10, lw=0)
        ph = phase[(phase.sea == sea)].drop_duplicates("altitude").sort_values("altitude")
        if len(ph):
            ax.plot(ph.altitude, ph.c_3px, color="#52514e", ls="--", lw=1, label="derived c(λ = 3 px)")
            ax.plot(ph.altitude, ph.c_16px, color="#52514e", ls=":", lw=1, label="derived c(λ = 16 px)")
            ax.axhline(ph["foam_drift_0.035U10"].iloc[0], color="#8a8984", ls="-.", lw=1, label="foam drift 0.035·U10")
        ax.axhline(BUDGET_MS, color="#e34948", lw=0.8, label="0.17 m/s (RSS share at 60 m/s)")
        ax.set_title(sea)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(alts)
        ax.set_xticklabels([f"{a:g}" for a in alts])
        _style(ax)
    for ax in axs[1]:
        ax.set_xlabel("altitude (m)")
    for ax in axs[:, 0]:
        ax.set_ylabel("|velocity bias| (m/s), median over runs")
    h, l = axs.flat[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("E1 core (nadir): velocity bias vs altitude, by sea state (band = interquartile range over speed, heading, sun)")
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.savefig(res / "fig_bias_vs_altitude.png", dpi=130)
    plt.close(fig)

    # 2. along/cross-track bias vs heading to the waves
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    heads = [60, 150, 240]
    w = 0.8 / max(1, len(ser))
    for j, comp in enumerate(["b_along", "b_cross"]):
        ax = axs[j]
        for k, (m, p, lab) in enumerate(ser):
            g = core[(core.method == m) & (core.param == p)].groupby("heading")[comp]
            med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
            x = np.arange(len(heads)) + (k - (len(ser) - 1) / 2) * w
            ax.bar(x, [med.get(h, np.nan) for h in heads], width=w * 0.9, color=COLORS[k], label=lab)
            ax.errorbar(x, [med.get(h, np.nan) for h in heads],
                        yerr=[[med.get(h) - q1.get(h) for h in heads], [q3.get(h) - med.get(h) for h in heads]],
                        fmt="none", ecolor="#52514e", lw=0.8, capsize=2)
        ax.axhline(0, color="#8a8984", lw=0.8)
        ax.set_xticks(range(len(heads)))
        ax.set_xticklabels([HEAD_NAME[h] for h in heads])
        ax.set_title({"b_along": "along-track bias (m/s; + = overestimates speed)",
                      "b_cross": "cross-track bias (m/s; + = to the right of track)"}[comp])
        _style(ax)
    axs[0].set_ylabel("median over runs (whiskers: IQR)")
    axs[1].legend(loc="upper right", frameon=False)
    fig.suptitle("E1 core (nadir): bias vs heading relative to the waves (waves travel toward 60°)")
    fig.tight_layout()
    fig.savefig(res / "fig_bias_vs_heading.png", dpi=130)
    plt.close(fig)

    # 3. vs ground speed: m/s and % of speed
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for k, (m, p, lab) in enumerate(ser):
        g = core[(core.method == m) & (core.param == p)]
        for j, col in enumerate(["b_abs", "b_pct"]):
            gg = g.groupby("airspeed")[col]
            med = gg.median()
            axs[j].plot(med.index, med.values, color=COLORS[k], marker=MARKERS[k], ms=5, lw=1.6, label=lab)
            axs[j].fill_between(med.index, gg.quantile(0.25).values, gg.quantile(0.75).values, color=COLORS[k],
                                alpha=0.10, lw=0)
    axs[0].axhline(BUDGET_MS, color="#e34948", lw=0.8, label="0.17 m/s")
    axs[1].axhline(BUDGET_RSS, color="#e34948", lw=0.8, label="0.28 % (RSS share)")
    axs[1].axhline(BUDGET_ALL, color="#e34948", lw=0.8, ls="--", label="0.5 % (whole budget)")
    for ax, yl in zip(axs, ["|velocity bias| (m/s)", "|velocity bias| (% of ground speed)"]):
        ax.set_yscale("log")
        ax.set_xlabel("ground speed (m/s)")
        ax.set_ylabel(yl + ", median over runs")
        ax.set_xticks([15, 30, 60])
        _style(ax)
    axs[1].legend(loc="upper right", frameon=False)
    fig.suptitle("E1 core (nadir): bias vs ground speed (band = IQR over sea, altitude, heading, sun)")
    fig.tight_layout()
    fig.savefig(res / "fig_bias_vs_speed.png", dpi=130)
    plt.close(fig)

    # 4. frozen control: per-run |b| on the frozen sea vs the matching live core runs
    fz = df[df.set == "frozen"]
    if len(fz):
        allser = [(f"flow:{m}", g_, f"{FLOW_LABEL[m]}, {g_}") for m in FLOW_MODELS for g_ in ("g=1", "g=3")]
        allser += [("shell:static", f"T={t}", f"shell static, T={t} s") for t in ("1.6", "3.2", "6.4", "12")]
        allser += [("shell:dispersive", f"T={t}", f"shell dispersive, T={t} s") for t in ("3.2", "12")]
        allser = [s for s in allser if ((fz.method == s[0]) & (fz.param == s[1])).any()]
        live = core[core.sea.isin(fz.sea.unique()) & core.altitude.isin(fz.altitude.unique())
                    & core.airspeed.isin(fz.airspeed.unique()) & core.heading.isin(fz.heading.unique())
                    & (core.light == "midday")]
        fig, ax = plt.subplots(figsize=(12, 4.8))
        for k, (m, p, lab) in enumerate(allser):
            for which, src, col in (("frozen", fz, "#2a78d6"), ("live", live, "#eb6834")):
                v = src[(src.method == m) & (src.param == p)].b_abs.values
                x = k + (-0.17 if which == "frozen" else 0.17) + np.random.default_rng(k).uniform(-0.08, 0.08, len(v))
                ax.scatter(x, v, s=12, color=col, alpha=0.8, lw=0, marker="o" if which == "frozen" else "s",
                           label=f"{which} sea (matching runs)" if k == 0 else None)
        ax.axhline(BUDGET_MS, color="#e34948", lw=0.8, label="0.17 m/s")
        ax.set_yscale("log")
        ax.set_xticks(range(len(allser)))
        ax.set_xticklabels([s[2] for s in allser], rotation=35, ha="right", fontsize=7)
        ax.set_ylabel("|velocity bias| per run (m/s)")
        ax.set_title("Frozen-sea control (render.seaRate 0): 24 runs; live = the same sea/altitude/speed/heading, midday, from core")
        ax.legend(frameon=False, loc="upper left")
        _style(ax)
        fig.tight_layout()
        fig.savefig(res / "fig_frozen_control.png", dpi=130)
        plt.close(fig)

    # 5. flow vs shell: % of speed by window length / gap, against the budget lines, by altitude
    fig, axs = plt.subplots(1, len(alts), figsize=(13, 4.4), sharey=True)
    cmp = [(f"flow:{m}", "g=1", FLOW_LABEL[m] + " g=1") for m in FLOW_MODELS]
    cmp += [("shell:static", "T=3.2", "shell static T=3.2")]
    cmp += [("shell:dispersive", f"T={t}", f"shell disp. T={t}") for t in ("1.6", "3.2", "6.4", "12")]
    cmp = [c for c in cmp if ((core.method == c[0]) & (core.param == c[1])).any()]
    for ax, alt in zip(np.atleast_1d(axs), alts):
        data = [core[(core.method == m) & (core.param == p) & (core.altitude == alt)].b_pct.dropna().values
                for m, p, _ in cmp]
        bp = ax.boxplot(data, widths=0.6, showfliers=True, patch_artist=True,
                        flierprops=dict(ms=2, mfc="#8a8984", mec="none"),
                        medianprops=dict(color="#0b0b0b", lw=1.2))
        for k, b in enumerate(bp["boxes"]):
            b.set_facecolor("#2a78d6" if cmp[k][0].startswith("flow") else ("#8a8984" if "static" in cmp[k][0] else "#1baf7a"))
            b.set_alpha(0.55)
        ax.axhline(BUDGET_RSS, color="#e34948", lw=0.8)
        ax.axhline(BUDGET_ALL, color="#e34948", lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_title(f"altitude {alt:g} m")
        ax.set_xticks(range(1, len(cmp) + 1))
        ax.set_xticklabels([c[2] for c in cmp], rotation=60, ha="right", fontsize=6.5)
        _style(ax)
    np.atleast_1d(axs)[0].set_ylabel("|velocity bias| per run (% of ground speed)")
    fig.suptitle("Flow (blue) vs shell fit (green: dispersive; grey: static) — red lines: 0.28 % RSS share, 0.5 % whole budget")
    fig.tight_layout()
    fig.savefig(res / "fig_flow_vs_shell.png", dpi=130)
    plt.close(fig)

    # 6. ECDF of % of speed, all core runs, against the budget lines
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for k, (m, p, lab) in enumerate(ser + [("shell:dispersive", "T=12", "shell fit, dispersive, T=12 s")]):
        v = np.sort(core[(core.method == m) & (core.param == p)].b_pct.dropna().values)
        if len(v):
            ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=COLORS[k % 8], lw=1.6, label=lab)
    ax.axvline(BUDGET_RSS, color="#e34948", lw=0.8, label="0.28 %")
    ax.axvline(BUDGET_ALL, color="#e34948", lw=0.8, ls="--", label="0.5 %")
    ax.set_xscale("log")
    ax.set_xlabel("|velocity bias| per run (% of ground speed)")
    ax.set_ylabel("fraction of core runs ≤ x")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("E1 core (432 runs): cumulative distribution of bias as % of speed")
    _style(ax)
    fig.tight_layout()
    fig.savefig(res / "fig_pct_budget.png", dpi=130)
    plt.close(fig)

    # 8. decomposition: the same 32 flights, flow and shell on the shaded image, the wave-slope
    #    layer alone (normals) and the foam layer alone
    dec = df[df.set.str.startswith("decomp-")].copy()
    if len(dec):
        dec["layer"] = dec.set.str.replace("decomp-", "", regex=False)
        dser = [("flow:searaft", "g=1", FLOW_LABEL["searaft"] + ", g=1"),
                ("flow:ufmrefine980", "g=1", FLOW_LABEL["ufmrefine980"] + ", g=1"),
                ("shell:static", "T=3.2", "shell static, T=3.2 s"),
                ("shell:dispersive", "T=6.4", "shell dispersive, T=6.4 s")]
        dser = [x for x in dser if ((dec.method == x[0]) & (dec.param == x[1])).any()]
        fig, axs = plt.subplots(1, len(dser), figsize=(4 * len(dser), 4.2), sharey=True)
        variants = [("shaded", "midday", "shaded image, midday"), ("shaded", "low-sun", "shaded image, low sun"),
                    ("normals", None, "wave slope only (normals)"), ("foam", None, "foam only (foam mask)")]
        for ax, (m, p, lab) in zip(np.atleast_1d(axs), dser):
            for k, (layer, light, vlab) in enumerate(variants):
                g = dec[(dec.method == m) & (dec.param == p) & (dec.layer == layer) & (dec.sea != "calm" if layer == "foam" else True)]
                if light:
                    g = g[g.light == light]
                med = g.groupby("altitude").b_abs.median()
                if med.empty:
                    continue
                ax.plot(med.index, med.values, color=COLORS[k], marker=MARKERS[k], ms=5, lw=1.6, label=vlab)
            hh = np.array(alts, dtype=float)
            ax.plot(hh, np.sqrt(G * 3 * hh / F_PX / (2 * np.pi)), color="#52514e", ls="--", lw=1, label="derived c(λ = 3 px)")
            ax.axhline(GATE_SPEED, color="#6250d6", ls=":", lw=1, label="break-gate scroll 1.98 m/s")
            ax.axhline(FOAM_K * 7, color="#8a8984", ls="-.", lw=1, label="foam drift 0.035·U10 (U10 = 7)")
            ax.axhline(BUDGET_MS, color="#e34948", lw=0.8, label="0.17 m/s")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks(alts)
            ax.set_xticklabels([f"{a:g}" for a in alts])
            ax.set_xlabel("altitude (m)")
            ax.set_title(lab)
            _style(ax)
        np.atleast_1d(axs)[0].set_ylabel("|velocity bias| (m/s), median over seas")
        np.atleast_1d(axs)[-1].legend(frameon=False, fontsize=6.5, loc="lower right")
        fig.suptitle("Decomposition (32 flights, 30 m/s, heading 60): which image layer carries the bias "
                     "(foam: calm excluded, it has none)")
        fig.tight_layout()
        fig.savefig(res / "fig_decomposition.png", dpi=130)
        plt.close(fig)

    # 7. phase speed: measured flow |a| vs camera height, with derived c(λ = n px) and foam drift
    f = core[(core.estimator == "flow") & (core.g == 1)]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    hh = np.geomspace(25, 300, 50)
    for k, m in enumerate([m for m in FLOW_MODELS if (f.model == m).any()]):
        g = f[f.model == m]
        med = g.groupby("cam_u").b_abs.median()
        axs[0].scatter(g.cam_u * np.exp(np.random.default_rng(k).uniform(-0.04, 0.04, len(g))), g.b_abs, s=4,
                       color=COLORS[k], alpha=0.25, lw=0)
        axs[0].plot(med.index, med.values, color=COLORS[k], marker=MARKERS[k], ms=6, lw=1.8, label=FLOW_LABEL[m] + " (median)")
        pm = phase[phase.model == m].groupby("altitude").lambda_eq_px.median()
        axs[1].plot(pm.index, pm.values, color=COLORS[k], marker=MARKERS[k], ms=6, lw=1.6, label=FLOW_LABEL[m])
    for n, ls in ((3, "--"), (4, "-."), (16, ":")):
        axs[0].plot(hh, np.sqrt(G * n * hh / F_PX / (2 * np.pi)), color="#52514e", ls=ls, lw=1, label=f"derived c(λ = {n} px)")
    for u, ls in ((3, "-"), (12, "--")):
        axs[0].axhline(FOAM_K * u, color="#8a8984", ls=ls, lw=1, label=f"foam drift 0.035·U10, U10 = {u} m/s")
    axs[0].axhline(GATE_SPEED, color="#6250d6", lw=1, ls=":", label="break-gate scroll 1.98 m/s")
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("camera height (m)")
    axs[0].set_ylabel("|apparent pattern velocity| per run (m/s), g=1")
    axs[0].legend(frameon=False, fontsize=6.5, loc="upper left")
    axs[0].set_title("Measured |a| (all core runs) vs derived phase speeds and drifts")
    axs[1].set_xscale("log")
    axs[1].set_xticks(alts)
    axs[1].set_xticklabels([f"{a:g}" for a in alts])
    axs[1].set_xlabel("altitude (m)")
    axs[1].set_ylabel("λ_eq = 2π|a|²/g, in ground pixels (median over seas)")
    axs[1].set_title("Wavelength whose phase speed equals |a| (derived)")
    axs[1].legend(frameon=False)
    for ax in axs:
        _style(ax)
    fig.tight_layout()
    fig.savefig(res / "fig_phase_speed.png", dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)
    res = Path(a.results)
    df = per_run(res)
    df.to_csv(res / "per_run.csv", index=False, float_format="%.6g")
    s = summarise(df)
    s.to_csv(res / "summary_by_slice.csv", index=False, float_format="%.5g")
    ph = phase_speed_table(df)
    ph.to_csv(res / "phase_speed.csv", index=False, float_format="%.5g")
    ex = altitude_exponent(df)
    ex.to_csv(res / "altitude_exponent.csv", index=False, float_format="%.4g")
    audit_checks(df, res).to_csv(res / "audit_checks.csv", index=False, float_format="%.5g")
    dec = df[df.set.str.startswith("decomp-")]
    if len(dec):
        t = dec.groupby(["set", "method", "param", "light", "sea", "altitude"]).agg(
            b_abs=("b_abs", "median"), b_along_wave=("b_along_wave", "median"),
            b_along_swell=("b_along_swell", "median"), a_bearing=("a_bearing", "median"),
            n=("n", "sum"), n_failed=("n_failed", "sum")).reset_index()
        t.to_csv(res / "decomp_by_layer.csv", index=False, float_format="%.5g")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 400)
    cols = ["set", "method", "param", "n_runs", "runs_excluded_all_flagged", "flagged_pairs", "b_abs_median", "b_abs_p90", "b_along_wave_median",
            "pct_median", "pct_p90", "frac_runs_le_0.28pct", "frac_runs_le_0.5pct"]
    print(s[s.slice == "all"][cols].round(4).to_string(index=False))
    if not a.no_plots:
        plots(df, res, ph)
    return 0


if __name__ == "__main__":
    sys.exit(main())
