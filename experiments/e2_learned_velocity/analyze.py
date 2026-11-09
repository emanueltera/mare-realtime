"""E2 cross-method comparison: the CSVs and figures that results/RESULTS.md quotes.

Reads only what the methods left on disk under results/ (the harness's per_run.csv and
slices.csv per method and set, the prediction CSVs, E1's per-run values joined in each
method's e1-core/e1_compare.csv) and writes, into results/:

- comparison_by_set.csv    every method x set x K: the harness's 'all' row, recomputed here from
                           per_run.csv with e2.evaluate.summarise and checked against slices.csv;
                           E1's estimators on e1-core added as rows (statistic = E1's own per-run b)
- comparison_by_slice.csv  every slice row of every method's slices.csv, labelled
- e1_core_compare.csv      e1-core by sun and by sun x altitude, E2 methods next to E1's (derived)
- k_ablation.csv           median b per method family, K and set
- rate_diagnostic.csv      test-rate per method and seaRate: b, signed along-wind error, the paired
                           change against B0 on the same run, and the derived constant-offset model
- calibration.csv          window-level sigma coverage per method and set (from the predictions),
                           plus the run-level rank correlation of sigma with b
- costs.csv                wall time of each render set and training run (from their logs); dollars derived
- fig_*.png                the figures listed in RESULTS.md

Run locally from this directory:
    uv run --no-project --with numpy --with pandas --with matplotlib python analyze.py
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e2.evaluate import BUDGET_MS, BUDGET_PCT, DRIFT_PCT, summarise  # noqa: E402

RES = HERE / "results"
OUT = RES
SETS = ["val", "test-iid", "test-sea", "test-alt", "test-rate", "e1-core"]
SUMMARY_COLS = ["runs", "b_median", "b_p90", "b_pct_median", "b_pct_p90", "rms_median", "drift_pct_median",
                "drift_pct_p90", "frac_b_le_017", "frac_bpct_le_028", "frac_bpct_le_05", "frac_drift_le_05"]


@dataclass(frozen=True)
class Method:
    id: str          # short id used in every CSV
    label: str       # legend label
    family: str      # A / B / C (learned) or B0..B3 (baselines)
    role: str        # main | ablation
    K: int
    eval_dir: str    # results/<eval_dir>/<set>/{per_run,slices}.csv
    pred_dir: str | None = None
    selected_on: str = "val"


METHODS = [
    # baselines (results/baselines/NOTES.md)
    Method("B0", "B0 phase correlation", "B0", "main", 8, "baselines/b0-phasecorr", "baselines/b0-phasecorr/preds", "none (no parameters)"),
    Method("B1", "B1 SEA-RAFT-M", "B1", "main", 8, "baselines/b1-searaft", "baselines/b1-searaft/preds", "none (no parameters)"),
    Method("B2", "B2 shell fit (truth-centred prior)", "B2", "main", 32, "baselines/b2-shell-dispersive", None, "E1 defaults"),
    Method("B3", "B3 conditions-only MLP", "B3", "main", 8, "baselines/b3-mlp", None),
    # learned models
    Method("A", "A e2e (cost volume)", "A", "main", 8, "e2e/eval", "e2e/preds"),
    Method("B", "B flowhead (frozen SEA-RAFT)", "B", "main", 8, "flowhead", "flowhead/preds"),
    Method("C", "C spectral (learned shell)", "C", "main", 32, "spectral", "spectral/preds", "a priori (val prefers K=16)"),
    # ablations
    Method("A-K2", "A e2e, K=2", "A", "ablation", 2, "e2e-K2/eval", "e2e-K2/preds"),
    Method("A-K4", "A e2e, K=4", "A", "ablation", 4, "e2e-K4/eval", "e2e-K4/preds"),
    Method("B-gap1", "B flowhead, gap 1 only", "B", "ablation", 8, "flowhead-gap1", "flowhead-gap1/preds"),
    Method("B-onthefly", "B flowhead, trained on the fly", "B", "ablation", 8, "flowhead-onthefly", None),
    Method("B-ft", "B flowhead, update block fine-tuned", "B", "ablation", 8, "flowhead-ft", "flowhead-ft/preds"),
    Method("C-K16", "C spectral, K=16", "C", "ablation", 16, "spectral-k16", "spectral-k16/preds"),
    Method("C-noreg", "C spectral, no pattern registration", "C", "ablation", 32, "spectral-noreg", "spectral-noreg/preds"),
    Method("C-tta", "C spectral, 8-fold test-time averaging", "C", "ablation", 32, "spectral-tta", None),
    Method("C-classical", "C pipeline, unlearned shell objective", "C", "ablation", 32, "spectral-classical", None, "none"),
    Method("B0-K2", "B0 phase correlation, K=2", "B0", "ablation", 2, "baselines/b0-phasecorr", None, "none (no parameters)"),
    Method("B0-K16", "B0 phase correlation, K=16", "B0", "ablation", 16, "baselines/b0-phasecorr", None, "none (no parameters)"),
    Method("B1-K2", "B1 SEA-RAFT-M, K=2", "B1", "ablation", 2, "baselines/b1-searaft", None, "none (no parameters)"),
    Method("B1-K16", "B1 SEA-RAFT-M, K=16", "B1", "ablation", 16, "baselines/b1-searaft", None, "none (no parameters)"),
    Method("B2-K64", "B2 shell fit, K=64 (truth-centred prior)", "B2", "ablation", 64, "baselines/b2-shell-dispersive", None, "E1 defaults"),
    Method("B2-static", "B2 static shell, K=32 (truth-centred prior)", "B2", "ablation", 32, "baselines/b2-shell-static", None, "E1 defaults"),
    Method("B3-const", "B1 + train-mean constant", "B3", "ablation", 8, "baselines/b3-const", None),
    Method("B3-full", "B3 MLP with direction features", "B3", "ablation", 8, "baselines/b3-mlp-full", None),
]
BY_ID = {m.id: m for m in METHODS}
MAIN = [m.id for m in METHODS if m.role == "main"]

# One colour per method family, fixed everywhere (categorical slots of the dataviz reference palette,
# validated with its validate_palette.js; the grey baselines are recessive on purpose).
COLOR = {"A": "#2a78d6", "C": "#eb6834", "B": "#1baf7a", "B2": "#6250d6", "B3": "#eda100",
         "B0": "#52514e", "B1": "#a3a29b", "E1": "#898781", "offset": "#0b0b0b"}
MARKER = {"A": "o", "B": "P", "C": "X", "B0": "s", "B1": "v", "B2": "D", "B3": "^"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

E1_COLS = {  # e1_compare.csv column -> label (E1's own per-run b: a median over windows, shell with a truth-centred prior)
    "e1_flow_searaft_g1": "E1 SEA-RAFT-M (g=1)",
    "e1_flow_ufmrefine980_g1": "E1 UFM-Refine-980 (g=1)",
    "e1_shell_dispersive_T3.2": "E1 shell fit T=3.2 s",
    "e1_shell_dispersive_T6.4": "E1 shell fit T=6.4 s",
}


# ----------------------------------------------------------------------------- loading

def per_run(m: Method, s: str) -> pd.DataFrame | None:
    p = RES / m.eval_dir / s / "per_run.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df = df[df.K == m.K]
    return df if len(df) else None


def harness_slices(m: Method, s: str) -> pd.DataFrame | None:
    p = RES / m.eval_dir / s / "slices.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df = df[df.K == m.K]
    return df if len(df) else None


def preds(m: Method, s: str) -> pd.DataFrame | None:
    if m.pred_dir is None:
        return None
    for name in (f"{s}.csv", f"{s}.csv.gz"):
        p = RES / m.pred_dir / name
        if p.exists():
            df = pd.read_csv(p)
            return df[df.K == m.K]
    return None


def rate_of(run_id: str) -> float:
    r = re.search(r"-rate([0-9p]+)$", run_id).group(1)
    return float(r.replace("p", "."))


def condition_of(run_id: str) -> str:
    return re.search(r"-(c\d{3})-rate", run_id).group(1)


# ----------------------------------------------------------------------------- tables

def comparison_by_set() -> pd.DataFrame:
    rows = []
    for m in METHODS:
        for s in SETS:
            pr, sl = per_run(m, s), harness_slices(m, s)
            if pr is None:
                continue
            mine = summarise(pr)
            if sl is not None:  # the harness's own row must match the recomputation
                ref = sl[(sl.slice == "all")].iloc[0]
                for c in SUMMARY_COLS:
                    a, b = float(mine[c]), float(ref[c])
                    assert (np.isnan(a) and np.isnan(b)) or abs(a - b) <= 1e-9 * max(1, abs(b)), (m.id, s, c, a, b)
            rows.append({"method": m.id, "label": m.label, "family": m.family, "role": m.role, "K": m.K, "set": s,
                         "statistic": "E2 harness: b = |mean window error| per run", "selected_on": m.selected_on,
                         "source": f"results/{m.eval_dir}/{s}/per_run.csv", **mine})
    # E1's estimators on the same 432 renders (their own per-run statistic; b% derived from the E2 ground speed)
    e1 = pd.read_csv(RES / "e2e/eval/e1-core/e1_compare.csv")
    for col, label in E1_COLS.items():
        b = e1[col].dropna()
        bp = 100 * e1.loc[b.index, col] / e1.loc[b.index, "ground_speed"]
        rows.append({"method": col, "label": label, "family": "E1", "role": "reference", "K": np.nan, "set": "e1-core",
                     "statistic": "E1 per-run b (median over windows; shell with E1's truth-centred prior); b% derived",
                     "selected_on": "E1", "source": "results/e2e/eval/e1-core/e1_compare.csv",
                     "runs": len(b), "b_median": b.median(), "b_p90": float(np.percentile(b, 90)),
                     "b_pct_median": bp.median(), "b_pct_p90": float(np.percentile(bp, 90)),
                     "frac_b_le_017": float((b <= BUDGET_MS).mean()), "frac_bpct_le_028": float((bp <= BUDGET_PCT).mean()),
                     "frac_bpct_le_05": float((bp <= DRIFT_PCT).mean())})
    return pd.DataFrame(rows)


def comparison_by_slice() -> pd.DataFrame:
    out = []
    for m in METHODS:
        for s in SETS:
            sl = harness_slices(m, s)
            if sl is None:
                continue
            sl = sl.copy()
            sl.insert(0, "method", m.id)
            sl.insert(1, "label", m.label)
            sl.insert(2, "role", m.role)
            out.append(sl)
    df = pd.concat(out, ignore_index=True)
    df["bin_lo"] = df["bin"].map(_bin_lo)
    return df


def _bin_lo(b: str) -> float:
    m = re.match(r"^\(([-0-9.]+),", str(b))
    if m:
        return float(m.group(1))
    try:
        return float(b)
    except ValueError:
        return np.nan


def e1_core_compare() -> pd.DataFrame:
    """e1-core by sun (7 deg 'low sun', 60 deg 'midday') and sun x altitude. Derived from each method's
    e1_compare.csv (column b = the harness's per-run b) and E1's per-run values in the same file."""
    series = {}
    for mid in ["B0", "B1", "B2", "B3", "A", "B", "C", "A-K2", "A-K4", "B-ft", "C-K16", "C-noreg", "B2-K64"]:
        m = BY_ID[mid]
        pr = per_run(m, "e1-core")
        if pr is not None:
            series[mid] = pr[["run_id", "b", "altitude", "sun_elevation", "ground_speed"]].assign(label=m.label)
    e1 = pd.read_csv(RES / "e2e/eval/e1-core/e1_compare.csv")
    for col, label in E1_COLS.items():
        series[col] = e1[["run_id", "altitude", "sun_elevation", "ground_speed"]].assign(b=e1[col], label=label)
    rows = []
    for mid, df in series.items():
        df = df.dropna(subset=["b"]).assign(sun=lambda d: np.where(d.sun_elevation >= 35, "midday", "low sun"))
        groups = [("all", "all", df)]
        groups += [(sun, "all", g) for sun, g in df.groupby("sun")]
        groups += [(sun, f"{alt:g}", g) for (sun, alt), g in df.groupby(["sun", "altitude"])]
        groups += [("all", f"speed {v:g}", g) for v, g in df.groupby(df.ground_speed.round())]
        for sun, alt, g in groups:
            rows.append({"method": mid, "label": g.label.iloc[0], "sun": sun, "altitude": alt, "runs": len(g),
                         "b_median": g.b.median(), "b_p25": g.b.quantile(0.25), "b_p75": g.b.quantile(0.75),
                         "b_p90": float(np.percentile(g.b, 90)), "frac_b_le_017": float((g.b <= BUDGET_MS).mean())})
    return pd.DataFrame(rows)


def k_ablation(cmp: pd.DataFrame) -> pd.DataFrame:
    fam = cmp[cmp.family.isin(["A", "C", "B0", "B1", "B2"]) & ~cmp.method.isin(["C-noreg", "C-tta", "C-classical", "B2-static"])]
    keep = ["family", "method", "K", "set", "runs", "b_median", "b_p90", "frac_b_le_017", "frac_bpct_le_028"]
    df = fam[keep].copy()
    # rate-1 runs of test-rate, the physical sea, as its own column
    rows = []
    for m in METHODS:
        if m.id not in set(df.method):
            continue
        pr = per_run(m, "test-rate")
        if pr is None:
            continue
        g = pr[pr.sea_rate == 1]
        sm = summarise(g)
        rows.append({"family": m.family, "method": m.id, "K": m.K, "set": "test-rate, seaRate 1", **{k: sm[k] for k in keep[4:]}})
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    return df.sort_values(["family", "K", "set"]).reset_index(drop=True)


def rate_diagnostic() -> pd.DataFrame:
    """test-rate: 100 conditions x seaRate 0, 0.5, 1, 2 (same seeds, same theta). For each method and rate:
    b, the signed along-wind error (negative = pulled the way phase correlation is pulled), and the paired
    change against B0 (K=8) on the same run. 'constant offset' is derived: B0 minus B0's own rate-1 error
    vector of the same condition, i.e. a model that subtracts the rate-1 wave pull whatever the sea does."""
    b0 = per_run(BY_ID["B0"], "test-rate").set_index("run_id")
    b0["cond"] = [condition_of(r) for r in b0.index]
    b0_r1 = b0[b0.sea_rate == 1].set_index("cond")
    rows = []

    def add(mid, label, pr):
        pr = pr.set_index("run_id")
        for r, g in pr.groupby("sea_rate"):
            pc = b0.loc[g.index]
            d_along = g.b_along_wind - pc.b_along_wind
            sm = summarise(g.reset_index())
            rows.append({"method": mid, "label": label, "sea_rate": r, "runs": len(g), "b_median": sm["b_median"],
                         "b_p90": sm["b_p90"], "frac_b_le_017": sm["frac_b_le_017"],
                         "along_wind_median": g.b_along_wind.median(), "abs_along_wind_median": g.b_along_wind.abs().median(),
                         "abs_cross_wind_median": g.b_cross_wind.abs().median(),
                         "minus_B0_along_median": d_along.median(), "minus_B0_along_p10": d_along.quantile(0.1),
                         "minus_B0_along_p90": d_along.quantile(0.9)})

    for mid in ["B0", "B1", "B3", "A", "A-K4", "A-K2", "B", "B-ft", "C", "C-K16", "C-noreg", "B2", "B2-static"]:
        pr = per_run(BY_ID[mid], "test-rate")
        if pr is not None:
            add(mid, BY_ID[mid].label, pr)
    # derived constant-offset model
    ref = b0_r1.loc[b0.cond]
    off = pd.DataFrame({"run_id": b0.index, "sea_rate": b0.sea_rate.values,
                        "b_e": b0.b_e.values - ref.b_e.values, "b_n": b0.b_n.values - ref.b_n.values,
                        "b_along_wind": b0.b_along_wind.values - ref.b_along_wind.values,
                        "b_cross_wind": b0.b_cross_wind.values - ref.b_cross_wind.values})
    off["b"] = np.hypot(off.b_e, off.b_n)
    off["b_pct"] = 100 * off.b / b0.ground_speed.values
    off["drift_pct"] = np.nan
    off["rms"] = np.nan
    add("offset", "constant offset (derived: B0 minus its rate-1 error)", off)
    return pd.DataFrame(rows)


def calibration() -> pd.DataFrame:
    rows = []
    for mid in ["A", "A-K2", "A-K4", "B", "B-gap1", "B-ft", "C", "C-K16", "C-noreg"]:
        m = BY_ID[mid]
        for s in SETS:
            p = preds(m, s)
            if p is None or p[["sigma_x", "sigma_y"]].isna().all().all():
                continue
            pr = per_run(m, s)
            groups = [(s, p, pr)]
            if s == "test-rate":
                p = p.assign(rate=[rate_of(r) for r in p.run_id])
                groups = [(f"test-rate, seaRate {r:g}", g, pr[pr.sea_rate == r]) for r, g in p.groupby("rate")]
            for name, g, gr in groups:
                ex = (g.v_est_x - g.v_true_x).to_numpy(), (g.v_est_y - g.v_true_y).to_numpy()
                sx, sy = g.sigma_x.to_numpy(), g.sigma_y.to_numpy()
                z = np.abs(np.concatenate([ex[0] / sx, ex[1] / sy]))
                z = z[np.isfinite(z)]
                e2d, s2d = np.hypot(*ex), np.hypot(sx, sy)
                ok = np.isfinite(e2d) & np.isfinite(s2d)
                rows.append({"method": mid, "label": m.label, "set": name, "windows": len(g),
                             "within_1sigma_axis": float((z <= 1).mean()), "within_2sigma_axis": float((z <= 2).mean()),
                             "z_rms_axis": float(np.sqrt((z ** 2).mean())),
                             "sigma_2d_median": float(np.median(s2d[ok])), "error_2d_median": float(np.median(e2d[ok])),
                             "spearman_window_sigma_error": _spearman(s2d[ok], e2d[ok]),
                             "runs": len(gr), "spearman_run_sigma_b": _spearman(gr.sigma_mean.to_numpy(), gr.b.to_numpy())})
    return pd.DataFrame(rows)


def _spearman(a, b) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    ra, rb = pd.Series(a[ok]).rank().to_numpy(), pd.Series(b[ok]).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def costs() -> pd.DataFrame:
    """Wall time per render set (data_summary.csv) and per training run (the runs' own logs). The GPU was
    shared by up to four jobs at once, so these do not add up to box time. usd_at_5.33_per_h is derived."""
    rows = []
    ds = pd.read_csv(RES / "data_summary.csv")
    for r in ds[ds.render_wall_s.notna()].itertuples():
        rows.append({"item": f"render {r.set}", "kind": "render", "hours": r.render_wall_s / 3600,
                     "source": "results/data_summary.csv render_wall_s"})
    for name, d in [("A e2e K=8", "e2e"), ("A e2e K=2", "e2e-K2"), ("A e2e K=4", "e2e-K4")]:
        log = pd.read_csv(RES / d / "train/log.csv")
        rows.append({"item": f"train {name}", "kind": "train", "hours": log.hours.max(), "steps": log.step.max(),
                     "peak_gpu_gb": log.gpu_mem_gb.max(), "source": f"results/{d}/train/log.csv"})
    for name, d in [("C spectral K=32", "k32"), ("C spectral K=16", "k16"), ("C spectral noreg", "k32-noreg")]:
        c = pd.read_csv(RES / "spectral/training" / d / "curve.csv")
        rows.append({"item": f"train {name}", "kind": "train", "hours": c.minutes.max() / 60, "steps": c.step.max(),
                     "peak_gpu_gb": c.max_mem_gib.max() * 1.073741824, "source": f"results/spectral/training/{d}/curve.csv"})
    for name, d in [("B flowhead on the fly", "onthefly-main"), ("B flowhead cached head (selected)", "head-g1247-lr5e-4"),
                    ("B flowhead cached head lr 1e-3", "head-g1247-lr1e-3"), ("B flowhead gap-1 head", "head-g1-lr5e-4"),
                    ("B flowhead gap-1 head lr 1e-3", "head-g1-lr1e-3"), ("B flowhead fine-tune", "ft-update")]:
        t = pd.read_csv(RES / "flowhead/curves" / d / "train_log.csv")
        rows.append({"item": f"train {name}", "kind": "train", "hours": t.time_s.max() / 3600, "steps": t.step.max(),
                     "peak_gpu_gb": t["mem_gb"].max() if "mem_gb" in t else np.nan, "source": f"results/flowhead/curves/{d}/train_log.csv"})
    df = pd.DataFrame(rows)
    df["usd_at_5.33_per_h"] = df.hours * 5.33
    return df


# ----------------------------------------------------------------------------- figures

def _style(plt):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 10,
        "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb", "legend.frameon": False,
    })


def _budget(ax, y, text, horizontal=True, color=INK, left=False, below=False):
    if horizontal:
        ax.axhline(y, color=color, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.annotate(text, (0.0 if left else 1.0, y), xycoords=("axes fraction", "data"),
                    xytext=(2 if left else -2, -2 if below else 2), textcoords="offset points",
                    ha="left" if left else "right", va="top" if below else "bottom", fontsize=7.5, color=INK2,
                    bbox=dict(boxstyle="square,pad=0.1", fc="#fcfcfb", ec="none", alpha=0.85), zorder=4)
    else:
        ax.axvline(y, color=color, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.annotate(text, (y, 1.0), xycoords=("data", "axes fraction"), xytext=(2, -2), textcoords="offset points",
                    ha="left", va="top", fontsize=7.5, color=INK2)


def fig_bias_by_set(plt):
    sets = ["val", "test-iid", "test-sea", "test-alt", "e1-core"]
    fig, axes = plt.subplots(2, len(sets), figsize=(15, 7.2), sharey="row")
    rng = np.random.default_rng(0)
    lo, hi = (0.005, 30.0), (0.01, 100.0)
    for j, s in enumerate(sets):
        for row, (col, lim) in enumerate([("b", lo), ("b_pct", hi)]):
            ax = axes[row, j]
            for i, mid in enumerate(MAIN):
                m = BY_ID[mid]
                pr = per_run(m, s)
                if pr is None:
                    continue
                v = pr[col].dropna().to_numpy()
                vc = np.clip(v, lim[0], lim[1])
                x = i + rng.uniform(-0.28, 0.28, len(v))
                ax.scatter(x, vc, s=5, color=COLOR[m.family], alpha=0.35, linewidths=0, zorder=2)
                med = np.median(v)
                ax.plot([i - 0.38, i + 0.38], [med, med], color=INK, lw=1.6, zorder=3, solid_capstyle="round")
            ax.set_yscale("log")
            ax.set_ylim(*lim)
            ax.set_xticks(range(len(MAIN)))
            ax.set_xticklabels(MAIN, fontsize=8)
            ax.grid(axis="x", visible=False)
            if row == 0:
                ax.set_title(s + (" (E1's 432 runs)" if s == "e1-core" else ""))
                _budget(ax, BUDGET_MS, "0.17 m/s", left=True, below=True)
            else:
                _budget(ax, BUDGET_PCT, "0.28 %", left=True, below=True)
                _budget(ax, DRIFT_PCT, "0.5 %", color=MUTED, left=True)
        axes[0, 0].set_ylabel("per-run bias b (m/s, log)")
        axes[1, 0].set_ylabel("b as % of ground speed (log)")
    key = "   ".join(f"{mid} = {BY_ID[mid].label}" for mid in MAIN[:4]) + "\n" + "   ".join(
        f"{mid} = {BY_ID[mid].label}" for mid in MAIN[4:])
    fig.suptitle("Per-run bias by method and set (dots = runs, bar = median; values outside the axis drawn at its edge)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, 0.005, key + " (K = 32, chosen a priori; val prefers K = 16).\nB2 reads the label (E1's truth-centred prior) and "
             "scores ~4 windows per run on the E2 sets; it is not a working classical reference on E2 flights "
             "(it also fails on the frozen sea; cause not isolated).",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(OUT / "fig_bias_by_set.png", dpi=150)
    plt.close(fig)


def fig_bias_vs_conditions(plt, sl: pd.DataFrame):
    show = ["B0", "B2", "A", "B", "C"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    panels = [("altitude", ["test-iid", "test-alt"], "altitude bin (m)"),
              ("ground_speed", ["test-iid"], "ground-speed bin (m/s)"),
              ("sun_elevation", ["test-iid"], "sun-elevation bin (deg)")]
    for ax, (slc, sets, xl) in zip(axes, panels):
        bins = (sl[(sl.slice == slc) & sl.set.isin(sets) & (sl.method == "A")]
                .sort_values(["bin_lo"])[["set", "bin", "bin_lo"]].drop_duplicates(["set", "bin"]))
        bins = bins.sort_values(["set", "bin_lo"], key=lambda c: c.map({s: i for i, s in enumerate(sets)}) if c.name == "set" else c)
        pos = {(r.set, r.bin): i for i, r in enumerate(bins.itertuples())}
        for mid in show:
            m = BY_ID[mid]
            for s in sets:
                d = sl[(sl.method == mid) & (sl.slice == slc) & (sl.set == s)].sort_values("bin_lo")
                xs = [pos[(s, b)] for b in d["bin"]]
                ax.plot(xs, d.b_median, color=COLOR[m.family], marker=MARKER[m.family], ms=5, lw=1.6,
                        label=m.label if s == sets[0] else None,
                        ls="-" if mid != "B2" else (0, (1, 1.5)))
        if "test-alt" in sets:
            first_alt = min(i for (s, _), i in pos.items() if s == "test-alt")
            ax.axvspan(first_alt - 0.5, len(pos) - 0.5, color="#f0efec", zorder=0)
            ax.text(first_alt - 0.4, 22, "test-alt (held out)", fontsize=7.5, color=INK2, va="top")
            ax.text(-0.4, 22, "test-iid", fontsize=7.5, color=INK2, va="top")
        ax.set_xticks(range(len(pos)))
        ax.set_xticklabels([_short_bin(b) for (_, b) in pos], fontsize=7.5)
        ax.set_xlabel(xl)
        ax.set_yscale("log")
        ax.set_ylim(0.05, 25)
        ax.grid(axis="x", visible=False)
        _budget(ax, BUDGET_MS, "0.17 m/s", left=True, below=True)
    axes[0].set_ylabel("median per-run b (m/s, log)")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, 0.93), fontsize=7.5)
    fig.suptitle("Median per-run bias against altitude, ground speed and sun (test-iid; altitude adds test-alt). "
                 "B2 dotted: uses the label.", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "fig_bias_vs_conditions.png", dpi=150)
    plt.close(fig)


def _short_bin(b: str) -> str:
    m = re.match(r"^\(([-0-9.]+), ([-0-9.]+)\]$", b)
    if not m:
        return b
    a, c = float(m.group(1)), float(m.group(2))
    if a <= -89:
        return f"≤ {round(c):g}"
    if (a, c) == (65, 200):  # the harness's last ground-speed bin is open-ended
        return f"> {a:g}"
    return f"{max(a, 0):g}–{round(c):g}"


def fig_e1_core(plt, e1: pd.DataFrame):
    order = ["e1_flow_searaft_g1", "e1_shell_dispersive_T3.2", "e1_shell_dispersive_T6.4", "B0", "B2", "B3", "A", "B", "C", "C-K16"]
    names = ["E1\nSEA-RAFT", "E1 shell\nT=3.2", "E1 shell\nT=6.4", "B0", "B2\n(same tiles)", "B3", "A", "B",
             "C K=32\n(a priori)", "C K=16\n(val pick)"]
    fam = {"e1_flow_searaft_g1": "E1", "e1_shell_dispersive_T3.2": "E1", "e1_shell_dispersive_T6.4": "E1"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, sun in zip(axes, ["midday", "low sun"]):
        for i, mid in enumerate(order):
            r = e1[(e1.method == mid) & (e1.sun == sun) & (e1.altitude == "all")].iloc[0]
            c = COLOR[fam.get(mid, BY_ID[mid].family if mid in BY_ID else "E1")]
            ax.plot([i, i], [r.b_p25, r.b_p75], color=c, lw=3, solid_capstyle="round", alpha=0.6)
            ax.plot(i, r.b_median, marker=MARKER.get(BY_ID[mid].family, "o") if mid in BY_ID else "o", color=c, ms=7,
                    mec="#fcfcfb", mew=1.2)
            ax.annotate(f"{r.b_median:.2f}", (i, r.b_median), xytext=(6, 0), textcoords="offset points",
                        fontsize=7.5, color=INK2, va="center")
        ax.axvline(2.5, color=GRID, lw=1)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(names, fontsize=8)
        ax.set_yscale("log")
        ax.set_ylim(0.02, 8)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"e1-core, {sun} (216 runs = 6 preset seas, one sea seed; sun {'60' if sun == 'midday' else '7'} deg)",
                     fontsize=9.5)
        _budget(ax, BUDGET_MS, "0.17 m/s", left=True, below=True)
    axes[0].set_ylabel("per-run b (m/s, log): median and interquartile range")
    fig.suptitle("E1's 432 renders: E1's estimators (left of the rule; E1's per-run statistic, shell with a truth-centred prior) "
                 "against E2's", x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig_e1_core.png", dpi=150)
    plt.close(fig)


def fig_k_ablation(plt, k: pd.DataFrame):
    sets = ["val", "test-iid", "test-sea", "test-alt", "e1-core", "test-rate, seaRate 1"]
    labels = {"A": "A e2e", "C": "C spectral", "B0": "B0 phase corr.", "B1": "B1 SEA-RAFT-M", "B2": "B2 shell (label)"}
    fig, axes = plt.subplots(1, len(sets), figsize=(16, 4.0), sharey=True)
    for ax, s in zip(axes, sets):
        for f in ["B1", "B0", "B2", "C", "A"]:
            d = k[(k.family == f) & (k.set == s)].sort_values("K")
            if d.empty:
                continue
            ax.plot(d.K, d.b_median, color=COLOR[f], marker=MARKER[f], ms=5, lw=1.6, label=labels[f],
                    ls="-" if f != "B2" else (0, (1, 1.5)))
        ax.set_xscale("log", base=2)
        ax.set_xticks([2, 4, 8, 16, 32, 64])
        ax.set_xticklabels(["2", "4", "8", "16", "32", "64"])
        ax.set_yscale("log")
        ax.set_ylim(0.1, 5)
        ax.set_title(s, fontsize=9.5)
        ax.set_xlabel("K (frames per window, 10 fps)")
        _budget(ax, BUDGET_MS, "0.17 m/s", left=True, below=True)
    axes[0].set_ylabel("median per-run b (m/s, log)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h[::-1], lab[::-1], loc="lower center", ncol=5, fontsize=8)
    fig.suptitle("Window length K: learned models against the baselines (one training run per K; C's K=16 had 2.5x less training)",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.savefig(OUT / "fig_k_ablation.png", dpi=150)
    plt.close(fig)


def fig_sea_rate(plt, rd: pd.DataFrame):
    show = [("B0", "B0"), ("B3", "B3"), ("A", "A"), ("A-K2", "A"), ("B", "B"), ("C", "C"), ("offset", "offset")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for mid, f in show:
        d = rd[rd.method == mid].sort_values("sea_rate")
        lab = d.label.iloc[0]
        kw = dict(color=COLOR[f], marker=MARKER.get(f, None), ms=5, lw=1.6, label=lab)
        if mid == "A-K2":
            kw.update(ls=(0, (4, 2)), mfc="#fcfcfb")
        if mid == "offset":
            kw.update(ls=(0, (1, 1.5)), marker=None, lw=1.4)
        if mid == "offset":  # exact at r = 1 by construction, so only the other rates, unjoined
            d0 = d[d.sea_rate != 1]
            axes[0].plot(d0.sea_rate, d0.b_median, **{**kw, "ls": "none", "marker": "_", "ms": 16, "mew": 2})
        else:
            axes[0].plot(d.sea_rate, d.b_median, **kw)
        axes[1].plot(d.sea_rate, d.along_wind_median, **kw)
    axes[0].set_yscale("log")
    axes[0].set_ylim(0.03, 6)
    _budget(axes[0], BUDGET_MS, "0.17 m/s", left=True, below=True)
    axes[0].set_ylabel("median per-run b (m/s, log)")
    axes[0].set_title("Size of the bias")
    axes[1].axhline(0, color=INK2, lw=0.8)
    axes[1].set_ylabel("median signed error along the wind sea (m/s)")
    axes[1].set_title("Direction: negative = pulled with the waves, like B0")
    for ax in axes:
        ax.set_xticks([0, 0.5, 1, 2])
        ax.set_xlabel("seaRate r (sea clock; 1 = real gravity, 0 = frozen)")
        ax.axvline(1, color=GRID, lw=6, zorder=0)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle("test-rate: the same 100 conditions with the sea's clock at 0, 0.5, 1 and 2. "
                 "A model that measured the water's motion would stay flat at its rate-1 value.",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.1, 1, 0.93))
    fig.savefig(OUT / "fig_sea_rate.png", dpi=150)
    plt.close(fig)


def fig_calibration(plt, cal: pd.DataFrame):
    sets = ["val", "test-iid", "test-sea", "test-alt", "e1-core", "test-rate, seaRate 1", "test-rate, seaRate 0",
            "test-rate, seaRate 0.5", "test-rate, seaRate 2"]
    show = ["A", "B", "C"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, (col, nominal) in zip(axes, [("within_1sigma_axis", 0.683), ("within_2sigma_axis", 0.954)]):
        for k, mid in enumerate(show):
            d = cal[cal.method == mid].set_index("set").reindex(sets)
            y = np.arange(len(sets)) + (k - 1) * 0.22
            ax.scatter(d[col], y, color=COLOR[mid], marker=MARKER[mid], s=36, label=BY_ID[mid].label, zorder=3,
                       edgecolors="#fcfcfb", linewidths=0.8)
        ax.axvline(nominal, color=INK, lw=1.0, ls=(0, (4, 3)))
        ax.annotate(f"Gaussian {nominal:.1%}", (nominal, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -2),
                    textcoords="offset points", fontsize=7.5, color=INK2, va="top")
        ax.axhspan(4.5, len(sets) - 0.5, color="#f0efec", zorder=0)
        ax.set_xlim(0.2, 1.0)
        ax.set_xlabel(f"fraction of per-axis window errors within {col[7]} sigma")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(sets)))
    axes[0].set_yticklabels(sets, fontsize=8)
    axes[0].invert_yaxis()
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle("Window-level sigma: coverage by set (shaded: test-rate by seaRate; 0, 0.5 and 2 are off the training physics)",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(OUT / "fig_calibration.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- main

def main():
    cmp = comparison_by_set()
    cmp.to_csv(OUT / "comparison_by_set.csv", index=False)
    sl = comparison_by_slice()
    sl.to_csv(OUT / "comparison_by_slice.csv", index=False)
    e1 = e1_core_compare()
    e1.to_csv(OUT / "e1_core_compare.csv", index=False)
    k = k_ablation(cmp)
    k.to_csv(OUT / "k_ablation.csv", index=False)
    rd = rate_diagnostic()
    rd.to_csv(OUT / "rate_diagnostic.csv", index=False)
    cal = calibration()
    cal.to_csv(OUT / "calibration.csv", index=False)
    costs().to_csv(OUT / "costs.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    fig_bias_by_set(plt)
    fig_bias_vs_conditions(plt, sl)
    fig_e1_core(plt, e1)
    fig_k_ablation(plt, k)
    fig_sea_rate(plt, rd)
    fig_calibration(plt, cal)

    pd.set_option("display.width", 200)
    main_rows = cmp[cmp.role != "ablation"].pivot_table(index=["method"], columns="set", values="b_median")
    print(main_rows.round(3))
    print(f"wrote {len(cmp)} by-set rows, {len(sl)} slice rows, {len(rd)} rate rows, {len(cal)} calibration rows to {OUT}")


if __name__ == "__main__":
    main()
