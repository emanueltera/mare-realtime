"""Summaries and plots for the dispersion-shell fit (shell_fit.py and the synthetic sweep).

    python analyze_shell.py --pilot results/shell-pilot.csv --synthetic results/shell-synthetic.csv \
        --outdir results

Writes shell-summary-pilot.csv, shell-summary-synthetic.csv (and shell-summary-<name>.csv for
any --runs CSV, e.g. the core grid) and PNG plots. Every number in them is a statistic of
measured fit errors, except the columns prefixed res_ (derived resolution limits).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SLICE_RUN = ["model", "sea_rate", "sea", "light", "pitch", "altitude", "airspeed", "heading", "T"]


def agg(df: pd.DataFrame, keys) -> pd.DataFrame:
    g = df.groupby(keys, dropna=False)
    out = g.agg(
        n=("err_e", "size"),
        mean_err_e=("err_e", "mean"), mean_err_n=("err_n", "mean"),
        mean_err_along=("err_along", "mean"), mean_err_cross=("err_cross", "mean"),
        mean_err_along_wave=("err_along_wave", "mean"),
        rms_err_along=("err_along", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        rms_err_cross=("err_cross", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        median_abs_err=("err_abs", "median"), p90_abs_err=("err_abs", lambda x: float(np.quantile(x, 0.9))),
        max_abs_err=("err_abs", "max"),
        mean_speed=("speed", "mean"),
        shell_frac=("shell_frac", "mean"), band_occupancy=("band_occupancy", "mean"),
        lambda_c=("lambda_c", "mean"),
        res_bin_lambda_c=("res_bin_lambda_c", "mean"), res_dwell_lambda_c=("res_dwell_lambda_c", "mean"),
    ).reset_index()
    out["bias_abs"] = np.hypot(out.mean_err_e, out.mean_err_n)  # |mean error vector| (the bias)
    out["bias_pct"] = 100 * out.bias_abs / out.mean_speed
    out["rms_abs"] = np.hypot(out.rms_err_along, out.rms_err_cross)
    out["rms_pct"] = 100 * out.rms_abs / out.mean_speed
    return out


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def plot_synthetic(df: pd.DataFrame, out: Path, name: str = "synthetic"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cases = [("live", "dispersive", "Live sea, dispersive model"), ("frozen", "static", "Frozen sea, static model (control)"),
             ("live", "static", "Live sea, static model (pattern tracker)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor=SURF)
    for ax, (sea, model, title) in zip(axes, cases):
        _style(ax)
        d = df[(df.sea == sea) & (df.model == model)]
        if d.empty:
            ax.set_visible(False)
            continue
        for i, spd in enumerate(sorted(d.speed.unique())):
            g = d[d.speed == spd].groupby("T").err_abs
            med, p90 = g.median(), g.quantile(0.9)
            ax.plot(med.index, med.values, "-o", color=SERIES[i], lw=2, ms=5, label=f"{spd:g} m/s median")
            ax.plot(p90.index, p90.values, "--", color=SERIES[i], lw=1.2, label=f"{spd:g} m/s 90th pct")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([1.6, 3.2, 6.4, 12.0], ["1.6", "3.2", "6.4", "12"])
        ax.axhline(0.05, color=INK2, lw=1, ls=":")
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("window T (s)", fontsize=8, color=INK2)
    axes[0].set_ylabel("|V_est − V_true| (m/s)", fontsize=8, color=INK2)
    axes[0].legend(fontsize=7, frameon=False)
    q = df["quad"].iloc[0] if "quad" in df else 0
    kind = "linear slope image" if not q else f"slope image + {q:g}·slope² (nonlinear)"
    fig.suptitle(f"Synthetic sea, {kind} (42 m tile, 256², 10 fps): velocity error vs window; dotted line 0.05 m/s",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURF)
    plt.close(fig)


def plot_pilot(df: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = (df[["heading", "altitude", "airspeed"]].drop_duplicates().sort_values(["altitude", "heading"]).values.tolist())
    seas = sorted(df.sea.unique())
    fig, axes = plt.subplots(len(seas), len(runs), figsize=(4.2 * len(runs), 3.4 * len(seas)), facecolor=SURF, squeeze=False)
    cases = [(1, "dispersive", "live · dispersive"), (1, "static", "live · static"), (0, "static", "frozen · static"),
             (0, "dispersive", "frozen · dispersive")]
    for r, sea in enumerate(seas):
        for c, (hd, alt, spd) in enumerate(runs):
            ax = axes[r][c]
            _style(ax)
            for i, (rate, model, lab) in enumerate(cases):
                d = df[(df.sea == sea) & (df.heading == hd) & (df.altitude == alt) & (df.airspeed == spd)
                       & (df.sea_rate == rate) & (df.model == model)]
                if d.empty:
                    continue
                g = d.groupby("T").err_along
                x = np.array(g.mean().index) * (1 + 0.03 * (i - 1.5))
                ax.errorbar(x, g.mean().values, yerr=g.std().fillna(0).values, fmt="-o", color=SERIES[i], lw=1.6,
                            ms=4, capsize=2, label=lab)
            ax.axhline(0, color=INK2, lw=0.8)
            ax.set_xscale("log", base=2)
            ax.set_xticks([1.6, 3.2, 6.4, 12.0], ["1.6", "3.2", "6.4", "12"])
            ax.set_title(f"{sea} · {alt:g} m · {spd:g} m/s · hdg {hd:g}", fontsize=9, color=INK, loc="left")
            if r == len(seas) - 1:
                ax.set_xlabel("window T (s)", fontsize=8, color=INK2)
            if c == 0:
                ax.set_ylabel("along-track error (m/s)", fontsize=8, color=INK2)
    axes[0][0].legend(fontsize=7, frameon=True, framealpha=0.9, edgecolor=GRID, loc="lower right")
    fig.suptitle("Pilot renders: shell-fit along-track error (mean ± sd over window starts)", fontsize=10, color=INK,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURF)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot")
    ap.add_argument("--synthetic", nargs="*", default=[])
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()
    od = Path(a.outdir)
    od.mkdir(parents=True, exist_ok=True)
    for f in a.synthetic:
        s = pd.read_csv(f)
        name = Path(f).stem.replace("shell-", "")
        agg(s, ["sea", "model", "speed", "heading", "T"]).to_csv(od / f"shell-summary-{name}.csv", index=False)
        agg(s, ["sea", "model", "speed", "T"]).to_csv(od / f"shell-summary-{name}-by-speed.csv", index=False)
        plot_synthetic(s, od / f"shell-{name}.png", name)
    if a.pilot:
        p = pd.read_csv(a.pilot)
        agg(p, SLICE_RUN).to_csv(od / "shell-summary-pilot.csv", index=False)
        plot_pilot(p, od / "shell-pilot.png")
    for f in a.runs:
        d = pd.read_csv(f)
        name = Path(f).stem.replace("shell-", "")
        agg(d, SLICE_RUN).to_csv(od / f"shell-summary-{name}.csv", index=False)
        for k in ["sea", "altitude", "airspeed", "heading", "light"]:
            agg(d, ["model", "sea_rate", "pitch", k, "T"]).to_csv(od / f"shell-summary-{name}-by-{k}.csv", index=False)


if __name__ == "__main__":
    main()
