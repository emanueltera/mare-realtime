"""E3 figures from the scored sets copied to results/eval/<set>/ (per_run.csv, per_step.csv),
the training logs (results/{net,blind}_log.csv) and, for the track and error-growth figures,
the per-pair predictions and run tables under E3_DATA (default below): preds/net-<set>.npz and
sets/<set>/<run>/ (run tables, no frames needed).

    uv run --no-project --with numpy --with pandas --with matplotlib python analyze.py

Writes results/fig_*.png, and results/analysis/{tracks_selected,error_growth}.csv.
Colours: network blue, sensors-only orange, airspeed dead reckoning grey, truth black.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from e3.data import DT, NOISE, Run, list_runs, run_seed, simulate
from e3.evaluate import chain

HERE = Path(__file__).parent
R = HERE / "results"
A = R / "analysis"
DATA = Path(os.environ.get("E3_DATA", "/private/tmp/claude-501/-Users-emanuel-repos-mayfly-demo/"
                           "02eac86e-e939-4778-9996-84cb2607a222/scratchpad/e3data"))
COL = {"net": "#2a6fdb", "blind": "#e07b00", "airspeed": "#888888"}
LABEL = {"net": "network (images + sensors)", "blind": "sensors only (no images)", "airspeed": "airspeed dead reckoning (no camera)"}
ATT = {"true": "true attitude", "ahrs": "AHRS attitude (with yaw error)", "chain": "own rotations"}
REF = 0.5  # reference line drawn on the drift plots, %
REF_KW = dict(color="k", ls=":", lw=1)

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "legend.fontsize": 9, "axes.spines.top": False, "axes.spines.right": False})


def plain_log(ax, axis="y"):
    """Log axis labelled with plain numbers (0.5, 1, 2, 5 ...) instead of powers of ten."""
    from matplotlib.ticker import FuncFormatter, LogLocator
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(LogLocator(subs=(1, 2, 5)))
    a.set_minor_locator(LogLocator(subs=np.arange(1, 10)))
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_formatter(FuncFormatter(lambda v, _: ""))


def cdf(ax, x, **kw):
    x = np.sort(x)
    ax.plot(x, np.arange(1, len(x) + 1) / len(x) * 100, **kw)


def boot_median_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """95 % percentile-bootstrap CI of the median over runs (rows of x), per column."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    m = np.median(x[idx], axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5], axis=0)
    return lo, hi


# ---- figures from the scored sets ------------------------------------------------------------


def fig_drift_cdf(pr: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
    for ax, att in zip(axes, ATT):
        txt = []
        for m in COL:
            d = pr[(pr.method == m) & (pr.att == att)]
            cdf(ax, d.drift_pct.clip(lower=1e-3), color=COL[m], lw=2, label=LABEL[m])
            txt.append(f"{m if m != 'net' else 'network'}: {d.drift_pct.median():.1f} %")
        ax.axvline(REF, **REF_KW, label=f"{REF} % reference line")
        ax.text(0.03, 0.97, "median drift\n" + "\n".join(txt), transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="0.8"))
        ax.set_xscale("log")
        plain_log(ax, "x")
        ax.set_title(f"steps placed with {ATT[att]}")
        ax.set_xlabel("drift: horizontal end error / distance flown (%)")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("flights with this drift or less (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=4, frameon=False)
    fig.suptitle("Drift over a minute, 120 simulated test flights of 60 s", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(R / "fig_drift_cdf.png", dpi=130)
    plt.close(fig)


def fig_drift_vs_conditions(pr: pd.DataFrame) -> None:
    d = pr[pr.att == "true"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    cols = [("altitude", "altitude above the sea (m)"), ("ground_speed", "ground speed (m/s)"), ("wind", "wind speed at the sea surface (m/s)")]
    for ax, (col, xl) in zip(axes, cols):
        for m in ("airspeed", "blind", "net"):
            dm = d[d.method == m]
            ax.scatter(dm[col], dm.drift_pct.clip(lower=1e-3), s=12, color=COL[m], alpha=0.75, label=LABEL[m])
        ax.axhline(REF, **REF_KW, label=f"{REF} % reference line")
        ax.set_yscale("log")
        plain_log(ax)
        ax.set_xlabel(xl)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("drift with true attitude (%)")
    h, l = axes[0].get_legend_handles_labels()
    order = [2, 1, 0, 3]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=4, frameon=False, markerscale=1.6)
    fig.suptitle("Drift against flight conditions, 120 test flights of 60 s (one dot per flight and method)", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(R / "fig_drift_vs_conditions.png", dpi=130)
    plt.close(fig)


def fig_sea_rate() -> None:
    rate_p = R / "eval/test-rate/per_run.csv"
    if not rate_p.exists():
        return
    rr = pd.read_csv(rate_p)
    rr = rr[rr.att == "true"]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    dodge = {"net": 0.0, "blind": -0.07, "airspeed": 0.07}
    for m in COL:
        g = rr[rr.method == m].groupby("sea_rate").drift_pct
        med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
        x = med.index.values + dodge[m]
        ax.errorbar(x, med.values, yerr=[med - q1, q3 - med], color=COL[m], marker="o", capsize=4, lw=2,
                    ls="-" if m == "net" else "none", label=LABEL[m])
        left = m == "blind"
        for xi, v in zip(x, med.values):
            ax.annotate(f"{v:.1f}", (xi, v), textcoords="offset points", xytext=(-9 if left else 9, 0),
                        ha="right" if left else "left", va="center", fontsize=8, color="0.25")
    ax.axhline(REF, **REF_KW, label=f"{REF} % reference line")
    ax.text(0.02, 0.6, "the two baselines use no images:\nthe same at every wave speed", transform=ax.transAxes, fontsize=8, color="0.3")
    ax.set_yscale("log")
    plain_log(ax)
    ax.set_xticks([0, 1, 2], ["0\nfrozen sea", "1\nas in training", "2\ndouble speed"])
    ax.set_xlim(-0.4, 2.4)
    ax.set_xlabel("speed of the simulated waves")
    ax.set_ylabel("drift with true attitude (%): median, bar = IQR")
    ax.set_title("The same 30 flights at three wave speeds (20 s each)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="center right", fontsize=8, bbox_to_anchor=(1, 0.6))
    fig.tight_layout()
    fig.savefig(R / "fig_sea_rate.png", dpi=130)
    plt.close(fig)


# ---- (c) training ----------------------------------------------------------------------------


def fig_training() -> None:
    logs = {m: pd.read_csv(R / f"{m}_log.csv") for m in ("net", "blind") if (R / f"{m}_log.csv").exists()}
    if not logs:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    xl = "epoch (one pass over the 360 k training pairs)"
    panels = [("val_t_err_m", "mean translation error per step (m)", "{:.3f} m"),
              ("val_t_err_med_pct", "median translation error per step (% of step)", "{:.1f} %")]
    for ax, (col, yl, fmt) in zip(axes[:2], panels):
        for m, lg in logs.items():
            ax.plot(lg.epoch + 1, lg[col], marker="o", ms=4, lw=2, color=COL[m], label=f"{LABEL[m]}, {lg.hours.iloc[-1]:.2g} h")
            ax.annotate(fmt.format(lg[col].iloc[-1]), (lg.epoch.iloc[-1] + 1, lg[col].iloc[-1]), textcoords="offset points",
                        xytext=(0, 8), ha="right", fontsize=9, color="0.2")
        ax.set_yscale("log")
        plain_log(ax)
        ax.set_ylabel(yl)
        ax.set_xlabel(xl)
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(loc="center right")
    axes[0].set_title("Translation, metres")
    axes[1].set_title("Translation, relative to the step")
    ax = axes[2]
    for m, lg in logs.items():
        ax.plot(lg.epoch + 1, lg.val_rot_net_deg * 1000, marker="o", ms=4, lw=2, color=COL[m], label=LABEL[m])
    g = next(iter(logs.values())).val_rot_gyro_deg.iloc[0] * 1000
    ax.axhline(g, color="k", ls="--", lw=1.2, label=f"gyro alone ({g:.2f})")
    ax.set_ylabel("mean rotation error per step (thousandths of a degree)")
    ax.set_xlabel(xl)
    ax.set_title("Rotation (gyro + learned correction)")
    ax.set_ylim(g - 0.1, g + 0.1)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Training curves, scored on the 100 validation flights (12 s each)", y=0.99)
    fig.tight_layout()
    fig.savefig(R / "fig_training.png", dpi=130)
    plt.close(fig)
    for m in ("net", "blind"):
        (R / f"fig_training_{m}.png").unlink(missing_ok=True)


# ---- (a, b) tracks rebuilt from the predictions ----------------------------------------------


def rebuild(set_name: str) -> dict[str, dict]:
    """Per run: truth and the chained tracks (net true / net ahrs / airspeed ahrs), built as e3.evaluate builds them."""
    z = np.load(DATA / f"preds/net-{set_name}.npz")
    runs = [Run.load(d) for d in list_runs(DATA / "sets" / set_name)]
    assert list(z["runs"]) == [r.id for r in runs]
    out = {}
    for ri, run in enumerate(runs):
        s = simulate(run, np.random.default_rng(run_seed(run)), NOISE)
        m = z["run"] == ri
        o = np.argsort(z["i"][m])
        assert (z["i"][m][o] == np.arange(run.n - 1)).all()
        dp_net = z["dp_pred"][m][o]
        dp_air = run.nose[None] * s.airspeed[:-1, None] * DT
        tr = {"net-true": chain(run.p[0], run.R[:-1], dp_net), "net-ahrs": chain(run.p[0], s.R_ahrs[:-1], dp_net),
              "airspeed-true": chain(run.p[0], run.R[:-1], dp_air), "airspeed-ahrs": chain(run.p[0], s.R_ahrs[:-1], dp_air)}
        out[run.id] = dict(run=run, p=run.p[:, :2], tr={k: v[:, :2] for k, v in tr.items()})
    return out


TRACK_STYLE = {"net-true": dict(color=COL["net"], lw=1.8, label="network, true attitude"),
               "net-ahrs": dict(color=COL["net"], lw=1.8, ls="--", label="network, AHRS attitude"),
               "airspeed-ahrs": dict(color=COL["airspeed"], lw=1.8, label="airspeed dead reckoning, AHRS attitude")}


def pick_runs(pr: pd.DataFrame) -> list[tuple[str, str]]:
    """Best, median and worst by network drift (true attitude), then the median flight of three other patterns."""
    d = pr[(pr.method == "net") & (pr.att == "true")].sort_values("drift_pct").reset_index(drop=True)
    picks = [(d.run.iloc[0], "best"), (d.run.iloc[len(d) // 2], "median"), (d.run.iloc[-1], "worst")]
    for pat in ("straight", "lawnmower", "figure8"):
        dp = d[(d.pattern == pat) & ~d.run.isin([p[0] for p in picks])]
        j = (dp.drift_pct - dp.drift_pct.median()).abs().idxmin()
        picks.append((dp.run.loc[j], f"typical {pat}"))
    return picks


def fig_tracks(pr: pd.DataFrame, tracks: dict) -> None:
    picks = pick_runs(pr)
    rows = []
    fig, axes = plt.subplots(4, 3, figsize=(15, 17.5), gridspec_kw=dict(height_ratios=[1.35, 0.7, 1.35, 0.7]))
    for k, (rid, why) in enumerate(picks):
        T = tracks[rid]
        run, p0 = T["run"], T["p"][0]
        ax, ae = axes[2 * (k // 3), k % 3], axes[2 * (k // 3) + 1, k % 3]
        t = np.arange(run.n) * DT
        dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(T["p"], axis=0), axis=1))])
        ax.plot(*(T["p"] - p0).T, color="k", lw=2.6, label="true track", zorder=1)
        for key, st in TRACK_STYLE.items():
            ax.plot(*(T["tr"][key] - p0).T, zorder=2, **st)
            ax.plot(*(T["tr"][key][-1] - p0), "o", ms=5, color=st["color"], zorder=3)
            ae.plot(t, np.linalg.norm(T["tr"][key] - T["p"], axis=1), **{**st, "label": None})
        ax.plot(0, 0, "k^", ms=8, zorder=4)
        ax.plot(*(T["p"][-1] - p0), "ks", ms=6, zorder=4)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        ax.set_xlabel("east (m)")
        ax.set_ylabel("north (m)")
        q = pr[pr.run == rid].set_index(["method", "att"])
        pat = q.pattern.iloc[0]
        head = f"{why} flight" if pat in why else f"{why} flight ({pat})"
        ax.set_title(f"{head}: {q.altitude.iloc[0]:.0f} m up, {q.ground_speed.iloc[0]:.0f} m/s over the ground,\n"
                     f"wind {q.wind.iloc[0]:.0f} m/s, {dist[-1]:.0f} m flown. End drift: network {q.loc[('net', 'true'), 'drift_pct']:.2f} %\n"
                     f"(true attitude), {q.loc[('net', 'ahrs'), 'drift_pct']:.2f} % (AHRS); airspeed {q.loc[('airspeed', 'ahrs'), 'drift_pct']:.0f} % (AHRS)",
                     fontsize=9.5)
        ae.set_yscale("log")
        ae.set_ylim(0.1, 3000)
        ae.set_xlim(0, t[-1])
        ae.grid(alpha=0.3, which="both")
        ae.set_xlabel("time (s)")
        ae.set_ylabel("horizontal error (m)")
        rows.append(dict(run=rid, why=why, pattern=pat, path_m=dist[-1],
                         **{f"end_err_{key}_m": np.linalg.norm(T["tr"][key][-1] - T["p"][-1]) for key in TRACK_STYLE},
                         **{f"max_err_{key}_m": np.linalg.norm(T["tr"][key] - T["p"], axis=1).max() for key in TRACK_STYLE},
                         drift_net_true_pct=q.loc[("net", "true"), "drift_pct"], max_drift_net_true_pct=q.loc[("net", "true"), "max_drift_pct"]))
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=4, frameon=False, fontsize=10)
    fig.suptitle("Six 60 s test flights: true and estimated tracks (triangle = start, square = true end, dot = estimated end);\n"
                 "below each, the horizontal error against time", y=0.995, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(R / "fig_tracks.png", dpi=110)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(A / "tracks_selected.csv", index=False)


def fig_error_growth(tracks: dict) -> None:
    ids = sorted(tracks)
    n = min(T["run"].n for T in tracks.values())
    tt = np.arange(1, n) * DT  # t = 0 has no distance yet
    curves = {}
    for key in ("net-true", "net-ahrs"):
        rel = []
        for rid in ids:
            T = tracks[rid]
            dist = np.cumsum(np.linalg.norm(np.diff(T["p"], axis=0), axis=1))[: n - 1]
            e = np.linalg.norm(T["tr"][key] - T["p"], axis=1)[1:n]
            rel.append(e / dist * 100)
        curves[key] = np.array(rel)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    rows = []
    style = {"net-true": dict(ls="-", alpha=0.25, label="network, true attitude"),
             "net-ahrs": dict(ls="--", alpha=0.12, label="network, AHRS attitude")}
    for key, c in curves.items():
        med, q1, q3 = np.median(c, 0), np.percentile(c, 25, 0), np.percentile(c, 75, 0)
        lo, hi = boot_median_ci(c)
        st = style[key]
        ax.fill_between(tt, q1, q3, color=COL["net"], alpha=st["alpha"], lw=0.8, ls=st["ls"], edgecolor=COL["net"])
        ax.plot(tt, med, color=COL["net"], lw=2, ls=st["ls"], label=f"{st['label']}: median, band = middle half of flights")
        ax.annotate(f"{med[-1]:.2f} %\n95% CI {lo[-1]:.2f}-{hi[-1]:.2f}", (tt[-1], med[-1]), textcoords="offset points",
                    xytext=(6, 0), va="center", fontsize=8.5, color="0.2", annotation_clip=False)
        rows += [dict(method=key, t_s=round(tt[j], 1), median_pct=med[j], q25_pct=q1[j], q75_pct=q3[j],
                      median_ci_lo=lo[j], median_ci_hi=hi[j]) for j in range(len(tt))]
    ax.set_yscale("log")
    plain_log(ax)
    ax.set_xlim(0, tt[-1])
    ax.set_xlabel("time since start (s)")
    ax.set_ylabel("horizontal error / distance flown so far (%)")
    ax.set_title(f"Error relative to distance flown, {len(ids)} test flights of 60 s\n(median CI: bootstrap over flights)", fontsize=10.5)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right")
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    fig.savefig(R / "fig_error_growth.png", dpi=130)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(A / "error_growth.csv", index=False)


def main() -> None:
    A.mkdir(parents=True, exist_ok=True)
    pr = pd.read_csv(R / "eval/test-long/per_run.csv")
    fig_drift_cdf(pr)
    fig_drift_vs_conditions(pr)
    fig_sea_rate()
    fig_training()
    if not (DATA / "preds/net-test-long.npz").exists():
        print(f"no predictions under {DATA}: skipping fig_tracks and fig_error_growth")
        return
    tracks = rebuild("test-long")
    # the rebuilt tracks must reproduce evaluate's end errors
    worst = 0.0
    for r in pr[pr.method.isin(["net", "airspeed"]) & pr.att.isin(["true", "ahrs"])].itertuples():
        T = tracks[r.run]
        worst = max(worst, abs(np.linalg.norm(T["tr"][f"{r.method}-{r.att}"][-1] - T["p"][-1]) - r.end_err_m))
    print(f"rebuilt tracks vs per_run.csv: max end-error difference {worst:.2e} m")
    assert worst < 1e-3
    fig_tracks(pr, tracks)
    fig_error_growth(tracks)
    print("figures written to", R)


if __name__ == "__main__":
    main()
