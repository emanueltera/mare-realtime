"""Figures for RESULTS.md: flight 0087 against its GNSS, and drift on the simulated test sets.

    python plots.py <extract_real.py output dir> <e3.real output dir, --gyro att> <e3.real output dir, --gyro gyro>

Writes results/fig_real_tracks.png, fig_real_along_flight.png, fig_real_drift_by_stretch.png and
fig_sim_drift.png. Needs numpy, pandas and matplotlib.
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "results"
real, ev_att, ev_gyro = map(Path, sys.argv[1:4])

# Reference palette (light): categorical slots in fixed order, ink and chrome.
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False, "lines.linewidth": 2,
})

idx = json.loads((real / "index.json").read_text())["stretches"]
za, zg = np.load(ev_att / "preds.npz"), np.load(ev_gyro / "preds.npz")
pa = pd.read_csv(ev_att / "per_stretch.csv").set_index("stretch")
pg = pd.read_csv(ev_gyro / "per_stretch.csv").set_index("stretch")


def load(name):
    r = np.load(real / name / "real.npz")
    dp = za[f"{name}/dp"]
    world = np.einsum("nij,nj->ni", r["R"][:-1], dp)[:, :2]
    return r, world


def angle(a, b):  # angle from b to a in degrees, positive counterclockwise (a left of b)
    return np.degrees(np.arctan2(b[:, 0] * a[:, 1] - b[:, 1] * a[:, 0], (a * b).sum(1)))


def rolling_median(x, n):
    return pd.Series(x).rolling(n, center=True, min_periods=n // 2).median().to_numpy()


# 1. Tracks: GNSS against the network's steps placed with the attitude, per stretch.
fig, axs = plt.subplots(3, 3, figsize=(11, 11))
for ax, st in zip(axs.flat, idx):
    name = st["name"]
    r, world = load(name)
    en = r["en"] / 1000
    net = np.vstack([en[:1], en[0] + np.cumsum(world, 0) / 1000])
    fit = np.vstack([en[:1], en[0] + np.cumsum(world * pa.loc[name, "scale_fit"], 0) / 1000])
    ax.plot(en[:, 0], en[:, 1], color=INK, lw=2, label="GNSS")
    ax.plot(net[:, 0], net[:, 1], color=S1, lw=2, label="network, attitude heading")
    ax.plot(fit[:, 0], fit[:, 1], color=S2, lw=1.2, ls=(0, (4, 2)), label="network, scale fitted")
    ax.plot(*en[0], "o", color=INK, ms=6, mec=SURF, mew=2)
    ax.set_aspect("equal", adjustable="datalim")
    k = int(name.split("-")[1])
    ax.set_title(f"stretch {k} · {st['seconds'] / 60:.1f} min · {pa.loc[name, 'path_km']:.1f} km", loc="left")
    ax.text(0.02, 0.02, f"drift {pa.loc[name, 'drift_att']:.1f} %\nscale fitted {pa.loc[name, 'drift_scaled']:.1f} %",
            transform=ax.transAxes, color=INK2, fontsize=8, va="bottom")
    ax.tick_params(labelsize=7)
for ax in axs[-1]:
    ax.set_xlabel("east, km")
for ax in axs[:, 0]:
    ax.set_ylabel("north, km")
h, lab = axs[0, 0].get_legend_handles_labels()
fig.legend(h, lab, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.965))
fig.suptitle("Flight 0087: the network's track against GNSS, 9 over-water stretches (dot = start)",
             x=0.06, ha="left", y=0.99, fontsize=12, fontweight="bold")
fig.text(0.06, 0.008, "Network = cam3k-noair, trained on simulated frames only. Steps placed with the recording's attitude "
         "(heading GNSS-aided);\nstep scale from the recording's height, which was derived using GNSS speed. "
         "Scale fitted = the same steps times one least-squares scale per stretch, which removes scale entirely.",
         color=MUTED, fontsize=8, ha="left", va="bottom")
fig.tight_layout(rect=(0, 0.045, 1, 0.95))
fig.savefig(OUT / "fig_real_tracks.png", dpi=150)
plt.close(fig)

# 2. Along the flight: direction of the predicted step against GNSS, crab, and per-step scale.
fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True)
for k, st in enumerate(idx):
    name = st["name"]
    r, world = load(name)
    true = np.diff(r["en"], axis=0)
    nose = -r["R"][:-1, :2, 1]
    tmin = r["frame"][:-1] / 10 / 60
    dirn = -angle(world, true)  # positive = prediction to the right of the GNSS track
    crab = -angle(true, nose)   # positive = track to the right of the nose
    ratio = za[f"{name}/ratio"]
    lab = k == 0
    a1.plot(tmin, rolling_median(dirn, 300), color=S1, lw=2, label="predicted step, right of GNSS track" if lab else None)
    a1.plot(tmin, rolling_median(crab, 300), color=S2, lw=1.2, ls=(0, (4, 2)), label="crab: GNSS track, right of the nose" if lab else None)
    a2.plot(tmin, rolling_median(ratio, 300), color=S1, lw=2)
    for ax in (a1, a2):
        ax.axvspan(tmin[0], tmin[-1], color=GRID, alpha=0.35, lw=0, zorder=0)
    a1.text(tmin.mean(), 8.0, str(k), ha="center", color=MUTED, fontsize=8)
a1.axhline(0, color=AXIS, lw=1)
a2.axhline(1, color=AXIS, lw=1)
a1.set_ylim(-8, 9)
a1.set_ylabel("degrees, 30 s median")
a1.set_title("Direction of the network's steps against the GNSS track, with the crab angle for comparison", loc="left", pad=22)
a1.legend(loc="lower left", ncol=2, bbox_to_anchor=(0, 1.0), borderaxespad=0.2)
a2.set_ylabel("|network| / |GNSS| per step,\n30 s median")
a2.set_title("Step length against GNSS (scale is not independent here: the height was derived using GNSS speed)", loc="left")
a2.set_xlabel("minutes into the recording (shaded = scored over-water stretch, numbered above)")
fig.tight_layout()
fig.savefig(OUT / "fig_real_along_flight.png", dpi=150)
plt.close(fig)

# 3. Drift per stretch: attitude heading, scale fitted, heading from chaining the camera's gyro.
fig, ax = plt.subplots(figsize=(11, 4.6))
names = [s["name"] for s in idx]
x = np.arange(len(names))
w = 0.26
series = [(pa.drift_att, S1, "attitude heading"), (pa.drift_scaled, S2, "attitude heading, scale fitted"),
          (pg.drift_chain, S3, "heading from chaining the camera's gyro")]
for j, (vals, c, lab) in enumerate(series):
    v = vals.loc[names].to_numpy()
    ax.bar(x + (j - 1) * w, v, w - 0.03, color=c, label=lab, zorder=2)
    for xi, vi in zip(x + (j - 1) * w, v):
        if vi > 20:
            ax.text(xi, vi * 1.08, f"{vi:.0f}", ha="center", color=INK2, fontsize=7.5)
ax.set_yscale("log")
ax.set_ylim(0.1, 300)
ax.set_yticks([0.1, 0.5, 1, 2, 5, 10, 30, 100])
ax.set_yticklabels(["0.1", "0.5", "1", "2", "5", "10", "30", "100"])
ax.axhline(0.5, color=INK2, lw=1, ls=(0, (2, 2)), label="0.5 % target")
ax.set_xticks(x, [f"{n.split('-')[1].lstrip('0') or '0'}\n{pa.loc[n, 'path_km']:.0f} km" for n in names])
ax.set_xlabel("stretch")
ax.set_ylabel("end error / distance flown, % (log)")
ax.set_title("Flight 0087 drift per stretch: the network is a few %, the gyro alone cannot hold heading", loc="left")
ax.legend(loc="upper left", ncol=4)
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(OUT / "fig_real_drift_by_stretch.png", dpi=150)
plt.close(fig)


# 4. Simulated test sets: median drift by model, true attitude, with flight 0087 for reference.
def drift_table(s):
    rows = {}
    for line in (HERE / "results" / "eval" / s / "summary.md").read_text().split("## By pattern")[0].splitlines():
        c = [t.strip() for t in line.split("|")]
        if len(c) > 6 and c[3] == "true" and re.match(r"^[\d.]+$", c[5]):
            rows[c[2]] = float(c[5])
    return rows


fig, (b1, b2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw=dict(width_ratios=[3, 2.3]))
sets = ["test-long", "test-varied", "test-varied-fx"]
models = [("e3", S1, "e3 (E3's network)"), ("var3k", S2, "var3k"), ("var3k-s1", S3, "var3k-s1"), ("var6k", S4, "var6k")]
x = np.arange(len(sets))
w = 0.2
tabs = {s: drift_table(s) for s in sets}
for j, (m, c, lab) in enumerate(models):
    v = [tabs[s][m] for s in sets]
    xs = x + (j - 1.5) * w
    b1.bar(xs, v, w - 0.03, color=c, label=lab, zorder=2)
    for xi, vi in zip(xs, v):
        b1.text(xi, vi + 0.06, f"{vi:.1f}", ha="center", color=INK2, fontsize=7.5)
b1.set_xticks(x, ["E3's scenes\n(test-long)", "varied scenes\n(test-varied)", "varied + camera effects\n(test-varied-fx)"])
b1.set_title("Portrait camera (E3 format)", loc="left")
b1.set_ylabel("median drift, % of distance (true attitude)")
b1.legend(loc="upper left", ncol=2)

cam = {s: drift_table(s) for s in ["test-cam", "test-cam-fx"]}
groups = ["test-cam", "test-cam-fx"]
x2 = np.arange(len(groups))
for j, (m, c, lab) in enumerate([("cam3k", S1, "cam3k"), ("cam3k-noair", S2, "cam3k-noair")]):
    v = [cam[g][m] for g in groups]
    xs = x2 + (j - 0.5) * 0.3
    b2.bar(xs, v, 0.27, color=c, label=lab, zorder=2)
    for xi, vi in zip(xs, v):
        b2.text(xi, vi + 0.06, f"{vi:.1f}", ha="center", color=INK2, fontsize=7.5)
wts = pa.path_km / pa.path_km.sum()
real_vals = [(wts * pa.drift_att).sum(), (wts * pa.drift_scaled).sum()]
xr = np.array([2.0, 2.35])
b2.bar(xr, real_vals, 0.27, color=S2, zorder=2, hatch="////", edgecolor=SURF, lw=0)
for xi, vi, t in zip(xr, real_vals, ["attitude\nheading", "scale\nfitted"]):
    b2.text(xi, vi + 0.06, f"{vi:.1f}", ha="center", color=INK2, fontsize=7.5)
    b2.text(xi, -0.62, t, ha="center", va="top", color=MUTED, fontsize=7)
b2.set_xticks(list(x2) + [2.175], ["simulated\n(test-cam)", "simulated + effects\n(test-cam-fx)", "flight 0087\n(real, cam3k-noair)"])
b2.set_title("Onboard camera's format, and the real flight", loc="left")
b2.legend(loc="upper left")
for ax in (b1, b2):
    ax.axhline(0.5, color=INK2, lw=1, ls=(0, (2, 2)), label="0.5 % target")
    ax.set_ylim(0, 4.6)
    ax.grid(axis="x", visible=False)
b1.legend(loc="upper left", ncol=2)
b2.legend(loc="upper left")
fig.text(0.01, 0.005, "Simulated: median over 120 flights of 60 s, true attitude. Flight 0087: distance-weighted over 9 stretches "
         "(2–11 min each),\nattitude from the recording; its scale is not GNSS-free (height derived using GNSS speed).",
         color=MUTED, fontsize=8, ha="left", va="bottom")
fig.tight_layout(rect=(0, 0.075, 1, 1))
fig.savefig(OUT / "fig_sim_drift.png", dpi=150)
plt.close(fig)
print("wrote", *sorted(p.name for p in OUT.glob("fig_*.png")))
