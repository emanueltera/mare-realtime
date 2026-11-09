"""Training curves of model A: python -m e2e.plots --runs main=<dir> abl-K2=<dir> … --out <png>.

Top: training L1 (m/s, mean over axes and windows, 100-step averages) of the start and of
the last iteration. Bottom: the val median per-run bias b of the EMA weights (val windows
every 4th start, as e2e.train scores them), with the 0.17 m/s budget."""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m e2e.plots")
    ap.add_argument("--runs", nargs="+", required=True, help="name=dir with log.csv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for i, spec in enumerate(a.runs):
        name, d = spec.split("=", 1)
        log = pd.read_csv(f"{d}/log.csv")
        last = sorted(c for c in log if c.startswith("l1_it"))[-1]
        c = f"C{i}"
        ax1.plot(log.step, log[last], color=c, label=f"{name}: last iteration")
        ax1.plot(log.step, log.l1_init, color=c, ls=":", lw=1, label=f"{name}: phase-correlation start")
        v = log.dropna(subset=["val_b_median"])
        ax2.plot(v.step, v.val_b_median, "o-", color=c, label=name)
    ax2.axhline(0.17, color="k", lw=0.8, ls="--", label="0.17 m/s budget")
    ax1.set_ylabel("train L1 per axis (m/s)")
    ax2.set_ylabel("val median per-run b (m/s)")
    ax2.set_xlabel("step (batch 16)")
    ax2.set_yscale("log")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(a.out, dpi=120)


if __name__ == "__main__":
    main()
