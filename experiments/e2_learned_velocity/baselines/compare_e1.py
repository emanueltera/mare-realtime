"""Cross-check on e1-core: B1 / B2 on E2's tiles against E1's per-run numbers for the same runs.

E1 and E2 see the same renders but measure differently:
- B1 vs E1 flow:searaft g=1: E1 ran SEA-RAFT-M on the full 720 × 1280 frame and back-projected
  each pixel's flow onto the sea with the true poses; 23 pairs per run (every 5th frame from
  frame 5), b from the median over pairs. B1 runs it on the 384² tile (0.62 × height square,
  random θ), all 120 pairs, b from the harness's mean over windows.
- B2 vs E1 shell T = 3.2 / 6.4 s: E1's tile was 256 cells over ~0.70 × the height,
  ENU-aligned; E2's is 384 cells over 0.62 × the height at θ; same fit, prior and window
  starts (every 1.6 s). E1's b is from the median over windows, the harness's from the mean.
Both are compared as vectors in ENU (b_e, b_n here; b_E, b_N in E1) and as |b|.

    python -m baselines.compare_e1 --eval /workspace/data/e2/eval --out results/baselines/e1_crosscheck
writes per_run.csv (one row per run and pairing) and summary.csv (medians per pairing, and
per pairing × E1 light / sea).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

E1_PER_RUN = Path(__file__).resolve().parents[2] / "e1_wave_bias" / "results" / "per_run.csv"

# (E2 method, K) ↔ (E1 method, param)
PAIRINGS = [
    ("b1-searaft", 2, "flow:searaft", "g=1"),
    ("b1-searaft", 8, "flow:searaft", "g=1"),
    ("b2-shell-dispersive", 32, "shell:dispersive", "T=3.2"),
    ("b2-shell-dispersive", 64, "shell:dispersive", "T=6.4"),
    ("b2-shell-static", 32, "shell:static", "T=3.2"),
    ("b2-shell-static", 64, "shell:static", "T=6.4"),
    # ablation: the shell restricted to E1's physical wavelength range (kmax_frac 0.475)
    ("b2-shell-dispersive-kmax0475", 32, "shell:dispersive", "T=3.2"),
    ("b2-shell-dispersive-kmax0475", 64, "shell:dispersive", "T=6.4"),
    ("b2-shell-static-kmax0475", 32, "shell:static", "T=3.2"),
    ("b2-shell-static-kmax0475", 64, "shell:static", "T=6.4"),
]


def compare(eval_root: Path, e1_csv: Path = E1_PER_RUN) -> pd.DataFrame:
    e1 = pd.read_csv(e1_csv)
    e1 = e1[e1.set == "core"]
    rows = []
    for m2, K, m1, p1 in PAIRINGS:
        f = eval_root / m2 / "e1-core" / "per_run.csv"
        if not f.exists():
            continue
        a = pd.read_csv(f)
        a = a[a.K == K][["run_id", "b", "b_e", "b_n", "ground_speed"]]
        b = e1[(e1.method == m1) & (e1.param == p1)][["run_id", "b_abs", "b_E", "b_N", "light", "sea", "altitude", "airspeed"]]
        j = a.merge(b, on="run_id", how="inner")
        j.insert(0, "pairing", f"{m2} K={K} vs E1 {m1} {p1}")
        j["d_b"] = j.b - j.b_abs  # E2 − E1, m/s
        j["d_vec"] = np.hypot(j.b_e - j.b_E, j.b_n - j.b_N)  # |b_E2 − b_E1| as vectors, m/s
        rows.append(j)
    return pd.concat(rows, ignore_index=True)


def summarise(pr: pd.DataFrame) -> pd.DataFrame:
    def stats(g):
        ok = g[["b", "b_abs"]].notna().all(axis=1)
        g = g[ok]
        return pd.Series({
            "runs": len(g), "b_e2_median": g.b.median(), "b_e1_median": g.b_abs.median(),
            "d_b_median": g.d_b.median(), "abs_d_b_median": g.d_b.abs().median(),
            "d_vec_median": g.d_vec.median(), "d_vec_p90": g.d_vec.quantile(0.9),
            "pearson_b": g.b.corr(g.b_abs) if len(g) > 2 else np.nan,
            "spearman_b": g.b.corr(g.b_abs, method="spearman") if len(g) > 2 else np.nan,
            "frac_e2_le_017": (g.b <= 0.17).mean(), "frac_e1_le_017": (g.b_abs <= 0.17).mean(),
        })

    out = [pr.groupby("pairing").apply(stats, include_groups=False).reset_index().assign(slice="all", bin="all")]
    for col in ("light", "sea"):
        s = pr.groupby(["pairing", col]).apply(stats, include_groups=False).reset_index().rename(columns={col: "bin"})
        out.append(s.assign(slice=col))
    cols = ["pairing", "slice", "bin"]
    o = pd.concat(out, ignore_index=True)
    return o[cols + [c for c in o if c not in cols]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="python -m baselines.compare_e1")
    ap.add_argument("--eval", default="/workspace/data/e2/eval")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pr = compare(Path(a.eval))
    pr.to_csv(out / "per_run.csv", index=False)
    sm = summarise(pr)
    sm.to_csv(out / "summary.csv", index=False)
    print(sm[sm.slice == "all"].to_string(index=False))
