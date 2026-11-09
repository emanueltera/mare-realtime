"""B0 / B1: pair velocities → K-frame window predictions.

A pair file is a standard prediction CSV with K = 2 rows, one per consecutive pair (i, i + 1)
of a run (e2.baselines for B0, baselines.flowpairs for B1). A window of K frames starting at
w holds the K − 1 pairs w … w + K − 2; its estimate is their component-wise median (NaN
pairs ignored), the pattern velocity over the window. K = 2 is the pair itself.

    python -m baselines.windows --pairs pairs/searaft/val.csv --set-root /workspace/data/e2/sets/val \\
        --out /workspace/data/e2/preds/b1-searaft/val.csv --K 2 8 16
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from e2.data import PRED_COLUMNS, TileSet, prediction_frame  # noqa: E402


def pair_arrays(pairs: pd.DataFrame) -> dict[str, np.ndarray]:
    """{run_id: [F − 1, 2] pair velocities in tile axes}, indexed by the pair's first frame."""
    out = {}
    for rid, g in pairs[pairs.K == 2].groupby("run_id", sort=False):
        g = g.sort_values("w_start")
        v = np.full((int(g.w_start.max()) + 1, 2), np.nan)
        v[g.w_start.to_numpy(int)] = g[["v_est_x", "v_est_y"]].to_numpy()
        out[rid] = v
    return out


def window_medians(v: np.ndarray, K: int) -> np.ndarray:
    """[F − 1, 2] pairs → [F − K + 1, 2] medians over each window's K − 1 pairs."""
    n = v.shape[0] - (K - 1) + 1
    if n <= 0:
        return np.zeros((0, 2))
    stack = np.stack([v[j:j + n] for j in range(K - 1)])  # [K − 1, n, 2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows stay NaN
        return np.nanmedian(stack, axis=0)


def window_predictions(ts: TileSet, pairs: pd.DataFrame, Ks=(2, 8)) -> pd.DataFrame:
    arrs = pair_arrays(pairs)
    keys, est = [], []
    for rid, v in arrs.items():
        t = ts._fr[rid]["t"].to_numpy()
        for K in Ks:
            m = window_medians(v, K)
            keys += [(ts.name, rid, w, K, float(t[w + K - 1] - t[w])) for w in range(len(m))]
            est.append(m)
    return prediction_frame(ts, keys, np.concatenate(est))[PRED_COLUMNS]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="python -m baselines.windows")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--set-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, nargs="+", default=[2, 8])
    a = ap.parse_args()
    df = window_predictions(TileSet(a.set_root), pd.read_csv(a.pairs), a.K)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df)} rows to {a.out}")
