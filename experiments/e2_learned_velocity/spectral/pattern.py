"""The classical pattern velocity that centres the learned search (v = v_pattern + δ).

Masked phase correlation between consecutive tiles (e2.baselines, no learning): one camera
velocity per pair (w, w + 1), in tile axes, written in the standard prediction format with
K = 2. A window's pattern velocity is the median over the K − 1 pairs inside it. It is biased
by the waves' own motion (that is what δ removes), but it is computed from the tiles alone.

The shared baseline already covers val, e1-core and test-rate (/workspace/data/e2/eval/
phasecorr/<set>.csv); the other sets are written to /workspace/data/e2/spectral/pattern/
by the same command (see NOTES.md).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

SOURCES = [Path(p) for p in os.environ.get("SPECTRAL_PATTERN_DIRS", "").split(":") if p] + [
    Path("/workspace/data/e2/eval/phasecorr"), Path("/workspace/data/e2/spectral/pattern")]


def pair_csv(set_name: str) -> Path:
    for d in SOURCES:
        p = d / f"{set_name}.csv"
        if p.exists():
            return p
    raise FileNotFoundError(f"no phase-correlation pairs for {set_name} in {SOURCES}")


def load_pairs(set_name: str, path: Path | None = None) -> dict[str, np.ndarray]:
    """run_id → [F − 1, 2] pair velocities (tile axes, NaN where phase correlation failed)."""
    df = pd.read_csv(path or pair_csv(set_name), usecols=["run_id", "w_start", "K", "v_est_x", "v_est_y"])
    df = df[df.K == 2]
    out = {}
    for rid, g in df.groupby("run_id", sort=False):
        n = int(g.w_start.max()) + 1
        a = np.full((n, 2), np.nan)
        a[g.w_start.to_numpy()] = g[["v_est_x", "v_est_y"]].to_numpy()
        out[rid] = a
    return out


def window_pattern(pairs: dict[str, np.ndarray], index: pd.DataFrame) -> np.ndarray:
    """Median pair velocity over each window's pairs w_start … w_start + (K − 1)·stride − 1.
    [len(index), 2]; NaN where every pair failed (the caller falls back to 0)."""
    out = np.full((len(index), 2), np.nan)
    for i, (rid, w, K, st) in enumerate(zip(index.run_id, index.w_start, index.K, index.stride)):
        a = pairs.get(rid)
        if a is None:
            continue
        seg = a[int(w):int(w) + (int(K) - 1) * int(st)]
        if np.isfinite(seg).all(-1).any():
            out[i] = np.nanmedian(seg, axis=0)
    return out
