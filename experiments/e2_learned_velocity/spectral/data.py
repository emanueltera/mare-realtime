"""Windows for the spectral model: raw float16 tiles (NaN = masked; the spectrum does its own
normalisation), gsd, dt, the window's pattern velocity and the label, with the same exact
dihedral augmentation as e2.data applied to tiles, label and pattern velocity alike (the
vectors here in the worker, the tiles on the GPU by augment_tiles)."""

from __future__ import annotations

import numpy as np
import torch

from e2.data import DIHEDRAL, TileSet, augment_vector

from .pattern import load_pairs, window_pattern


class SpectralWindows(torch.utils.data.Dataset):
    """K-frame windows of a TileSet. `register=False` gives a zero pattern velocity (the
    no-registration ablation). Items: tiles [K, N, N] float16, gsd, dt, centre [2], target [2],
    aug, i (row of self.index)."""

    def __init__(self, ts: TileSet, K: int, *, step: int = 1, stride: int = 1, augment: bool = False,
                 register: bool = True, runs=None, pairs=None, seed: int = 0):
        self.ts, self.K, self.stride, self.augment = ts, K, stride, augment
        self.index = ts.windows(K, stride, step, 0.0, runs).reset_index(drop=True)
        if register:
            vp = window_pattern(pairs if pairs is not None else load_pairs(ts.name), self.index)
        else:
            vp = np.zeros((len(self.index), 2))
        self.index["pat_missing"] = ~np.isfinite(vp).all(-1)
        vp = np.where(np.isfinite(vp), vp, 0.0)
        self.index["vpat_x"], self.index["vpat_y"] = vp[:, 0], vp[:, 1]
        self._cols = {c: self.index[c].to_numpy() for c in ("run_id", "w_start", "gsd", "dt", "v_true_x", "v_true_y",
                                                            "vpat_x", "vpat_y")}
        self.seed = seed

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        c = self._cols
        # Opened per item and dropped (not TileSet.tiles, which keeps every run's map, and
        # with it a file descriptor, open in each worker).
        a = np.load(self.ts.root / "tiles" / f"{c['run_id'][i]}.npy", mmap_mode="r")
        w = int(c["w_start"][i])
        x = np.array(a[w:w + (self.K - 1) * self.stride + 1:self.stride])
        tgt = np.array([c["v_true_x"][i], c["v_true_y"][i]])
        vp = np.array([c["vpat_x"][i], c["vpat_y"][i]])
        g = 0
        if self.augment:
            g = int(np.random.randint(8))
            tgt, vp = augment_vector(tgt, g), augment_vector(vp, g)
        return {"tiles": torch.from_numpy(x), "gsd": np.float32(c["gsd"][i]),
                "dt": np.float32(c["dt"][i]), "centre": vp.astype(np.float32), "target": tgt.astype(np.float32),
                "aug": g, "i": i}


def augment_tiles(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Dihedral element g[b] on the last two axes of x[b] ([B, K, n, n], row = y, col = x),
    exactly as e2.data.augment_array."""
    out = []
    for xb, gb in zip(x, g.tolist()):
        fx, fy, tr = DIHEDRAL[int(gb)]
        if fx:
            xb = xb.flip(-1)
        if fy:
            xb = xb.flip(-2)
        if tr:
            xb = xb.transpose(-1, -2)
        out.append(xb)
    return torch.stack(out)


def worker_init(wid):
    np.random.seed((torch.initial_seed() + wid) % 2**32)
