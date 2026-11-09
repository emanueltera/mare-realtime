"""Window loading for model A: raw float16 tiles (NaN kept as the mask) straight from the
set's memory-mapped .npy files, so the page cache holds the train set once for every
process on the box; normalisation and everything else happen on the GPU (e2e.model.prep).

Per item: the K tiles of one window, its gsd (m/px) and dt (s) — the allowed inputs — and
the label (m/s, tile axes). Augmentations (training only), both exact symmetries of the
problem:
- the 8 rotations / flips of the square (e2.data.augment_array / augment_vector), the
  target transformed with the tiles;
- scale: 2 × 2 binning (cells averaged, gsd doubled). A tile of cells twice as large over
  the same water is close to what a camera twice as high sees (a cell ≈ one image pixel at
  the tile's own height), so it widens the gsd range seen in training to 0.085–0.66 m/px.
  The whole batch is binned or not (one tile size per batch): `ScaleBatches`.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

from e2.data import TileSet, augment_array, augment_vector


class Windows(torch.utils.data.Dataset):
    """All K-frame windows (stride 1, every `step` frames) of a TileSet. Items are indexed by
    an int or by (int, bin)."""

    def __init__(self, ts: TileSet, K: int, step: int = 1, runs=None, augment: bool = False, seed: int = 0):
        self.ts, self.K, self.augment, self.seed = ts, K, augment, seed
        self.index = ts.windows(K, 1, step, 0.0, runs).reset_index(drop=True)
        self.run_id = self.index["run_id"].to_numpy()
        self.w_start = self.index["w_start"].to_numpy(int)
        self.gsd = self.index["gsd"].to_numpy(np.float32)
        self.dt = self.index["dt"].to_numpy(np.float32)
        self.target = self.index[["v_true_x", "v_true_y"]].to_numpy(np.float32)
        self._rng = None

    def __len__(self):
        return len(self.index)

    def rng(self):
        if self._rng is None:  # one stream per worker process
            info = torch.utils.data.get_worker_info()
            self._rng = np.random.default_rng([self.seed, info.id if info else 0, torch.initial_seed() % 2**31])
        return self._rng

    def __getitem__(self, i):
        i, b = (i if isinstance(i, tuple) else (i, 1))
        w = self.w_start[i]
        # open the run's memmap per item: caching 4500 maps per worker runs out of file handles
        a = np.array(np.load(self.ts.root / "tiles" / f"{self.run_id[i]}.npy", mmap_mode="r")[w:w + self.K])
        gsd = float(self.gsd[i])
        if b > 1:
            n = a.shape[-1] // b
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)              # all-NaN blocks stay NaN
                a = np.nanmean(a[:, :n * b, :n * b].astype(np.float32).reshape(self.K, n, b, n, b), axis=(2, 4))
            a = a.astype(np.float16)
            gsd *= b
        target = self.target[i]
        g = 0
        if self.augment:
            g = int(self.rng().integers(8))
            a = augment_array(a, g)
            target = augment_vector(target, g).astype(np.float32)
        return {"tiles": np.ascontiguousarray(a), "gsd": np.float32(gsd), "dt": self.dt[i],
                "target": target, "aug": g, "i": i}


class ScaleBatches(torch.utils.data.Sampler):
    """Random batches of (index, bin) with one bin per batch: 2 with probability p_bin."""

    def __init__(self, n: int, batch: int, steps: int, p_bin: float = 0.0, seed: int = 0):
        self.n, self.batch, self.steps, self.p_bin, self.seed = n, batch, steps, p_bin, seed

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        for _ in range(self.steps):
            b = 2 if rng.random() < self.p_bin else 1
            yield [(int(i), b) for i in rng.integers(self.n, size=self.batch)]


def to_device(batch: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
