"""E2 tile sets: K-frame windows for training and evaluation.

A set directory (written by e2.rectify) holds
    tiles/<run_id>.npy  float16 [F, n, n] linear-light luminance, NaN = no image there
    frames.csv          run_id, frame, t, gsd, valid_frac                 (model-side)
    labels.csv          run_id, frame, t, cam_e, cam_n, cam_u, cam_x, cam_y, …   (labels only)
    runs.csv            per-run metadata (tile θ, c_run, sea, sky, flight …)    (labels only)

What a model may see, per window: the K tiles, their validity mask, the tile's ground
sampling distance (m/px) and the frame interval dt. Everything else (positions, run
metadata, the run id, the window's place in the run) is for labels and bookkeeping.

    from e2.data import TileSet, WindowDataset
    ts = TileSet("/workspace/data/e2/sets/train")
    ds = WindowDataset(ts, K=8, stride=1, step=2, augment=True, crop=256)
    x = ds[0]   # {'tiles': [K, c, c] float32, 'mask': [K, c, c] bool, 'gsd': float, 'dt': float,
                #  'target': [2] float32 (m/s, tile axes, augmented like the tiles), 'key': (...)}

Augmentations are exact symmetries of the problem: the 8 rotations / mirror flips of the
square tile, with the target transformed the same way, and crops (a crop of a translating
pattern translates the same). Time reversal is not offered: foam decays, so a reversed
sequence is not one the sea makes.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import geom

FPS_DEFAULT = 10.0

# The dihedral group of the square as (flip_x, flip_y, transpose) applied in that order to
# tile coordinates (x = column, y = row). On a vector (x, y): flip_x negates x, flip_y
# negates y, transpose swaps them; the array ops are the same maps about the tile centre.
DIHEDRAL = [(fx, fy, tr) for tr in (False, True) for fy in (False, True) for fx in (False, True)]


def augment_array(a: np.ndarray, g: int) -> np.ndarray:
    """Apply dihedral element g (0..7, 0 = identity) to the last two axes [.., row = y, col = x]."""
    fx, fy, tr = DIHEDRAL[g]
    if fx:
        a = a[..., :, ::-1]
    if fy:
        a = a[..., ::-1, :]
    if tr:
        a = np.swapaxes(a, -1, -2)
    return np.ascontiguousarray(a)


def augment_vector(v: np.ndarray, g: int) -> np.ndarray:
    """The same element on (…, 2) tile-axis vectors (x, y)."""
    fx, fy, tr = DIHEDRAL[g]
    v = np.array(v, dtype=np.float64, copy=True)
    if fx:
        v[..., 0] = -v[..., 0]
    if fy:
        v[..., 1] = -v[..., 1]
    if tr:
        v = v[..., ::-1].copy()
    return v


def unaugment_vector(v: np.ndarray, g: int) -> np.ndarray:
    """Inverse of augment_vector (to map a prediction made on augmented tiles back)."""
    fx, fy, tr = DIHEDRAL[g]
    v = np.array(v, dtype=np.float64, copy=True)
    if tr:
        v = v[..., ::-1].copy()
    if fy:
        v[..., 1] = -v[..., 1]
    if fx:
        v[..., 0] = -v[..., 0]
    return v


class TileSet:
    """One rectified set on disk. `preload=True` reads every run's tiles into RAM (float16)."""

    def __init__(self, root, preload: bool = False):
        self.root = Path(root)
        self.info = json.loads((self.root / "set.json").read_text())
        self.name = self.info["set"]
        self.n = int(self.info["n"])
        self.runs = pd.read_csv(self.root / "runs.csv").set_index("run_id", drop=False)
        self.frames = pd.read_csv(self.root / "frames.csv")
        self.labels = pd.read_csv(self.root / "labels.csv")
        self.run_ids = list(self.runs.index)
        self._fr = {rid: g.sort_values("frame") for rid, g in self.frames.groupby("run_id", sort=False)}
        self._lb = {rid: g.sort_values("frame") for rid, g in self.labels.groupby("run_id", sort=False)}
        self._tiles: dict[str, np.ndarray] = {}
        if preload:
            for rid in self.run_ids:
                self._tiles[rid] = np.load(self.root / "tiles" / f"{rid}.npy")

    def tiles(self, run_id: str) -> np.ndarray:
        """[F, n, n] float16 (memory-mapped unless preloaded)."""
        a = self._tiles.get(run_id)
        if a is None:
            a = np.load(self.root / "tiles" / f"{run_id}.npy", mmap_mode="r")
            self._tiles[run_id] = a
        return a

    def gsd(self, run_id: str) -> float:
        return float(self.runs.at[run_id, "gsd"])

    def theta(self, run_id: str) -> float:
        return float(self.runs.at[run_id, "theta_deg"])

    def fps(self, run_id: str) -> float:
        return float(self.runs.at[run_id, "fps"]) if "fps" in self.runs else FPS_DEFAULT

    def windows(self, K: int, stride: int = 1, step: int = 1, min_valid: float = 0.0, runs=None) -> pd.DataFrame:
        """Every window of K frames, `stride` frames apart, starting every `step` frames.
        Columns: set, run_id, w_start, K, stride, T, dt, gsd, valid_min, and the label
        v_true_x, v_true_y (tile axes), v_true_e, v_true_n (ENU): the mean camera velocity
        over the window, (p_end − p_start) / (t_end − t_start)."""
        out = []
        for rid in (runs if runs is not None else self.run_ids):
            fr, lb = self._fr[rid], self._lb[rid]
            t = lb["t"].to_numpy()
            pe = lb[["cam_e", "cam_n"]].to_numpy()
            px = lb[["cam_x", "cam_y"]].to_numpy()
            vf = fr["valid_frac"].to_numpy()
            span = (K - 1) * stride
            starts = np.arange(0, len(t) - span, step)
            if len(starts) == 0:
                continue
            ends = starts + span
            T = t[ends] - t[starts]
            vmin = np.array([vf[s:e + 1:stride].min() for s, e in zip(starts, ends)])
            keep = vmin >= min_valid
            ve = (pe[ends] - pe[starts]) / T[:, None]
            vx = (px[ends] - px[starts]) / T[:, None]
            out.append(pd.DataFrame({
                "set": self.name, "run_id": rid, "w_start": starts, "K": K, "stride": stride, "T": T,
                "dt": T / max(K - 1, 1), "gsd": self.gsd(rid), "valid_min": vmin,
                "v_true_x": vx[:, 0], "v_true_y": vx[:, 1], "v_true_e": ve[:, 0], "v_true_n": ve[:, 1],
            })[keep])
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

    def window_tiles(self, run_id: str, w_start: int, K: int, stride: int = 1) -> np.ndarray:
        """[K, n, n] float32 with NaN where there is no image."""
        a = self.tiles(run_id)
        return np.asarray(a[w_start:w_start + (K - 1) * stride + 1:stride], dtype=np.float32)

    def to_enu(self, run_id: str, v_xy: np.ndarray) -> np.ndarray:
        return geom.tile_to_enu(v_xy, self.theta(run_id))


@dataclass
class WindowDataset:
    """K-frame windows of a TileSet as model inputs (a torch-style Dataset: len and getitem).

    crop: side of a square crop (random when augment, else centred); None keeps the tile.
    bin:  average b × b cells (gsd × b); applied after the crop.
    augment: random dihedral element per item; the target is transformed with it.
    normalize: 'window' subtracts the window's valid mean and divides by its valid std.
    fill: value written where the mask is False (after normalisation).
    """

    ts: TileSet
    K: int
    stride: int = 1
    step: int = 1
    min_valid: float = 0.0
    augment: bool = False
    crop: int | None = None
    bin: int = 1
    normalize: str | None = "window"
    fill: float = 0.0
    seed: int = 0
    runs: list | None = None

    def __post_init__(self):
        self.index = self.ts.windows(self.K, self.stride, self.step, self.min_valid, self.runs)
        self.rng = np.random.default_rng(self.seed)

    def __len__(self):
        return len(self.index)

    def item(self, i: int, g: int = 0, crop_at: tuple[int, int] | None = None) -> dict:
        r = self.index.iloc[i].to_dict()
        a = self.ts.window_tiles(r["run_id"], int(r["w_start"]), self.K, self.stride)
        n = a.shape[-1]
        c = self.crop or n
        if c < n:
            if crop_at is None:
                crop_at = ((n - c) // 2, (n - c) // 2)
            y0, x0 = crop_at
            a = a[:, y0:y0 + c, x0:x0 + c]
        gsd = float(r["gsd"])
        if self.bin > 1:
            b = self.bin
            m = a.shape[-1] // b
            a = a[:, :m * b, :m * b].reshape(self.K, m, b, m, b)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN blocks stay NaN
                a = np.nanmean(a, axis=(2, 4))
            gsd *= b
        mask = np.isfinite(a)
        if self.normalize == "window" and mask.any():
            vals = a[mask]
            a = (a - vals.mean()) / max(float(vals.std()), 1e-6)
        a = np.where(mask, a, self.fill).astype(np.float32)
        target = np.array([r["v_true_x"], r["v_true_y"]])
        if g:
            a = augment_array(a, g)
            mask = augment_array(mask, g)
            target = augment_vector(target, g)
        return {"tiles": a, "mask": mask, "gsd": np.float32(gsd), "dt": np.float32(r["dt"]),
                "target": target.astype(np.float32), "aug": g,
                "key": (r["set"], r["run_id"], int(r["w_start"]), int(r["K"]), float(r["T"]))}

    def __getitem__(self, i: int) -> dict:
        if not self.augment:
            return self.item(i)
        g = int(self.rng.integers(8))
        n = self.ts.n
        c = self.crop or n
        crop_at = (int(self.rng.integers(n - c + 1)), int(self.rng.integers(n - c + 1))) if c < n else None
        return self.item(i, g, crop_at)


def collate(items: list[dict]) -> dict:
    """Stack items (numpy); keys stay a list. Wrap with torch.as_tensor as needed."""
    out = {k: np.stack([it[k] for it in items]) for k in ("tiles", "mask", "gsd", "dt", "target", "aug")}
    out["key"] = [it["key"] for it in items]
    return out


def prediction_frame(ts: TileSet, keys, v_est_xy, sigma_xy=None) -> pd.DataFrame:
    """The standard prediction rows for a list of window keys (set, run_id, w_start, K, T)
    and estimates in (un-augmented) tile axes: fills v_true from the labels and ENU from θ."""
    v_est_xy = np.asarray(v_est_xy, dtype=np.float64).reshape(-1, 2)
    sig = np.full_like(v_est_xy, np.nan) if sigma_xy is None else np.asarray(sigma_xy, dtype=np.float64).reshape(-1, 2)
    rows = []
    for (s, rid, w, K, T), ve, sg in zip(keys, v_est_xy, sig):
        lb = ts._lb[rid]
        t = lb["t"].to_numpy()
        stride = int(round(T * ts.fps(rid) / max(K - 1, 1)))
        e = w + (K - 1) * stride
        Tw = t[e] - t[w]
        vx = (lb[["cam_x", "cam_y"]].to_numpy()[e] - lb[["cam_x", "cam_y"]].to_numpy()[w]) / Tw
        vn = (lb[["cam_e", "cam_n"]].to_numpy()[e] - lb[["cam_e", "cam_n"]].to_numpy()[w]) / Tw
        en = ts.to_enu(rid, ve)
        rows.append({"set": s, "run_id": rid, "w_start": w, "K": K, "T": Tw,
                     "v_true_x": vx[0], "v_true_y": vx[1], "v_est_x": ve[0], "v_est_y": ve[1],
                     "sigma_x": sg[0], "sigma_y": sg[1],
                     "v_true_e": vn[0], "v_true_n": vn[1], "v_est_e": en[0], "v_est_n": en[1]})
    return pd.DataFrame(rows)


PRED_COLUMNS = ["set", "run_id", "w_start", "K", "T", "v_true_x", "v_true_y", "v_est_x", "v_est_y", "sigma_x", "sigma_y",
                "v_true_e", "v_true_n", "v_est_e", "v_est_n"]

