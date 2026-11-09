"""E2 tile geometry: rectify onboard frames onto a ground tile that translates exactly with
the camera, in a per-run random orientation.

Conventions (docs/ARCHITECTURE.md; the same as experiments/e1_wave_bias/e1geom.py):
- World: ENU metres, z up, mean sea level at z = 0.
- Camera: OpenCV axes (x right, y down, z along the optical axis). R_wc (R00..R22 in
  frames.csv, row-major) maps camera vectors to ENU; t_wc = (cam_e, cam_n, cam_u).
  A world point P projects as p_c = R_wcᵀ (P − t_wc), u = fx·x/z + cx, v = fy·y/z + cy.
- Pixel (0, 0) is the top-left corner of the top-left pixel; the centre of pixel
  (row r, col c) is (u, v) = (c + 0.5, r + 0.5).

The tile (E2's fixed design): frame i samples the ground points
    p = cam_pos_i + c_run + R(θ_run)·(x, y),    on the plane up = 0,
with c_run a per-run constant horizontal offset (the run-mean footprint centre, relative to
the camera) and θ_run a per-run orientation. Tile axes in ENU are
    x̂ = (cos θ, sin θ),   ŷ = (−sin θ, cos θ)        (θ counter-clockwise from east),
so a vector with tile components (a, b) is a·x̂ + b·ŷ in (E, N), and v_tile = R(θ)ᵀ v_enu.
The array is [row, col] = [y index, x index], both increasing toward +y / +x (row 0 is the
−y edge), cell centres at (k + ½)·gsd − side/2. Because c_run is constant within a run the
tile moves exactly with the camera: ground content moves through it at −v (plus the
water's own motion). Only attitude (R_wc), K and the camera's height cam_u enter the
sampling; cam_e / cam_n cancel because every cell is placed relative to the camera.
"""

from __future__ import annotations

import csv
import json
import math
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

LUMA = np.array([0.2126, 0.7152, 0.0722])
# sRGB 8-bit code → linear light, indexed by the code (B, G, R order irrelevant: per channel).
_SRGB_LUT = np.where(np.arange(256) / 255.0 <= 0.04045, np.arange(256) / 255.0 / 12.92,
                     ((np.arange(256) / 255.0 + 0.055) / 1.055) ** 2.4).astype(np.float32)


# ---- runs on disk --------------------------------------------------------------------------


def read_frames(run_dir) -> list[dict[str, float]]:
    """frames.csv as a list of {column: float} rows."""
    with open(Path(run_dir) / "frames.csv") as f:
        return [{k: float(v) for k, v in r.items() if v not in ("", None)} for r in csv.DictReader(f)]


def load_run(run_dir) -> dict:
    return json.loads((Path(run_dir) / "run.json").read_text())


def pose(row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """K (3×3), R_wc (3×3, camera → ENU), t_wc (3,) from one frames.csv row."""
    R = np.array([row[f"R{i}{j}"] for i in range(3) for j in range(3)], dtype=np.float64).reshape(3, 3)
    t = np.array([row["cam_e"], row["cam_n"], row["cam_u"]], dtype=np.float64)
    K = np.array([[row["fx"], 0.0, row["cx"]], [0.0, row["fy"], row["cy"]], [0.0, 0.0, 1.0]])
    return K, R, t


def luminance(path) -> np.ndarray:
    """Linear-light luminance (Rec. 709 weights on decoded 8-bit sRGB), float32, [row, col]."""
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    lin = _SRGB_LUT[bgr]
    return (lin[..., 2] * LUMA[0] + lin[..., 1] * LUMA[1] + lin[..., 0] * LUMA[2]).astype(np.float32)


# ---- geometry ------------------------------------------------------------------------------


def rotation(theta_deg: float) -> np.ndarray:
    """R(θ): columns are the tile axes x̂, ŷ in (E, N)."""
    c, s = math.cos(math.radians(theta_deg)), math.sin(math.radians(theta_deg))
    return np.array([[c, -s], [s, c]])


def enu_to_tile(v_en: np.ndarray, theta_deg: float) -> np.ndarray:
    """(…, 2) (E, N) vectors → tile components (x, y)."""
    return np.asarray(v_en, dtype=np.float64) @ rotation(theta_deg)


def tile_to_enu(v_xy: np.ndarray, theta_deg: float) -> np.ndarray:
    return np.asarray(v_xy, dtype=np.float64) @ rotation(theta_deg).T


def theta_for(key: str) -> float:
    """A per-run tile orientation in [0, 360), uniform, from a stable hash of `key` (the set
    and run, or the condition for paired runs). It depends on nothing in the scene."""
    rng = np.random.default_rng(zlib.crc32(("e2-theta:" + key).encode()))
    return float(rng.uniform(0.0, 360.0))


def ray_to_plane(K, R, h, u, v):
    """Horizontal offset (dE, dN) from the camera to where pixel (u, v)'s ray meets z = 0
    (camera at height h); NaN where the ray does not point down."""
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    dc = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    dw = dc @ R.T
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(dw[..., 2] < -1e-9, -h / dw[..., 2], np.nan)
    return dw[..., 0] * s, dw[..., 1] * s


def footprint_centre_offset(row: dict) -> np.ndarray:
    """(dE, dN) from the camera to where the principal ray meets the sea (NaN if it misses)."""
    K, R, t = pose(row)
    e, n = ray_to_plane(K, R, t[2], K[0, 2], K[1, 2])
    return np.array([float(e), float(n)])


@dataclass(frozen=True)
class Tile:
    """A run's tile: n × n cells of `gsd` metres, orientation θ (degrees), centre offset
    c_run (E, N) from the camera's ground point."""

    n: int
    gsd: float
    theta: float
    c_run: tuple[float, float]

    @property
    def side(self) -> float:
        return self.n * self.gsd

    def cell_offsets(self) -> np.ndarray:
        """(n, n, 2) cell-centre offsets (dE, dN) from the camera's ground point; [row = y, col = x]."""
        c = (np.arange(self.n) + 0.5) * self.gsd - self.side / 2
        x, y = np.meshgrid(c, c)  # x varies along columns, y along rows
        en = np.stack([x, y], -1) @ rotation(self.theta).T
        return en + np.asarray(self.c_run)


def run_tile(rows: list[dict], n: int, theta: float, gsd_scale: float = 1.0) -> Tile:
    """The tile for a run: cell = gsd_scale × the image's nadir GSD at the run-mean camera
    height (cam_u / fx), centre = the run-mean footprint centre (principal ray ∩ sea)."""
    fx = rows[0]["fx"]
    h = float(np.mean([r["cam_u"] for r in rows]))
    offs = np.array([footprint_centre_offset(r) for r in rows])
    ok = np.isfinite(offs).all(1)
    if not ok.any():
        raise ValueError("the principal ray misses the sea in every frame")
    c = offs[ok].mean(0)
    return Tile(n=n, gsd=gsd_scale * h / fx, theta=theta, c_run=(float(c[0]), float(c[1])))


def pixel_map(row: dict, tile: Tile, offsets: np.ndarray | None = None):
    """(u, v) image coordinates of every tile cell in this frame (NaN where the cell is
    behind the camera), and the image pixels spanned by one tile cell at most (the larger
    singular direction, over the valid cells) — > 1 means the tile undersamples the image."""
    K, R, t = pose(row)
    off = tile.cell_offsets() if offsets is None else offsets
    d = np.concatenate([off, np.full(off.shape[:2] + (1,), -t[2])], -1)
    pc = d @ R  # row-vector form of R_wcᵀ d
    z = pc[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(z > 0, K[0, 0] * pc[..., 0] / z + K[0, 2], np.nan)
        v = np.where(z > 0, K[1, 1] * pc[..., 1] / z + K[1, 2], np.nan)
    # pixels per cell along x (columns) and y (rows)
    dux, duy = np.gradient(u, axis=1), np.gradient(u, axis=0)
    dvx, dvy = np.gradient(v, axis=1), np.gradient(v, axis=0)
    ppc = np.fmax(np.hypot(dux, dvx), np.hypot(duy, dvy))
    ratio = float(np.nanmax(ppc)) if np.isfinite(ppc).any() else float("nan")
    return u, v, ratio


def rectify_frame(img: np.ndarray, u: np.ndarray, v: np.ndarray, ratio: float, antialias: bool = True,
                  margin: float = 0.5) -> np.ndarray:
    """Bilinear (exact, float64 coordinates) inverse-map sample of img at (u, v). If a tile
    cell spans more than one image pixel (ratio > 1), the image is first blurred with a
    Gaussian of σ = ½·sqrt(ratio² − 1) px to limit aliasing. NaN where the sample is not
    inside the image (pixel centres from `margin` to size − margin)."""
    if antialias and ratio > 1.0:
        img = ndimage.gaussian_filter(img, 0.5 * math.sqrt(ratio * ratio - 1.0), mode="nearest")
    h, w = img.shape
    bad = ~np.isfinite(u) | ~np.isfinite(v) | (u < margin) | (v < margin) | (u > w - margin) | (v > h - margin)
    uu = np.where(bad, 0.0, u) - 0.5
    vv = np.where(bad, 0.0, v) - 0.5
    out = ndimage.map_coordinates(img, [vv, uu], order=1, mode="nearest")
    out = out.astype(np.float32)
    out[bad] = np.nan
    return out
