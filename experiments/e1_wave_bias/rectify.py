"""Rectify onboard frames onto a sea-plane tile fixed to the camera (E1 dispersion-shell fit).

Self-contained (numpy + scipy.ndimage) so shell_fit.py does not depend on e1geom.py; the
two must agree, and `tests/test_shell.py::test_projection_matches_e1geom` checks that they do.

Conventions (docs/ARCHITECTURE.md):
- World: ENU metres, z up, mean sea level at z = 0.
- Camera: OpenCV axes (x right, y down, z along the optical axis). R_wc (R00..R22 in
  frames.csv, row-major) maps camera vectors to ENU; t_wc = (cam_e, cam_n, cam_u).
  A world point P projects as p_c = R_wcᵀ (P − t_wc), u = fx·x/z + cx, v = fy·y/z + cy.
- Pixel (0, 0) is the top-left corner of the top-left pixel; the centre of pixel
  (row r, col c) is (u, v) = (c + 0.5, r + 0.5).
- Use cam_u (the camera's height above z = 0), not the trajectory altitude.

The tile: an n × n grid of square cells, axis-aligned with east and north, `side` metres,
whose centre is fixed relative to the camera (the nadir point for a nadir camera, the
point where the principal ray meets the sea for an oblique one). Array layout is
[row = north index, col = east index], with BOTH indices increasing toward +north / +east
(row 0 is the SOUTH edge), so array axes map to (y = north, x = east) with no sign flips.
Only attitude (R) and height (cam_u) enter; cam_e / cam_n cancel because every cell is
placed relative to the camera.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

LUMA = np.array([0.2126, 0.7152, 0.0722])


def read_frames(run_dir) -> list[dict[str, float]]:
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


def project_rel(K, R, d):
    """Project world offsets d = P − t_wc (..., 3) → (u, v, z)."""
    pc = d @ R  # row-vector form of R_wcᵀ d
    z = pc[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * pc[..., 0] / z + K[0, 2]
        v = K[1, 1] * pc[..., 1] / z + K[1, 2]
    return u, v, z


def ray_to_plane(K, R, h, u, v):
    """Horizontal offset (dE, dN) from the camera to where pixel (u, v)'s ray meets z = 0
    (camera at height h); NaN if the ray does not point down."""
    dc = np.stack([(np.asarray(u, float) - K[0, 2]) / K[0, 0], (np.asarray(v, float) - K[1, 2]) / K[1, 1],
                   np.ones_like(np.asarray(u, float))], -1)
    dw = dc @ R.T
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(dw[..., 2] < -1e-9, -h / dw[..., 2], np.nan)
    return dw[..., 0] * s, dw[..., 1] * s


@dataclass(frozen=True)
class TileGeom:
    offset: tuple[float, float]  # tile centre − camera ground position (E, N), metres
    side: float                  # metres
    n: int
    footprint: tuple[float, float]  # (across, along) image-edge-midpoint distances on the plane

    @property
    def cell(self) -> float:
        return self.side / self.n


def tile_geometry(row: dict, n: int = 256, frac: float = 0.6) -> TileGeom:
    """The camera-fixed tile for this frame's attitude and height: centred on the point the
    principal ray meets the sea (= the nadir point for pitch −90), side = frac × the
    smaller footprint dimension (distance between the left/right and top/bottom edge
    midpoints, projected onto the sea)."""
    K, R, t = pose(row)
    W, H = row["width"], row["height"]
    h = t[2]
    ce, cn = ray_to_plane(K, R, h, K[0, 2], K[1, 2])
    le, ln = ray_to_plane(K, R, h, 0.0, K[1, 2])
    re, rn = ray_to_plane(K, R, h, W, K[1, 2])
    te, tn = ray_to_plane(K, R, h, K[0, 2], 0.0)
    be, bn = ray_to_plane(K, R, h, K[0, 2], H)
    across = float(math.hypot(re - le, rn - ln))
    along = float(math.hypot(te - be, tn - bn))
    if not np.isfinite([ce, cn, across, along]).all():
        raise ValueError("footprint does not lie on the sea (a ray misses the plane)")
    return TileGeom(offset=(float(ce), float(cn)), side=frac * min(across, along), n=n, footprint=(across, along))


def tile_offsets(geom: TileGeom) -> np.ndarray:
    """Cell-centre offsets from the camera's ground position, shape (n, n, 2) = (dE, dN);
    [row, col] → row = north index (south edge first), col = east index."""
    c = (np.arange(geom.n) + 0.5) * geom.cell - geom.side / 2
    dE, dN = np.meshgrid(geom.offset[0] + c, geom.offset[1] + c)  # dE varies along cols
    return np.stack([dE, dN], -1)


def srgb_to_linear(a):
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def luminance(path) -> np.ndarray:
    """Linear-light luminance (Rec. 709 weights on decoded sRGB), float32, [row, col]."""
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return (srgb_to_linear(a) @ LUMA.astype(np.float32)).astype(np.float32)


def normals_slope(path, bearing_deg: float = 60.0) -> np.ndarray:
    """From the 'normals' label mode (RGB = 0.5 + 0.5·n, R = east, G = north component of the
    unit normal, written without sRGB encoding or dither): the normal's horizontal component
    toward `bearing_deg`, a scalar linear in the surface slope for small slopes. No lighting,
    foam, clouds or glint enter it."""
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    b = math.radians(bearing_deg)
    return ((a[..., 0] - 0.5) * math.sin(b) + (a[..., 1] - 0.5) * math.cos(b) + 1.0).astype(np.float32)


def load_image(run_dir: Path, frame: int, mode: str = "shaded") -> np.ndarray:
    if mode == "shaded":
        return luminance(run_dir / "frames" / f"{frame:06d}.png")
    if mode == "normals":
        return normals_slope(run_dir / "frames" / "normals" / f"{frame:06d}.png")
    return luminance(run_dir / "frames" / mode / f"{frame:06d}.png")


def pixel_map(row: dict, geom: TileGeom):
    """(u, v) image coordinates of every tile cell for this frame, and the ground sample
    distance at the tile centre (m/px)."""
    K, R, t = pose(row)
    off = tile_offsets(geom)
    d = np.concatenate([off, np.full(off.shape[:2] + (1,), -t[2])], -1)
    u, v, z = project_rel(K, R, d)
    u = np.where(z > 0, u, np.nan)
    v = np.where(z > 0, v, np.nan)
    # GSD at the tile centre: metres per pixel along the image x axis.
    c = np.array([[geom.offset[0], geom.offset[1], -t[2]], [geom.offset[0] + 0.01, geom.offset[1], -t[2]],
                  [geom.offset[0], geom.offset[1] + 0.01, -t[2]]])
    uc, vc, _ = project_rel(K, R, c)
    px_per_m = max(math.hypot(uc[1] - uc[0], vc[1] - vc[0]), math.hypot(uc[2] - uc[0], vc[2] - vc[0])) / 0.01
    return u, v, 1.0 / px_per_m


def rectify(img: np.ndarray, u: np.ndarray, v: np.ndarray, gsd: float, cell: float, antialias: bool = True):
    """Bilinear inverse-map sample of img at (u, v) (pixel-corner convention). If a tile
    cell is coarser than a pixel, the image is first blurred with a Gaussian of
    σ = ½·sqrt(r² − 1) px (r = cell / gsd) to limit aliasing. NaN outside the image."""
    if antialias:
        r = cell / gsd
        if r > 1.0:
            img = ndimage.gaussian_filter(img, 0.5 * math.sqrt(r * r - 1.0), mode="nearest")
    h, w = img.shape
    out = ndimage.map_coordinates(img, [v - 0.5, u - 0.5], order=1, mode="constant", cval=np.nan)
    bad = ~np.isfinite(u) | ~np.isfinite(v) | (u < 0.5) | (v < 0.5) | (u > w - 0.5) | (v > h - 0.5)
    out[bad] = np.nan
    return out.astype(np.float32)


def rectify_run(run_dir, n: int = 256, frac: float = 0.6, frames=None, antialias: bool = True, mode: str = "shaded"):
    """Rectify every frame (or the listed ones) of a run onto its camera-fixed tile.
    Returns (stack [nt, n, n] float32, rows used, TileGeom). The tile geometry (side) is
    set from frame 0 and held fixed; the pixel map is recomputed per frame from that
    frame's attitude and height."""
    run_dir = Path(run_dir)
    rows = read_frames(run_dir)
    geom0 = tile_geometry(rows[0], n=n, frac=frac)
    idx = range(len(rows)) if frames is None else frames
    stack, used = [], []
    for i in idx:
        r = rows[i]
        g = tile_geometry(r, n=n, frac=frac)
        g = TileGeom(offset=g.offset, side=geom0.side, n=n, footprint=g.footprint)
        u, v, gsd = pixel_map(r, g)
        img = load_image(run_dir, int(r["frame"]), mode)
        stack.append(rectify(img, u, v, gsd, g.cell, antialias))
        used.append(r)
    return np.stack(stack), used, geom0
