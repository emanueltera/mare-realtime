"""Shared E1 geometry: camera poses from a run's frames.csv, projection, and back-projection
of pixels onto a horizontal sea plane.

Conventions (docs/ARCHITECTURE.md, README "frames.csv"):
- World: ENU metres (x east, y north, z up above mean sea level).
- Camera: OpenCV axes (x right, y down, z along the optical axis). R_wc (row-major
  R00..R22 in frames.csv) maps camera-frame vectors to ENU; t_wc = (cam_e, cam_n, cam_u).
- A world point P projects as p_c = R_wcᵀ (P − t_wc), u = fx·p_c.x/p_c.z + cx,
  v = fy·p_c.y/p_c.z + cy.
- Pixel coordinates are continuous with (0, 0) at the top-left CORNER of the top-left
  pixel, so the centre of pixel (row r, column c) is (u, v) = (c + 0.5, r + 0.5). That is
  the convention under which cx = width/2 is the image centre (frames.csv has cx = 360
  for a 720-wide frame). Dense flow from a model is a displacement between pixel centres,
  so it adds to these coordinates directly.
- Use cam_u (the camera's height), not the trajectory altitude: the camera sits ~0.22 m
  below the aircraft.

Everything works on numpy arrays; `backproject_torch` is the same math on torch tensors so
flow_bias.py can do it on the GPU.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Pose:
    """One frame's camera: K (3×3), R_wc (3×3, camera → ENU), t_wc (3,), and frame time."""

    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    time: float = 0.0
    width: int = 0
    height: int = 0


def read_frames(run_dir: str | Path) -> list[dict[str, float]]:
    """frames.csv as a list of {column: float} rows."""
    with open(Path(run_dir) / "frames.csv") as f:
        return [{k: float(v) for k, v in r.items() if v != ""} for r in csv.DictReader(f)]


def pose_from_row(row: dict[str, float]) -> Pose:
    R = np.array([row[f"R{i}{j}"] for i in range(3) for j in range(3)], dtype=np.float64).reshape(3, 3)
    t = np.array([row["cam_e"], row["cam_n"], row["cam_u"]], dtype=np.float64)
    K = np.array([[row["fx"], 0.0, row["cx"]], [0.0, row["fy"], row["cy"]], [0.0, 0.0, 1.0]])
    return Pose(K=K, R=R, t=t, time=row["t"], width=int(row.get("width", 0)), height=int(row.get("height", 0)))


def load_poses(run_dir: str | Path) -> list[Pose]:
    return [pose_from_row(r) for r in read_frames(run_dir)]


def load_run(run_dir: str | Path) -> dict:
    return json.loads((Path(run_dir) / "run.json").read_text())


def project(pose: Pose, P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World points P (..., 3) → pixel (u, v) and camera depth z (..., each)."""
    pc = (np.asarray(P, dtype=np.float64) - pose.t) @ pose.R  # row-vector form of R_wcᵀ (P − t)
    z = pc[..., 2]
    u = pose.K[0, 0] * pc[..., 0] / z + pose.K[0, 2]
    v = pose.K[1, 1] * pc[..., 1] / z + pose.K[1, 2]
    return u, v, z


def rays(pose: Pose, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Unit-free ray directions in ENU (..., 3) through pixel coordinates (u, v)."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    dc = np.stack([(u - pose.K[0, 2]) / pose.K[0, 0], (v - pose.K[1, 2]) / pose.K[1, 1], np.ones_like(u)], -1)
    return dc @ pose.R.T  # R_wc · d_c for row vectors


def backproject(pose: Pose, u, v, plane_up: float = 0.0, max_range: float | None = None,
                max_incidence_deg: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Pixels (u, v) → their ray's intersection with the plane up = plane_up.

    Returns (X (..., 3) ENU, valid (...) bool). Invalid where the ray does not go down to the
    plane (or the camera is not above it), where the slant range exceeds max_range, or where
    the ray meets the plane more than max_incidence_deg from the vertical. Invalid X are NaN.
    """
    d = rays(pose, u, v)
    h = pose.t[2] - plane_up
    dz = d[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = -h / dz
    valid = (dz < 0) & (h > 0)
    X = pose.t + s[..., None] * d
    if max_range is not None:
        valid &= s * np.linalg.norm(d, axis=-1) <= max_range
    if max_incidence_deg is not None:
        cos_inc = -dz / np.linalg.norm(d, axis=-1)
        valid &= cos_inc >= math.cos(math.radians(max_incidence_deg))
    X = np.where(valid[..., None], X, np.nan)
    return X, valid


def pixel_grid(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-centre coordinates (u, v), each (height, width)."""
    return np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)


def nadir_point(pose: Pose) -> np.ndarray:
    """(E, N) under the camera."""
    return pose.t[:2].copy()


def footprint_centre(pose: Pose, plane_up: float = 0.0) -> np.ndarray:
    """(E, N) where the optical axis meets the plane (principal point back-projected)."""
    X, ok = backproject(pose, np.array(pose.K[0, 2]), np.array(pose.K[1, 2]), plane_up)
    if not ok:
        raise ValueError("optical axis does not meet the sea plane")
    return X[:2]


def track_axes(heading_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors (E, N) along track (toward heading) and cross track (to the right of it)."""
    h = math.radians(heading_deg)
    return np.array([math.sin(h), math.cos(h)]), np.array([math.cos(h), -math.sin(h)])


def bearing_unit(bearing_deg: float) -> np.ndarray:
    """Unit (E, N) vector toward a compass bearing."""
    b = math.radians(bearing_deg)
    return np.array([math.sin(b), math.cos(b)])


# ---- torch versions (same math, used on the GPU by flow_bias.py) -------------------------


def backproject_torch(K, R, t, u, v, plane_up: float = 0.0):
    """Batched back-projection. K, R: (B, 3, 3); t: (B, 3); u, v: (B, H, W) or broadcastable.
    Returns X (B, H, W, 2) in (E, N), slant range s·|d| (B, H, W), cos of incidence
    (B, H, W), and validity (ray goes down, camera above the plane). float64 recommended."""
    import torch

    fx = K[:, 0, 0].view(-1, 1, 1)
    fy = K[:, 1, 1].view(-1, 1, 1)
    cx = K[:, 0, 2].view(-1, 1, 1)
    cy = K[:, 1, 2].view(-1, 1, 1)
    xc = (u - cx) / fx
    yc = (v - cy) / fy
    # d = R · (xc, yc, 1)
    Rv = R.view(-1, 1, 1, 3, 3)
    dE = Rv[..., 0, 0] * xc + Rv[..., 0, 1] * yc + Rv[..., 0, 2]
    dN = Rv[..., 1, 0] * xc + Rv[..., 1, 1] * yc + Rv[..., 1, 2]
    dU = Rv[..., 2, 0] * xc + Rv[..., 2, 1] * yc + Rv[..., 2, 2]
    h = (t[:, 2] - plane_up).view(-1, 1, 1)
    s = -h / dU
    valid = (dU < 0) & (h > 0)
    E = t[:, 0].view(-1, 1, 1) + s * dE
    N = t[:, 1].view(-1, 1, 1) + s * dN
    norm = torch.sqrt(dE * dE + dN * dN + dU * dU)
    rng = s * norm
    cos_inc = -dU / norm
    return torch.stack([E, N], -1), rng, cos_inc, valid
