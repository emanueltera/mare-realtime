"""The E2 tile geometry on a synthetic frozen ground texture seen by a pinhole camera that
flies a known path with changing attitude and height: the tile samples the right ground
points, and the texture moves through it at exactly −v·dt per frame in tile axes (sign and
rotation convention), however θ_run is set."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import ndimage

from e2 import geom
from e2.baselines import phase_correlation

W, H, FX = 720, 1280, 618.04


_CELL = 0.3  # m
_ORIGIN = -150.0
_GRID = ndimage.spline_filter(ndimage.gaussian_filter(np.random.default_rng(5).normal(size=(1000, 1000)), 1.2), order=3)


def texture(e, n):
    """A smooth random frozen ground texture (a cubic spline through Gaussian noise on a
    0.3 m grid), at any (E, N) within ±150 m."""
    rows = (np.asarray(n) - _ORIGIN) / _CELL
    cols = (np.asarray(e) - _ORIGIN) / _CELL
    return 0.5 + 0.5 * ndimage.map_coordinates(_GRID, [rows, cols], order=3, prefilter=False, mode="nearest")


def r_wc(yaw_deg, pitch_deg, roll_deg):
    """Camera → ENU for a camera whose optical axis is `pitch_deg` below the horizon (−90
    nadir) toward compass `yaw_deg`, rolled by `roll_deg` about its optical axis."""
    y = math.radians(yaw_deg)
    fwd_h = np.array([math.sin(y), math.cos(y), 0.0])
    right_h = np.array([math.cos(y), -math.sin(y), 0.0])
    p = math.radians(-pitch_deg)  # depression angle
    z = math.cos(p) * fwd_h + np.array([0, 0, -math.sin(p)])
    x = right_h
    yv = np.cross(z, x)
    r = math.radians(roll_deg)
    x2 = math.cos(r) * x + math.sin(r) * yv
    y2 = -math.sin(r) * x + math.cos(r) * yv
    R = np.stack([x2, y2, z], 1)
    assert np.allclose(R.T @ R, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)
    return R


def render(row):
    K, R, t = geom.pose(row)
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    de, dn = geom.ray_to_plane(K, R, t[2], u, v)
    return texture(t[0] + de, t[1] + dn).astype(np.float32)


def flight(v_en, n_frames=4, dt=0.1, h=40.0, pitch=-75.0):
    rows = []
    for i in range(n_frames):
        R = r_wc(30 + 2.0 * i, pitch + 1.5 * math.sin(i), 2.5 * math.cos(1.3 * i))
        row = {"frame": i, "t": i * dt, "cam_e": 5.0 + v_en[0] * i * dt, "cam_n": -3.0 + v_en[1] * i * dt,
               "cam_u": h + 0.3 * math.sin(0.7 * i), "fx": FX, "fy": FX, "cx": W / 2, "cy": H / 2,
               "width": W, "height": H}
        for a in range(3):
            for b in range(3):
                row[f"R{a}{b}"] = R[a, b]
        rows.append(row)
    return rows


@pytest.mark.parametrize("theta", [0.0, 37.0, 200.0])
def test_tile_samples_ground_and_moves_at_minus_v(theta):
    v_en = np.array([21.0, -9.0])  # m/s
    dt = 0.1
    rows = flight(v_en, dt=dt)
    tile = geom.run_tile(rows, 256, theta)
    off = tile.cell_offsets()
    tiles = []
    for r in rows:
        u, v, ratio = geom.pixel_map(r, tile, off)
        t = geom.rectify_frame(render(r), u, v, ratio, antialias=False)
        tiles.append(t)
        # the tile holds the texture at cam_pos + c_run + R(θ)(x, y)
        truth = texture(r["cam_e"] + off[..., 0], r["cam_n"] + off[..., 1])
        ok = np.isfinite(t)
        assert ok.mean() > 0.99
        assert np.abs(t[ok] - truth[ok]).max() < 4e-3
    # orientation: column index grows along x̂ = (cos θ, sin θ), row index along ŷ
    d_col = off[0, 1] - off[0, 0]
    d_row = off[1, 0] - off[0, 0]
    th = math.radians(theta)
    assert np.allclose(d_col / tile.gsd, [math.cos(th), math.sin(th)])
    assert np.allclose(d_row / tile.gsd, [-math.sin(th), math.cos(th)])
    # content moves at −v: measured shift per frame = −v_tile·dt / gsd pixels (x = col, y = row)
    v_tile = geom.enu_to_tile(v_en, theta)
    for i in range(len(tiles) - 1):
        d, _ = phase_correlation(tiles[i], tiles[i + 1])
        expect = -v_tile * dt / tile.gsd
        assert np.abs(d - expect).max() < 0.03, (d, expect)
    # and the label rotation round-trips
    assert np.allclose(geom.tile_to_enu(v_tile, theta), v_en)


def test_c_run_is_constant_and_tile_centre_ignores_attitude():
    rows = flight(np.array([30.0, 0.0]), n_frames=6, pitch=-65.0)
    tile = geom.run_tile(rows, 128, 10.0)
    offs = np.array([geom.footprint_centre_offset(r) for r in rows])
    assert np.allclose(tile.c_run, offs.mean(0))
    # cell offsets are relative to the camera only: they do not depend on the frame
    assert tile.cell_offsets().shape == (128, 128, 2)
    assert abs(tile.gsd - np.mean([r["cam_u"] for r in rows]) / FX) < 1e-12


def test_theta_is_stable_and_spread():
    a = [geom.theta_for(f"train:run{i}") for i in range(2000)]
    assert geom.theta_for("train:run7") == a[7]
    assert 0 <= min(a) and max(a) < 360
    hist, _ = np.histogram(a, bins=8, range=(0, 360))
    assert hist.min() > 180  # roughly uniform
