"""E3 checks on rendered runs (set E3_SET to a set directory, e.g. the pilot)."""
import os
from pathlib import Path

import numpy as np
import pytest

from e3.data import MIRRORS, NO_NOISE, PairDataset, Run, exp_so3, list_runs, log_so3, mirror_rot, simulate
from e3.evaluate import chain, chain_rot

SET = os.environ.get("E3_SET")
need = pytest.mark.skipif(not SET, reason="E3_SET not set")


def test_so3_roundtrip():
    w = np.random.default_rng(0).normal(0, 0.5, (100, 3))
    assert np.allclose(log_so3(exp_so3(w)), w, atol=1e-9)


def test_mirror_rot_matches_conjugation():
    w = np.random.default_rng(1).normal(0, 0.3, (20, 3))
    for k, M in enumerate(MIRRORS):
        assert np.allclose(log_so3(M @ exp_so3(w) @ M), mirror_rot(w, k), atol=1e-9)


@need
def test_labels_chain_back_to_truth():
    for d in list_runs(SET):
        run = Run.load(d)
        dp, rot = run.labels()
        assert np.allclose(chain(run.p[0], run.R[:-1], dp), run.p, atol=1e-6)
        assert np.allclose(chain_rot(run.R[0], rot), run.R, atol=1e-6)


@need
def test_noiseless_sensors_are_truth():
    run = Run.load(list_runs(SET)[0])
    s = simulate(run, np.random.default_rng(0), NO_NOISE)
    assert np.allclose(s.R_ahrs, run.R) and np.allclose(s.baro, run.p[:, 2]) and np.allclose(s.gyro, run.labels()[1])


def test_fx_default_shape_is_e3s():
    from e3.fx import draw_fx
    a, b = draw_fx(np.random.default_rng(3)), draw_fx(np.random.default_rng(3), (512, 288))
    assert a.dirt.shape == (512, 288) and np.array_equal(a.dirt, b.dirt) and a.jpeg == b.jpeg
    c = draw_fx(np.random.default_rng(3), (288, 512))
    assert c.dirt.shape == (288, 512)


@need
def test_set_camera_fits_frames():
    from e3.data import E3_CAM
    ds = PairDataset(SET, train=False)
    run = ds.runs[0]
    assert ds.cam.h == run.frames.shape[1] and ds.cam.w == run.frames.shape[2]
    assert abs(ds.cam.f / ds.cam.w - run.full["fx"] / run.full["width"]) < 1e-9
    if (ds.cam.h, ds.cam.w) == (E3_CAM.h, E3_CAM.w):
        assert ds.cam.matches(E3_CAM)


@need
def test_eval_targets_and_fixed_noise():
    ds = PairDataset(SET, train=False)
    img, x, y, h, r, i = ds[5]
    assert img.shape == (2, ds.cam.h, ds.cam.w) and x.shape == (14,)
    dp, _ = ds.runs[r].labels()
    assert np.allclose(y[:3] * h / 10, dp[i], rtol=1e-5)
    assert np.array_equal(ds[5][1], PairDataset(SET, train=False)[5][1])


FROZEN = os.environ.get("E3_FROZEN")  # a seaRate-0 run directory


@pytest.mark.skipif(not FROZEN, reason="E3_FROZEN not set")
def test_frames_move_as_the_pose_says():
    """On a frozen sea, warping frame i+1 into frame i through the sea plane with the true
    pose lines the images up better than with the translation mirrored in x or y, and far
    better than reversed or zero (a mirror changes little when that component is small)."""
    import cv2
    run = Run.load(Path(FROZEN))
    CX, CY, FX = run.cam.cx, run.cam.cy, run.cam.f
    K = np.array([[FX, 0, CX], [0, FX, CY], [0, 0, 1.0]])
    dp, _ = run.labels()

    def residual(i, flip):
        R0, R1 = run.R[i], run.R[i + 1]
        t = run.p[i + 1] - run.p[i]
        t = (R0 @ (flip @ (R0.T @ t)))
        # plane z = 0 in world; pixel in frame i -> ray -> ground point -> frame i+1
        h, w = run.frames.shape[1:]
        v, u = np.mgrid[0:h, 0:w] + 0.5
        d = np.stack([(u - CX) / FX, (v - CY) / FX, np.ones_like(u)], -1) @ R0.T
        s = -run.p[i][2] / d[..., 2]
        P = run.p[i] + s[..., None] * d
        c = (P - (run.p[i] + t)) @ R1
        uu, vv = c[..., 0] / c[..., 2] * FX + CX - 0.5, c[..., 1] / c[..., 2] * FX + CY - 0.5
        ok = (s > 0) & (c[..., 2] > 0) & (uu > 2) & (uu < w - 3) & (vv > 2) & (vv < h - 3)
        warped = cv2.remap(run.frames[i + 1].astype(np.float32), uu.astype(np.float32), vv.astype(np.float32), cv2.INTER_LINEAR)
        return np.abs(warped - run.frames[i].astype(np.float32))[ok].mean()

    for i in (10, 50, 100):
        good = residual(i, np.eye(3))
        assert good <= residual(i, np.diag([-1.0, 1, 1])), i
        assert good <= residual(i, np.diag([1.0, -1, 1])), i
        assert good < 0.6 * residual(i, -np.eye(3)), i
        assert good < 0.6 * residual(i, np.zeros((3, 3))), i


@need
def test_training_samples_are_finite():
    ds = PairDataset(SET, train=True)
    for _ in range(20):
        img, x, y, h, r, i = ds[np.random.randint(len(ds))]
        assert np.isfinite(img).all() and np.isfinite(x).all() and np.isfinite(y).all()
