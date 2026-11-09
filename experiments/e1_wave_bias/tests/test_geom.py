"""Unit tests for e1geom (and the flow → sea-velocity step of flow_bias.py when torch is present).

Run: python -m pytest experiments/e1_wave_bias/tests -q
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import e1geom  # noqa: E402


def rot_from_ypr(yaw_deg, pitch_deg, roll_deg=0.0):
    """R_wc for a camera whose optical axis points along bearing `yaw`, tilted `pitch` from the
    horizontal (−90 = nadir), image x to the right of the view, then rolled about the axis."""
    yaw, pitch, roll = map(math.radians, (yaw_deg, pitch_deg, roll_deg))
    fwd_h = np.array([math.sin(yaw), math.cos(yaw), 0.0])
    right_h = np.array([math.cos(yaw), -math.sin(yaw), 0.0])
    z = math.cos(pitch) * fwd_h + np.array([0, 0, math.sin(pitch)])  # optical axis
    x = right_h
    y = np.cross(z, x)  # OpenCV: x right, y down, z forward; y = z × x
    # roll about z
    c, s = math.cos(roll), math.sin(roll)
    x2 = c * x + s * y
    y2 = -s * x + c * y
    R = np.stack([x2, y2, z], 1)  # columns are camera axes in ENU
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)
    return R


def make_pose(yaw=37.0, pitch=-90.0, roll=0.0, t=(12.3, -45.6, 59.778), w=720, h=1280, fovy=92.0):
    f = (h / 2) / math.tan(math.radians(fovy) / 2)
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    return e1geom.Pose(K=K, R=rot_from_ypr(yaw, pitch, roll), t=np.array(t, float), width=w, height=h)


@pytest.mark.parametrize("yaw,pitch,roll", [(0, -90, 0), (60, -90, 0), (37, -90, 3), (240, -60, 0), (150, -45, -5)])
def test_project_backproject_round_trip(yaw, pitch, roll):
    rng = np.random.default_rng(1)
    pose = make_pose(yaw, pitch, roll)
    u, v = e1geom.pixel_grid(pose.width, pose.height)
    u, v = u[::37, ::23], v[::37, ::23]
    X, ok = e1geom.backproject(pose, u, v, max_incidence_deg=85)
    assert ok.mean() > 0.5
    assert np.allclose(X[ok][:, 2], 0.0, atol=1e-9)
    u2, v2, z = e1geom.project(pose, X[ok])
    assert np.max(np.abs(u2 - u[ok])) < 1e-6 and np.max(np.abs(v2 - v[ok])) < 1e-6
    # world → image → world, for random points on the sea plane in view
    P = np.zeros((500, 3))
    P[:, :2] = X[ok][rng.integers(0, ok.sum(), 500), :2] + rng.normal(0, 0.5, (500, 2))
    uu, vv, zz = e1geom.project(pose, P)
    assert np.all(zz > 0)
    Xb, okb = e1geom.backproject(pose, uu, vv)
    assert okb.all()
    assert np.max(np.linalg.norm(Xb - P, axis=-1)) < 1e-6


def test_conventions_nadir_image_up_is_heading():
    # A nadir camera with image-up along heading 60 (the renders' convention): moving the world
    # point north-east along the heading moves it UP the image (smaller v).
    pose = make_pose(yaw=60, pitch=-90)
    c = pose.t[:2]
    u0, v0, _ = e1geom.project(pose, np.array([c[0], c[1], 0.0]))
    assert abs(u0 - 360) < 1e-9 and abs(v0 - 640) < 1e-9
    fwd = e1geom.bearing_unit(60)
    u1, v1, _ = e1geom.project(pose, np.array([c[0] + 10 * fwd[0], c[1] + 10 * fwd[1], 0.0]))
    assert v1 < v0 and abs(u1 - u0) < 1e-9
    right = e1geom.track_axes(60)[1]
    u2, v2, _ = e1geom.project(pose, np.array([c[0] + 10 * right[0], c[1] + 10 * right[1], 0.0]))
    assert u2 > u0 and abs(v2 - v0) < 1e-9
    # metres per pixel at nadir = h / f
    assert abs((v0 - v1) - 10 * pose.K[1, 1] / pose.t[2]) < 1e-9


def test_rows_pose_matches_frames_csv_layout():
    row = {"t": 0.0, "cam_e": 0.478791, "cam_n": 0.284513, "cam_u": 59.77813, "R00": 0.5, "R01": -0.866025404,
           "R02": 0, "R10": -0.866025404, "R11": -0.5, "R12": 0, "R20": 0, "R21": 0, "R22": -1,
           "fx": 618.040816, "fy": 618.040816, "cx": 360, "cy": 640, "width": 720, "height": 1280}
    p = e1geom.pose_from_row(row)
    assert p.R[0, 1] == row["R01"] and p.R[1, 0] == row["R10"]
    # optical axis straight down; image up (−y) points toward bearing 60 (heading of that run)
    assert np.allclose(p.R[:, 2], [0, 0, -1])
    up = -p.R[:, 1]
    assert abs(math.degrees(math.atan2(up[0], up[1])) % 360 - 60) < 1e-6


def test_backproject_rejects_sky_and_far_rays():
    pose = make_pose(yaw=0, pitch=-10)  # nearly horizontal: top of the image sees sky
    u, v = e1geom.pixel_grid(pose.width, pose.height)
    X, ok = e1geom.backproject(pose, u[:5], v[:5])
    assert not ok.any() and np.isnan(X).all()
    # the centre rows look 10° below the horizon: on the sea, but 80° from the vertical
    X2, ok2 = e1geom.backproject(pose, u[636:644], v[636:644], max_incidence_deg=75)
    X3, ok3 = e1geom.backproject(pose, u[636:644], v[636:644])
    assert ok3.all() and not ok2.any()


torch = pytest.importorskip("torch")


def test_torch_backproject_matches_numpy():
    pose = make_pose(yaw=150, pitch=-60, roll=2)
    u, v = e1geom.pixel_grid(pose.width, pose.height)
    X, ok = e1geom.backproject(pose, u, v)
    T = lambda a: torch.tensor(np.asarray(a)[None], dtype=torch.float64)  # noqa: E731
    Xt, rng, ci, okt = e1geom.backproject_torch(T(pose.K), T(pose.R), T(pose.t), T(u), T(v))
    okt = okt[0].numpy()
    assert (okt == ok).all()
    assert np.max(np.abs(Xt[0].numpy()[ok] - X[ok][:, :2])) < 1e-6


@pytest.mark.parametrize("W", [(0.0, 0.0), (1.0, -0.5), (-2.0, 3.0)])
@pytest.mark.parametrize("pitch", [-90, -60])
def test_apparent_velocity_recovers_sea_motion_from_exact_flow(W, pitch):
    """Exact flow for a camera moving at 60 m/s over a sea whose pattern moves at W: the
    pipeline must return a = W (sign and magnitude)."""
    import flow_bias

    dt = 0.3
    p1 = make_pose(yaw=240, pitch=pitch, t=(100.0, 200.0, 119.788), w=720, h=1280)
    v_cam = 60 * e1geom.bearing_unit(240)
    p2 = e1geom.Pose(K=p1.K, R=p1.R, t=p1.t + np.array([v_cam[0] * dt, v_cam[1] * dt, 0.0]), time=dt,
                     width=720, height=1280)
    u, v = e1geom.pixel_grid(720, 1280)
    X1, ok = e1geom.backproject(p1, u, v)
    X1m = X1.copy()
    X1m[..., 0] += W[0] * dt
    X1m[..., 1] += W[1] * dt
    u2, v2, z = e1geom.project(p2, X1m)
    flow = np.stack([u2 - u, v2 - v])[None]
    flow = torch.tensor(np.nan_to_num(flow, nan=1e9), dtype=torch.float32)
    valid = torch.ones((1, 1280, 720), dtype=torch.bool)
    out = flow_bias.apparent_velocity(flow, valid, [p1], [p2], np.array([dt]))
    assert out["valid_frac"][0] > 0.3
    # float32 flow: ~1e-4 px of rounding over ≥ 50 px of flow → well under 1 mm/s
    assert abs(out["a_E"][0] - W[0]) < 2e-3 and abs(out["a_N"][0] - W[1]) < 2e-3
    if W == (0.0, 0.0):
        assert out["static_res_px"][0] < 1e-3
    along, cross, alw, als = flow_bias.decompose(np.array([1.0]), np.array([0.0]), 90.0)
    assert abs(along[0] - 1) < 1e-12 and abs(cross[0]) < 1e-12  # east is along-track for heading 90
    along, cross, alw, als = flow_bias.decompose(np.array([0.0]), np.array([-1.0]), 90.0)
    assert abs(cross[0] - 1) < 1e-12  # south is to the right of heading 90
    assert np.isnan(als[0])  # no swell → no swell component
    # cross-swell: wind sea toward 60°, swell toward 140° (run.json; directions are 'toward')
    along, cross, alw, als = flow_bias.decompose(np.array([1.0]), np.array([0.0]), 90.0, 60.0, 140.0)
    assert abs(alw[0] - np.sin(np.radians(60))) < 1e-12 and abs(als[0] - np.sin(np.radians(140))) < 1e-12
