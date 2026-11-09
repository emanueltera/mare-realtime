"""Checks for the dispersion-shell fit (shell_fit.py, rectify.py, synth.py).

pytest:  pytest tests/test_shell.py          (geometry + synthetic recovery at T = 3.2 s)
sweep:   python tests/test_shell.py sweep --out results/shell-synthetic.csv --jobs 27
         (speeds × headings × seeds × T ∈ {1.6, 3.2, 6.4, 12}, live and frozen synthetic seas)
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import rectify as rc  # noqa: E402
import shell_fit as sf  # noqa: E402
import synth  # noqa: E402


def fake_row(heading=60.0, pitch=-90.0, h=59.778, e=10.0, n=-4.0, W=720, H=1280, fov=92.0):
    """A frames.csv-like row for a stabilised gimbal camera: image top along `heading`,
    pitched `pitch` from the horizon (−90 = nadir). Built independently of the renderer:
    camera x (right) = right of track, camera y (down) and z (optical axis) from the pitch."""
    hd = math.radians(heading)
    fwd = np.array([math.sin(hd), math.cos(hd), 0.0])
    right = np.array([math.cos(hd), -math.sin(hd), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    p = math.radians(pitch)
    z = math.cos(p) * fwd + math.sin(p) * up          # optical axis
    y = np.cross(z, right)                              # image down (x × y = z)
    R = np.stack([right, y, z], 1)  # columns: camera x, y, z in ENU
    f = (H / 2) / math.tan(math.radians(fov) / 2)
    row = {f"R{i}{j}": R[i, j] for i in range(3) for j in range(3)}
    row.update(cam_e=e, cam_n=n, cam_u=h, fx=f, fy=f, cx=W / 2, cy=H / 2, width=W, height=H, t=0.0, frame=0)
    return row


def test_fake_row_matches_real_frame0():
    # Frame 0 of e1-pilot-000 (heading 60, nadir): R row-major as written by the renderer.
    real = [0.5, -0.866025404, 0, -0.866025404, -0.5, 0, 0, 0, -1]
    row = fake_row(heading=60)
    got = [row[f"R{i}{j}"] for i in range(3) for j in range(3)]
    assert np.allclose(got, real, atol=1e-8)


def test_tile_projects_back_to_its_cells():
    for pitch in (-90.0, -60.0):
        row = fake_row(pitch=pitch)
        g = rc.tile_geometry(row, n=64)
        u, v, gsd = rc.pixel_map(row, g)
        off = rc.tile_offsets(g)
        K, R, t = rc.pose(row)
        dE, dN = rc.ray_to_plane(K, R, t[2], u, v)
        assert np.allclose(dE, off[..., 0], atol=1e-6) and np.allclose(dN, off[..., 1], atol=1e-6)
        assert np.isfinite(u).all() and (u > 0).all() and (u < 720).all() and (v > 0).all() and (v < 1280).all()
    # Nadir: tile centred under the camera, side = 0.6 × footprint width.
    row = fake_row(pitch=-90.0)
    g = rc.tile_geometry(row)
    assert abs(g.offset[0]) < 1e-9 and abs(g.offset[1]) < 1e-9
    assert abs(g.side - 0.6 * 2 * row["cam_u"] * 360 / row["fx"]) < 1e-9


def test_projection_matches_e1geom():
    try:
        import e1geom
    except Exception:  # pragma: no cover
        import pytest
        pytest.skip("e1geom.py not importable")
    for pitch in (-90.0, -60.0):
        row = fake_row(pitch=pitch, heading=150.0)
        pose = e1geom.pose_from_row(row)
        g = rc.tile_geometry(row, n=16)
        u, v, _ = rc.pixel_map(row, g)
        X, ok = e1geom.backproject(pose, u, v)
        assert ok.all()
        off = rc.tile_offsets(g)
        assert np.allclose(X[..., 0] - row["cam_e"], off[..., 0], atol=1e-6)
        assert np.allclose(X[..., 1] - row["cam_n"], off[..., 1], atol=1e-6)
        assert np.allclose(e1geom.footprint_centre(pose) - [row["cam_e"], row["cam_n"]], g.offset, atol=1e-6)


def test_rectify_pixel_convention():
    # An image whose value is its own column centre u = c + 0.5 must rectify to u.
    row = fake_row()
    g = rc.tile_geometry(row, n=32)
    u, v, gsd = rc.pixel_map(row, g)
    img = np.tile(np.arange(720, dtype=np.float32) + 0.5, (1280, 1))
    out = rc.rectify(img, u, v, gsd, g.cell, antialias=False)
    assert np.allclose(out, u, atol=1e-3)


def _synthetic_case(speed, heading, T, seed=1, frozen=False, model="dispersive", prior_off=(3.1, -2.2)):
    wf = synth.WaveField(seed=seed, frozen=frozen)
    V = speed * sf.unit(heading)
    st = synth.moving_tile_stack(wf, V, nt=int(round(T * 10)))
    sp = sf.prepare(st, 0.1, 42.0 / 256)
    Ve, _, _ = sf.fit_velocity(sp, V + np.array(prior_off), model=model)
    return Ve - V


def test_synthetic_live_recovers_v_at_3p2s():
    for speed, heading in ((15, 240), (30, 150), (60, 60)):
        err = _synthetic_case(speed, heading, 3.2)
        assert np.hypot(*err) < 0.05, (speed, heading, err)


def test_synthetic_scale_fit_recovers_one():
    wf = synth.WaveField(seed=1)
    V = 30 * sf.unit(150)
    st = synth.moving_tile_stack(wf, V, nt=64)
    sp = sf.prepare(st, 0.1, 42.0 / 256)
    V0, _, _ = sf.fit_velocity(sp, V + np.array([3.1, -2.2]))
    sc, V1, _, _ = sf.fit_scale(sp, V0)
    assert abs(sc - 1) < 0.02 and np.hypot(*(V1 - V)) < 0.05, (sc, V1 - V)


def test_synthetic_frozen_static_model_recovers_v():
    err = _synthetic_case(30, 150, 3.2, frozen=True, model="static")
    assert np.hypot(*err) < 0.05, err


# --------------------------------------------------------------------------------------
# Sweep (CLI)


def _sweep_one(args):
    seed, frozen, speed, heading, Ts, pad_frames, opts = args
    wf = synth.WaveField(seed=seed, frozen=frozen, peak_wavelength=opts["peak_wavelength"])
    V = speed * sf.unit(heading)
    nmax = int(round(max(Ts) * 10))
    st = synth.moving_tile_stack(wf, V, nt=nmax, side=opts["side"], quad=opts["quad"], noise=opts["noise"], seed=seed)
    rows = []
    for T, s0, nw in sf.windows(nmax, 10.0, Ts, 1.6):
        sp = sf.prepare(st[s0:s0 + nw], 0.1, opts["side"] / 256)
        off = sf.prior_offset(f"synth-{seed}-{frozen}-{speed}-{heading}", s0 / 10, T, 5.0, 1)
        for model in ("dispersive", "static"):
            Ve, J, info = sf.fit_velocity(sp, V + off, model=model)
            fr, occ, lam = sf.shell_stats(sp, Ve, 1.0, model)
            r = {"seed": seed, "sea": "frozen" if frozen else "live", "speed": speed, "heading": heading,
                 "side": opts["side"], "quad": opts["quad"], "noise": opts["noise"],
                 "T": T, "t_start": s0 / 10, "model": model, "v_true_e": V[0], "v_true_n": V[1],
                 "v_est_e": Ve[0], "v_est_n": Ve[1]}
            r.update(sf.errors_row(Ve - V, V, heading))
            r.update({"shell_frac": fr, "band_occupancy": occ, "lambda_c": lam,
                      "res_bin_lambda_c": lam / T, "res_dwell_lambda_c": lam * speed / opts["side"],
                      "J_second_over_J": info["J_second"] / J, "quad_ok": info["quad_ok"]})
            rows.append(r)
    print(f"synthetic seed {seed} {'frozen' if frozen else 'live'} {speed} m/s hdg {heading}: done", flush=True)
    return rows


def sweep(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--speeds", type=float, nargs="+", default=[15, 30, 60])
    ap.add_argument("--headings", type=float, nargs="+", default=[60, 150, 240])
    ap.add_argument("--T", type=float, nargs="+", default=[1.6, 3.2, 6.4, 12.0])
    ap.add_argument("--side", type=float, default=42.0)
    ap.add_argument("--quad", type=float, default=0.0)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--peak-wavelength", type=float, default=40.0)
    ap.add_argument("--frozen", action="store_true", help="also run frozen seas")
    a = ap.parse_args(argv)
    opts = {"side": a.side, "quad": a.quad, "noise": a.noise, "peak_wavelength": a.peak_wavelength}
    tasks = [(s, fz, v, h, tuple(a.T), None, opts) for s in a.seeds for fz in ((False, True) if a.frozen else (False,))
             for v in a.speeds for h in a.headings]
    rows = []
    if a.jobs > 1:
        import multiprocessing as mp
        import os
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        os.environ.setdefault("SHELL_FFT_WORKERS", "4")
        with mp.get_context("spawn").Pool(a.jobs) as pool:
            for r in pool.imap_unordered(_sweep_one, tasks):
                rows.extend(r)
    else:
        for t in tasks:
            rows.extend(_sweep_one(t))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        sweep(sys.argv[2:])
    else:
        print(__doc__)
