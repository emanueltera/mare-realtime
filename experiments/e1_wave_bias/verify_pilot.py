"""Check the E1 pilot renders: image size, intrinsics, camera motion, and the frozen-sea
control (render.seaRate 0) — that consecutive frozen frames are the same sea translated
by exactly the camera's own motion.

Usage: python3 verify_pilot.py <job dir> [<job dir> ...]   (a job dir holds the run dirs)

Needs numpy and Pillow only. Conventions (docs/ARCHITECTURE.md): ENU world, OpenCV camera
axes, R_wc row-major (camera → ENU), K = [fx 0 cx; 0 fy cy; 0 0 1], pixel (0, 0) at the
top-left corner of the top-left pixel (so pixel centres are at integer + 0.5).

The registration: each frame is rectified onto an east/north grid fixed to the camera
(centred on its nadir point, plane up = 0, true pose only), and two frames' grids are
phase-correlated. A pattern fixed on the water at world X sits at X − c in a grid centred
on c, so between frames i and j it moves by s = −(c_j − c_i) when the sea is still. The
apparent pattern velocity is a = (s + c_j − c_i) / Δt: ≈ 0 for a frozen sea, the motion
of the water's pattern otherwise.
"""

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def read_frames(run_dir):
    with open(run_dir / "frames.csv") as f:
        rows = list(csv.DictReader(f))
    return [{k: float(v) for k, v in r.items()} for r in rows]


def pose(row):
    R = np.array([row[f"R{i}{j}"] for i in range(3) for j in range(3)]).reshape(3, 3)
    t = np.array([row["cam_e"], row["cam_n"], row["cam_u"]])
    K = np.array([[row["fx"], 0, row["cx"]], [0, row["fy"], row["cy"]], [0, 0, 1.0]])
    return K, R, t


def gray(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    return a @ np.array([0.2126, 0.7152, 0.0722])


def bilinear(img, u, v):
    """Sample img at pixel coordinates (u right, v down; pixel centres at +0.5); NaN outside."""
    h, w = img.shape
    x = u - 0.5
    y = v - 0.5
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    ok = (x0 >= 0) & (y0 >= 0) & (x0 < w - 1) & (y0 < h - 1)
    x0c = np.clip(x0, 0, w - 2)
    y0c = np.clip(y0, 0, h - 2)
    fx = x - x0c
    fy = y - y0c
    out = (img[y0c, x0c] * (1 - fx) * (1 - fy) + img[y0c, x0c + 1] * fx * (1 - fy)
           + img[y0c + 1, x0c] * (1 - fx) * fy + img[y0c + 1, x0c + 1] * fx * fy)
    out[~ok] = np.nan
    return out


def rectify(img, K, R, t, center, side, n):
    """The image on an east/north grid of n×n cells, `side` metres, centred on `center` (E, N), plane up = 0.
    Row 0 is the north edge, column 0 the west edge."""
    cell = side / n
    offs = (np.arange(n) + 0.5) * cell - side / 2
    E, N = np.meshgrid(center[0] + offs, center[1] - offs)
    P = np.stack([E, N, np.zeros_like(E)], -1) - t
    pc = P @ R  # R^T (P - t) for row vectors
    z = pc[..., 2]
    u = K[0, 0] * pc[..., 0] / z + K[0, 2]
    v = K[1, 1] * pc[..., 1] / z + K[1, 2]
    out = bilinear(img, u, v)
    out[z <= 0] = np.nan
    return out


def phase_correlate(a, b):
    """Shift (dx east, dy north) in cells that moves a's pattern onto b's, sub-cell, and the peak height."""
    n = a.shape[0]
    win = np.outer(np.hanning(n), np.hanning(n))
    a = np.nan_to_num(a - np.nanmean(a)) * win
    b = np.nan_to_num(b - np.nanmean(b)) * win
    F = np.fft.fft2(b) * np.conj(np.fft.fft2(a))
    F /= np.abs(F) + 1e-12
    r = np.fft.ifft2(F).real
    iy, ix = np.unravel_index(np.argmax(r), r.shape)

    def sub(c, m1, p1):
        d = m1 - 2 * c + p1
        return 0.0 if d == 0 else 0.5 * (m1 - p1) / d

    dy = iy + sub(r[iy, ix], r[(iy - 1) % n, ix], r[(iy + 1) % n, ix])
    dx = ix + sub(r[iy, ix], r[iy, (ix - 1) % n], r[iy, (ix + 1) % n])
    dy = dy - n if dy > n / 2 else dy
    dx = dx - n if dx > n / 2 else dx
    # Rows run southward: a pattern moving north moves to lower rows.
    return dx, -dy, r[iy, ix]


def check_run(run_dir, n=512, gaps=(1, 3), every=5, start=5):
    run = json.loads((run_dir / "run.json").read_text())
    rows = read_frames(run_dir)
    tr = run["trajectory"]
    rend = run["render"]
    out = {"id": run["id"], "seaRate": rend.get("seaRate", 1), "sea": run["sea"]["windSpeed"],
           "altitude": tr["altitude"], "airspeed": tr["airspeed"], "heading": tr["start"]["heading"]}

    # Image size and intrinsics.
    im0 = Image.open(run_dir / "frames" / "000000.png")
    out["png"] = f"{im0.size[0]}x{im0.size[1]}"
    W, H = rend["width"], rend["height"]
    fov = tr["gimbal"]["fov"]
    f_exp = (H / 2) / math.tan(math.radians(fov) / 2)
    r0 = rows[0]
    out["fx"], out["fy"], out["cx"], out["cy"] = r0["fx"], r0["fy"], r0["cx"], r0["cy"]
    out["f_expected"] = f_exp
    out["K_err_px"] = max(abs(r0["fx"] - f_exp), abs(r0["fy"] - f_exp), abs(r0["cx"] - W / 2), abs(r0["cy"] - H / 2))
    out["frames"] = len(rows)

    # Camera motion: speed and course from the camera centre, and its attitude.
    T = np.array([r["t"] for r in rows])
    C = np.array([[r["cam_e"], r["cam_n"], r["cam_u"]] for r in rows])
    v = np.diff(C, axis=0) / np.diff(T)[:, None]
    gs = np.hypot(v[:, 0], v[:, 1])
    out["cam_speed_min"], out["cam_speed_max"] = gs.min(), gs.max()
    out["cam_vu_maxabs"] = np.abs(v[:, 2]).max()
    crs = np.degrees(np.arctan2(v[:, 0], v[:, 1])) % 360
    out["cam_course_mean"] = crs.mean()
    out["cam_u_mean"] = C[:, 2].mean()
    out["seaTime_first"], out["seaTime_last"] = rows[0]["seaTime"], rows[-1]["seaTime"]
    zaxis = np.array([[r["R02"], r["R12"], r["R22"]] for r in rows])
    out["nadir_err_deg"] = np.degrees(np.arccos(np.clip(-zaxis[:, 2], -1, 1))).max()
    # Image up (−y camera) on the ground, as a compass bearing.
    up = -np.array([r0["R01"], r0["R11"], r0["R21"]])
    out["image_up_bearing"] = math.degrees(math.atan2(up[0], up[1])) % 360

    # Registration of rectified frame pairs.
    pitch = tr["gimbal"]["pitch"]
    alt = C[0, 2]
    half_w = alt * (W / 2) / r0["fx"]
    side = 0.6 * 2 * half_w  # 60 % of the smaller footprint dimension (the width, portrait)
    out["tile_side_m"] = side
    out["tile_cell_m"] = side / n
    res = []
    cache = {}

    def rect(i):
        if i not in cache:
            K, R, t = pose(rows[i])
            centre = t[:2]
            if pitch != -90:
                raise SystemExit("oblique registration not implemented here")
            cache[i] = (rectify(gray(run_dir / "frames" / f"{i:06d}.png"), K, R, t, centre, side, n), centre)
        return cache[i]

    for g in gaps:
        for i in range(start, len(rows) - g, every):
            a, ca = rect(i)
            b, cb = rect(i + g)
            dx, dy, peak = phase_correlate(a, b)
            s = np.array([dx, dy]) * side / n
            dc = cb - ca
            dt = rows[i + g]["t"] - rows[i]["t"]
            app = (s + dc) / dt
            # Residual after the true shift: NCC of b against a moved by −dc, over the overlap.
            res.append({"g": g, "i": i, "dt": dt, "shift_e": s[0], "shift_n": s[1], "cam_de": dc[0], "cam_dn": dc[1],
                        "a_e": app[0], "a_n": app[1], "peak": peak})
        for k in list(cache):
            if k < i - 3:
                del cache[k]
    return out, res


def ncc_after_true_shift(run_dir, i, j, n=512):
    """NCC between frame j and frame i resampled at the same world points (true poses, plane up = 0)."""
    rows = read_frames(run_dir)
    Ki, Ri, ti = pose(rows[i])
    Kj, Rj, tj = pose(rows[j])
    alt = ti[2]
    side = 0.6 * 2 * alt * (rows[i]["width"] / 2) / rows[i]["fx"]
    centre = (ti[:2] + tj[:2]) / 2  # world-fixed patch seen by both
    a = rectify(gray(run_dir / "frames" / f"{i:06d}.png"), Ki, Ri, ti, centre, side * 0.5, n)
    b = rectify(gray(run_dir / "frames" / f"{j:06d}.png"), Kj, Rj, tj, centre, side * 0.5, n)
    ok = np.isfinite(a) & np.isfinite(b)
    a = a[ok] - a[ok].mean()
    b = b[ok] - b[ok].mean()
    return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum())), float(np.abs(a - b).mean())


def registration_checks(run_dir, n=512, gaps=(1, 3), every=5, start=5):
    """Split the camera-fixed registration's residual into its two sources, per pair:
    - `world`: both frames rectified onto the SAME world-fixed grid (centred between the two
      nadir points); a still sea gives zero shift, so the shift found is what the render
      itself changed between the frames (view-dependent shading, parallax), as m/s;
    - `synth`: ONE frame rectified at its own nadir and again at the next frame's nadir, so
      the pattern is static by construction; the error is the registration's own (sub-cell
      interpolation, windowing), as m/s."""
    run = json.loads((run_dir / "run.json").read_text())
    rows = read_frames(run_dir)
    W = run["render"]["width"]
    out = []
    for g in gaps:
        for i in range(start, len(rows) - g, every):
            j = i + g
            Ki, Ri, ti = pose(rows[i])
            Kj, Rj, tj = pose(rows[j])
            side = 0.6 * 2 * ti[2] * (W / 2) / rows[i]["fx"]
            dt = rows[j]["t"] - rows[i]["t"]
            gi = gray(run_dir / "frames" / f"{i:06d}.png")
            gj = gray(run_dir / "frames" / f"{j:06d}.png")
            mid = (ti[:2] + tj[:2]) / 2
            dx, dy, _ = phase_correlate(rectify(gi, Ki, Ri, ti, mid, side, n), rectify(gj, Kj, Rj, tj, mid, side, n))
            world = np.array([dx, dy]) * side / n / dt
            dc = tj[:2] - ti[:2]
            dx, dy, _ = phase_correlate(rectify(gi, Ki, Ri, ti, ti[:2], side, n), rectify(gi, Ki, Ri, ti, ti[:2] + dc, side, n))
            synth = (np.array([dx, dy]) * side / n + dc) / dt
            out.append({"g": g, "i": i, "world_e": world[0], "world_n": world[1], "synth_e": synth[0], "synth_n": synth[1]})
    return out


def main():
    summary = []
    for job in sys.argv[1:]:
        for run_dir in sorted(p for p in Path(job).iterdir() if (p / "frames.csv").exists()):
            info, res = check_run(run_dir)
            ncc1, mad1 = ncc_after_true_shift(run_dir, 60, 61)
            ncc3, mad3 = ncc_after_true_shift(run_dir, 60, 63)
            info.update({"ncc_true_g1": ncc1, "ncc_true_g3": ncc3, "mad_true_g1": mad1, "mad_true_g3": mad3})
            checks = registration_checks(run_dir)
            for g in (1, 3):
                c = [x for x in checks if x["g"] == g]
                for k in ("world", "synth"):
                    e = np.median([x[f"{k}_e"] for x in c])
                    n_ = np.median([x[f"{k}_n"] for x in c])
                    info[f"g{g}_{k}_e_med"], info[f"g{g}_{k}_n_med"] = float(e), float(n_)
                    info[f"g{g}_{k}_abs_med"] = float(math.hypot(e, n_))
                r = [x for x in res if x["g"] == g]
                ae = np.array([x["a_e"] for x in r])
                an = np.array([x["a_n"] for x in r])
                info[f"g{g}_pairs"] = len(r)
                info[f"g{g}_a_e_med"] = float(np.median(ae))
                info[f"g{g}_a_n_med"] = float(np.median(an))
                info[f"g{g}_a_abs_med"] = float(math.hypot(np.median(ae), np.median(an)))
                info[f"g{g}_a_abs_maxpair"] = float(np.hypot(ae, an).max())
                info[f"g{g}_peak_med"] = float(np.median([x["peak"] for x in r]))
            summary.append(info)
            print(json.dumps(info, default=float))
            sys.stdout.flush()
    return summary


if __name__ == "__main__":
    main()
