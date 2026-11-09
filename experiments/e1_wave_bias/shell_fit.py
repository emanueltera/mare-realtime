"""E1 dispersion-shell velocity fit on a camera-fixed sea tile (classical, no network).

Each frame is rectified onto an ENU-aligned n × n tile fixed to the camera (rectify.py: true
attitude and height only). Over a window of T seconds the tile stack gets a Hann window in
x, y and t and a 3-D FFT. A wave of wavevector k and intrinsic frequency Ω = ±√(g|k|) seen
from a tile moving at V relative to the water appears at

    ω' = ±√(g·s·|k|) − k·V          (s = 1 with the true height; see "scale" below)

with the convention that a component e^{i(k·x' − ω' t)} lies at (k, ω'). V is found by
maximising the (weighted) spectral energy within a band around that shell, both branches,
over a ±box around a prior (truth + a random offset), then refined below the grid step.

Two models are fitted:
- 'dispersive': the shell above. A static-sea assumption is not made; this is the E1
  estimator of velocity relative to the water.
- 'static':     ω' = −k·V (a frozen pattern). On a frozen sea this is the right model (the
  pipeline control); on a live sea it is a spectral pattern tracker, whose error is
  −(apparent pattern velocity), the bias E1 is about.

Departures from the plain design, and why:
- Temporal aliasing: at 10 fps the tile moves V/10 per frame (6 m at 60 m/s), so k·V
  exceeds the temporal Nyquist (π·fps) for all but the longest waves. The shell is
  therefore evaluated modulo 2π·fps (wrapped), which the FFT does anyway.
- Exposure: each rectified tile is divided by its mean (the renderer's auto-exposure
  changes gain from frame to frame), and the per-cell temporal mean over the window is
  removed (fixed-in-image structure such as the midday glint patch and vignetting lies
  at ω' = 0 for every k and would otherwise pull the fit).
- Scale: with a height error h' = s·h the tile is stretched by s, so in tile units
  k_m = k/s, V_m = s·V and the shell is ω' = ±√(g·s·k_m) − k_m·V_m. `--fit-scale`
  fits s by a 1-D search after V and refits V at the best s.

Resolution limits (derived, reported per row): one temporal bin is Δω = 2π/T, worth
λ/T of velocity for a component of wavelength λ. In a camera-fixed tile water crosses the
tile in L/|V| s, which smears each k over ±|V|·Δk in ω' (Δk = 2π/L) — a dwell-limited
velocity resolution of λ·|V|/L. Many components give sub-bin precision; these are not
error bars.

Usage:
    python shell_fit.py RUN_DIR [RUN_DIR ...] --out results/shell-pilot.csv [--fit-scale]
    (a directory holding run directories is expanded)
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.fft

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rectify as rc  # noqa: E402

G = 9.81
WAVE_DIR_DEG = 60.0  # fallback only: every E1 preset's wind sea travels toward 60°; run.json is read per run
FFT_WORKERS = int(os.environ.get("SHELL_FFT_WORKERS", "-1"))


def unit(bearing_deg: float) -> np.ndarray:
    b = math.radians(bearing_deg)
    return np.array([math.sin(b), math.cos(b)])  # (E, N)


# --------------------------------------------------------------------------------------
# Spectrum


@dataclass
class Spectrum:
    k: np.ndarray        # [nk, 2] (kE, kN) rad/m, half plane (kE ≥ 0)
    kmag: np.ndarray     # [nk]
    S: np.ndarray        # [nk, ntp] power; column j ↔ ω' = j·dwp (circular)
    Sband: np.ndarray    # [nk, ntp] S summed over ±hb columns (circular)
    w: np.ndarray        # [nk] objective weight per k (incl. ½ on the kE = 0 / Nyquist columns)
    half: np.ndarray     # [nk] ½ on duplicated columns, else 1 (for energy fractions)
    dw: float            # unpadded bin, 2π / T  (rad/s)
    dwp: float           # padded bin (rad/s)
    ntp: int
    hb: int              # band half-width in padded bins
    band: float          # band half-width (rad/s)
    T: float
    L: float             # tile side (m)
    dk: float            # 2π / L
    extra: dict = field(default_factory=dict)


def prepare(stack: np.ndarray, dt: float, cell: float, *, pad: int = 4, kmin_bins: float = 3.0,
            kmax_frac: float = 0.8, band_bins: float = 1.0, weight: str = "whiten",
            lambda_max: float | None = None, normalize: bool = True, remove_static: bool = True) -> Spectrum:
    """Windowed 3-D power spectrum of a tile stack [nt, n(north), n(east)]."""
    x = np.asarray(stack, dtype=np.float64)
    nt, n, n2 = x.shape
    assert n == n2
    if normalize:
        m = np.nanmean(x, axis=(1, 2))
        x = x / m[:, None, None] - 1.0
    x = np.nan_to_num(x, nan=0.0)
    if remove_static:
        x = x - x.mean(axis=0, keepdims=True)
    wt = np.hanning(nt + 2)[1:-1]  # strictly positive Hann (no wasted end frames)
    ws = np.hanning(n + 2)[1:-1]
    x = x * wt[:, None, None] * ws[None, :, None] * ws[None, None, :]
    ntp = nt * pad
    F = scipy.fft.rfftn(x, s=(ntp, n, n), axes=(0, 1, 2), workers=FFT_WORKERS)
    P = (F.real ** 2 + F.imag ** 2)
    L = n * cell
    kN = 2 * np.pi * np.fft.fftfreq(n, cell)
    kE = 2 * np.pi * np.fft.rfftfreq(n, cell)
    KE, KN = np.meshgrid(kE, kN)  # [n, n//2+1]
    km = np.hypot(KE, KN)
    dk = 2 * np.pi / L
    kmax = kmax_frac * np.pi / cell
    kmin = kmin_bins * dk
    if lambda_max:
        kmin = max(kmin, 2 * np.pi / lambda_max)
    sel = (km >= kmin) & (km <= kmax)
    half = np.where((KE == 0) | (np.isclose(KE, kE[-1]) & (n % 2 == 0)), 0.5, 1.0)[sel]
    Psel = P[:, sel].T  # [nk, ntp] indexed by numpy's ω (e^{-iωt})
    # Physical ω' = −ω_numpy: column j of S is ω' = j·dwp.
    S = Psel[:, (-np.arange(ntp)) % ntp]
    dw = 2 * np.pi / (nt * dt)
    dwp = dw / pad
    hb = max(0, int(round(band_bins * pad)))
    Sband = S.copy()
    for j in range(1, hb + 1):
        Sband += np.roll(S, j, axis=1) + np.roll(S, -j, axis=1)
    Ek = (S * 1.0).sum(1)
    if weight == "energy":
        w = np.ones_like(Ek)
    elif weight == "sqrt":
        w = 1.0 / np.sqrt(Ek + 1e-30)
    elif weight == "whiten":
        w = 1.0 / (Ek + 1e-30)
    else:
        raise ValueError(weight)
    w = w * half
    w = w / w.sum()
    k = np.stack([KE[sel], KN[sel]], -1)
    return Spectrum(k=k, kmag=km[sel], S=S.astype(np.float32), Sband=Sband.astype(np.float32),
                    w=w.astype(np.float64), half=half, dw=dw, dwp=dwp, ntp=ntp, hb=hb, band=hb * dwp,
                    T=nt * dt, L=L, dk=dk)


def shell_freqs(sp: Spectrum, V: np.ndarray, s: float = 1.0, model: str = "dispersive") -> list[np.ndarray]:
    """Predicted ω' per branch, each [m, nk], for velocities V [m, 2]."""
    dop = V @ sp.k.T  # [m, nk]
    if model == "static":
        return [-dop]
    om = np.sqrt(G * s * sp.kmag)[None, :]
    return [om - dop, -om - dop]


def objective(sp: Spectrum, V: np.ndarray, s: float = 1.0, model: str = "dispersive", chunk: int = 256) -> np.ndarray:
    """Weighted band energy on the shell for each V [m, 2] (linear interpolation along ω')."""
    V = np.atleast_2d(np.asarray(V, dtype=np.float64))
    if _torch_device() is not None:
        return _objective_torch(sp, V, s, model)
    out = np.empty(len(V))
    flat = sp.Sband.ravel()
    base = (np.arange(len(sp.kmag)) * sp.ntp)[None, :]
    for a in range(0, len(V), chunk):
        Vc = V[a:a + chunk]
        acc = np.zeros(len(Vc))
        for om in shell_freqs(sp, Vc, s, model):
            f = np.mod(om / sp.dwp, sp.ntp)
            i0 = np.floor(f).astype(np.int64)
            fr = f - i0
            i0 %= sp.ntp
            i1 = (i0 + 1) % sp.ntp
            val = flat[base + i0] * (1 - fr) + flat[base + i1] * fr
            acc += val @ sp.w
        out[a:a + chunk] = acc
    return out


_TORCH = {"checked": False, "device": None}


def _torch_device():
    """CUDA device for the objective if SHELL_DEVICE != 'cpu' and torch sees a GPU, else None."""
    if not _TORCH["checked"]:
        _TORCH["checked"] = True
        if os.environ.get("SHELL_DEVICE", "auto") != "cpu":
            try:
                import torch
                if torch.cuda.is_available():
                    _TORCH["device"] = torch.device(os.environ.get("SHELL_CUDA", "cuda"))
            except Exception:
                pass
    return _TORCH["device"]


def _objective_torch(sp: Spectrum, V: np.ndarray, s: float, model: str, chunk: int = 2048) -> np.ndarray:
    """Same as objective() on the GPU (frequencies in float64, spectrum gathers in float32)."""
    import torch
    dev = _torch_device()
    cache = sp.extra.get("_torch")
    if cache is None:
        cache = {
            "flat": torch.as_tensor(sp.Sband, device=dev).reshape(-1),
            "k": torch.as_tensor(sp.k, device=dev, dtype=torch.float64),
            "om": torch.sqrt(G * torch.as_tensor(sp.kmag, device=dev, dtype=torch.float64)),
            "w": torch.as_tensor(sp.w, device=dev, dtype=torch.float32),
            "base": (torch.arange(len(sp.kmag), device=dev) * sp.ntp)[None, :],
        }
        sp.extra["_torch"] = cache
    out = []
    Vt = torch.as_tensor(V, device=dev, dtype=torch.float64)
    for a in range(0, len(V), chunk):
        dop = Vt[a:a + chunk] @ cache["k"].T
        oms = [-dop] if model == "static" else [math.sqrt(s) * cache["om"][None, :] - dop,
                                                 -math.sqrt(s) * cache["om"][None, :] - dop]
        acc = torch.zeros(dop.shape[0], device=dev, dtype=torch.float32)
        for om in oms:
            f = torch.remainder(om / sp.dwp, sp.ntp)
            i0 = torch.floor(f)
            fr = (f - i0).to(torch.float32)
            i0 = i0.to(torch.int64) % sp.ntp
            i1 = (i0 + 1) % sp.ntp
            val = cache["flat"][cache["base"] + i0] * (1 - fr) + cache["flat"][cache["base"] + i1] * fr
            acc += val @ cache["w"]
        out.append(acc.double().cpu().numpy())
    return np.concatenate(out)


def _grid(center, half, step):
    o = np.arange(-half, half + 1e-9, step)
    gx, gy = np.meshgrid(center[0] + o, center[1] + o)
    return np.stack([gx.ravel(), gy.ravel()], -1), len(o)


def _quad_refine(J2, x0, y0, step):
    """Sub-grid peak of a 2-D grid of objective values around its argmax (5×5 quadratic LSQ)."""
    m = J2.shape[0]
    iy, ix = np.unravel_index(np.argmax(J2), J2.shape)
    best = np.array([x0[ix], y0[iy]])
    r = 2
    if iy < r or ix < r or iy >= m - r or ix >= m - r:
        return best, False
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    z = J2[iy - r:iy + r + 1, ix - r:ix + r + 1].ravel()
    A = np.stack([np.ones(z.size), xx.ravel(), yy.ravel(), xx.ravel() ** 2, xx.ravel() * yy.ravel(), yy.ravel() ** 2], -1)
    c, *_ = np.linalg.lstsq(A, z, rcond=None)
    H = np.array([[2 * c[3], c[4]], [c[4], 2 * c[5]]])
    if not (np.all(np.linalg.eigvalsh(H) < 0)):
        return best, False
    d = -np.linalg.solve(H, c[1:3])
    if np.any(np.abs(d) > 1.5):
        return best, False
    return best + d * step, True


def fit_velocity(sp: Spectrum, prior, *, box: float = 8.0, coarse: float = 0.2, fine: float = 0.02,
                 fine_half: float = 0.4, s: float = 1.0, model: str = "dispersive"):
    """Grid search over prior ± box (coarse), then a fine grid around the best, then a
    quadratic sub-grid refinement. Returns (V_est, J_max, info)."""
    prior = np.asarray(prior, dtype=np.float64)
    Vg, m = _grid(prior, box, coarse)
    Jc = objective(sp, Vg, s, model)
    i = int(np.argmax(Jc))
    c1 = Vg[i]
    # Second-best separated peak (ambiguity measure): best J outside 1 m/s of the max.
    far = np.hypot(*(Vg - c1).T) > 1.0
    J2nd = float(Jc[far].max()) if far.any() else float("nan")
    Vf, mf = _grid(c1, fine_half, fine)
    Jf = objective(sp, Vf, s, model)
    o = np.arange(-fine_half, fine_half + 1e-9, fine)
    V, ok = _quad_refine(Jf.reshape(mf, mf), c1[0] + o, c1[1] + o, fine)
    Jmax = float(objective(sp, V[None], s, model)[0])
    at_edge = bool(np.any(np.abs(c1 - prior) > box - coarse * 1.5))
    return V, Jmax, {"J_coarse_max": float(Jc[i]), "J_second": J2nd, "J_median": float(np.median(Jc)),
                     "quad_ok": ok, "at_edge": at_edge}


def fit_scale(sp: Spectrum, V, *, lo: float = 0.7, hi: float = 1.3, step: float = 0.01,
              vhalf: float = 1.5, vstep: float = 0.05):
    """Joint (V, s) fit: for each s on a grid, the best V on a local grid (±vhalf around
    the s = 1 estimate V); s and V are then refined (parabola in s; fit_velocity at that s).
    Returns (s, V, J, info). s = 1 means the dispersion matches the true height."""
    V = np.asarray(V, dtype=np.float64)
    ss = np.arange(lo, hi + 1e-9, step)
    Vg, m = _grid(V, vhalf, vstep)
    best = np.array([objective(sp, Vg, sc, "dispersive").max() for sc in ss])
    i = int(np.argmax(best))
    sc = float(ss[i])
    if 0 < i < len(ss) - 1:
        a, b, c = best[i - 1], best[i], best[i + 1]
        d = a - 2 * b + c
        if d < 0:
            sc = float(ss[i] + 0.5 * (a - c) / d * step)
    Jg = objective(sp, Vg, sc, "dispersive")
    V2, J, info = fit_velocity(sp, Vg[int(np.argmax(Jg))], box=0.5, coarse=0.05, s=sc, model="dispersive")
    info["scale_at_bound"] = i in (0, len(ss) - 1)
    return sc, V2, J, info


def shell_stats(sp: Spectrum, V, s: float = 1.0, model: str = "dispersive"):
    """Fraction of the (selected-k) spectral energy within ±band of the shell at V, the
    fraction a featureless spectrum would give (band occupancy), and the energy-weighted
    mean wavelength of the in-band energy."""
    oms = shell_freqs(sp, np.asarray(V, dtype=np.float64)[None], s, model)
    j = np.arange(sp.ntp)[None, :]
    mask = np.zeros(sp.S.shape, dtype=bool)
    for om in oms:
        f = np.mod(om[0] / sp.dwp, sp.ntp)[:, None]
        d = np.abs(j - f)
        d = np.minimum(d, sp.ntp - d)
        mask |= d <= sp.hb + 0.5
    E = sp.S * sp.half[:, None]
    tot = float(E.sum())
    inb = float((E * mask).sum())
    Ein = (E * mask).sum(1)
    lam = float((Ein * (2 * np.pi / sp.kmag)).sum() / max(Ein.sum(), 1e-30))
    return inb / max(tot, 1e-30), float(mask.mean()), lam


# --------------------------------------------------------------------------------------
# Runs


def run_meta(run_dir: Path) -> dict:
    run = rc.load_run(run_dir)
    meta = {}
    mp = run_dir / "meta.json"
    if mp.exists():
        import json
        meta = json.loads(mp.read_text())
    summ = (meta.get("summary") or "").split(" · ")
    tr = run["trajectory"]
    return {
        "run_id": run["id"],
        "job": meta.get("job", ""),
        "sea": summ[0] if summ else "",
        "light": summ[1] if len(summ) > 1 else "",
        "hs": meta.get("seaStats", {}).get("significantHeight", float("nan")),
        "peak_wavelength": meta.get("seaStats", {}).get("peakWavelength", float("nan")),
        "wind_speed": run["sea"]["windSpeed"],
        # Sea directions are compass degrees the waves travel TOWARD (src/ocean/scene.js).
        "wind_dir": run["sea"].get("windDirection", WAVE_DIR_DEG),
        "swell_dir": run["sea"].get("swellDirection") if (run["sea"].get("swellHeight") or 0) > 0 else float("nan"),
        "altitude": tr["altitude"],
        "airspeed": tr["airspeed"],
        "heading": tr["start"]["heading"],
        "pitch": tr["gimbal"]["pitch"],
        "sun_elevation": run["light"]["sunElevation"],
        "sun_azimuth": run["light"]["sunAzimuth"],
        "sea_rate": run["render"].get("seaRate", 1),
        "fps": run["render"]["fps"],
    }


def prior_offset(run_id: str, t0: float, T: float, half: float, seed: int) -> np.ndarray:
    key = f"{seed}|{run_id}|{t0:.3f}|{T:.3f}".encode()
    rng = np.random.default_rng(zlib.crc32(key))
    return rng.uniform(-half, half, 2)


def windows(nframes: int, fps: float, Ts, step_s: float):
    step = max(1, int(round(step_s * fps)))
    for T in Ts:
        nw = int(round(T * fps))
        for s in range(0, nframes - nw + 1, step):
            yield T, s, nw


def errors_row(err: np.ndarray, Vt: np.ndarray, heading: float, wave_dir: float = WAVE_DIR_DEG,
               swell_dir: float = float("nan")) -> dict:
    """err_along_wave is along the wind sea's travel direction; err_along_swell along the swell's
    (NaN when the sea has no swell)."""
    speed = float(np.hypot(*Vt))
    al = unit(heading)
    cr = unit(heading + 90.0)  # right of track
    wv = unit(wave_dir)
    return {
        "err_e": float(err[0]), "err_n": float(err[1]),
        "err_along": float(err @ al), "err_cross": float(err @ cr), "err_along_wave": float(err @ wv),
        "err_along_swell": float(err @ unit(swell_dir)) if np.isfinite(swell_dir) else float("nan"),
        "err_abs": float(np.hypot(*err)),
        "err_abs_pct": 100.0 * float(np.hypot(*err)) / speed if speed > 0 else float("nan"),
        "err_along_pct": 100.0 * float(err @ al) / speed if speed > 0 else float("nan"),
    }


def process_run(run_dir, *, Ts=(1.6, 3.2, 6.4, 12.0), step_s=1.6, models=("dispersive", "static"),
                n=256, frac=0.6, prior_half=5.0, box=8.0, seed=1, fit_scale_flag=False, prep_kw=None, mode="shaded",
                log=lambda m: print(m, flush=True)):
    run_dir = Path(run_dir)
    prep_kw = prep_kw or {}
    meta = run_meta(run_dir)
    t0 = time.time()
    stack, rows, geom = rc.rectify_run(run_dir, n=n, frac=frac, mode=mode)
    t_rect = time.time() - t0
    tt = np.array([r["t"] for r in rows])
    cam = np.array([[r["cam_e"], r["cam_n"]] for r in rows])
    fps = meta["fps"]
    dt = float(np.median(np.diff(tt)))
    out = []
    for T, s0, nw in windows(len(rows), fps, Ts, step_s):
        s1 = s0 + nw - 1
        Vt = (cam[s1] - cam[s0]) / (tt[s1] - tt[s0])
        speed = float(np.hypot(*Vt))
        off = prior_offset(meta["run_id"], tt[s0], T, prior_half, seed)
        prior = Vt + off
        sp = prepare(stack[s0:s0 + nw], dt, geom.cell, **prep_kw)
        lam_res = {}
        for model in models:
            V, J, info = fit_velocity(sp, prior, box=box, model=model)
            sc = float("nan")
            if fit_scale_flag and model == "dispersive":
                sc, V, J, info = fit_scale(sp, V)
            frac_in, occ, lam_c = shell_stats(sp, V, 1.0 if not np.isfinite(sc) else sc, model)
            frac_true, _, _ = shell_stats(sp, Vt, 1.0, model)
            Jt = float(objective(sp, Vt[None], 1.0, model)[0])
            err = V - Vt
            row = dict(meta)
            row.update({
                "model": model, "T": T, "t_start": float(tt[s0]), "frame_start": int(rows[s0]["frame"]),
                "n_frames": nw, "tile_side_m": geom.side, "tile_cell_m": geom.cell, "cam_u": float(rows[s0]["cam_u"]),
                "v_true_e": float(Vt[0]), "v_true_n": float(Vt[1]), "speed": speed,
                "prior_e": float(prior[0]), "prior_n": float(prior[1]),
                "v_est_e": float(V[0]), "v_est_n": float(V[1]),
            })
            row.update(errors_row(err, Vt, meta["heading"], meta["wind_dir"], meta["swell_dir"]))
            row.update({
                "scale": sc, "shell_frac": frac_in, "shell_frac_at_truth": frac_true, "band_occupancy": occ,
                "J_est": J, "J_truth": Jt, "J_second": info["J_second"], "J_median": info["J_median"],
                "quad_ok": info["quad_ok"], "at_edge": info["at_edge"], "scale_at_bound": info.get("scale_at_bound", ""),
                "lambda_c": lam_c,
                # Derived resolution limits (not error bars):
                "res_bin_lambda_c": lam_c / T,                       # λ_c / T
                "res_dwell_lambda_c": lam_c * speed / geom.side,     # λ_c · |V| / L
                "res_bin_2m": 2.0 / T, "dwell_s": geom.side / speed if speed > 0 else float("inf"),
                "band_rad_s": sp.band, "dw_rad_s": sp.dw, "nk": len(sp.kmag),
                "k_min": float(sp.kmag.min()), "k_max": float(sp.kmag.max()),
                "image_mode": mode, "weight": prep_kw.get("weight", "whiten"), "lambda_max": prep_kw.get("lambda_max") or "",
            })
            out.append(row)
    log(f"{meta['run_id']}: rectify {t_rect:.1f}s, fits {time.time() - t0 - t_rect:.1f}s, {len(out)} rows")
    return out


def expand_dirs(paths):
    out = []
    for p in map(Path, paths):
        if (p / "frames.csv").exists():
            out.append(p)
        else:
            out.extend(sorted(d for d in p.iterdir() if (d / "frames.csv").exists()))
    return out


def _worker(args):
    d, kw = args
    try:
        return process_run(d, **kw)
    except Exception as e:  # keep the batch going; record the failure
        print(f"FAILED {d}: {e!r}", flush=True)
        return []


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--T", type=float, nargs="+", default=[1.6, 3.2, 6.4, 12.0])
    ap.add_argument("--step", type=float, default=1.6, help="window start spacing, s")
    ap.add_argument("--models", nargs="+", default=["dispersive", "static"])
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--frac", type=float, default=0.6)
    ap.add_argument("--prior-half", type=float, default=5.0)
    ap.add_argument("--box", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fit-scale", action="store_true")
    ap.add_argument("--weight", default="whiten", choices=["energy", "sqrt", "whiten"])
    ap.add_argument("--band", type=float, default=1.0, help="band half-width in unpadded bins")
    ap.add_argument("--kmin-bins", type=float, default=3.0)
    ap.add_argument("--kmax-frac", type=float, default=0.8)
    ap.add_argument("--lambda-max", type=float, default=None,
                    help="use only wavelengths shorter than this (m); default: L / kmin-bins")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--mode", default="shaded", help="shaded (frames/), normals (slope toward 60° from "
                    "frames/normals/) or another label mode under frames/<mode>/ (its luminance)")
    a = ap.parse_args(argv)
    dirs = expand_dirs(a.runs)
    kw = dict(Ts=tuple(a.T), step_s=a.step, models=tuple(a.models), n=a.n, frac=a.frac,
              prior_half=a.prior_half, box=a.box, seed=a.seed, fit_scale_flag=a.fit_scale, mode=a.mode,
              prep_kw=dict(weight=a.weight, band_bins=a.band, kmin_bins=a.kmin_bins, kmax_frac=a.kmax_frac,
                           lambda_max=a.lambda_max))
    rows = []
    if a.jobs > 1:
        import multiprocessing as mp
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        os.environ.setdefault("SHELL_FFT_WORKERS", "4")
        with mp.get_context("spawn").Pool(a.jobs) as pool:
            for r in pool.imap_unordered(_worker, [(d, kw) for d in dirs]):
                rows.extend(r)
    else:
        for d in dirs:
            rows.extend(_worker((d, kw)))
    if not rows:
        sys.exit("no rows")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
