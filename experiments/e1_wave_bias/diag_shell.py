"""Diagnostics for one run's shell fit: GPU vs numpy objective agreement, and the fit
restricted to wavelength bands (which wavelengths pull the estimate where).

    python diag_shell.py RUN_DIR [--T 6.4] [--start 0]
"""
import argparse, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rectify as rc, shell_fit as sf

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--T", type=float, default=6.4)
ap.add_argument("--start", type=int, default=0)
a = ap.parse_args()
meta = sf.run_meta(Path(a.run))
stack, rows, geom = rc.rectify_run(a.run)
nw = int(round(a.T * meta["fps"]))
cam = np.array([[r["cam_e"], r["cam_n"]] for r in rows]); tt = np.array([r["t"] for r in rows])
s0, s1 = a.start, a.start + nw - 1
Vt = (cam[s1] - cam[s0]) / (tt[s1] - tt[s0])
print(meta["run_id"], "T", a.T, "V_true", Vt.round(3), "tile", round(geom.side, 2), "cell", round(geom.cell, 4))
sp = sf.prepare(stack[s0:s0 + nw], 0.1, geom.cell)
Vs = Vt[None] + np.random.default_rng(0).uniform(-3, 3, (200, 2))
t = time.time(); Jg = sf.objective(sp, Vs); tg = time.time() - t
sf._TORCH["device"] = None
t = time.time(); Jc = sf.objective(sp, Vs); tc = time.time() - t
print(f"objective gpu vs numpy: max rel diff {np.max(np.abs(Jg - Jc) / np.abs(Jc)):.2e}  (gpu {tg:.3f}s, numpy {tc:.3f}s for 200 V)")
sf._TORCH["checked"] = False
al, cr, wv = sf.unit(meta["heading"]), sf.unit(meta["heading"] + 90), sf.unit(60)
lam_edges = [geom.side / 3, 16, 8, 4, 2, 1, 0.5, 4 * geom.cell / 0.8 / 2]
lam_edges = [l for l in lam_edges if l <= geom.side / 3 + 1e-9]
for model in ("dispersive", "static"):
    for lo, hi in zip(lam_edges[1:], lam_edges[:-1]):
        kmin, kmax = 2 * np.pi / hi, 2 * np.pi / lo
        sel = (sp.kmag >= kmin) & (sp.kmag < kmax)
        if sel.sum() < 20:
            continue
        sub = sf.Spectrum(k=sp.k[sel], kmag=sp.kmag[sel], S=sp.S[sel], Sband=sp.Sband[sel], w=sp.w[sel] / sp.w[sel].sum(),
                          half=sp.half[sel], dw=sp.dw, dwp=sp.dwp, ntp=sp.ntp, hb=sp.hb, band=sp.band, T=sp.T, L=sp.L, dk=sp.dk)
        V, J, info = sf.fit_velocity(sub, Vt + np.array([1.3, -0.7]), model=model)
        e = V - Vt
        fr, occ, _ = sf.shell_stats(sub, V, 1.0, model)
        frt, _, _ = sf.shell_stats(sub, Vt, 1.0, model)
        print(f"{model:10s} λ {lo:5.2f}-{hi:5.2f} m nk {sel.sum():5d}: err along {e @ al:+.3f} cross {e @ cr:+.3f} along-wave {e @ wv:+.3f}  frac {fr:.3f} (truth {frt:.3f}, occ {occ:.3f}) J2/J {info['J_second']/J:.3f}")
