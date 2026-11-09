"""A small synthetic tile set on disk, in e2.rectify's layout: frozen periodic texture
translated exactly (Fourier shift) by −v·dt per frame in tile axes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from e2 import geom


def fourier_shift(a: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """b(x) = a(x − d), periodic."""
    ky = np.fft.fftfreq(a.shape[0])[:, None]
    kx = np.fft.fftfreq(a.shape[1])[None, :]
    return np.fft.ifft2(np.fft.fft2(a) * np.exp(-2j * np.pi * (kx * dx + ky * dy))).real


def make_set(root: Path, name="fake", n=64, frames=21, fps=10.0, gsd=0.1,
             runs=(("r0", 0.0, (12.0, 0.0), "transit"), ("r1", 33.0, (-5.0, 20.0), "loiter"), ("r2", 250.0, (15.0, -5.0), "transit"))):
    root = Path(root)
    (root / "tiles").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    base = np.fft.ifft2(np.fft.fft2(rng.normal(size=(n, n))) * np.exp(-((np.fft.fftfreq(n)[:, None] ** 2 + np.fft.fftfreq(n)[None, :] ** 2) / 0.02))).real
    rrows, frows, lrows = [], [], []
    for k, (rid, theta, v_en, pattern) in enumerate(runs):
        v_en = np.array(v_en)
        v_xy = geom.enu_to_tile(v_en, theta)
        dt = 1 / fps
        t = np.arange(frames) * dt
        p = np.array([100.0 * k, -50.0]) + t[:, None] * v_en
        xy = geom.enu_to_tile(p, theta)
        tiles = np.stack([fourier_shift(base, *(-v_xy * ti / gsd)) for ti in t]).astype(np.float16)
        tiles[:, :2, :] = np.nan  # a masked strip
        np.save(root / "tiles" / f"{rid}.npy", tiles)
        speed = float(np.hypot(*v_en))
        rrows.append({"set": name, "run_id": rid, "n": n, "gsd": gsd, "side": n * gsd, "theta_deg": theta, "c_run_e": 0.0,
                      "c_run_n": 0.0, "n_frames": frames, "duration": t[-1], "fps": fps, "distance": speed * t[-1],
                      "ground_speed_mean": speed, "cam_u_mean": 50.0, "valid_frac_mean": 1 - 2 / n, "valid_frac_min": 1 - 2 / n,
                      "masked_frac_mean": 2 / n, "altitude": 50 + 60 * k, "airspeed": speed, "gimbal_pitch": -90 + 10 * k,
                      "stabilized": 1, "turbulence": 0, "pattern": pattern, "wind_speed": 5 + 6 * k, "wind_direction": 60.0,
                      "swell": k % 2, "swell_height": 0.5 * (k % 2), "swell_period": 10, "sea_hs": 1.0, "sea_tp": 6.0,
                      "sun_elevation": 10 + 20 * k, "overcast": 0, "haze": 0, "sea_rate": 1})
        for i in range(frames):
            frows.append({"run_id": rid, "frame": i, "t": t[i], "gsd": gsd, "valid_frac": 1 - 2 / n})
            lrows.append({"run_id": rid, "frame": i, "t": t[i], "cam_e": p[i, 0], "cam_n": p[i, 1], "cam_u": 50.0,
                          "cam_x": xy[i, 0], "cam_y": xy[i, 1], "aircraft_heading": 0, "aircraft_bank": 0})
    pd.DataFrame(rrows).to_csv(root / "runs.csv", index=False)
    pd.DataFrame(frows).to_csv(root / "frames.csv", index=False)
    pd.DataFrame(lrows).to_csv(root / "labels.csv", index=False)
    (root / "set.json").write_text(json.dumps({"set": name, "n": n, "runs": len(runs), "frames": frames * len(runs)}))
    return root
