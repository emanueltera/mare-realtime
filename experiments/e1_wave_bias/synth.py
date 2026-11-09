"""Synthetic linear sea for validating shell_fit.py: a directional JONSWAP-like random-phase
wave field on a periodic world grid, rendered as a slope image (optionally with a small
quadratic term and noise) and sampled into a camera-fixed tile that moves at a known V.

Layout matches rectify.py: [row = north, col = east], both increasing.
A component at grid wavevector k travels along +k: η = Re Σ A_k e^{i(k·x − ω t)},
ω = √(g|k|) (or 0 for a frozen sea).
"""

from __future__ import annotations

import math
import os

import numpy as np
import scipy.fft
from scipy import ndimage

G = 9.81


class WaveField:
    def __init__(self, *, nw: int = 2048, dx: float = 0.1, peak_wavelength: float = 40.0,
                 direction: float = 60.0, spread: float = 8.0, kcut_wavelength: float = 0.4,
                 slope_power: float = -3.5, gamma: float = 3.3, seed: int = 0, frozen: bool = False):
        self.nw, self.dx = nw, dx
        k1 = 2 * np.pi * np.fft.fftfreq(nw, dx)
        KE, KN = np.meshgrid(k1, k1)  # [row=north, col=east]
        k = np.hypot(KE, KN)
        kp = 2 * np.pi / peak_wavelength
        with np.errstate(divide="ignore", invalid="ignore"):
            # Omnidirectional ∝ k^p · exp(−1.25 (kp/k)²) with a JONSWAP-style peak enhancement.
            sig = np.where(k <= kp, 0.07, 0.09)
            r = np.exp(-((np.sqrt(k) - np.sqrt(kp)) ** 2) / (2 * sig ** 2 * kp))
            Fk = k ** slope_power * np.exp(-1.25 * (kp / k) ** 2) * gamma ** r
            Fk *= np.exp(-(k / (2 * np.pi / kcut_wavelength)) ** 4)
            bearing = np.arctan2(KE, KN)  # compass bearing of k
            D = np.cos((bearing - math.radians(direction)) / 2) ** (2 * spread)
            F2 = Fk / k * D
        F2[k == 0] = 0.0
        F2 = np.nan_to_num(F2)
        rng = np.random.default_rng(seed)
        A = np.sqrt(F2 / 2) * (rng.standard_normal(k.shape) + 1j * rng.standard_normal(k.shape))
        self.A = A
        self.KE, self.KN, self.k = KE, KN, k
        self.omega = np.zeros_like(k) if frozen else np.sqrt(G * k)

    def slope(self, t: float, sun_bearing: float = 180.0) -> np.ndarray:
        """Surface slope along the given compass bearing at time t, on the world grid."""
        b = math.radians(sun_bearing)
        ks = self.KE * math.sin(b) + self.KN * math.cos(b)
        H = self.A * np.exp(-1j * self.omega * t) * (1j * ks)
        return (scipy.fft.ifft2(H, workers=int(os.environ.get("SHELL_FFT_WORKERS", "-1"))).real * self.A.size).astype(np.float32)

    def sample(self, field: np.ndarray, E: np.ndarray, N: np.ndarray, order: int = 3) -> np.ndarray:
        """Sample a world-grid field at ENU positions (periodic)."""
        return ndimage.map_coordinates(field, [N / self.dx, E / self.dx], order=order, mode="grid-wrap").astype(np.float32)


def moving_tile_stack(wf: WaveField, V, *, n: int = 256, side: float = 42.0, nt: int = 32, dt: float = 0.1,
                      p0=(13.0, 7.0), sun_bearing: float = 180.0, quad: float = 0.0, noise: float = 0.0,
                      seed: int = 0) -> np.ndarray:
    """Tile stack [nt, n, n] of a slope image seen by a tile whose centre moves at V (E, N)
    m/s from p0. Image = 1 + s/σ + quad·(s/σ)² + noise·N(0, 1), σ = slope rms."""
    c = (np.arange(n) + 0.5) * side / n - side / 2
    dE, dN = np.meshgrid(c, c)
    rng = np.random.default_rng(seed + 991)
    out = []
    sig = None
    for i in range(nt):
        t = i * dt
        f = wf.slope(t, sun_bearing)
        if sig is None:
            sig = float(f.std())
        s = wf.sample(f, p0[0] + V[0] * t + dE, p0[1] + V[1] * t + dN) / sig
        img = 1.0 + 0.2 * (s + quad * s * s)
        if noise:
            img = img + noise * rng.standard_normal(img.shape).astype(np.float32)
        out.append(img)
    return np.stack(out)
