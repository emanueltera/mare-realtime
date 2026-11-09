"""What a real camera does to a frame after the scene, for E4's training and its harder test:
defocus, motion blur, vignette, lens dirt, tone curve, sensor noise and compression.

The optics and the tone curve are one draw per run (or per training sample), the same on both
frames of a pair; the noise and compression differ frame to frame. No effect moves a pixel's
geometry, so the labels stand. All numbers are choices for the augmentation, not measurements
of any camera.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

cv2.setNumThreads(0)  # the data loader's workers are the parallelism

H, W = 512, 288  # E3's shrunk frame, the default


@dataclass
class Fx:
    defocus: float      # Gaussian σ, px on the shrunk frame
    smear: float        # motion blur length, px
    smear_dir: float    # radians
    vignette: float     # edge darkening at the corner, 0–1
    dirt: np.ndarray    # (H, W) multiplicative, 1 where clean
    gain: float
    gamma: float
    flicker: float      # σ of the frame-to-frame gain
    read: float         # read noise σ, in [0, 1] units
    shot: float         # shot noise: variance per unit of signal
    jpeg: int           # quality, 0 for none


def draw_fx(rng: np.random.Generator, shape: tuple[int, int] = (H, W)) -> Fx:
    """One draw of the effects for frames of `shape` (height, width)."""
    h, w = shape
    dirt = np.ones((h, w), np.float32)
    if rng.random() < 0.3:
        low = rng.normal(0, 1, (7, 4) if h >= w else (4, 7)).astype(np.float32)
        dirt = 1 - rng.uniform(0.01, 0.08) * np.clip(np.abs(cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)), 0, 2)
    return Fx(
        defocus=rng.uniform(0.3, 1.2) if rng.random() < 0.5 else 0.0,
        smear=rng.uniform(1, 4) if rng.random() < 0.3 else 0.0,
        smear_dir=rng.uniform(0, np.pi),
        vignette=rng.uniform(0, 0.5),
        dirt=dirt,
        gain=np.exp(rng.uniform(np.log(0.6), np.log(1.5))),
        gamma=rng.uniform(0.7, 1.4),
        flicker=rng.uniform(0, 0.03),
        read=rng.uniform(0.5, 5) / 255,
        shot=rng.uniform(0, 0.002),
        jpeg=int(rng.integers(30, 96)) if rng.random() < 0.5 else 0,
    )


def _optics(fx: Fx) -> np.ndarray:
    h, w = fx.dirt.shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((x - w / 2 + 0.5) / (w / 2)) ** 2 + ((y - h / 2 + 0.5) / (h / 2)) ** 2
    return (1 - fx.vignette * r2 / 2) * fx.dirt


def _smear_kernel(length: float, angle: float) -> np.ndarray:
    n = int(np.ceil(length)) | 1
    k = np.zeros((n, n), np.float32)
    c = n // 2
    for t in np.linspace(-length / 2, length / 2, 2 * n + 1):
        k[int(round(c + t * np.sin(angle))), int(round(c + t * np.cos(angle)))] += 1
    return k / k.sum()


def apply_fx(img: np.ndarray, fx: Fx, rng: np.random.Generator) -> np.ndarray:
    """img (2, H, W) float32 in [0, 1] → the same, through the camera. Only rng's draws differ by frame."""
    optics = _optics(fx)
    kernel = _smear_kernel(fx.smear, fx.smear_dir) if fx.smear > 0 else None
    out = np.empty_like(img)
    for f in range(img.shape[0]):
        a = img[f]
        if fx.defocus > 0:
            a = cv2.GaussianBlur(a, (0, 0), float(fx.defocus))
        if kernel is not None:
            a = cv2.filter2D(a, -1, kernel)
        a = np.clip(a * optics * fx.gain * (1 + rng.normal(0, fx.flicker)), 0, 1) ** fx.gamma
        var = fx.read**2 + fx.shot * a
        a = np.clip(a + np.sqrt(var) * rng.standard_normal(a.shape, dtype=np.float32), 0, 1)
        if fx.jpeg:
            ok, buf = cv2.imencode(".jpg", (a * 255 + 0.5).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, fx.jpeg])
            assert ok
            a = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        out[f] = a
    return out
