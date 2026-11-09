"""Model C's spectrum on a synthetic linear sea in tile space: the classical (uniform-weight)
shell cost peaks at the true velocity, a frozen pattern does not (dispersive shell), the
dihedral augmentation commutes with the estimate, and the network runs end to end."""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from e2.data import augment_array, augment_vector  # noqa: E402
from spectral import spectrum as spx  # noqa: E402
from spectral.model import SpectralNet, grid, losses  # noqa: E402

G = 9.81


def sea(V, *, n=128, K=32, gsd=0.2, dt=0.1, rate=1.0, seed=0, big=4):
    """Tile stack [K, n, n] of a linear sea seen by a camera moving at V (tile axes, m/s):
    content I_t(x) = W(x + V t). W is a random field on a periodic domain big·n cells wide
    (every lattice k carries a wave of frequency rate·√(g|k|), travelling mostly toward +x),
    shifted exactly in Fourier space; the tile is its first n × n cells."""
    rng = np.random.default_rng(seed)
    m = big * n
    fr = np.fft.fftfreq(m, gsd) * 2 * np.pi
    ky, kx = np.meshgrid(fr, fr, indexing="ij")
    km = np.hypot(kx, ky)
    ang = np.arctan2(ky, kx)
    spread = 0.25 + np.cos(0.5 * (ang - 0.3)) ** 8
    A = np.where(km > 0, km.clip(1e-9) ** -1.5 * spread, 0.0) * np.exp(2j * np.pi * rng.uniform(size=km.shape))
    A *= rng.rayleigh(size=km.shape)
    om = rate * np.sqrt(G * km)
    out = np.zeros((K, n, n))
    for i in range(K):
        t = i * dt
        W = np.fft.ifft2(A * np.exp(1j * (kx * V[0] * t + ky * V[1] * t - om * t))).real
        out[i] = W[:n, :n]
    return 1.0 + 0.3 * out / out.std()


def classical(tiles, V0, gsd=0.2, dt=0.1, half=4.0, step=0.1):
    t = torch.as_tensor(tiles, dtype=torch.float32)[None]
    sp = spx.prepare(t, torch.tensor([gsd]), torch.tensor([dt]), bin=1, pad=2)
    w = torch.ones(1, sp.k.shape[1]) / sp.k.shape[1]
    o = grid(half, step)
    c = torch.as_tensor(V0, dtype=torch.float32)[None]
    C = spx.shell_cost(sp, w, c[:, None] + o[None])
    return (c + o[C.argmax(-1)])[0].numpy()


@pytest.mark.parametrize("V", [(12.0, -3.0), (-30.0, 20.0)])
def test_shell_peaks_at_truth(V):
    tiles = sea(np.array(V))
    est = classical(tiles, np.array(V) + np.array([1.5, -1.0]))
    assert np.hypot(*(est - np.array(V))) < 0.15, est


def test_frozen_sea_is_off_the_shell():
    """A frozen pattern (rate 0) moves at exactly −V; the dispersive shell is then wrong and
    the frozen line (Ω = 0) is what a pattern tracker sees. Checks the sign of Ω."""
    V = np.array([10.0, 4.0])
    frozen = sea(V, rate=0.0)
    est = classical(frozen, V + np.array([0.5, 0.5]))
    assert np.hypot(*(est - V)) > 0.2


def test_augmentation_commutes():
    V = np.array([8.0, -14.0])
    tiles = sea(V, seed=3)
    for g in (1, 5, 6):
        est = classical(augment_array(tiles, g), augment_vector(V, g) + np.array([1.0, 1.0]))
        assert np.hypot(*(est - augment_vector(V, g))) < 0.15, (g, est)


def test_network_runs_and_backprops():
    V = np.array([5.0, 2.0])
    tiles = torch.as_tensor(np.stack([sea(V, n=64, K=16), sea(V, n=64, K=16, seed=2)]),
                            dtype=torch.float32)
    tiles[:, :, :2] = float("nan")
    net = SpectralNet(ntp=32, stages=((2.0, 0.25, None), (0.4, 0.1, None)), hid=16, prep=dict(bin=1, pad=2))
    tgt = torch.as_tensor(np.stack([V, V]), dtype=torch.float32)
    out = net(tiles, torch.tensor([0.2, 0.2]), torch.tensor([0.1, 0.1]), tgt + 0.5, target=tgt, teacher=0.5)
    loss, parts = losses(out, tgt, net.stages)
    loss.backward()
    assert math.isfinite(float(loss))
    assert net.knets[0].a[0].weight.grad is not None and net.knets[0].a[0].weight.grad.abs().sum() > 0


def test_shell_cost_gradient():
    """The hand-written backward of the fused cost equals autograd through the plain expression."""
    g = torch.Generator().manual_seed(0)
    B, nk, ntp, M = 2, 50, 16, 7
    Sb = torch.rand(B, nk, ntp, generator=g)
    k = torch.randn(B, nk, 2, generator=g) * 5
    om = torch.sqrt(G * k.norm(dim=-1))
    dwp = torch.tensor([0.9, 1.1])
    V = torch.randn(B, M, 2, generator=g)
    w = torch.rand(B, nk, generator=g, requires_grad=True)
    up = torch.randn(B, M, generator=g)
    (spx._ShellCost.apply(w, Sb, k, om, dwp, V) * up).sum().backward()
    g1 = w.grad.clone()
    w.grad = None
    (spx._cost_fwd(Sb, k, om, dwp, V, w) * up).sum().backward()
    assert torch.allclose(g1, w.grad, atol=1e-5)


def test_gpu_augmentation_matches_e2():
    from spectral.data import augment_tiles

    x = np.random.default_rng(0).normal(size=(8, 3, 5, 5)).astype(np.float32)
    got = augment_tiles(torch.as_tensor(x), torch.arange(8)).numpy()
    for g in range(8):
        assert np.array_equal(got[g], augment_array(x[g], g))
