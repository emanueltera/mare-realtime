"""Camera-fixed tile stack → whitened 3-D power spectrum, and gathers along the dispersion shell.

Conventions (as E1's shell_fit.py, in tile axes): arrays are [row = y, col = x]; a content
component e^{i(k·x − ω' t)} lies at (k, ω'). Water content moves through the tile at −V, so a
wave of intrinsic frequency Ω(k) = √(g|k|) travelling along +k appears at

    ω'(k) = Ω(k) − k·V        (wrapped modulo 2π/dt: the sea is sampled at 10 Hz)

The full k plane is kept with the + branch only: for a real image S(−k, −ω') = S(k, ω'),
so the − branch at k is the + branch at −k (E1's half plane with both branches, same thing).

Preparation (E1's, on the GPU): 2 × 2 binning (mask-aware), each frame divided by its valid
mean (auto-exposure), each cell's temporal mean removed (fixed-in-image glint and vignetting
lie at ω' = 0), Hann in t, x and y, a spatial FFT and a temporal FFT zero-padded × pad.
Whitening: each k's power is divided by its total over ω', so every k contributes the
fraction of its energy that a candidate shell picks up. What a candidate V is worth is
J_k(V) = Sband_k(Ω_k − k·V), the whitened power summed over a ±band and read by linear
interpolation along ω' (circular).

Registration by a pattern velocity c is only a re-indexing of the ω' axis per k
(ω'' = ω' + k·c), so it is done inside the gathers: a stage centred on c reads the profile
Sn_k(Ω_k − k·c + u) and candidates c + o. No image is shifted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

G = 9.81


@dataclass
class Spectrum:
    Sn: torch.Tensor      # [B, nk, ntp] power / its sum over ω' (whitened); column j ↔ ω' = j·dwp
    Sband: torch.Tensor   # [B, nk, ntp] Sn summed over ±hb columns (circular)
    k: torch.Tensor       # [B, nk, 2] (kx, ky) rad/m in tile axes
    kmag: torch.Tensor    # [B, nk]
    omega: torch.Tensor   # [B, nk] √(g|k|), rad/s
    f: torch.Tensor       # [nk, 2] (fx, fy) cycles per (binned) cell, the same lattice for every window
    logE: torch.Tensor    # [B, nk] log of each k's total power relative to the window's median
    dwp: torch.Tensor     # [B] padded ω' bin, rad/s
    ntp: int
    hb: int
    T: torch.Tensor       # [B] window length K·dt, s
    L: torch.Tensor       # [B] tile side, m
    cell: torch.Tensor    # [B] cell size after binning, m


def kgrid(n: int, kmin_cells: float = 3.0, kmax_frac: float = 0.8, device=None):
    """Selected lattice points of an n × n FFT: kmin_cells/n ≤ |f| ≤ kmax_frac · ½ (cycles/cell).
    Returns (flat indices into [n·n] in (row, col) order, f [nk, 2] as (fx, fy))."""
    fr = torch.fft.fftfreq(n, device=device)
    fy, fx = torch.meshgrid(fr, fr, indexing="ij")
    fm = torch.hypot(fx, fy)
    sel = (fm >= kmin_cells / n) & (fm <= 0.5 * kmax_frac)
    idx = torch.nonzero(sel.reshape(-1), as_tuple=False)[:, 0]
    return idx, torch.stack([fx.reshape(-1)[idx], fy.reshape(-1)[idx]], -1)


def hann(n: int, device=None) -> torch.Tensor:
    """Strictly positive Hann of length n (as np.hanning(n + 2)[1:-1])."""
    i = torch.arange(1, n + 1, device=device, dtype=torch.float32)
    return 0.5 - 0.5 * torch.cos(2 * math.pi * i / (n + 1))


def prepare(tiles: torch.Tensor, gsd: torch.Tensor, dt: torch.Tensor, *, bin: int = 2, pad: int = 2,
            band_bins: float = 1.0, kmin_cells: float = 3.0, kmax_frac: float = 0.8) -> Spectrum:
    """tiles [B, K, N, N] (NaN = no image) on the GPU → Spectrum. gsd [B] m/px, dt [B] s."""
    x = tiles.float()
    B, K, N, _ = x.shape
    m = torch.isfinite(x)
    x = torch.where(m, x, torch.zeros((), device=x.device))
    if bin > 1:
        s = F.avg_pool2d(x.reshape(B * K, 1, N, N), bin) * bin * bin
        c = F.avg_pool2d(m.float().reshape(B * K, 1, N, N), bin) * bin * bin
        n = N // bin
        x = (s / c.clamp_min(1)).reshape(B, K, n, n)
        m = (c > 0).reshape(B, K, n, n)
    else:
        n = N
    mf = m.float()
    # Auto-exposure: each frame over its valid mean.
    mean = (x * mf).sum((2, 3)) / mf.sum((2, 3)).clamp_min(1)
    x = (x / mean.clamp_min(1e-12)[..., None, None] - 1.0) * mf
    # Fixed-in-image structure: each cell's temporal mean over its valid frames.
    x = (x - (x.sum(1, keepdim=True) / mf.sum(1, keepdim=True).clamp_min(1))) * mf
    wt, ws = hann(K, x.device), hann(n, x.device)
    x = x * wt[None, :, None, None] * ws[None, None, :, None] * ws[None, None, None, :]
    idx, f = kgrid(n, kmin_cells, kmax_frac, x.device)
    Fs = torch.fft.fft2(x).reshape(B, K, n * n)[:, :, idx]            # [B, K, nk]
    ntp = K * pad
    P = torch.fft.fft(Fs, n=ntp, dim=1).abs().square()                  # [B, ntp, nk], numpy-sign ω
    # Physical ω' = −ω_fft: column j of S is ω' = j·dwp.
    S = P[:, (-torch.arange(ntp, device=x.device)) % ntp, :].transpose(1, 2).contiguous()  # [B, nk, ntp]
    E = S.sum(-1)
    Sn = S / E.clamp_min(1e-30)[..., None]
    hb = max(0, int(round(band_bins * pad)))
    Sband = Sn.clone()
    for j in range(1, hb + 1):
        Sband += torch.roll(Sn, j, -1) + torch.roll(Sn, -j, -1)
    cell = gsd.float() * bin
    k = 2 * math.pi * f[None] / cell[:, None, None]                      # [B, nk, 2]
    kmag = k.norm(dim=-1)
    T = K * dt.float()
    logE = torch.log(E.clamp_min(1e-30))
    logE = logE - logE.median(dim=1, keepdim=True).values
    return Spectrum(Sn=Sn, Sband=Sband, k=k, kmag=kmag, omega=torch.sqrt(G * kmag), f=f, logE=logE,
                    dwp=2 * math.pi / (T * pad), ntp=ntp, hb=hb, T=T, L=n * cell, cell=cell)


def gather_omega(A: torch.Tensor, om: torch.Tensor, dwp: torch.Tensor) -> torch.Tensor:
    """Read A [B, nk, ntp] (circular in ω') at frequencies om [B, nk, M] (rad/s) by linear
    interpolation. Returns [B, nk, M]."""
    ntp = A.shape[-1]
    fpos = torch.remainder(om / dwp[:, None, None], ntp)
    i0 = torch.floor(fpos)
    fr = fpos - i0
    i0 = i0.long() % ntp
    i1 = (i0 + 1) % ntp
    return torch.gather(A, 2, i0) * (1 - fr) + torch.gather(A, 2, i1) * fr


def profile(sp: Spectrum, centre: torch.Tensor, kidx: torch.Tensor | None = None) -> torch.Tensor:
    """Each k's whitened ω' profile rolled so that column ntp/2 is the shell of the
    velocity `centre` [B, 2]: Sn_k(Ω_k − k·c + u_j), u_j = (j − ntp/2)·dwp. [B, nk, ntp]."""
    k, om, Sn = _sub(sp, kidx)
    u = (torch.arange(sp.ntp, device=Sn.device) - sp.ntp // 2).float()[None, None, :] * sp.dwp[:, None, None]
    base = om - (k * centre[:, None, :]).sum(-1)
    return gather_omega(Sn, base[..., None] + u, sp.dwp)


def shell_values(sp: Spectrum, V: torch.Tensor, kidx: torch.Tensor | None = None) -> torch.Tensor:
    """J_k(V) = Sband_k(Ω_k − k·V) for candidate velocities V [B, M, 2]. Returns [B, nk, M]."""
    k, om, Sb = _sub(sp, kidx, band=True)
    dop = torch.einsum("bkc,bmc->bkm", k, V)
    return gather_omega(Sb, om[..., None] - dop, sp.dwp)


def _sub(sp: Spectrum, kidx, band: bool = False):
    A = sp.Sband if band else sp.Sn
    if kidx is None:
        return sp.k, sp.omega, A
    return sp.k[:, kidx], sp.omega[:, kidx], A[:, kidx]


def _cost_fwd(Sband, k, om, dwp, V, w):
    """C[b, m] = Σ_k w[b, k] · J_k(V_bm). Written as one expression so that torch.compile fuses
    the gathers and the sum over k into one kernel (J, [B, nk, M], is never stored)."""
    ntp = Sband.shape[-1]
    dop = k[:, :, None, 0] * V[:, None, :, 0] + k[:, :, None, 1] * V[:, None, :, 1]
    f = torch.remainder((om[..., None] - dop) / dwp[:, None, None], ntp)
    i0 = torch.floor(f)
    fr = f - i0
    i0 = i0.long() % ntp
    i1 = (i0 + 1) % ntp
    J = torch.gather(Sband, 2, i0) * (1 - fr) + torch.gather(Sband, 2, i1) * fr
    return (J * w[..., None]).sum(1)


def _cost_bwd(Sband, k, om, dwp, V, g):
    """∂/∂w of Σ_m g[b, m] C[b, m]: Σ_m g[b, m] · J_k(V_bm), [B, nk] (J recomputed, fused)."""
    ntp = Sband.shape[-1]
    dop = k[:, :, None, 0] * V[:, None, :, 0] + k[:, :, None, 1] * V[:, None, :, 1]
    f = torch.remainder((om[..., None] - dop) / dwp[:, None, None], ntp)
    i0 = torch.floor(f)
    fr = f - i0
    i0 = i0.long() % ntp
    i1 = (i0 + 1) % ntp
    J = torch.gather(Sband, 2, i0) * (1 - fr) + torch.gather(Sband, 2, i1) * fr
    return (J * g[:, None, :]).sum(2)


torch._dynamo.config.cache_size_limit = 64
_cost_fwd_c = torch.compile(_cost_fwd, dynamic=False)
_cost_bwd_c = torch.compile(_cost_bwd, dynamic=False)


class _ShellCost(torch.autograd.Function):
    """C[b, m] = Σ_k w[b, k] · J_k(V_bm) with a fused (compiled) forward and backward on the
    GPU; plain torch on the CPU (tests)."""

    @staticmethod
    def forward(ctx, w, Sband, k, om, dwp, V):
        ctx.save_for_backward(Sband, k, om, dwp, V)
        fn = _cost_fwd_c if w.is_cuda else _cost_fwd
        return fn(Sband.contiguous(), k.contiguous(), om.contiguous(), dwp, V.contiguous(), w.float().contiguous())

    @staticmethod
    def backward(ctx, g):
        Sband, k, om, dwp, V = ctx.saved_tensors
        fn = _cost_bwd_c if g.is_cuda else _cost_bwd
        return fn(Sband.contiguous(), k.contiguous(), om.contiguous(), dwp, V.contiguous(), g.float().contiguous()), \
            None, None, None, None, None


def shell_cost(sp: Spectrum, w: torch.Tensor, V: torch.Tensor, kidx: torch.Tensor | None = None) -> torch.Tensor:
    """Weighted shell energy Σ_k w_k J_k(V) for candidates V [B, M, 2]; w [B, nk(sub)] (with
    gradient). Returns [B, M]."""
    k, om, Sb = _sub(sp, kidx, band=True)
    return _ShellCost.apply(w, Sb, k, om, sp.dwp, V)
