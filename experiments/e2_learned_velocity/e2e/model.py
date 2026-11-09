"""Model A ('e2e'): an end-to-end video network for the water-relative velocity.

The question it tests: given K rectified tiles of moving water, can a network learn to
tell the camera's motion (every scale of the texture translates together at −v) from the
waves' own motion (each wavelength travels at its own phase speed, √(g λ / 2π) in deep
water) without being told any of that?

Pipeline, per window (B windows of K frames, tiles [K, n, n]):

1. `prep`: log luminance, normalised over the window's valid cells (the mask is kept).
2. `Encoder`: one 2-D CNN shared by every frame, FiLM-conditioned on the tile's gsd (wave
   speed depends on metres, the tile is in pixels), gives feature maps at four scales
   (1/2, 1/4, 1/8, 1/16 of the tile). The network chooses what to look at at each scale.
3. `cost_volumes`: for every frame gap g = 1 … K − 1 and every scale, the masked global
   cross-correlation of the (unit-norm) features of frames k and k + g, averaged over k,
   at every lag (via FFT): one correlation surface per (scale, gap), plus the overlap at
   each lag. The camera's motion puts every scale's peak at the lag g·u; a wave train
   moves its own scale's peak by g·c(λ)·dt, more for long waves and long gaps — the
   dispersion is laid out across the surfaces.
4. Start (no learning, no gradient): `phase_corr_init`, the peak of the summed, softly
   whitened cross-power spectra of consecutive raw frames → u₀ px/frame. It is pulled by
   the waves exactly like the phase-correlation baseline; the network has to undo that.
5. Iterations (shared weights, `n_iter` times, RAFT-style): around the current u, a
   (2r + 1)² window of every surface is sampled at the lag g·u (sub-cell, bilinear); each
   window becomes a token (with a learned scale/gap embedding), a small transformer over
   the tokens plus a conditioning token (log gsd, dt, u, iteration) returns a correction Δu
   and log σ.
6. Velocity in tile axes: v = −u · gsd / dt (content moves through the tile at −v).

Inputs, all of them allowed by E2's design: the tiles, their validity mask, gsd (m/px) and
dt (s); the network's own u enters the read-out. Nothing about the run, the sea, the sky
or the aircraft enters.

Sign convention (as e2.baselines): b(x) = a(x − d) for a content shift d = (dx, dy) in
(col = x, row = y) pixels between consecutive frames; u is that d per frame.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_EPS = 1e-3      # log(luminance + eps): linear light spans dusk to glint
GSD_REF = 0.1       # m/px, the centre of the log-gsd conditioning
DT_REF = 0.1        # s (10 Hz)
U_REF = 32.0        # px/frame, the scale of u in the conditioning (train median ≈ 27)


# ---- inputs and the fixed start ------------------------------------------------------------


def prep(tiles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """tiles [..., H, W] (float16/32, NaN = no image) → (x, m): log luminance normalised over
    each frame's valid cells (0 where masked), and the mask as float. Per frame, so that a
    frame's features do not depend on the window it is in (inference reuses them)."""
    t = tiles.float()
    m = torch.isfinite(t)
    x = torch.log(torch.where(m, t, torch.ones_like(t)).clamp_min(0) + LOG_EPS)
    mf = m.float()
    cnt = mf.sum((-2, -1), keepdim=True).clamp_min(1)
    mean = (x * mf).sum((-2, -1), keepdim=True) / cnt
    var = (((x - mean) * mf) ** 2).sum((-2, -1), keepdim=True) / cnt
    x = (x - mean) / var.sqrt().clamp_min(1e-6) * mf
    return x, mf


def _hann(n: int, device) -> torch.Tensor:
    return torch.hann_window(n, periodic=False, device=device, dtype=torch.float32)


@torch.no_grad()
def pc_pairs(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Softly whitened cross-power spectra of consecutive frames: x, m [B, F, H, W] →
    [B, F − 1, H, W/2 + 1], pair k = frames (k, k + 1), b · conj(a) (peak at d, b(x) = a(x − d))."""
    B, Fr, H, W = x.shape
    w = (_hann(H, x.device)[:, None] * _hann(W, x.device)[None, :]) * m
    mean = (x * w).sum((2, 3), keepdim=True) / w.sum((2, 3), keepdim=True).clamp_min(1e-6)
    f = torch.fft.rfft2((x - mean) * w)
    r = f[:, 1:] * f[:, :-1].conj()
    mag = r.abs()
    return r / (mag + mag.mean((2, 3), keepdim=True) + 1e-12)    # soft whitening, as e2.baselines


@torch.no_grad()
def pc_peak(r: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """r [B, H, W/2 + 1] summed pair spectra → shift [B, 2] (dx, dy) px: Gaussian low-pass,
    integer peak, then a parabola through its neighbours."""
    B = r.shape[0]
    fy = torch.fft.fftfreq(H, device=r.device)[:, None]
    fx = torch.fft.rfftfreq(W, device=r.device)[None, :]
    cc = torch.fft.irfft2(r * torch.exp(-0.5 * (fx * fx + fy * fy) / 0.25 ** 2), s=(H, W))
    flat = cc.flatten(1).argmax(1)
    py, px = flat // W, flat % W
    b = torch.arange(B, device=r.device)

    def sub(c0, cm, cp):  # vertex of the parabola through (−1, cm), (0, c0), (1, cp)
        den = cm - 2 * c0 + cp
        return torch.where(den.abs() > 1e-12, 0.5 * (cm - cp) / den, torch.zeros_like(den)).clamp(-0.5, 0.5)

    c0 = cc[b, py, px]
    sx = sub(c0, cc[b, py, (px - 1) % W], cc[b, py, (px + 1) % W])
    sy = sub(c0, cc[b, (py - 1) % H, px], cc[b, (py + 1) % H, px])
    dx = torch.where(px > W // 2, px - W, px).float() + sx
    dy = torch.where(py > H // 2, py - H, py).float() + sy
    return torch.stack([dx, dy], 1)


def phase_corr_init(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """First guess of the content shift per frame, u₀ [B, 2] (dx, dy) px/frame, from the sum
    over the window's consecutive pairs of softly whitened cross-power spectra (one peak if
    the motion is constant over the window)."""
    return pc_peak(pc_pairs(x, m).sum(1), x.shape[-2], x.shape[-1])


def align(x: torch.Tensor, m: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Resample frame k at p + (k − c)·u, c = (K − 1)/2, so content moving at u per frame
    stands still on the middle frame's grid: frame_k(p) = frame_0(p − k·u) ⇒ aligned_k(p) =
    frame_0(p − c·u) for every k. Not used by the model (its read-out moves the lag window
    instead); kept for the tests and for looking at stacks."""
    B, K, H, W = x.shape
    k = torch.arange(K, device=x.device, dtype=torch.float32) - (K - 1) / 2
    s = k[None, :, None] * u[:, None, :].float()                      # [B, K, 2] (dx, dy)
    gy, gx = torch.meshgrid(torch.arange(H, device=x.device, dtype=torch.float32) + 0.5,
                            torch.arange(W, device=x.device, dtype=torch.float32) + 0.5, indexing="ij")
    px = gx[None, None] + s[..., 0, None, None]
    py = gy[None, None] + s[..., 1, None, None]
    grid = torch.stack([2 * px / W - 1, 2 * py / H - 1], -1).view(B * K, H, W, 2)
    xa = F.grid_sample(x.reshape(B * K, 1, H, W), grid, mode="bicubic", padding_mode="zeros", align_corners=False)
    ma = F.grid_sample(m.reshape(B * K, 1, H, W), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    ma = (ma > 0.999).float()
    return (xa * ma).view(B, K, H, W), ma.view(B, K, H, W)


def condition(gsd: torch.Tensor, dt: torch.Tensor, u: torch.Tensor | None = None) -> torch.Tensor:
    """[B, 2] log gsd and dt; with u, [B, 5] adding u (px/frame) and |u|."""
    c = [torch.log(gsd / GSD_REF), dt / DT_REF - 1]
    if u is not None:
        un = u / U_REF
        c += [un[:, 0], un[:, 1], un.norm(dim=1)]
    return torch.stack(c, 1).float()


# ---- encoder and cost volumes --------------------------------------------------------------


class Block(nn.Module):
    """2-D residual block, FiLM (from log gsd, dt) on the second norm."""

    def __init__(self, c: int, cdim: int = 2):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)
        self.n1 = nn.GroupNorm(8, c)
        self.n2 = nn.GroupNorm(8, c)
        self.film = nn.Linear(cdim, 2 * c)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, c):
        h = F.gelu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        g, b = self.film(c)[:, :, None, None].chunk(2, 1)
        return F.gelu(x + h * (1 + g) + b)


class Encoder(nn.Module):
    """Frames [N, 2, H, W] (image × mask, mask) → feature maps at 1/2, 1/4, 1/8, 1/16 with
    `dims` channels each, for correlation."""

    def __init__(self, widths=(32, 64, 96, 128), depths=(1, 2, 2, 2), dims=(8, 16, 32, 32)):
        super().__init__()
        self.stem = nn.Conv2d(2, widths[0], 4, stride=2, padding=1)
        self.downs = nn.ModuleList([nn.Conv2d(widths[i - 1], widths[i], 2, stride=2) for i in range(1, len(widths))])
        self.levels = nn.ModuleList([nn.ModuleList([Block(c) for _ in range(d)]) for c, d in zip(widths, depths)])
        self.proj = nn.ModuleList([nn.Conv2d(c, d, 1) for c, d in zip(widths, dims)])

    def forward(self, x, c):
        h = self.stem(x.contiguous(memory_format=torch.channels_last))   # NHWC: what bf16 convs want
        out = []
        for i, blocks in enumerate(self.levels):
            if i:
                h = self.downs[i - 1](h)
            for blk in blocks:
                h = blk(h, c)
            out.append(self.proj[i](h).contiguous())
        return out


def level_masks(m: torch.Tensor, n_levels: int, erode: int = 2) -> list[torch.Tensor]:
    """m [N, 1, H, W] → validity at each scale (a cell is valid if all its pixels are), eroded
    by `erode` cells and with the outer `erode` cells of the tile dropped, so that features
    made from the zero padding at a tile or mask edge — which stay put while the content
    moves — are left out of the correlations."""
    out = []
    for i in range(n_levels):
        v = -F.max_pool2d(-m, 2 ** (i + 1))
        v = -F.max_pool2d(-v, 2 * erode + 1, stride=1, padding=erode)
        v = F.pad(v[..., erode:-erode, erode:-erode], (erode,) * 4)
        out.append(v)
    return out


def spectra(f: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-padded (to 2h × 2w, so the correlation is linear, not circular) spectra of masked
    unit-norm features f [N, D, h, w] and of their validity v [N, 1, h, w]."""
    s = (2 * f.shape[-2], 2 * f.shape[-1])
    return torch.fft.rfft2(f.float() * v, s=s), torch.fft.rfft2(v.float(), s=s)


def surfaces(num: torch.Tensor, den: torch.Tensor, pairs: torch.Tensor, h: int, w: int):
    """Summed pair spectra num, den [..., G, 2h, w + 1] → (corr, overlap) [..., G, 2h, 2w], lag 0
    at [h, w]: corr = masked mean cosine similarity at each lag, overlap = the fraction of the
    pairs' cells that overlap there; `pairs` [G] the number of frame pairs in each sum."""
    s = (2 * h, 2 * w)
    num = torch.fft.fftshift(torch.fft.irfft2(num, s=s), dim=(-2, -1))
    den = torch.fft.fftshift(torch.fft.irfft2(den, s=s), dim=(-2, -1))
    ov = (den / (pairs[:, None, None] * h * w)).clamp(0, 1)
    corr = num / den.clamp_min(8.0)     # below 8 overlapping cells, shrunk toward 0
    return corr, ov


def cost_volumes(ff: torch.Tensor, fm: torch.Tensor, gaps, h: int, w: int):
    """Windows' frame spectra ff [B, K, D, 2h, w + 1], fm [B, K, 1, 2h, w + 1] → for every gap g
    the sums over k of Σ_d F_{k+g} conj(F_k) (peak at lag d when frame k + g is frame k moved
    by d), as (corr, overlap) [B, G, 2h, 2w] (see `surfaces`)."""
    B, K = ff.shape[:2]

    def cross(f):  # Σ_k Σ_c F_{k+g} conj(F_k) for every g: [B, G, 2h, w + 1]
        f2 = f.flatten(3).permute(0, 3, 1, 2)                          # [B, F, K, C]
        sm = f2 @ f2.conj().transpose(-1, -2)                          # S[j, k] = Σ_c F_j conj(F_k)
        g = torch.stack([torch.diagonal(sm, -g, -2, -1).sum(-1) for g in gaps], 1)
        return g.view(B, len(gaps), *f.shape[-2:])

    pairs = torch.tensor([K - g for g in gaps], device=ff.device, dtype=torch.float32)
    return surfaces(cross(ff), cross(fm), pairs, h, w)


def run_cost_volumes(ff: torch.Tensor, fm: torch.Tensor, gaps, K: int, starts: torch.Tensor, h: int, w: int):
    """The same surfaces for windows of K frames starting at `starts` within one run whose
    frame spectra are ff [F, D, 2h, w + 1], fm [F, 1, 2h, w + 1]: every frame pair's product is
    made once and the window sums are differences of cumulative sums. → [Nw, G, 2h, 2w] ×2."""
    nums, dens = [], []
    for g in gaps:
        for f, out in ((ff, nums), (fm, dens)):
            p = (f[g:] * f[:-g].conj()).sum(1)                           # pair (k, k + g), k = 0 … F − g − 1
            c = torch.cat([torch.zeros_like(p[:1]), p.cumsum(0)])
            out.append(c[starts + K - g] - c[starts])                    # Σ over k = s … s + K − g − 1
    pairs = torch.tensor([K - g for g in gaps], device=ff.device, dtype=torch.float32)
    return surfaces(torch.stack(nums, 1), torch.stack(dens, 1), pairs, h, w)


def sample_windows(surf: torch.Tensor, centre: torch.Tensor, r: int) -> torch.Tensor:
    """surf [B, G, C, 2h, 2w] (lag 0 at [h, w]), centre [B, G, 2] lag (dx, dy) in cells →
    [B, G, C, (2r + 1)²]: the surfaces at the integer offsets around the centre (bilinear)."""
    B, G, C, H2, W2 = surf.shape
    o = torch.arange(-r, r + 1, device=surf.device, dtype=torch.float32)
    oy, ox = torch.meshgrid(o, o, indexing="ij")
    px = centre[..., 0, None, None] + ox + W2 // 2
    py = centre[..., 1, None, None] + oy + H2 // 2
    grid = torch.stack([2 * px / (W2 - 1) - 1, 2 * py / (H2 - 1) - 1], -1)
    v = F.grid_sample(surf.reshape(B * G, C, H2, W2), grid.reshape(B * G, 2 * r + 1, 2 * r + 1, 2),
                      mode="bilinear", padding_mode="zeros", align_corners=True)
    return v.reshape(B, G, C, -1)


# ---- read-out ------------------------------------------------------------------------------


class ReadOut(nn.Module):
    """Tokens = sampled windows of every (scale, gap) surface; a transformer over them and a
    conditioning token; the conditioning token's output → Δu (px/frame) and log σ."""

    def __init__(self, n_levels: int, max_gap: int, r: int, d: int = 128, layers: int = 3, heads: int = 4,
                 max_step: float = 12.0):
        super().__init__()
        self.max_step = max_step
        n = (2 * r + 1) ** 2
        self.tok = nn.Sequential(nn.Linear(2 * n + 2, d), nn.GELU(), nn.Linear(d, d))
        self.emb_level = nn.Embedding(n_levels, d)
        self.emb_gap = nn.Embedding(max_gap + 1, d)
        self.cond = nn.Sequential(nn.Linear(5, d), nn.GELU(), nn.Linear(d, d))
        self.emb_iter = nn.Embedding(16, d)   # which iteration: early ones see larger residuals
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, 4)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, win, frac, level, gap, c, it: int = 0):
        """win [B, T, 2n] (corr, overlap at each offset), frac [B, T, 2] the sub-cell part of
        each window's centre (its sampling phase), level / gap [T] ids, c [B, 5], it the
        iteration."""
        t = self.tok(torch.cat([win, frac], -1)) + self.emb_level(level) + self.emb_gap(gap)
        ct = self.cond(c) + self.emb_iter.weight[it]
        h = self.tf(torch.cat([ct[:, None], t], 1))
        o = self.out(self.norm(h[:, 0]))
        du = self.max_step * torch.tanh(o[:, :2] / self.max_step)
        return du, o[:, 2:].clamp(-8, 6)


class E2E(nn.Module):
    """The whole estimator: encoder and cost volumes once per window, phase-correlation
    start, then n_iter learned corrections that read the cost volumes around the estimate."""

    def __init__(self, K: int = 8, n_iter: int = 4, radius: int = 4, max_gap: int = 7, enc: dict | None = None,
                 readout: dict | None = None):
        super().__init__()
        self.n_iter, self.r = n_iter, radius
        self.gaps = list(range(1, min(K - 1, max_gap) + 1))
        self.enc = Encoder(**(enc or {})).to(memory_format=torch.channels_last)
        self.n_levels = len(self.enc.proj)
        self.readout = ReadOut(self.n_levels, max(self.gaps), radius, **(readout or {}))
        lv, gp = zip(*[(lvl, g) for lvl in range(self.n_levels) for g in self.gaps])
        self.register_buffer("tok_level", torch.tensor(lv), persistent=False)
        self.register_buffer("tok_gap", torch.tensor(gp), persistent=False)

    def frame_spectra(self, x, m, gsd, dt, amp=True):
        """Frames x, m [N, H, W] (prepped), gsd, dt [N] → per scale (ff, fm, h, w)."""
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and x.is_cuda):
            feats = self.enc(torch.stack([x, m], 1), condition(gsd, dt))
        out = []
        for f, v in zip(feats, level_masks(m[:, None], self.n_levels)):
            out.append((*spectra(F.normalize(f.float(), dim=1), v), *f.shape[-2:]))
        return out

    def iterate(self, vols, u, gsd, dt, n_iter=None, amp=True):
        """The read-out: from the start u [B, 2] and the per-scale (corr, overlap) surfaces,
        n_iter corrections. Returns the dict described in `forward`."""
        gaps = torch.tensor(self.gaps, device=u.device, dtype=torch.float32)
        out = {"u0": u, "us": []}
        ls = None
        for it in range(n_iter or self.n_iter):
            u = u.detach()
            wins, fracs = [], []
            for lvl, (corr, ov) in enumerate(vols):
                centre = gaps[None, :, None] * u[:, None, :] / 2 ** (lvl + 1)     # [B, G, 2] cells
                wins.append(sample_windows(torch.stack([corr, ov], 2), centre, self.r).flatten(2))
                fracs.append(centre - centre.round())
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and u.is_cuda):
                du, ls = self.readout(torch.cat(wins, 1), torch.cat(fracs, 1), self.tok_level, self.tok_gap,
                                      condition(gsd, dt, u), it)
            u = u + du.float()
            ls = ls.float()
            out["us"].append(u)
        scale = (gsd / dt)[:, None]
        out["log_sigma"] = ls
        out["v"] = -u * scale
        out["sigma_v"] = torch.exp(ls) * scale
        return out

    def forward(self, tiles: torch.Tensor, gsd: torch.Tensor, dt: torch.Tensor, n_iter: int | None = None,
                amp: bool = True):
        """tiles [B, K, H, W] (NaN = masked), gsd [B] m/px, dt [B] s. Returns a dict with u0 (the
        phase-correlation start, px/frame), us (u after each iteration; u is detached between
        iterations, RAFT-style), log_sigma (px/frame) and v / sigma_v (m/s, tile axes) of the
        last iteration."""
        B, K, H, W = tiles.shape
        x, m = prep(tiles)
        u = phase_corr_init(x, m)
        sp = self.frame_spectra(x.flatten(0, 1), m.flatten(0, 1), gsd.repeat_interleave(K), dt.repeat_interleave(K), amp)
        vols = [cost_volumes(ff.view(B, K, *ff.shape[1:]), fm.view(B, K, *fm.shape[1:]), self.gaps, h, w)
                for ff, fm, h, w in sp]
        return self.iterate(vols, u, gsd, dt, n_iter, amp)

    @torch.no_grad()
    def forward_run(self, tiles: torch.Tensor, gsd: float, dt: float, K: int, starts: torch.Tensor,
                    n_iter: int | None = None, amp: bool = True, chunk: int = 32):
        """Every window of K frames starting at `starts` in one run, tiles [F, H, W]: the same
        estimate as `forward` on each window (per-frame features and pair spectra made once
        per run; window sums by cumulative sums), much faster for overlapping windows."""
        Fr, H, W = tiles.shape
        x, m = prep(tiles)
        r = pc_pairs(x[None], m[None])[0]                                    # [F − 1, H, W/2 + 1]
        c = torch.cat([torch.zeros_like(r[:1]), r.cumsum(0)])
        u = pc_peak(c[starts + K - 1] - c[starts], H, W)
        gsd_f = torch.full((Fr,), gsd, device=tiles.device)
        dt_f = torch.full((Fr,), dt, device=tiles.device)
        parts = [self.frame_spectra(x[i:i + chunk], m[i:i + chunk], gsd_f[i:i + chunk], dt_f[i:i + chunk], amp)
                 for i in range(0, Fr, chunk)]
        vols = []
        for lvl in range(self.n_levels):
            ff = torch.cat([p[lvl][0] for p in parts])
            fm = torch.cat([p[lvl][1] for p in parts])
            vols.append(run_cost_volumes(ff, fm, self.gaps, K, starts, *parts[0][lvl][2:]))
        n = len(starts)
        return self.iterate(vols, u, torch.full((n,), gsd, device=tiles.device), torch.full((n,), dt, device=tiles.device),
                            n_iter, amp)


def to_velocity(u: torch.Tensor, gsd: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """Content shift u (px/frame) → camera velocity in tile axes (m/s): v = −u · gsd / dt."""
    return -u * (gsd / dt)[:, None]


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
