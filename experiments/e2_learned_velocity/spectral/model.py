"""The learned dispersion-shell fit.

A window's spectrum (spectrum.prepare) is searched in stages. Each stage s has a centre c_s
(the classical pattern velocity for the first stage, the previous stage's peak afterwards) and
a square grid of candidate velocities c_s + o (o on ±half at `step`):

1. Per k, a small network reads the k's whitened ω' profile rolled to the shell of c_s
   (where the energy sits relative to the shell), its |k| (rad/m and cycles/cell), direction,
   relative power, and where the fixed-in-image line and the wrap fall, plus a context vector
   pooled over all k (PointNet style), and outputs a weight w_k ≥ 0: E1's whitening, learned.
2. The cost volume C(o) = Σ_k w_k J_k(c_s + o) / Σ_k w_k (spectrum.shell_cost), standardised
   over the grid, goes through a small CNN to logits over the grid.
3. Coarse stages: the argmax (detached) is the next centre. The last (fine) stage: soft-argmax
   over its grid gives δ, v = c_s + δ; σ comes from a head on the fine distribution's spread
   and the coarse stage's confidence.

Only the inputs E2 allows reach the network: the tiles and mask (through the spectrum), gsd,
dt and the pattern velocity computed from the tiles. Velocities are in (augmented) tile axes.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import spectrum as spx


def grid(half: float, step: float, device=None) -> torch.Tensor:
    """Offsets [m·m, 2] (x, y) on ±half at step; row-major with y as the row."""
    o = torch.arange(-half, half + 1e-6, step, device=device)
    oy, ox = torch.meshgrid(o, o, indexing="ij")
    return torch.stack([ox.reshape(-1), oy.reshape(-1)], -1)


class KNet(nn.Module):
    """Per-k weights from per-k features with a pooled context (two passes)."""

    def __init__(self, nin: int, hid: int = 96):
        super().__init__()
        self.a = nn.Sequential(nn.Linear(nin, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU())
        self.b = nn.Sequential(nn.Linear(2 * hid + 3, hid), nn.GELU(), nn.Linear(hid, 1))

    def forward(self, feat: torch.Tensor, glob: torch.Tensor) -> torch.Tensor:
        h = self.a(feat)                                       # [B, nk, hid]
        ctx = h.mean(1, keepdim=True).expand(-1, h.shape[1], -1)
        g = glob[:, None, :].expand(-1, h.shape[1], -1)
        return self.b(torch.cat([h, ctx, g], -1))[..., 0]       # [B, nk] logit


class VolNet(nn.Module):
    """Cost volume [B, 1, m, m] → logits [B, m·m]."""

    def __init__(self, ch: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=2, dilation=2), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=4, dilation=4), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(ch, 1, 3, padding=1))
        self.scale = nn.Parameter(torch.tensor(4.0))

    def forward(self, C: torch.Tensor) -> torch.Tensor:
        B, m, _ = C.shape
        z = (C - C.mean((1, 2), keepdim=True)) / C.std((1, 2), keepdim=True).clamp_min(1e-6)
        # The standardised volume itself is a logit (E1's objective); the CNN adds to it.
        return (self.scale * z + self.net(z[:, None])[:, 0]).reshape(B, m * m)


class SpectralNet(nn.Module):
    """stages: list of (half, step, kmax_cycles or None). The last stage is the fine one."""

    def __init__(self, ntp: int, stages=((6.0, 0.25, None), (0.6, 0.05, None)), hid: int = 96,
                 prep: dict | None = None):
        super().__init__()
        self.ntp = ntp
        self.stages = [tuple(s) for s in stages]
        self.prep = {**dict(bin=2, pad=2, band_bins=1.0, kmin_cells=3.0, kmax_frac=0.8), **(prep or {})}
        nin = ntp + 9
        self.knets = nn.ModuleList([KNet(nin, hid) for _ in stages])
        self.vols = nn.ModuleList([VolNet() for _ in stages])
        # σ head: fine spread (log var x, y), coarse confidence (max p, entropy), log gsd, dwell.
        self.sig = nn.Sequential(nn.Linear(6, 32), nn.GELU(), nn.Linear(32, 2))

    def features(self, sp: spx.Spectrum, c: torch.Tensor, kidx):
        prof = spx.profile(sp, c, kidx)                                      # [B, nk, ntp]
        prof = torch.log(prof * self.ntp + 1e-3)
        k = sp.k if kidx is None else sp.k[:, kidx]
        km = sp.kmag if kidx is None else sp.kmag[:, kidx]
        om = sp.omega if kidx is None else sp.omega[:, kidx]
        f = sp.f if kidx is None else sp.f[kidx]
        logE = sp.logE if kidx is None else sp.logE[:, kidx]
        dt = sp.T / (self.ntp / self.prep["pad"])
        phi_shell = (om - (k * c[:, None, :]).sum(-1)) * dt[:, None]          # shell of c, rad/frame
        scal = torch.stack([
            torch.log(km), f.norm(dim=-1)[None].expand_as(km) * 4.0,
            k[..., 0] / km, k[..., 1] / km, logE.clamp(-8, 8) / 4.0,
            torch.cos(phi_shell), torch.sin(phi_shell),                       # where ω' = 0 (fixed in image) sits
            torch.cos(om * dt[:, None]), torch.sin(om * dt[:, None]),          # Ω per frame (wrap)
        ], -1)
        glob = torch.stack([torch.log(sp.cell), (c.norm(dim=-1) * sp.T / sp.L).clamp(max=10),
                            torch.log(sp.T)], -1)
        return torch.cat([prof, scal], -1), glob

    def forward(self, tiles, gsd, dt, centre, target=None, teacher: float = 0.0):
        """tiles [B, K, N, N] (NaN masked), gsd, dt [B], centre [B, 2] (pattern velocity, or 0).
        target [B, 2] only for teacher forcing of the stage centres (training).
        Returns dict(v [B, 2], sigma [B, 2], stages=[(centre, offsets, logits)], ...)."""
        sp = spx.prepare(tiles, gsd, dt, **self.prep)
        c = centre.float()
        out = {"stages": []}
        conf = None
        for s, (half, step, kmax) in enumerate(self.stages):
            kidx = None
            if kmax is not None:
                kidx = torch.nonzero(sp.f.norm(dim=-1) <= kmax)[:, 0]
            feat, glob = self.features(sp, c, kidx)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=feat.is_cuda):
                w = F.softplus(self.knets[s](feat, glob)).float() + 1e-4
            w = w / w.sum(1, keepdim=True)
            o = grid(half, step, c.device)
            m = int(round(math.sqrt(o.shape[0])))
            C = spx.shell_cost(sp, w, c[:, None, :] + o[None], kidx)
            logits = self.vols[s](C.reshape(-1, m, m))
            out["stages"].append((c, o, logits))
            p = logits.softmax(-1)
            if s < len(self.stages) - 1:
                if conf is None:
                    conf = torch.stack([p.max(-1).values.clamp_min(1e-6).log(),
                                        -(p * p.clamp_min(1e-12).log()).sum(-1) / math.log(p.shape[-1])], -1)
                nxt = c + o[logits.argmax(-1)]
                if target is not None and teacher > 0:
                    # Teacher forcing: some windows centre the next stage on the truth, jittered
                    # within half the next grid, so the fine stage learns before the coarse one is good.
                    nh = self.stages[s + 1][0]
                    tf = torch.rand(c.shape[0], device=c.device) < teacher
                    jit = (torch.rand_like(c) * 2 - 1) * 0.5 * nh
                    nxt = torch.where(tf[:, None], target + jit, nxt)
                c = nxt.detach()
            else:
                d = p @ o
                var = p @ o.square() - d.square()
                v = c + d
                if conf is None:
                    conf = torch.zeros(c.shape[0], 2, device=c.device)
                sf = torch.cat([var.clamp_min(1e-6).log(), conf, torch.log(sp.cell)[:, None],
                                (c.norm(dim=-1) * sp.T / sp.L).clamp(max=10)[:, None]], -1)
                out.update(v=v, logsig=self.sig(sf.detach()), var=var)
        return out


def ce_loss(c, o, logits, target, step, valid=None):
    """Cross-entropy against a Gaussian (σ = step) around the truth on the grid; windows whose
    truth lies outside the grid are left out. Returns (loss, fraction inside)."""
    rel = target - c                                                          # [B, 2]
    half = o.abs().max()
    inside = (rel.abs() <= half).all(-1)
    if valid is not None:
        inside = inside & valid
    d2 = ((o[None] - rel[:, None]) ** 2).sum(-1) / step ** 2
    q = torch.softmax(-0.5 * d2, -1)
    ce = -(q * logits.log_softmax(-1)).sum(-1)
    if inside.any():
        return ce[inside].mean(), inside.float().mean()
    return logits.sum() * 0.0, inside.float().mean()


def losses(out, target, stages, w_ce=1.0, w_reg=1.0, w_nll=0.1, beta=0.05):
    """Training loss: CE on every coarse stage, smooth-L1 on the fine estimate, Gaussian NLL
    of the (detached) error for σ."""
    parts = {}
    total = 0.0
    for s, (c, o, logits) in enumerate(out["stages"][:-1]):
        l, fin = ce_loss(c, o, logits, target, stages[s][1])
        parts[f"ce{s}"] = l.detach()
        parts[f"in{s}"] = fin.detach()
        total = total + w_ce * l
    c, o, logits = out["stages"][-1]
    lf, fin = ce_loss(c, o, logits, target, stages[-1][1])
    parts["ce_fine"] = lf.detach()
    parts["in_fine"] = fin.detach()
    err = out["v"] - target
    reg = F.smooth_l1_loss(out["v"], target, beta=beta, reduction="none").sum(-1).mean()
    ls = out["logsig"]
    nll = (ls + 0.5 * err.detach().square() * torch.exp(-2 * ls)).sum(-1).mean()
    parts.update(reg=reg.detach(), nll=nll.detach(), abs_err=err.detach().norm(dim=-1).mean())
    total = total + 0.3 * w_ce * lf + w_reg * reg + w_nll * nll
    return total, parts
