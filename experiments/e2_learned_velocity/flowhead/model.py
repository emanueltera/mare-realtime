"""Model B ('flowhead'): SEA-RAFT-M flow between tile pairs at several gaps + a learned head.

The window's K rectified tiles go through SEA-RAFT-M pair by pair: for every gap g in
`gaps`, a chain of pairs (i, i + g) spread over the window (make_pairs). Pretrained
SEA-RAFT returns about zero flow on this water once the displacement passes ~60 px (fast,
low runs, longer gaps), so each pair's flow starts from a global shift instead of
SEA-RAFT's own regressed guess: whitened phase correlation of every consecutive pair (an
integer shift), summed along the chain from i to j (phase_shift, chain_init); SEA-RAFT's
iterations refine it.

In tile axes the content moves at −v (plus the waves' own motion), so a flow of f tile
pixels over g frames is a raw velocity u = −f · gsd / (g · dt) in m/s: E1's failure mode,
biased by the waves' phase motion. The head sees, at 1/8 resolution, every pair's u,
the initial shift as a velocity, SEA-RAFT's own uncertainty (its mixture weights and log b),
the photometric residual after
warping, the fraction of each cell whose flow lands inside the tile, the tiles' appearance
(a small learned stem on the first and last frame: wavelength, orientation, glint), the
validity mask and log(gsd), and returns

    v = Σ_{pair, cell} softmax(w)_{pair, cell} · (u_{pair, cell} + Δ_{pair, cell})

with w and Δ predicted per cell from local features modulated by a global summary (FiLM),
so the head can both choose where to trust the flow and learn the correction the waves'
phase motion needs; plus σ (m/s, per tile axis) from the global summary. With Δ = 0 and
uniform w it is the plain mean of the raw flow velocity (Δ and w start there).

Inputs are only the allowed ones: tiles, mask, gsd and dt. Nothing about the run, the
aircraft or the sea reaches the model.

A pair's features depend only on the frames from i to j (each frame is z-scored on its own
valid cells; the initial shift chains the consecutive pairs in between), so prediction can
compute each pair of a run once and reuse it for every window that contains it
(FlowHeadNet.pair_features / frame_features / head).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

SEA_RAFT_DIR = Path(os.environ.get("SEA_RAFT_DIR", "/workspace/models/SEA-RAFT"))
SEA_RAFT_CKPT = "MemorySlices/Tartan-C-T-TSKH-spring540x960-M"

U_SCALE = 20.0      # m/s: raw velocities enter the head divided by this
DELTA_SCALE = 2.0   # m/s: the head's correction Δ is this × a unit-scale output
GSD_REF = 0.125     # m/px: log(gsd / GSD_REF) is the head's scale input (train median)
DT_REF = 0.1        # s


def make_pairs(K: int, gaps) -> list[tuple[int, int]]:
    """Pairs (i, i + g) for each gap g: ceil((K − 1) / g) starts spread evenly over
    [0, K − 1 − g], so each gap's chain covers the window. K = 8, gaps 1, 2, 4, 7 gives
    7 + 4 + 2 + 1 = 14 pairs."""
    out = []
    for g in gaps:
        if g < 1 or g > K - 1:
            raise ValueError(f"gap {g} does not fit a window of {K} frames")
        n = math.ceil((K - 1) / g)
        starts = sorted({round(s * (K - 1 - g) / max(n - 1, 1)) for s in range(n)})
        out += [(s, s + g) for s in starts]
    return out


def load_sea_raft(iters: int = 4):
    """SEA-RAFT-M with its pretrained weights (as E1's flow_bias.py loads it)."""
    sys.path.insert(0, str(SEA_RAFT_DIR / "core"))
    sys.path.insert(0, str(SEA_RAFT_DIR))
    from raft import RAFT  # noqa: E402

    args = argparse.Namespace(**json.loads((SEA_RAFT_DIR / "config/eval/spring-M.json").read_text()))
    args.iters = iters
    return RAFT.from_pretrained(SEA_RAFT_CKPT, args=args)


def phase_shift(za: torch.Tensor, zb: torch.Tensor) -> torch.Tensor:
    """Integer global shift d [N, 2] (px, (x, y)) with zb(p) ≈ za(p − d), i.e. the content's
    displacement from a to b, by whitened phase correlation (Hann-tapered, masked cells are
    0 in z). Shifts wrap at ± half the tile."""
    N, H, W = za.shape
    with torch.autocast("cuda", enabled=False):
        win = torch.outer(torch.hann_window(H, device=za.device), torch.hann_window(W, device=za.device))
        fa = torch.fft.rfft2(za.float() * win)
        fb = torch.fft.rfft2(zb.float() * win)
        r = fa * fb.conj()
        r = torch.fft.irfft2(r / (r.abs() + 1e-6), s=(H, W))
        k = r.reshape(N, -1).argmax(-1)
        py, px = k // W, k % W
        # the peak sits at −d (mod the tile)
        dx = -(((px + W // 2) % W) - W // 2)
        dy = -(((py + H // 2) % H) - H // 2)
    return torch.stack([dx, dy], -1).float()


def chain_init(shift1: torch.Tensor, ia: torch.Tensor, ib: torch.Tensor) -> torch.Tensor:
    """Initial flow of pairs (ia → ib) [N, 2] from the consecutive shifts shift1 [F − 1, 2]
    of the same frames (pair k is frame k → k + 1): the sum along the chain."""
    cs = torch.cat([torch.zeros_like(shift1[:1]), shift1.cumsum(0)])
    return cs[ib] - cs[ia]


def zscore_frames(tiles: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Each frame [.., H, W] standardised over its own valid cells; 0 where masked."""
    m = mask.float()
    n = m.sum((-1, -2), keepdim=True).clamp_min(1.0)
    mu = (tiles * m).sum((-1, -2), keepdim=True) / n
    var = (((tiles - mu) * m) ** 2).sum((-1, -2), keepdim=True) / n
    return (tiles - mu) / var.sqrt().clamp_min(1e-6) * m


class ResBlock(nn.Module):
    def __init__(self, c: int, dilation: int):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=dilation, dilation=dilation)
        self.c2 = nn.Conv2d(c, c, 3, padding=dilation, dilation=dilation)
        self.n1 = nn.GroupNorm(8, c)
        self.n2 = nn.GroupNorm(8, c)

    def forward(self, x):
        y = self.c1(F.gelu(self.n1(x)))
        y = self.c2(F.gelu(self.n2(y)))
        return x + y


BACKBONE_UPDATE = ("init_conv", "upsample_weight", "flow_head", "update_block")


class FlowHeadNet(nn.Module):
    """SEA-RAFT-M backbone + the velocity head.

    backbone: 'frozen' runs SEA-RAFT without gradients (pretrained weights as they are);
    'update' fine-tunes its recurrent update, flow head and upsampler through the velocity
    loss while its two ResNet encoders stay pretrained and frozen; 'all' fine-tunes
    everything. SEA-RAFT's BatchNorm layers are kept in eval mode in every case (the pairs of
    one window are far from independent samples). The feature encoder runs once per frame,
    not once per pair."""

    PAIR_CH = 2 + 2 + 4 + 2   # u, the initial shift as a velocity, SEA-RAFT info (mixture weights 2,
                              # raw log b 2), photometric residual, in-tile fraction
    APP_CH = 16            # appearance stem channels per frame

    def __init__(self, K: int = 8, gaps=(1, 2, 4, 7), iters: int = 4, backbone: str = "frozen",
                 width: int = 128, blocks: int = 6):
        super().__init__()
        self.K, self.gaps, self.iters, self.backbone = K, tuple(gaps), iters, backbone
        self.pairs = make_pairs(K, gaps)
        self.P = len(self.pairs)
        self.raft = load_sea_raft(iters)
        for n, p in self.raft.named_parameters():
            p.requires_grad_(backbone == "all" or (backbone == "update" and n.startswith(BACKBONE_UPDATE)))
        from corr import CorrBlock  # SEA-RAFT's (on the path after load_sea_raft)
        from utils.utils import coords_grid
        self._corr_block, self._coords_grid = CorrBlock, coords_grid
        # appearance: full resolution → 1/8, on a single frame
        self.stem = nn.Sequential(
            nn.Conv2d(1, 24, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, self.APP_CH, 3, stride=2, padding=1),
        )
        cin = self.P * self.PAIR_CH + 2 * self.APP_CH + 1 + 2   # + valid fraction + log gsd, dt
        self.inp = nn.Conv2d(cin, width, 3, padding=1)
        dil = [1, 2, 4, 8, 1, 2, 4, 8][:blocks]
        self.trunk = nn.ModuleList([ResBlock(width, d) for d in dil])
        self.film = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2 * width))
        self.out = nn.Sequential(nn.GroupNorm(8, width), nn.GELU(), nn.Conv2d(width, 3 * self.P, 1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)
        self.sigma = nn.Sequential(nn.Linear(width + 2, 64), nn.GELU(), nn.Linear(64, 2))

    def train(self, mode: bool = True):
        super().train(mode)
        for m in self.raft.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    # ---- SEA-RAFT on pairs of frames (its forward in test mode, split so that the feature
    # ---- encoder runs once per frame) ---------------------------------------------------

    @staticmethod
    def to_image(z):
        """z-scored frames [N, H, W] → SEA-RAFT's input range, [N, 3, H, W] in −1..1."""
        return ((127.5 + 50.0 * z).clamp(0, 255) / 127.5 - 1.0).unsqueeze(1).expand(-1, 3, -1, -1)

    def encode_frames(self, z):
        """SEA-RAFT's feature map of every frame [N, H, W] → [N, 256, H/8, W/8]."""
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.backbone == "all"):
            return self.raft.fnet(self.to_image(z))

    def pair_flow(self, z, fmap, ia, ib, init):
        """Flow (px) and SEA-RAFT's info [N, 4, H, W] for pairs (z[ia] → z[ib]), its
        iterations starting from the global shift init [N, 2] (px)."""
        r, a = self.raft, self.raft.args
        grad = torch.is_grad_enabled()
        im1, im2 = self.to_image(z[ia]), self.to_image(z[ib])
        with torch.set_grad_enabled(grad and self.backbone == "all"):
            cnet = r.cnet(torch.cat([im1, im2], dim=1))
        with torch.set_grad_enabled(grad and self.backbone != "frozen"):
            cnet = r.init_conv(cnet)
            net, context = torch.split(cnet, [a.dim, a.dim], dim=1)
            N, _, h, w = net.shape
            upd = r.flow_head(net)
            flow8 = (init / 8.0).reshape(N, 2, 1, 1).expand(N, 2, h, w).to(upd.dtype)
            info8 = upd[:, 2:]
            corr_fn = self._corr_block(fmap[ia], fmap[ib], a)
            dilation = torch.ones(N, 1, h, w, device=z.device)
            for _ in range(self.iters):
                flow8 = flow8.detach()
                coords2 = (self._coords_grid(N, h, w, device=z.device) + flow8).detach()
                with torch.backends.cudnn.flags(enabled=False):   # cuDNN's grid_sample refuses large batches
                    corr = corr_fn(coords2, dilation=dilation)
                net = r.update_block(net, context, corr, flow8)
                upd = r.flow_head(net)
                flow8, info8 = flow8 + upd[:, :2], upd[:, 2:]
            flow, info = r.upsample_data(flow8, info8, 0.25 * r.upsample_weight(net))
        return flow.float(), info.float()

    # ---- pair and frame features (cacheable per run) -------------------------------------

    def pair_features(self, z, mask, fmap, ia, ib, init, gsd, dt):
        """Features of pairs (z[ia] → z[ib]) [N, PAIR_CH, H/8, W/8] and the raw velocity
        u [N, 2, H/8, W/8] (m/s, tile axes). z, mask: frames [F, H, W]; fmap: encode_frames(z);
        ia, ib: index tensors [N]; init: initial flow [N, 2] (px, chain_init); gsd, dt: [N]."""
        flow, info = self.pair_flow(z, fmap, ia, ib, init)
        # content moves at −v: a flow of f px over g frames is −v·g·dt/gsd
        gap = (ib - ia).to(flow.dtype).reshape(-1, 1, 1, 1)
        u = -flow * (gsd.reshape(-1, 1, 1, 1) / (gap * dt.reshape(-1, 1, 1, 1)))
        # photometric residual: frame b warped back onto a with the (detached) flow
        N, _, H, W = flow.shape
        yy, xx = torch.meshgrid(torch.arange(H, device=flow.device), torch.arange(W, device=flow.device), indexing="ij")
        f = flow.detach()
        grid = torch.stack([(xx + f[:, 0]) / (W - 1) * 2 - 1, (yy + f[:, 1]) / (H - 1) * 2 - 1], -1)
        wb = F.grid_sample(z[ib].unsqueeze(1).float(), grid, align_corners=True, padding_mode="zeros")
        resid = (z[ia].unsqueeze(1) - wb).abs() * mask[ia].unsqueeze(1)
        inside = ((grid.abs() <= 1).all(-1).float() * mask[ia]).unsqueeze(1)
        info = torch.cat([info[:, :2], info[:, 2:].clamp(-10, 10)], 1)
        u8 = F.avg_pool2d(u, 8)
        u0 = (-init.reshape(-1, 2, 1, 1) * (gsd.reshape(-1, 1, 1, 1) / (gap * dt.reshape(-1, 1, 1, 1)))).expand_as(u8)
        feats = torch.cat([u8 / U_SCALE, u0 / U_SCALE, F.avg_pool2d(info, 8), F.avg_pool2d(resid, 8),
                           F.avg_pool2d(inside, 8)], 1)
        return feats, u8

    def frame_features(self, z):
        """Appearance stem on frames [N, H, W] → [N, APP_CH, H/8, W/8]."""
        return self.stem(z.unsqueeze(1))

    # ---- the head ------------------------------------------------------------------------

    def head(self, pf, u8, app0, app1, valid8, gsd, dt):
        """pf [B, P, 7, h, w], u8 [B, P, 2, h, w], app0/app1 [B, APP_CH, h, w] (first and
        last frame), valid8 [B, 1, h, w] (fraction of cells valid in every frame), gsd, dt [B].
        Returns v [B, 2] (m/s, tile axes), sigma [B, 2], and diagnostics."""
        B, P, _, h, w = pf.shape
        lg = torch.log(gsd / GSD_REF).reshape(B, 1, 1, 1).expand(B, 1, h, w)
        ld = (dt / DT_REF).reshape(B, 1, 1, 1).expand(B, 1, h, w)
        x = torch.cat([pf.reshape(B, P * self.PAIR_CH, h, w), app0, app1, valid8, lg, ld], 1)
        x = self.inp(x)
        for blk in self.trunk:
            x = blk(x)
        g = x.mean((-1, -2))
        scale, shift = self.film(g).chunk(2, -1)
        y = x * (1 + scale[..., None, None]) + shift[..., None, None]
        o = self.out(y).reshape(B, P, 3, h, w)
        logit = o[:, :, 0] + torch.log(valid8.clamp_min(1e-3))          # cells with no image get ~no weight
        wgt = torch.softmax(logit.reshape(B, -1), -1).reshape(B, P, 1, h, w)
        delta = DELTA_SCALE * o[:, :, 1:]
        v = (wgt * (u8 + delta)).sum((1, 3, 4))
        v_raw = (wgt * u8).sum((1, 3, 4))
        spread = ((wgt * (u8 - v_raw[:, None, :, None, None]) ** 2).sum((1, 3, 4)) + 1e-6).sqrt()
        log_sigma = self.sigma(torch.cat([g, torch.log(spread)], -1))
        sigma = F.softplus(log_sigma) + 0.01
        return v, sigma, {"v_raw": v_raw, "u_mean": u8.mean((1, 3, 4))}

    def forward(self, tiles, mask, gsd, dt):
        """tiles [B, K, H, W] (any per-window normalisation; re-standardised per frame),
        mask [B, K, H, W] bool, gsd, dt [B] → v [B, 2], sigma [B, 2], diagnostics."""
        B, K, H, W = tiles.shape
        z = zscore_frames(tiles.float(), mask).reshape(B * K, H, W)
        m = mask.reshape(B * K, H, W)
        fmap = self.encode_frames(z)
        base = torch.arange(B, device=z.device).repeat_interleave(self.P) * K
        ia = base + torch.tensor([i for i, _ in self.pairs], device=z.device).repeat(B)
        ib = base + torch.tensor([j for _, j in self.pairs], device=z.device).repeat(B)
        # consecutive-pair shifts of every window, chained to each pair's initial flow
        k = torch.arange(B * K, device=z.device).reshape(B, K)
        s1 = phase_shift(z[k[:, :-1].reshape(-1)], z[k[:, 1:].reshape(-1)]).reshape(B, K - 1, 2)
        s1 = torch.cat([s1, torch.zeros_like(s1[:, :1])], 1).reshape(B * K, 2)[:-1]   # no chain across windows
        init = chain_init(s1, ia, ib)
        rep = lambda t: t.reshape(B, 1).expand(B, self.P).reshape(-1)  # noqa: E731
        pf, u8 = self.pair_features(z, m, fmap, ia, ib, init, rep(gsd), rep(dt))
        h, w = pf.shape[-2:]
        zz = z.reshape(B, K, H, W)
        app = self.frame_features(torch.stack([zz[:, 0], zz[:, -1]], 1).reshape(-1, H, W)).reshape(B, 2, -1, h, w)
        valid8 = F.avg_pool2d(mask.all(1, keepdim=True).float(), 8)
        return self.head(pf.reshape(B, self.P, -1, h, w), u8.reshape(B, self.P, 2, h, w), app[:, 0], app[:, 1],
                         valid8, gsd, dt)
