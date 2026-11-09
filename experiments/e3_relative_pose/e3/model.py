"""E3's relative-pose network: two frames and the sensor vector in, the translation between
them (in tenths of the baro height, camera axes of the first frame) and a correction to the
gyro's rotation (centidegrees) out, each with a Laplace scale.

A shared convolutional encoder gives features at 1/8 and 1/16; local correlation volumes
(radius 10 cells at 1/8, 6 at 1/16, i.e. ±80 and ±96 px) compare the frames; the pixel's
ray direction is appended so the read-out knows where in the (wide, tilted) view each cell
sits; the sensor vector enters at 1/16 (broadcast) and again in the final MLP. `blind=True`
gives the sensor-only baseline: no images, the same MLP.

The network is built for one camera (`cam`, E3's portrait by default): the rays come from
it, and the read-out's size from its frame.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import E3_CAM, SENSOR_DIM, Camera


def conv(cin, cout, k=3, s=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, k // 2, bias=False), nn.GroupNorm(8, cout), nn.GELU())


class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.a = conv(c, c)
        self.b = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1, bias=False), nn.GroupNorm(8, c))

    def forward(self, x):
        return F.gelu(x + self.b(self.a(x)))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.s2 = nn.Sequential(conv(1, 32, 7, 2), Res(32))
        self.s4 = nn.Sequential(conv(32, 64, 3, 2), Res(64))
        self.s8 = nn.Sequential(conv(64, 96, 3, 2), Res(96), Res(96))
        self.s16 = nn.Sequential(conv(96, 128, 3, 2), Res(128), Res(128))

    def forward(self, x):
        f8 = self.s8(self.s4(self.s2(x)))
        return f8, self.s16(f8)


_WINDOWS: dict = {}


def _window(h: int, w: int, r: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """For each cell, the flat index of every cell within [−r, r]² of it (clamped) and whether it is inside."""
    key = (h, w, r, str(device))
    if key not in _WINDOWS:
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        d = torch.arange(-r, r + 1)
        dy, dx = torch.meshgrid(d, d, indexing="ij")
        yy = y.reshape(-1, 1) + dy.reshape(1, -1)
        xx = x.reshape(-1, 1) + dx.reshape(1, -1)
        inside = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        idx = yy.clamp(0, h - 1) * w + xx.clamp(0, w - 1)
        _WINDOWS[key] = (idx.to(device), inside.to(device))
    return _WINDOWS[key]


def local_corr(a: torch.Tensor, b: torch.Tensor, r: int) -> torch.Tensor:
    """Cosine correlation of a with b shifted by every (dy, dx) in [−r, r]² (zero outside b):
    (B, (2r+1)², h, w), channel (dy + r)·(2r+1) + (dx + r). All pairs by one matmul, then a gather."""
    B, C, h, w = a.shape
    a = F.normalize(a, dim=1).flatten(2).transpose(1, 2)          # B, hw, C
    b = F.normalize(b, dim=1).flatten(2)                           # B, C, hw
    full = torch.bmm(a, b)                                         # B, hw, hw
    idx, inside = _window(h, w, r, a.device)
    out = torch.gather(full, 2, idx[None].expand(B, -1, -1)) * inside
    return out.transpose(1, 2).reshape(B, -1, h, w)


def rays(h: int, w: int, cam: Camera = E3_CAM) -> torch.Tensor:
    """Normalised ray direction (x/z, y/z) at the centres of an h × w grid over cam's frame."""
    v = (torch.arange(h) + 0.5) * (cam.h / h)
    u = (torch.arange(w) + 0.5) * (cam.w / w)
    return torch.stack([((u - cam.cx) / cam.f)[None].expand(h, w), ((v - cam.cy) / cam.f)[:, None].expand(h, w)])


def camera_of(ck: dict) -> Camera:
    """The camera a checkpoint was trained for (E3's if it predates the field)."""
    return Camera(**ck["cam"]) if "cam" in ck else E3_CAM


class PoseNet(nn.Module):
    def __init__(self, blind: bool = False, r8: int = 10, r16: int = 6, cam: Camera = E3_CAM):
        super().__init__()
        self.blind = blind
        self.cam = cam
        self.r8, self.r16 = r8, r16
        H, W = cam.h, cam.w
        self.sens = nn.Sequential(nn.Linear(SENSOR_DIM, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU())
        if not blind:
            self.enc = Encoder()
            c8 = (2 * r8 + 1) ** 2
            c16 = (2 * r16 + 1) ** 2
            self.p8 = nn.Conv2d(96, 32, 1)
            self.p16 = nn.Conv2d(128, 32, 1)
            self.t8 = nn.Sequential(conv(c8 + 32 + 2, 128), Res(128), conv(128, 160, 3, 2))
            self.t16 = nn.Sequential(conv(160 + c16 + 32 + 2 + 16, 192), Res(192), Res(192), conv(192, 192, 3, 2), Res(192))
            self.s16 = nn.Linear(64, 16)
            self.squeeze = nn.Conv2d(192, 32, 1)
            self.register_buffer("ray8", rays(H // 8, W // 8, cam), persistent=False)
            self.register_buffer("ray16", rays(H // 16, W // 16, cam), persistent=False)
            feat = 32 * (H // 32) * (W // 32)
        else:
            feat = 0
        self.head = nn.Sequential(nn.Linear(feat + 64, 512), nn.GELU(), nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 12))

    def forward(self, img: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """img (B, 2, H, W) in [0, 1], x (B, SENSOR_DIM) → mean (B, 6), log-scale (B, 6)."""
        # The sensors and the read-out run in float32: bf16 would round the gyro and gravity
        # inputs near their noise and the outputs by ~0.4 %, the same way every step.
        with torch.autocast("cuda", enabled=False):
            s = self.sens(x.float())
        if self.blind:
            z = s
        else:
            B = img.shape[0]
            if tuple(img.shape[-2:]) != (self.cam.h, self.cam.w):
                raise ValueError(f"frames {tuple(img.shape[-2:])} do not fit the network's camera {self.cam}")
            f8, f16 = self.enc((img.reshape(B * 2, 1, self.cam.h, self.cam.w) - 0.45) / 0.25)
            f8 = f8.reshape(B, 2, *f8.shape[1:])
            f16 = f16.reshape(B, 2, *f16.shape[1:])
            c8 = local_corr(f8[:, 0], f8[:, 1], self.r8).to(f8.dtype)
            c16 = local_corr(f16[:, 0], f16[:, 1], self.r16).to(f16.dtype)
            ray8 = self.ray8.to(f8.dtype)[None].expand(B, -1, -1, -1)
            ray16 = self.ray16.to(f16.dtype)[None].expand(B, -1, -1, -1)
            y = self.t8(torch.cat([c8, self.p8(f8[:, 0]), ray8], 1))
            sb = self.s16(s).to(y.dtype)[:, :, None, None].expand(-1, -1, *y.shape[-2:])
            y = self.t16(torch.cat([y, c16, self.p16(f16[:, 0]), ray16, sb], 1))
            z = torch.cat([self.squeeze(y).flatten(1).float(), s], 1)
        with torch.autocast("cuda", enabled=False):
            out = self.head(z.float())
        return out[:, :6], out[:, 6:].clamp(-7, 5)


def laplace_nll(mu, log_b, y, w=(1, 1, 1, 0.3, 0.3, 0.3)):
    w = torch.tensor(w, device=y.device)
    return (((y - mu).abs() * torch.exp(-log_b) + log_b) * w).sum(1).mean()
