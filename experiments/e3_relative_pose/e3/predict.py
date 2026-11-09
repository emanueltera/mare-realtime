"""Run a checkpoint over every consecutive pair of a set, with the set's fixed per-run sensor
noise, and save what it predicts.

    python -m e3.predict --ckpt <dir>/best.pt --set <sets>/<set> --out <preds>.npz

The npz holds, per pair: run (index into `runs`), i, dp_pred and sig_t (metres, camera
axes of frame i; sig is the Laplace scale), rot_pred and sig_r (radians; the gyro's rotation
plus the predicted correction).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import D, PairDataset, parse_noise
from .model import PoseNet, camera_of


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--set", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--noise", action="append", help="sensor noise override key=value (see data.parse_noise); repeatable")
    p.add_argument("--fx", action="store_true", help="through the camera effects, one draw per run (e3/fx.py)")
    a = p.parse_args()
    ck = torch.load(a.ckpt, map_location="cuda", weights_only=False)
    cam = camera_of(ck)
    model = PoseNet(blind=ck["blind"], cam=cam).cuda().eval()
    model.load_state_dict(ck["model"])
    noise = parse_noise(a.noise)
    air = ck.get("airspeed", True)
    ds = PairDataset(a.set, train=False, images=not ck["blind"], noise=noise, fx=a.fx, airspeed=air)
    if not ck["blind"] and not ds.cam.matches(cam):
        raise SystemExit(f"{a.ckpt} was trained for camera {cam}; {a.set} has {ds.cam}")
    dl = DataLoader(ds, 128, shuffle=False, num_workers=24, pin_memory=True)
    out = {k: [] for k in ("run", "i", "dp_pred", "sig_t", "rot_pred", "sig_r")}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for img, x, y, h, r, i in dl:
            mu, lb = model(img.cuda(non_blocking=True), x.cuda())
            mu, b = mu.float().cpu().numpy(), torch.exp(lb).float().cpu().numpy()
            hm = h.numpy()[:, None] / 10
            gyro = np.stack([ds.sensors(int(rr)).gyro[int(ii)] for rr, ii in zip(r, i)])
            out["run"].append(r.numpy())
            out["i"].append(i.numpy())
            out["dp_pred"].append(mu[:, :3] * hm)
            out["sig_t"].append(b[:, :3] * hm)
            out["rot_pred"].append(gyro + mu[:, 3:] / 100 * D)
            out["sig_r"].append(b[:, 3:] / 100 * D)
    arrs = {k: np.concatenate(v) for k, v in out.items()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, runs=np.array([r.id for r in ds.runs]), ckpt=str(a.ckpt), epoch=ck["epoch"], noise=json.dumps(noise), fx=a.fx, airspeed=air, **arrs)
    print(f"{a.out}: {len(arrs['i'])} pairs from {len(ds.runs)} runs", flush=True)


if __name__ == "__main__":
    main()
