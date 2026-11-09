"""Run a camera-format checkpoint on a recorded flight's over-water stretches (written by
experiments/e4_varied_scenes/extract_real.py) and score its steps against the flight's GNSS.

    python -m e3.real --ckpt <dir>/best.pt --real <out>/real/<name> --out <dir> [--gyro gyro|att]

What the network gets per pair, as data.sensor_vector builds it in simulation (no GNSS here):
  gravity   from the recording's image attitude (only its tilt enters: R̃ᵀ(0, 0, −1))
  gyro      --gyro gyro (default): the camera's own gyro and stabilisation record (rot_gyro),
            less a constant bias, the median of rot_gyro − rot_att over the stretch's first 10 s
            (rot_att comes from the attitude, whose heading uses GNSS: the start of each stretch
            leans on GNSS for this one constant, as a system would calibrate before losing it);
            --gyro att: rot_att itself
  height    the recording's fused height above the sea, which it derives from the video; the
            recording's notes put its scale at ±10–15 %, so every metre the network reports
            carries that scale too
  airspeed  none: the checkpoint must have been trained with --no-airspeed

Scores per stretch, horizontal only, against GNSS (en, the camera's step = GNSS step):
  dir_err   angle between the predicted step, turned into the world by the attitude, and the
            GNSS step, deg: median of |·| and the median signed value (a steady bias)
  scale     |predicted step| / |GNSS step|: median, 10th and 90th percentiles
  drift     end error / distance flown, %, chaining the steps with the attitude ("att"; its
            heading uses GNSS) or with the network's own rotations from the first attitude
            ("chain"); drift_scaled is "att" after one best-fit scale per stretch, which
            separates the direction and shape of the track from its scale
Writes per_stretch.csv, summary.md and preds.npz in --out.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import D
from .evaluate import chain_rot
from .model import PoseNet, camera_of

DOWN = np.array([0.0, 0.0, -1.0])
BIAS_S = 10.0


def sensor_vectors(R: np.ndarray, gyro: np.ndarray, height: np.ndarray) -> np.ndarray:
    """(N−1, 14) as data.sensor_vector, airspeed entries zero."""
    n = len(gyro)
    x = np.zeros((n, 14), np.float32)
    x[:, 0:3] = np.einsum("nji,j->ni", R[:-1], DOWN)
    x[:, 3:6] = np.einsum("nji,j->ni", R[1:], DOWN)
    x[:, 6:9] = gyro / 0.05
    x[:, 9] = np.log(np.maximum(height[:-1], 5.0) / 100)
    return x


def predict(model, frames: np.ndarray, x: np.ndarray, batch: int = 128) -> tuple[np.ndarray, np.ndarray]:
    mus, lbs = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for a in range(0, len(x), batch):
            b = min(a + batch, len(x))
            img = np.stack([frames[a:b], frames[a + 1:b + 1]], 1).astype(np.float32) / 255.0
            mu, lb = model(torch.from_numpy(img).cuda(non_blocking=True), torch.from_numpy(x[a:b]).cuda())
            mus.append(mu.float().cpu().numpy())
            lbs.append(lb.float().cpu().numpy())
    return np.concatenate(mus), np.concatenate(lbs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--real", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gyro", choices=("gyro", "att"), default="gyro")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cuda", weights_only=False)
    if ck["blind"] or ck.get("airspeed", True):
        raise SystemExit(f"{a.ckpt}: needs a camera network trained with --no-airspeed (the recording has no airspeed sensor)")
    cam = camera_of(ck)
    model = PoseNet(cam=cam).cuda().eval()
    model.load_state_dict(ck["model"])
    real = Path(a.real)
    index = json.loads((real / "index.json").read_text())["stretches"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, saved = [], {}
    for st in index:
        sd = real / st["name"]
        z = np.load(sd / "real.npz")
        frames = np.load(sd / "frames.npy", mmap_mode="r")
        if frames.shape[1:] != (cam.h, cam.w):
            raise SystemExit(f"{sd}: frames {frames.shape[1:]} do not fit the network's camera {cam}")
        R, t, height, en = z["R"], z["t"], z["height"], z["en"]
        if a.gyro == "att":
            gyro = z["rot_att"]
        else:
            first = (t[:-1] - t[0]) < BIAS_S
            gyro = z["rot_gyro"] - np.median(z["rot_gyro"][first] - z["rot_att"][first], 0)
        x = sensor_vectors(R, gyro, height)
        mu, lb = predict(model, frames, x)
        h = np.maximum(height[:-1], 5.0)[:, None]
        dp = mu[:, :3] * h / 10                                   # metres, camera axes of frame i
        rot = gyro + mu[:, 3:] / 100 * D
        step_true = np.diff(en, axis=0)
        n = len(step_true)
        dist = np.linalg.norm(step_true, axis=1).sum()

        def drift(R_list, scale=1.0):
            p = np.cumsum(np.einsum("nij,nj->ni", R_list, dp)[:, :2] * scale, 0)   # positions of frames 1..N−1
            errs = np.linalg.norm(p - (en[1:] - en[0]), axis=1)
            return errs[-1] / dist * 100, errs.max() / dist * 100

        steps_att = np.einsum("nij,nj->ni", R[:-1], dp)[:, :2]
        ang = np.degrees(np.arctan2(steps_att[:, 0] * step_true[:, 1] - steps_att[:, 1] * step_true[:, 0], (steps_att * step_true).sum(1)))
        ratio = np.linalg.norm(steps_att, axis=1) / np.maximum(np.linalg.norm(step_true, axis=1), 1e-6)
        s_fit = (steps_att * step_true).sum() / (steps_att**2).sum()
        d_att, m_att = drift(R[:-1])
        d_scaled, m_scaled = drift(R[:-1], s_fit)
        R_chain = chain_rot(R[0], rot)[:-1]
        d_chain, m_chain = drift(R_chain)
        rows.append(dict(stretch=st["name"], seconds=st["seconds"], path_km=dist / 1000, height_med=float(np.median(height)),
                         dir_err_abs_med=float(np.median(np.abs(ang))), dir_err_signed_med=float(np.median(ang)),
                         scale_med=float(np.median(ratio)), scale_p10=float(np.percentile(ratio, 10)), scale_p90=float(np.percentile(ratio, 90)),
                         scale_fit=float(s_fit), drift_att=d_att, max_drift_att=m_att, drift_scaled=d_scaled, max_drift_scaled=m_scaled,
                         drift_chain=d_chain, max_drift_chain=m_chain, steps=n))
        saved[st["name"]] = dict(dp=dp, rot=rot, sig=np.exp(lb), ang=ang, ratio=ratio)
        print(json.dumps(rows[-1]), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_stretch.csv", index=False)
    np.savez(out / "preds.npz", ckpt=str(a.ckpt), gyro=a.gyro, **{f"{k}/{f}": v for k, d in saved.items() for f, v in d.items()})
    w = df.path_km / df.path_km.sum()
    with open(out / "summary.md", "w") as f:
        f.write(f"# {real.name}: {len(df)} over-water stretches, {df.path_km.sum():.1f} km, {df.seconds.sum() / 60:.1f} min\n\n")
        f.write(f"Checkpoint {a.ckpt} (epoch {ck['epoch']}), gyro input: {a.gyro}. Heights are the recording's video-derived "
                "height (scale ±10–15 %); the attitude's heading uses GNSS.\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".3f"))
        f.write(f"\n\nDistance-weighted: drift_att {np.sum(w * df.drift_att):.2f} %, drift_scaled {np.sum(w * df.drift_scaled):.2f} %, "
                f"drift_chain {np.sum(w * df.drift_chain):.2f} %, scale_fit {np.sum(w * df.scale_fit):.3f}\n")
    print(open(out / "summary.md").read())


if __name__ == "__main__":
    main()
