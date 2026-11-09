"""E1 (a): apparent sea-pattern velocity from off-the-shelf dense optical flow.

For each run, frame pairs (i, i+g), g in {1, 3}, i = 5, 10, 15, ... A pretrained flow
model (UFM-Base, SEA-RAFT-M) gives the flow f at every pixel p1 of frame i. With the TRUE
poses from frames.csv, p1 is back-projected (frame-i pose) and p1 + f (frame-(i+g) pose)
onto the plane up = 0 (mean sea level). d = X2 − X1 is how far the matched pattern moved
over the sea; the apparent pattern velocity is a = median(d) / Δt (component-wise median
over valid pixels). An estimator that assumes a static sea would report v_est = v_true − a,
so the velocity bias is −a.

Usage (on the GPU box, inside /workspace/venv):

  python flow_bias.py run --data /workspace/data/e1/pilot-frozen --out results/flow-pilot-frozen.csv
  python flow_bias.py run --data /workspace/data/e1/core --models ufm,searaft --out ...
  python flow_bias.py synth --run <frozen job dir> --frames 20,60,100 --out results/flow-synth.csv

`--data` may be a job directory, a directory of job directories, or a run directory; every
directory below it (depth ≤ 3) with run.json and frames.csv is a run. `--runs` filters run
ids by substring (comma-separated).

Models and the resolution each runs at:
- ufm      UFM-Base (Hugging Face infinity1096/UFM-Base) runs at its single trained
           resolution, 560×420 (W×H), and resizes whatever it gets to that, without keeping
           the aspect ratio. The 720×1280 portrait frame is therefore cropped to its central
           720×960 (rows 160–1119: a 3:4 window, the outer 160 rows at each end of the
           track are dropped) and transposed to 960×720 landscape, which UFM scales by
           0.583 in both axes. The flow is transposed back. Valid pixels: covisibility > 0.5.
- searaft  SEA-RAFT M (princeton-vl/SEA-RAFT, checkpoint
           MemorySlices/Tartan-C-T-TSKH-spring540x960-M, config spring-M, 4 iterations) on
           the full 720×1280 frame (scale 0). `searaft_half` runs it at 360×640 (scale −1,
           as the spring config does for 1080p). Its per-pixel Laplace scale is recorded
           (`conf_med`, expected log b); no confidence gate unless --searaft-max-logb.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e1geom  # noqa: E402

WAVE_DIR_DEG = 60.0  # fallback only: every E1 preset's wind sea travels toward 60°; run.json is read per run
SEA_RAFT_DIR = Path(os.environ.get("SEA_RAFT_DIR", "/workspace/models/SEA-RAFT"))
SEA_RAFT_CKPT = "MemorySlices/Tartan-C-T-TSKH-spring540x960-M"
UFM_CKPT = "infinity1096/UFM-Base"


# ---- run discovery and metadata ---------------------------------------------------------


def find_runs(paths, max_depth=3):
    runs = []
    for p in paths:
        p = Path(p)
        if (p / "run.json").exists() and (p / "frames.csv").exists():
            runs.append(p)
            continue
        for depth in range(1, max_depth + 1):
            for rj in sorted(p.glob("/".join(["*"] * depth) + "/run.json")):
                if (rj.parent / "frames.csv").exists():
                    runs.append(rj.parent)
    seen, out = set(), []
    for r in runs:
        if r.resolve() not in seen:
            seen.add(r.resolve())
            out.append(r)
    return out


def run_meta(run_dir: Path) -> dict:
    run = e1geom.load_run(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text()) if (run_dir / "meta.json").exists() else {}
    summary = [s.strip() for s in meta.get("summary", "").split("·")]
    tr = run["trajectory"]
    sea = run["sea"]
    light = run["light"]
    return {
        "set": meta.get("job", run_dir.parent.name),
        "run_id": run["id"],
        "sea": summary[0] if summary and summary[0] else "",
        "light": summary[1] if len(summary) > 1 else "",
        "sea_rate": run["render"].get("seaRate", 1),
        "wind_speed": sea.get("windSpeed"),
        "hs": (meta.get("seaStats") or {}).get("significantHeight"),
        "swell_height": sea.get("swellHeight"),
        # Sea directions are compass degrees the waves travel TOWARD (src/ocean/scene.js).
        "wind_dir": sea.get("windDirection", WAVE_DIR_DEG),
        "swell_dir": sea.get("swellDirection") if (sea.get("swellHeight") or 0) > 0 else float("nan"),
        "altitude": tr["altitude"],
        "airspeed": tr["airspeed"],
        "heading": tr["start"]["heading"],
        "pitch": tr["gimbal"]["pitch"],
        "sun_elevation": light.get("sunElevation"),
        "sun_azimuth": light.get("sunAzimuth"),
    }


def pair_list(n_frames, gaps=(1, 3), start=5, every=5):
    return [(i, g) for g in gaps for i in range(start, n_frames - g, every)]


# ---- frame loading ------------------------------------------------------------------------


def read_png(path):
    import cv2

    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


class RunDataset:
    """torch-style dataset: one item per run, all frames its pairs need, decoded."""

    def __init__(self, runs, gaps, start, every, mode="shaded"):
        self.runs, self.gaps, self.start, self.every, self.mode = runs, gaps, start, every, mode

    def __len__(self):
        return len(self.runs)

    def __getitem__(self, k):
        import cv2

        cv2.setNumThreads(1)
        rd = self.runs[k]
        rows = e1geom.read_frames(rd)
        pairs = pair_list(len(rows), self.gaps, self.start, self.every)
        need = sorted({i for i, _ in pairs} | {i + g for i, g in pairs})
        fdir = rd / "frames" if self.mode == "shaded" else rd / "frames" / self.mode
        imgs = np.stack([read_png(fdir / f"{i:06d}.png") for i in need])
        return {"run_dir": str(rd), "rows": rows, "pairs": pairs, "need": need, "imgs": imgs}


# ---- models -------------------------------------------------------------------------------


UFM_VARIANTS = {
    # name: (Hugging Face checkpoint, refinement head?)
    "ufm": (UFM_CKPT, False),
    "ufm980": ("infinity1096/UFM-Base-980", False),
    "ufmrefine": ("infinity1096/UFM-Refine", True),
    "ufmrefine980": ("infinity1096/UFM-Refine-980", True),
}


class UFMModel:
    def __init__(self, device, crop_rows=(160, 1120), cov_thr=0.5, chunk=24, name="ufm"):
        from uniflowmatch.models.ufm import UniFlowMatchClassificationRefinement, UniFlowMatchConfidence

        ckpt, refine = UFM_VARIANTS[name]
        cls = UniFlowMatchClassificationRefinement if refine else UniFlowMatchConfidence
        self.m = cls.from_pretrained(ckpt).to(device).eval()
        self.name = name
        self.crop = crop_rows
        self.cov_thr = cov_thr
        self.chunk = chunk if "980" not in name else max(1, chunk // 4)
        res = getattr(getattr(self.m, "config", None), "inference_resolution", None) or getattr(
            self.m, "inference_resolution", None)
        self.desc = (f"{ckpt} at its inference resolution {res} (W×H) "
                     f"(crop rows {crop_rows[0]}-{crop_rows[1] - 1}, transposed)")

    def __call__(self, im1, im2):
        """im1, im2: (B, 3, H, W) uint8 on device. Returns flow (B, 2, H, W) float32 in pixels
        (NaN outside the crop), a validity mask (B, H, W) and a confidence (B, H, W)."""
        import torch

        B, _, H, W = im1.shape
        r0, r1 = self.crop
        flow = torch.full((B, 2, H, W), float("nan"), device=im1.device)
        conf = torch.zeros((B, H, W), device=im1.device)
        for s in range(0, B, self.chunk):
            a = im1[s:s + self.chunk, :, r0:r1, :].transpose(2, 3).contiguous()
            b = im2[s:s + self.chunk, :, r0:r1, :].transpose(2, 3).contiguous()
            with torch.no_grad():
                res = self.m.predict_correspondences_batched(source_image=a, target_image=b)
            f = res.flow.flow_output.float().transpose(2, 3)  # (b, 2, r1-r0, W): ch0 = d(transposed x) = dv
            flow[s:s + self.chunk, 0, r0:r1, :] = f[:, 1]
            flow[s:s + self.chunk, 1, r0:r1, :] = f[:, 0]
            conf[s:s + self.chunk, r0:r1, :] = res.covisibility.mask.float().transpose(1, 2)
        valid = conf > self.cov_thr
        return flow, valid, conf


class SeaRaftModel:
    def __init__(self, device, scale=0, iters=4, chunk=8, max_logb=None, name="searaft"):
        import torch

        sys.path.insert(0, str(SEA_RAFT_DIR / "core"))
        sys.path.insert(0, str(SEA_RAFT_DIR))
        from raft import RAFT  # noqa: E402

        args = argparse.Namespace(**json.loads((SEA_RAFT_DIR / "config/eval/spring-M.json").read_text()))
        args.iters = iters
        self.args = args
        self.m = RAFT.from_pretrained(SEA_RAFT_CKPT, args=args).to(device).eval()
        self.scale = scale
        self.chunk = chunk
        self.max_logb = max_logb
        self.name = name
        self.torch = torch
        self.desc = f"SEA-RAFT-M {SEA_RAFT_CKPT}, iters {iters}, scale 2^{scale}"

    def heat(self, info):
        a = self.args
        raw_b = info[:, 2:]
        log_b = self.torch.zeros_like(raw_b)
        weight = info[:, :2].softmax(dim=1)
        log_b[:, 0] = self.torch.clamp(raw_b[:, 0], min=0, max=a.var_max)
        log_b[:, 1] = self.torch.clamp(raw_b[:, 1], min=a.var_min, max=0)
        return (log_b * weight).sum(dim=1)

    def __call__(self, im1, im2):
        import torch
        import torch.nn.functional as F

        B, _, H, W = im1.shape
        flow = torch.empty((B, 2, H, W), device=im1.device)
        conf = torch.empty((B, H, W), device=im1.device)
        for s in range(0, B, self.chunk):
            a = im1[s:s + self.chunk].float()
            b = im2[s:s + self.chunk].float()
            if self.scale != 0:
                a = F.interpolate(a, scale_factor=2 ** self.scale, mode="bilinear", align_corners=False)
                b = F.interpolate(b, scale_factor=2 ** self.scale, mode="bilinear", align_corners=False)
            with torch.no_grad():
                out = self.m(a, b, iters=self.args.iters, test_mode=True)
            f = out["flow"][-1]
            info = out["info"][-1]
            if self.scale != 0:
                f = F.interpolate(f, size=(H, W), mode="bilinear", align_corners=False) * (0.5 ** self.scale)
                info = F.interpolate(info, size=(H, W), mode="area")
            flow[s:s + self.chunk] = f
            conf[s:s + self.chunk] = self.heat(info)
        valid = torch.ones((B, H, W), dtype=torch.bool, device=im1.device)
        if self.max_logb is not None:
            valid &= conf <= self.max_logb
        return flow, valid, conf


def make_models(names, device, args):
    out = []
    for n in names:
        if n in UFM_VARIANTS:
            out.append(UFMModel(device, cov_thr=args.ufm_cov, name=n))
        elif n.startswith("searaft"):
            # searaft (full frame), searaft_half (scale −1), or searaft_s<scale>_i<iters>, e.g. searaft_s-1_i12
            scale, iters = (-1 if n == "searaft_half" else 0), args.searaft_iters
            for tok in n.split("_")[1:]:
                if tok.startswith("s") and tok != "s":
                    scale = int(tok[1:])
                elif tok.startswith("i"):
                    iters = int(tok[1:])
                elif tok != "half":
                    raise SystemExit(f"bad model spec {n}")
            out.append(SeaRaftModel(device, scale=scale, iters=iters, max_logb=args.searaft_max_logb, name=n))
        else:
            raise SystemExit(f"unknown model {n}")
    return out


# ---- flow → apparent velocity on the sea plane --------------------------------------------


def pose_tensors(poses, device):
    import torch

    K = torch.tensor(np.stack([p.K for p in poses]), dtype=torch.float64, device=device)
    R = torch.tensor(np.stack([p.R for p in poses]), dtype=torch.float64, device=device)
    t = torch.tensor(np.stack([p.t for p in poses]), dtype=torch.float64, device=device)
    return K, R, t


def nanmedian_rows(x):
    """Median of each row of x (B, M), ignoring NaN; NaN if a row is all NaN."""
    return x.nanmedian(dim=1).values


def apparent_velocity(flow, valid_model, poses1, poses2, dt, margin=32.0, max_incidence_deg=75.0,
                      plane_up=0.0, stride=1):
    """flow (B, 2, H, W) in pixels (torch, any float); poses1/poses2 lists of e1geom.Pose; dt (B,)
    seconds. Returns a dict of (B,) numpy arrays: a_E, a_N (median d/Δt, m/s), the same without the
    model's confidence gate, the robust spread (1.4826·MAD) of each component, valid counts and
    fractions, the median flow magnitude and the median distance (px) between the model's flow and
    the flow a static sea would give (`static_res_px`)."""
    import torch

    dev = flow.device
    B, _, H, W = flow.shape
    vs = torch.arange(0, H, stride, device=dev, dtype=torch.float64) + 0.5
    us = torch.arange(0, W, stride, device=dev, dtype=torch.float64) + 0.5
    v1, u1 = torch.meshgrid(vs, us, indexing="ij")
    u1 = u1.expand(B, -1, -1)
    v1 = v1.expand(B, -1, -1)
    f = flow[:, :, ::stride, ::stride].double()
    vm = valid_model[:, ::stride, ::stride]
    u2 = u1 + f[:, 0]
    v2 = v1 + f[:, 1]
    K1, R1, t1 = pose_tensors(poses1, dev)
    K2, R2, t2 = pose_tensors(poses2, dev)
    X1, rng1, ci1, ok1 = e1geom.backproject_torch(K1, R1, t1, u1, v1, plane_up)
    X2, rng2, ci2, ok2 = e1geom.backproject_torch(K2, R2, t2, u2, v2, plane_up)
    cmin = math.cos(math.radians(max_incidence_deg))
    inside2 = (u2 >= margin) & (u2 <= W - margin) & (v2 >= margin) & (v2 <= H - margin)
    inside1 = (u1 >= margin) & (u1 <= W - margin) & (v1 >= margin) & (v1 <= H - margin)
    geo = ok1 & ok2 & (ci1 >= cmin) & (ci2 >= cmin) & inside1 & inside2 & torch.isfinite(f).all(1)
    full = geo & vm
    dtt = torch.as_tensor(np.asarray(dt, dtype=np.float64), device=dev).view(-1, 1, 1)
    d = (X2 - X1) / dtt[..., None]  # (B, h, w, 2) m/s

    def med(mask):
        dd = d.clone()
        dd[~mask] = float("nan")
        e = nanmedian_rows(dd[..., 0].reshape(B, -1))
        n = nanmedian_rows(dd[..., 1].reshape(B, -1))
        return e, n, dd

    aE, aN, dd = med(full)
    aE0, aN0, _ = med(geo)
    madE = nanmedian_rows((dd[..., 0] - aE.view(-1, 1, 1)).abs().reshape(B, -1)) * 1.4826
    madN = nanmedian_rows((dd[..., 1] - aN.view(-1, 1, 1)).abs().reshape(B, -1)) * 1.4826

    # The flow a static sea would give: X1 seen from pose 2.
    Pc = torch.stack([X1[..., 0] - t2[:, 0].view(-1, 1, 1), X1[..., 1] - t2[:, 1].view(-1, 1, 1),
                      -t2[:, 2].view(-1, 1, 1) + plane_up + 0 * X1[..., 0]], -1)
    pc = torch.einsum("bhwi,bij->bhwj", Pc, R2)  # R_wcᵀ (P − t) in row form
    us_ = K2[:, 0, 0].view(-1, 1, 1) * pc[..., 0] / pc[..., 2] + K2[:, 0, 2].view(-1, 1, 1)
    vs_ = K2[:, 1, 1].view(-1, 1, 1) * pc[..., 1] / pc[..., 2] + K2[:, 1, 2].view(-1, 1, 1)
    res = torch.hypot(u2 - us_, v2 - vs_)
    res[~full] = float("nan")
    fm = torch.hypot(f[:, 0], f[:, 1])
    fm[~full] = float("nan")
    n_full = full.reshape(B, -1).sum(1)
    tot = float(H * W) / (stride * stride)
    to = lambda x: x.detach().cpu().numpy()  # noqa: E731
    return {
        "a_E": to(aE), "a_N": to(aN), "a_E_nogate": to(aE0), "a_N_nogate": to(aN0),
        "spread_E": to(madE), "spread_N": to(madN),
        "n_valid": to(n_full), "valid_frac": to(n_full / tot), "geo_frac": to(geo.reshape(B, -1).sum(1) / tot),
        "flow_px_med": to(nanmedian_rows(fm.reshape(B, -1))),
        "static_res_px": to(nanmedian_rows(res.reshape(B, -1))),
    }


def decompose(aE, aN, heading, wave_dir=WAVE_DIR_DEG, swell_dir=float("nan")):
    """Along-track, cross-track (+ right of track), along the wind sea's travel direction
    (`wave_dir`, column a_along_wave) and along the swell's (`swell_dir`; NaN when there is no
    swell, column a_along_swell)."""
    along_u, cross_u = e1geom.track_axes(heading)
    w = e1geom.bearing_unit(wave_dir)
    a = np.stack([aE, aN], -1)
    sw = a @ e1geom.bearing_unit(swell_dir) if np.isfinite(swell_dir) else np.full(a.shape[:-1], np.nan)
    return a @ along_u, a @ cross_u, a @ w, sw


# ---- the run command ----------------------------------------------------------------------

FIELDS = ["set", "run_id", "model", "i", "g", "dt", "t_i", "sea", "light", "sea_rate", "wind_speed", "hs",
          "swell_height", "wind_dir", "swell_dir", "altitude", "cam_u", "airspeed", "heading", "pitch", "sun_elevation", "sun_azimuth",
          "ground_speed", "cam_vE", "cam_vN", "a_E", "a_N", "a_abs", "a_along", "a_cross", "a_along_wave",
          "a_along_swell", "a_pct_of_speed", "bias_along", "bias_cross", "a_E_nogate", "a_N_nogate", "spread_E", "spread_N",
          "valid_frac", "geo_frac", "n_valid", "flow_px_med", "static_res_px", "conf_med", "flow_failed",
          "model_ms_per_pair", "mode"]
# A pair is flagged as a flow failure when the model's flow has collapsed toward zero, i.e. the
# apparent pattern velocity a is within FAIL_COLLAPSE·ground speed of the camera's own velocity
# (SEA-RAFT at full resolution on ~190 px of flow returns a near-zero flow, so a ≈ v_cam), or when
# the per-pixel velocities scatter by more than FAIL_SPREAD m/s (1.4826·MAD, either component):
# partial failures, where part of the frame locked onto the wrong displacement. On the pilot,
# pairs that are not failures have a spread of at most 1.94 m/s; partial failures 2.3–38 m/s.
# (The pilot rule was |a| ≥ 0.25·ground speed; it was replaced before the full runs because a
# real pattern speed of a few m/s at 15 m/s ground speed and 250 m would trip it. The pilot
# failures all satisfy the new rule too.) Rows keep every column, so a different rule can be
# applied afterwards.
FAIL_COLLAPSE = 0.25
FAIL_SPREAD = 3.0


def is_failed(a_E, a_N, cam_vE, cam_vN, spread_E, spread_N):
    gs = np.hypot(cam_vE, cam_vN)
    collapse = np.hypot(np.asarray(a_E) - cam_vE, np.asarray(a_N) - cam_vN) < FAIL_COLLAPSE * gs
    scatter = np.maximum(spread_E, spread_N) > FAIL_SPREAD
    return np.asarray(collapse | scatter, dtype=bool)


def process_run(item, models, device, writer, args):
    import torch

    rd = Path(item["run_dir"])
    meta = run_meta(rd)
    rows = item["rows"]
    poses = [e1geom.pose_from_row(r) for r in rows]
    pairs = item["pairs"]
    idx = {f: k for k, f in enumerate(item["need"])}
    imgs = torch.from_numpy(item["imgs"]).to(device, non_blocking=True).permute(0, 3, 1, 2).contiguous()
    i1 = torch.tensor([idx[i] for i, g in pairs], device=device)
    i2 = torch.tensor([idx[i + g] for i, g in pairs], device=device)
    im1, im2 = imgs[i1], imgs[i2]
    p1 = [poses[i] for i, g in pairs]
    p2 = [poses[i + g] for i, g in pairs]
    dt = np.array([b.time - a.time for a, b in zip(p1, p2)])
    cam_v = np.array([(b.t[:2] - a.t[:2]) for a, b in zip(p1, p2)]) / dt[:, None]
    gs = np.hypot(cam_v[:, 0], cam_v[:, 1])
    timings = {}
    for m in models:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        flow, valid, conf = m(im1, im2)
        torch.cuda.synchronize()
        t_model = time.perf_counter() - t0
        if args.mask_min_lum is not None:
            # e.g. the foam label mode: only pixels of frame i brighter than the threshold count
            valid = valid & (im1.float().mean(1) > args.mask_min_lum)
        out = apparent_velocity(flow, valid, p1, p2, dt, margin=args.margin, max_incidence_deg=args.max_incidence,
                                stride=args.stride)
        confm = conf.clone().float()
        confm[~valid] = float("nan")
        conf_med = confm.reshape(len(pairs), -1).nanmedian(1).values.cpu().numpy()
        torch.cuda.synchronize()
        timings[m.name] = (t_model, time.perf_counter() - t0)
        along, cross, alw, als = decompose(out["a_E"], out["a_N"], meta["heading"], meta["wind_dir"],
                                           meta["swell_dir"])
        for k, (i, g) in enumerate(pairs):
            aE, aN = float(out["a_E"][k]), float(out["a_N"][k])
            row = dict(meta)
            row.update({
                "model": m.name, "i": i, "g": g, "dt": dt[k], "t_i": p1[k].time, "cam_u": p1[k].t[2],
                "ground_speed": gs[k], "cam_vE": cam_v[k, 0], "cam_vN": cam_v[k, 1],
                "a_E": aE, "a_N": aN, "a_abs": math.hypot(aE, aN), "a_along": along[k], "a_cross": cross[k],
                "a_along_wave": alw[k], "a_along_swell": als[k], "a_pct_of_speed": 100 * math.hypot(aE, aN) / gs[k],
                "bias_along": -along[k], "bias_cross": -cross[k],
                "flow_failed": int(is_failed(aE, aN, cam_v[k, 0], cam_v[k, 1], float(out["spread_E"][k]),
                                             float(out["spread_N"][k]))),
                "conf_med": conf_med[k], "model_ms_per_pair": 1000 * t_model / len(pairs), "mode": args.mode,
            })
            for key in ["a_E_nogate", "a_N_nogate", "spread_E", "spread_N", "valid_frac", "geo_frac", "n_valid",
                        "flow_px_med", "static_res_px"]:
                row[key] = float(out[key][k])
            writer.writerow({k2: (f"{v:.6g}" if isinstance(v, float) else v) for k2, v in row.items() if k2 in FIELDS})
        if args.dump_dir:
            dump = Path(args.dump_dir) / f"{meta['run_id']}__{m.name}.npz"
            dump.parent.mkdir(parents=True, exist_ok=True)
            k = 0
            np.savez_compressed(dump, flow=flow[k, :, ::4, ::4].float().cpu().numpy(),
                                conf=conf[k, ::4, ::4].float().cpu().numpy(), pair=np.array(pairs[k]))
    return timings, len(pairs)


def cmd_run(args):
    import torch

    runs = find_runs(args.data)
    if args.runs:
        keys = args.runs.split(",")
        runs = [r for r in runs if any(k in r.name for k in keys)]
    if args.limit:
        runs = runs[: args.limit]
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        runs = runs[k::n]
    if not runs:
        raise SystemExit("no runs found")
    device = torch.device("cuda")
    models = make_models(args.models.split(","), device, args)
    for m in models:
        print(f"[model] {m.name}: {m.desc}", flush=True)
    gaps = tuple(int(g) for g in args.gaps.split(","))
    ds = RunDataset(runs, gaps, args.start, args.every, args.mode)
    loader = torch.utils.data.DataLoader(ds, batch_size=None, shuffle=False, num_workers=args.workers,
                                         prefetch_factor=2 if args.workers else None,
                                         collate_fn=lambda x: x, persistent_workers=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    tot_pairs = 0
    agg = {m.name: [0.0, 0.0] for m in models}
    t_wait = 0.0
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        t_last = time.perf_counter()
        for k, item in enumerate(loader):
            t_wait += time.perf_counter() - t_last
            timings, n = process_run(item, models, device, w, args)
            fh.flush()
            tot_pairs += n
            for name, (tm, tt) in timings.items():
                agg[name][0] += tm
                agg[name][1] += tt
            el = time.perf_counter() - t_start
            print(f"[{k + 1}/{len(runs)}] {Path(item['run_dir']).name} pairs {n} "
                  + " ".join(f"{nm} {1000 * tm / n:.1f}ms/pair" for nm, (tm, tt) in timings.items())
                  + f" elapsed {el:.0f}s", flush=True)
            t_last = time.perf_counter()
    el = time.perf_counter() - t_start
    summary = {"runs": len(runs), "pairs_per_model": tot_pairs, "wall_s": el, "loader_wait_s": t_wait,
               "models": {nm: {"model_ms_per_pair": 1000 * a / tot_pairs, "total_ms_per_pair": 1000 * b / tot_pairs,
                               "desc": next(m.desc for m in models if m.name == nm)} for nm, (a, b) in agg.items()},
               "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    print("[timing] " + json.dumps(summary), flush=True)
    Path(str(out) + ".timing.json").write_text(json.dumps(summary, indent=2))


# ---- the synth command: a frozen frame warped by a known camera move and sea motion -------


def synth_pair(img, pose1, D, W_sea, dt, plane_up=0.0):
    """Frame 2 seen from pose1 translated by D (3,), over a sea whose pattern moved by W_sea·dt
    (E, N) since frame 1 (img, pose1). Pixels of frame 2 that come from outside frame 1 are 0
    and flagged. Returns (img2, pose2, covered mask)."""
    import cv2

    pose2 = e1geom.Pose(K=pose1.K, R=pose1.R, t=pose1.t + np.asarray(D, float), time=pose1.time + dt,
                        width=pose1.width, height=pose1.height)
    Hh, Ww = img.shape[:2]
    u, v = e1geom.pixel_grid(Ww, Hh)
    X, ok = e1geom.backproject(pose2, u, v, plane_up)
    Xs = X.copy()
    Xs[..., 0] -= W_sea[0] * dt
    Xs[..., 1] -= W_sea[1] * dt
    us, vs, z = e1geom.project(pose1, Xs)
    mapx = (us - 0.5).astype(np.float32)  # cv2.remap samples at pixel-index coordinates
    mapy = (vs - 0.5).astype(np.float32)
    out = cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    covered = ok & (z > 0) & (us >= 0.5) & (us <= Ww - 0.5) & (vs >= 0.5) & (vs <= Hh - 0.5)
    out[~covered] = 0
    return out, pose2, covered


def cmd_synth(args):
    """Each listed frame of each run, warped by the run's own camera move over g frames plus a known
    sea-pattern motion W: the pipeline must return a = W. The warp is exact geometry (no shading
    change), so what remains is the flow model's own error."""
    import torch

    device = torch.device("cuda")
    models = make_models(args.models.split(","), device, args)
    run_dirs = []
    for r in args.run:
        run_dirs += find_runs([r])
    frames = [int(f) for f in args.frames.split(",")]
    W_cases = [(0.0, 0.0), (1.0, 0.0), (0.0, -1.5), (0.7, 0.7)]
    out_rows = []
    for rd in run_dirs:
        rows = e1geom.read_frames(rd)
        meta = run_meta(rd)
        for fr in frames:
            p1 = e1geom.pose_from_row(rows[fr])
            img = read_png(rd / "frames" / f"{fr:06d}.png")
            for g in (1, 3):
                pg = e1geom.pose_from_row(rows[fr + g])
                D = pg.t - p1.t  # the run's own camera move over g frames
                dt = pg.time - p1.time
                for W_sea in W_cases:
                    img2, pose2, cov = synth_pair(img, p1, D, W_sea, dt)
                    im1 = torch.from_numpy(img).permute(2, 0, 1)[None].to(device)
                    im2 = torch.from_numpy(img2).permute(2, 0, 1)[None].to(device)
                    for m in models:
                        flow, valid, conf = m(im1, im2)
                        o = apparent_velocity(flow, valid, [p1], [pose2], np.array([dt]), margin=args.margin,
                                              max_incidence_deg=args.max_incidence)
                        gs = float(np.hypot(D[0], D[1]) / dt)
                        aE, aN = float(o["a_E"][0]), float(o["a_N"][0])
                        eal, ecr, _, _ = decompose(np.array([aE - W_sea[0]]), np.array([aN - W_sea[1]]), meta["heading"])
                        r = {"run_id": meta["run_id"], "altitude": meta["altitude"], "airspeed": meta["airspeed"],
                             "heading": meta["heading"], "sea": meta["sea"], "frame": fr, "g": g, "model": m.name,
                             "D_E": D[0], "D_N": D[1], "dt": dt, "ground_speed": gs, "W_E": W_sea[0], "W_N": W_sea[1],
                             "a_E": aE, "a_N": aN, "err_E": aE - W_sea[0], "err_N": aN - W_sea[1],
                             "err_along": float(eal[0]), "err_cross": float(ecr[0]),
                             "err_abs": math.hypot(aE - W_sea[0], aN - W_sea[1]),
                             "failed": int(is_failed(aE, aN, D[0] / dt, D[1] / dt, float(o["spread_E"][0]),
                                                     float(o["spread_N"][0]))),
                             "valid_frac": float(o["valid_frac"][0]), "flow_px_med": float(o["flow_px_med"][0])}
                        out_rows.append(r)
            print(f"[synth] {meta['run_id']} frame {fr} done", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        for r in out_rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--models", default="ufm,searaft_half,searaft")
    common.add_argument("--margin", type=float, default=32.0, help="exclude pixels this close to the image edge (px)")
    common.add_argument("--max-incidence", type=float, default=75.0, help="max ray angle from vertical at the sea (deg)")
    common.add_argument("--ufm-cov", type=float, default=0.5, help="UFM covisibility threshold")
    common.add_argument("--searaft-iters", type=int, default=4)
    common.add_argument("--searaft-max-logb", type=float, default=None, help="SEA-RAFT gate on expected log b")
    common.add_argument("--out", required=True)
    r = sub.add_parser("run", parents=[common])
    r.add_argument("--data", nargs="+", required=True)
    r.add_argument("--runs", default="")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--gaps", default="1,3")
    r.add_argument("--start", type=int, default=5)
    r.add_argument("--every", type=int, default=5)
    r.add_argument("--stride", type=int, default=1, help="evaluate every n-th pixel (both axes)")
    r.add_argument("--workers", type=int, default=12)
    r.add_argument("--dump-dir", default="")
    r.add_argument("--mode", default="shaded", help="frames/ (shaded) or a label mode written under frames/<mode>/")
    r.add_argument("--mask-min-lum", type=float, default=None,
                   help="count only pixels of frame i whose mean RGB exceeds this (0-255), e.g. foam in the foam mode")
    r.add_argument("--shard", default="", help="k/n: process every n-th run starting at k (run n shards side by side)")
    s = sub.add_parser("synth", parents=[common])
    s.add_argument("--run", nargs="+", required=True, help="run dirs or dirs holding runs")
    s.add_argument("--frames", default="20,60,100")
    args = ap.parse_args(argv)
    {"run": cmd_run, "synth": cmd_synth}[args.cmd](args)


if __name__ == "__main__":
    main()
