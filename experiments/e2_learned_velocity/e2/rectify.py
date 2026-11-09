"""Rectify rendered runs into an E2 tile set (see geom.py for the tile and README.md for the
layout on disk).

    # stream: rectify each run as soon as the generator marks it complete, verify, delete its PNGs
    python -m e2.rectify stream --render /workspace/data/e2/render/train/e2-train \
        --out /workspace/data/e2/sets/train --set train --keep 5 --workers 12 --done-flag <file>
    # batch: every complete run now (E1 core: keep all PNGs)
    python -m e2.rectify batch --render /workspace/data/e1/core/e1-core --out /workspace/data/e2/sets/e1-core \
        --set e1-core --keep all --workers 24
    # tables: (re)build frames.csv, labels.csv, runs.csv and set.json from the per-run files
    python -m e2.rectify finalize --out /workspace/data/e2/sets/train --set train

Per run it writes tiles/<run_id>.npy (float16 [F, N, N], linear luminance, NaN = no image
there) and meta/<run_id>.json (run metadata, tile geometry and per-frame truth). A run's PNGs
are deleted only after its tiles are verified: the .npy reloads with the right shape, its
valid fractions match, and two frames rectified again from the PNGs match it.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from . import geom

N_DEFAULT = 384
FORMAT_VERSION = 1


def theta_key(set_name: str, run_id: str) -> str:
    """test-rate runs of one condition (…-c007-rate0p5) share their tile orientation."""
    m = re.search(r"-(c\d+)-rate", run_id)
    return f"{set_name}:{m.group(1)}" if m else f"{set_name}:{run_id}"


def rectify_run(run_dir: Path, set_name: str, n: int = N_DEFAULT, frames=None):
    """→ (tiles float32 [F, n, n] with NaN, meta dict)."""
    run_dir = Path(run_dir)
    rows = geom.read_frames(run_dir)
    spec = geom.load_run(run_dir)
    theta = geom.theta_for(theta_key(set_name, spec["id"]))
    tile = geom.run_tile(rows, n, theta)
    offsets = tile.cell_offsets()
    idx = range(len(rows)) if frames is None else frames
    tiles, valid, ratio = [], [], []
    for i in idx:
        r = rows[i]
        u, v, rat = geom.pixel_map(r, tile, offsets)
        img = geom.luminance(run_dir / "frames" / f"{int(r['frame']):06d}.png")
        t = geom.rectify_frame(img, u, v, rat)
        tiles.append(t)
        valid.append(float(np.isfinite(t).mean()))
        ratio.append(rat)
    tiles = np.stack(tiles)
    meta = run_meta(run_dir, spec, rows, tile, set_name)
    meta["frames"]["valid_frac"] = [round(x, 6) for x in valid]
    meta["frames"]["aa_ratio"] = [round(x, 5) for x in ratio]
    return tiles, meta


def run_meta(run_dir: Path, spec: dict, rows: list[dict], tile: geom.Tile, set_name: str) -> dict:
    p = np.array([[r["cam_e"], r["cam_n"]] for r in rows])
    t = np.array([r["t"] for r in rows])
    steps = np.diff(p, axis=0)
    dist = float(np.linalg.norm(steps, axis=1).sum())
    traj = spec["trajectory"]
    pattern = traj["pattern"]
    return {
        "format": FORMAT_VERSION,
        "set": set_name,
        "run_id": spec["id"],
        "render_dir": str(run_dir),
        "tile": {"n": tile.n, "gsd": tile.gsd, "side": tile.side, "theta_deg": tile.theta,
                 "c_run_e": tile.c_run[0], "c_run_n": tile.c_run[1]},
        "run": spec,
        "summary": {
            "n_frames": len(rows), "duration": float(t[-1] - t[0]), "distance": dist,
            "ground_speed_mean": dist / float(t[-1] - t[0]),
            "cam_u_mean": float(np.mean([r["cam_u"] for r in rows])),
            "pattern_type": pattern["type"] if isinstance(pattern, dict) else str(pattern),
            "fx": rows[0]["fx"], "width": int(rows[0]["width"]), "height": int(rows[0]["height"]),
        },
        "frames": {
            "frame": [int(r["frame"]) for r in rows],
            "t": [r["t"] for r in rows],
            "cam_e": [r["cam_e"] for r in rows],
            "cam_n": [r["cam_n"] for r in rows],
            "cam_u": [r["cam_u"] for r in rows],
            "aircraft_heading": [r.get("aircraft_heading", float("nan")) for r in rows],
            "aircraft_bank": [r.get("aircraft_bank", float("nan")) for r in rows],
        },
    }


def verify(run_dir: Path, out: Path, run_id: str, set_name: str, n: int, check_frames=(0, -1)) -> str | None:
    """None if the saved tiles are good, else what is wrong."""
    try:
        a = np.load(out / "tiles" / f"{run_id}.npy", mmap_mode="r")
        meta = json.loads((out / "meta" / f"{run_id}.json").read_text())
        nf = meta["summary"]["n_frames"]
        if a.shape != (nf, n, n) or a.dtype != np.float16:
            return f"shape/dtype {a.shape} {a.dtype}"
        vf = np.isfinite(a).reshape(nf, -1).mean(1)
        if np.abs(vf - np.array(meta["frames"]["valid_frac"])).max() > 1e-5:
            return "valid fractions differ"
        frames = sorted({f % nf for f in check_frames})
        again, _ = rectify_run(run_dir, set_name, n, frames=frames)
        saved = a[frames].astype(np.float32)
        both = np.isfinite(again) & np.isfinite(saved)
        if (np.isfinite(again) != np.isfinite(saved)).any():
            return "masks differ on re-rectification"
        err = np.abs(again[both] - saved[both]) / np.maximum(np.abs(again[both]), 1e-3)
        if both.any() and err.max() > 2e-3:  # float16 keeps ~1e-3 relative
            return f"re-rectified frames differ (max rel {err.max():.2e})"
    except Exception as e:  # noqa: BLE001 - reported, not raised
        return f"{type(e).__name__}: {e}"
    return None


def process(run_dir: str, out: str, set_name: str, n: int, delete_pngs: bool) -> dict:
    run_dir, out = Path(run_dir), Path(out)
    t0 = time.time()
    tiles, meta = rectify_run(run_dir, set_name, n)
    rid = meta["run_id"]
    (out / "tiles").mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    tmp = out / "tiles" / f".{rid}.tmp.npy"
    np.save(tmp, tiles.astype(np.float16))
    os.replace(tmp, out / "tiles" / f"{rid}.npy")
    (out / "meta" / f".{rid}.tmp").write_text(json.dumps(meta))
    os.replace(out / "meta" / f".{rid}.tmp", out / "meta" / f"{rid}.json")
    problem = verify(run_dir, out, rid, set_name, n)
    deleted = 0
    if problem is None:
        (out / "meta" / f"{rid}.ok").write_text("verified\n")
        if delete_pngs:
            for f in (run_dir / "frames").glob("*.png"):
                f.unlink()
                deleted += 1
    return {"run_id": rid, "seconds": time.time() - t0, "problem": problem, "deleted": deleted,
            "valid_mean": float(np.mean(meta["frames"]["valid_frac"]))}


def complete_runs(render: Path) -> list[Path]:
    out = []
    if not render.is_dir():
        return out
    for d in sorted(p for p in render.iterdir() if p.is_dir()):
        try:
            if json.loads((d / "meta.json").read_text()).get("status") == "complete":
                out.append(d)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    return out


def run_index(render: Path, run_dir: Path) -> int:
    """The run's index in its job (`<job>-<index>-…`)."""
    rest = run_dir.name[len(render.name) + 1:]
    return int(rest.split("-")[0])


def drive(args, stream: bool):
    render, out = Path(args.render), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keep = math.inf if args.keep == "all" else int(args.keep)
    submitted: set[str] = set()
    for f in (out / "meta").glob("*.ok") if (out / "meta").exists() else []:
        submitted.add(f.stem)
    log = open(out / "rectify.log", "a")
    n_done = n_bad = 0
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {}
        while True:
            flag_seen = (not stream) or (args.done_flag and Path(args.done_flag).exists())
            for d in complete_runs(render):
                rid = json.loads((d / "run.json").read_text())["id"]
                if rid in submitted:
                    continue
                submitted.add(rid)
                delete = run_index(render, d) >= keep
                futures[pool.submit(process, str(d), str(out), args.set, args.n, delete)] = rid
            for fut in [f for f in futures if f.done()]:
                rid = futures.pop(fut)
                try:
                    res = fut.result()
                except Exception as e:  # noqa: BLE001
                    res = {"run_id": rid, "problem": f"{type(e).__name__}: {e}"}
                if res.get("problem"):
                    n_bad += 1
                else:
                    n_done += 1
                log.write(json.dumps(res) + "\n")
                log.flush()
            if flag_seen and not futures:
                # one last scan after the flag: anything completed just before it
                if not [d for d in complete_runs(render) if json.loads((d / "run.json").read_text())["id"] not in submitted]:
                    break
            time.sleep(args.poll if stream else 0.5)
    log.close()
    print(f"rectify {args.set}: {n_done} runs verified, {n_bad} with problems (see {out / 'rectify.log'})")
    finalize(out, args.set)


def finalize(out: Path, set_name: str):
    """Set-level tables from the per-run meta files (verified runs only)."""
    out = Path(out)
    metas = []
    for f in sorted((out / "meta").glob("*.json")):
        if (out / "meta" / f"{f.stem}.ok").exists():
            metas.append(json.loads(f.read_text()))
    if not metas:
        print(f"finalize {set_name}: no verified runs")
        return
    index_cols = {}
    for m in metas:
        idx = Path(m["render_dir"]).parent / "index.csv"
        if idx not in index_cols and idx.exists():
            with open(idx) as fh:
                index_cols[idx] = {r["id"]: r for r in csv.DictReader(fh)}
    with open(out / "frames.csv", "w", newline="") as ff, open(out / "labels.csv", "w", newline="") as fl:
        wf = csv.writer(ff)
        wl = csv.writer(fl)
        wf.writerow(["run_id", "frame", "t", "gsd", "valid_frac"])
        wl.writerow(["run_id", "frame", "t", "cam_e", "cam_n", "cam_u", "cam_x", "cam_y", "aircraft_heading", "aircraft_bank"])
        for m in metas:
            fr = m["frames"]
            th = m["tile"]["theta_deg"]
            xy = geom.enu_to_tile(np.stack([fr["cam_e"], fr["cam_n"]], 1), th)
            for k in range(len(fr["frame"])):
                wf.writerow([m["run_id"], fr["frame"][k], f"{fr['t'][k]:.6f}", f"{m['tile']['gsd']:.8f}", f"{fr['valid_frac'][k]:.6f}"])
                wl.writerow([m["run_id"], fr["frame"][k], f"{fr['t'][k]:.6f}", f"{fr['cam_e'][k]:.6f}", f"{fr['cam_n'][k]:.6f}",
                             f"{fr['cam_u'][k]:.6f}", f"{xy[k, 0]:.6f}", f"{xy[k, 1]:.6f}",
                             f"{fr['aircraft_heading'][k]:.4f}", f"{fr['aircraft_bank'][k]:.4f}"])
    rows = []
    for m in metas:
        spec, s, tl = m["run"], m["summary"], m["tile"]
        sea, light, traj, rend = spec["sea"], spec["light"], spec["trajectory"], spec["render"]
        idx = index_cols.get(Path(m["render_dir"]).parent / "index.csv", {}).get(m["run_id"], {})
        vf = np.array(m["frames"]["valid_frac"])
        bank = np.abs(np.array(m["frames"]["aircraft_bank"], dtype=float))
        rows.append({
            "set": set_name, "run_id": m["run_id"],
            "n": tl["n"], "gsd": tl["gsd"], "side": tl["side"], "theta_deg": tl["theta_deg"],
            "c_run_e": tl["c_run_e"], "c_run_n": tl["c_run_n"],
            "n_frames": s["n_frames"], "duration": s["duration"], "fps": rend["fps"],
            "distance": s["distance"], "ground_speed_mean": s["ground_speed_mean"], "cam_u_mean": s["cam_u_mean"],
            "valid_frac_mean": float(vf.mean()), "valid_frac_min": float(vf.min()),
            "masked_frac_mean": float(1 - vf.mean()), "aa_ratio_max": float(np.max(m["frames"]["aa_ratio"])),
            "bank_abs_max": float(np.nanmax(bank)) if np.isfinite(bank).any() else float("nan"),
            "altitude": traj["altitude"], "airspeed": traj["airspeed"], "gimbal_pitch": traj["gimbal"]["pitch"],
            "gimbal_fov": traj["gimbal"]["fov"], "stabilized": traj["gimbal"]["stabilized"],
            "turbulence": traj["turbulence"], "pattern": s["pattern_type"], "start_heading": traj["start"]["heading"],
            "wind_aloft": json.dumps(traj["wind"]) if isinstance(traj["wind"], dict) else traj["wind"],
            "traj_seed": traj["seed"], "sea_seed": rend["seaSeed"], "sea_rate": rend.get("seaRate", 1),
            "wind_speed": sea["windSpeed"], "wind_direction": sea["windDirection"], "fetch": sea["fetch"],
            "swell_height": sea["swellHeight"], "swell_period": sea["swellPeriod"],
            "swell_direction": sea["swellDirection"], "choppiness": sea["choppiness"],
            "swell": int(sea["swellHeight"] > 0),
            "sea_hs": float(idx.get("sea_hs", "nan") or "nan"), "sea_tp": float(idx.get("sea_tp", "nan") or "nan"),
            "sun_elevation": light["sunElevation"], "sun_azimuth": light["sunAzimuth"],
            "overcast": light["overcast"], "haze": light["haze"],
        })
    with open(out / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    vf_all = np.array([r["valid_frac_mean"] for r in rows])
    info = {
        "format": FORMAT_VERSION, "set": set_name, "runs": len(rows),
        "frames": int(sum(r["n_frames"] for r in rows)), "n": int(rows[0]["n"]),
        "gsd_min": float(min(r["gsd"] for r in rows)), "gsd_max": float(max(r["gsd"] for r in rows)),
        "masked_frac_mean": float(1 - vf_all.mean()),
        "tiles": "tiles/<run_id>.npy: float16 [n_frames, n, n], linear-light luminance, NaN = outside the image",
        "inputs": "frames.csv (run_id, frame, t, gsd, valid_frac) + tiles; labels.csv and runs.csv are labels/metadata, never model input",
        "bytes": int(sum(f.stat().st_size for f in (out / "tiles").glob("*.npy"))),
    }
    (out / "set.json").write_text(json.dumps(info, indent=1) + "\n")
    print(f"finalize {set_name}: {info['runs']} runs, {info['frames']} frames, masked {info['masked_frac_mean']:.3f}, "
          f"{info['bytes'] / 1e9:.2f} GB")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m e2.rectify")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("stream", "batch"):
        p = sub.add_parser(name)
        p.add_argument("--render", required=True, help="<out>/<job> directory of the generator")
        p.add_argument("--out", required=True)
        p.add_argument("--set", required=True)
        p.add_argument("--n", type=int, default=N_DEFAULT)
        p.add_argument("--keep", default="5", help="keep the PNGs of runs with job index < keep ('all' keeps every run's)")
        p.add_argument("--workers", type=int, default=12)
        p.add_argument("--poll", type=float, default=10.0)
        p.add_argument("--done-flag", default=None, help="stream: stop once this file exists and every complete run is done")
    p = sub.add_parser("finalize")
    p.add_argument("--out", required=True)
    p.add_argument("--set", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "finalize":
        finalize(Path(args.out), args.set)
    else:
        drive(args, stream=args.cmd == "stream")


if __name__ == "__main__":
    sys.exit(main())
