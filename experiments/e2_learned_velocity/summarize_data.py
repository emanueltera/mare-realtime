"""Describe the E2 sets on disk: one row per set in results/data_summary.csv (sizes, windows,
tile GSD, masking, storage, render wall time and frame rate from the driver logs), and the
sea-seed disjointness check in results/seed_check.csv.

    /workspace/venv/bin/python summarize_data.py --sets /workspace/data/e2/sets --logs /workspace/data/e2/logs --out results
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

SETS = ["train", "val", "test-iid", "test-sea", "test-alt", "test-rate", "e1-core"]


def wall_time(log: Path):
    if not log.exists():
        return np.nan
    text = log.read_text()
    start = re.search(r"=== render \S+ start (\S+)", text)
    ends = re.findall(r"=== render \S+ attempt \d+ exit \d+ (\S+)", text)
    if not start or not ends:
        return np.nan
    return (dt.datetime.fromisoformat(ends[-1]) - dt.datetime.fromisoformat(start.group(1))).total_seconds()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="/workspace/data/e2/sets")
    ap.add_argument("--logs", default="/workspace/data/e2/logs")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    root, logs, out = Path(a.sets), Path(a.logs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, seeds = [], {}
    for s in SETS:
        d = root / s
        if not (d / "runs.csv").exists():
            continue
        r = pd.read_csv(d / "runs.csv")
        info = json.loads((d / "set.json").read_text())
        nf = r.n_frames.to_numpy()
        row = {"set": s, "runs": len(r), "frames": int(nf.sum()), "frames_per_run": int(np.median(nf)), "tile_n": int(r.n.iloc[0])}
        for K in (2, 4, 8, 16):
            row[f"windows_K{K}"] = int(np.clip(nf - (K - 1), 0, None).sum())
        row.update({
            "gsd_min": r.gsd.min(), "gsd_median": r.gsd.median(), "gsd_max": r.gsd.max(),
            "tile_side_min_m": r.side.min(), "tile_side_max_m": r.side.max(),
            "masked_frac_mean": r.masked_frac_mean.mean(), "masked_frac_run_p99": r.masked_frac_mean.quantile(0.99),
            "masked_frac_run_max": r.masked_frac_mean.max(), "valid_frac_frame_min": r.valid_frac_min.min(),
            "aa_ratio_max": r.aa_ratio_max.max(),
            "altitude_min": r.altitude.min(), "altitude_max": r.altitude.max(),
            "ground_speed_min": r.ground_speed_mean.min(), "ground_speed_max": r.ground_speed_mean.max(),
            "airspeed_min": r.airspeed.min(), "airspeed_max": r.airspeed.max(),
            "gimbal_pitch_min": r.gimbal_pitch.min(), "gimbal_pitch_max": r.gimbal_pitch.max(),
            "wind_speed_max": r.wind_speed.max(), "swell_period_max_with_swell": r.loc[r.swell == 1, "swell_period"].max() if (r.swell == 1).any() else np.nan,
            "held_out_sea_runs": int(((r.wind_speed >= 16) | ((r.swell == 1) & (r.swell_period >= 14))).sum()),
            "sun_elevation_min": r.sun_elevation.min(), "sun_elevation_max": r.sun_elevation.max(),
            "sea_rates": " ".join(str(x) for x in sorted(r.sea_rate.unique())),
            "tile_bytes": info["bytes"],
        })
        wt = wall_time(logs / f"driver-{s}.log")
        row["render_wall_s"] = wt
        row["render_fps"] = row["frames"] / wt if np.isfinite(wt) and s != "e1-core" else np.nan
        rows.append(row)
        seeds[s] = set(r.sea_seed)
    df = pd.DataFrame(rows)
    df.to_csv(out / "data_summary.csv", index=False)
    chk = []
    for i, s in enumerate(seeds):
        for t in list(seeds)[i + 1:]:
            chk.append({"set_a": s, "set_b": t, "shared_sea_seeds": len(seeds[s] & seeds[t])})
    pd.DataFrame(chk).to_csv(out / "seed_check.csv", index=False)
    print(df.T.to_string())
    print(pd.DataFrame(chk).to_string(index=False))


if __name__ == "__main__":
    main()
