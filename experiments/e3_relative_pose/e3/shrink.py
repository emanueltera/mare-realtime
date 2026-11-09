"""Shrink E3 renders as they finish: each complete run's PNGs become one uint8 array of
sRGB luma (area-averaged by 2.5), saved with the run's tables; the PNGs are deleted once the
array is written and read back with the right shape. 720×1280 portrait gives (N, 512, 288),
1280×720 landscape (N, 288, 512).

    python -m e3.shrink --render <out>/<job> --out <sets>/<set> [--done-flag FILE] [--workers 16]

With --done-flag it keeps scanning until that file exists and every complete run is done.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

# (height, width) rendered → (width, height) after shrinking, as cv2.resize takes it
SIZES = {(1280, 720): (288, 512), (720, 1280): (512, 288)}
TABLES = ("run.json", "meta.json", "frames.csv", "trajectory.csv", "imu.csv")


def complete_runs(render: Path) -> list[Path]:
    out = []
    for d in sorted(p for p in render.iterdir() if p.is_dir()):
        try:
            if json.loads((d / "meta.json").read_text()).get("status") == "complete":
                out.append(d)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    return out


def shrink_run(run: Path, out: Path) -> str:
    meta = json.loads((run / "meta.json").read_text())
    n = meta["frames"]["count"]
    dst = out / meta["id"]
    dst.mkdir(parents=True, exist_ok=True)
    pngs = [run / "frames" / f"{i:06d}.png" for i in range(n)]
    render = json.loads((run / "run.json").read_text())["render"]
    shape = (render["height"], render["width"])
    if shape not in SIZES:
        raise RuntimeError(f"{run.name}: no shrunk size for {shape[1]}×{shape[0]} renders")
    size = SIZES[shape]
    arr = np.empty((n, size[1], size[0]), np.uint8)
    for i, p in enumerate(pngs):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None or img.shape != shape:
            raise RuntimeError(f"{p}: unreadable or not {shape[1]}×{shape[0]}")
        arr[i] = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    tmp = dst / "frames.tmp.npy"
    np.save(tmp, arr)
    back = np.load(tmp, mmap_mode="r")
    if back.shape != arr.shape or not np.array_equal(back[-1], arr[-1]):
        raise RuntimeError(f"{run.name}: read-back mismatch")
    for t in TABLES:
        shutil.copy2(run / t, dst / t)
    tmp.rename(dst / "frames.npy")  # last: frames.npy marks a complete run
    shutil.rmtree(run / "frames")
    (run / "SHRUNK").write_text("ok\n")
    return meta["id"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--render", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--done-flag", default=None)
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    render, out = Path(a.render), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    submitted: set[str] = set()
    with ProcessPoolExecutor(a.workers) as ex:
        futs = []
        while True:
            flag = a.done_flag is None or Path(a.done_flag).exists()
            render.mkdir(parents=True, exist_ok=True)
            new = [d for d in complete_runs(render) if d.name not in submitted and not (d / "SHRUNK").exists()]
            for d in new:
                submitted.add(d.name)
                futs.append(ex.submit(shrink_run, d, out))
            if flag and not new:
                break
            time.sleep(5)
        failed = 0
        for f in futs:
            try:
                print("shrunk", f.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                failed += 1
                print("FAILED", e, flush=True)
    print(f"done: {len(futs) - failed} runs, {failed} failed", flush=True)


if __name__ == "__main__":
    main()
