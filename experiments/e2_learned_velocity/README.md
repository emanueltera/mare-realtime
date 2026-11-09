# E2 — learned velocity over moving water

Can a network trained on simulated sequences learn to remove the waves' own motion and
output the aircraft's velocity over the water (here equal to its ground velocity: the sim
has no current), well inside the 0.17 m/s budget E1 set, and does it generalise beyond its
training conditions? This directory holds E2's data, loader and evaluation harness; models
live in their own subdirectories and all use the same sets, windows and metrics.

**Results: [results/RESULTS.md](results/RESULTS.md).** In short, on simulated seas only: a
network that sees 8 rectified frames (model A, `e2e/`) removes most of the wave bias (median
per-run bias 0.191 m/s on test-iid against 1.356 m/s for phase correlation), but 43 % of runs
are within 0.17 m/s, not all of them. On a sea whose clock is changed it does not follow the
water's motion: it removes most of the bias at the training sea rate and not at other rates,
and how it does so is not established (the K = 2 and K = 4 ablations behave like a correction
keyed to the waves' appearance; A at K = 8 does not). Its accuracy is tied to the simulator's
waves, and partly to its foam (RESULTS §4.3, §5). Nothing here has seen real water.

| Method | Directory | What it is | Notes |
|---|---|---|---|
| A `e2e` | `e2e/` | end-to-end network: multi-scale, multi-gap cost volumes over K tiles, a transformer read-out | [results/e2e/NOTES.md](results/e2e/NOTES.md) |
| B `flowhead` | `flowhead/` | frozen pretrained SEA-RAFT-M on tile pairs + a learned weighting / correction head | [results/flowhead/NOTES.md](results/flowhead/NOTES.md) |
| C `spectral` | `spectral/` | learned weighting of a 3-D FFT dispersion-shell cost volume (√(g\|k\|) built in) | [results/spectral/NOTES.md](results/spectral/NOTES.md) |
| B0–B3 | `baselines/` | phase correlation, SEA-RAFT-M, E1's shell fit (reads the label), a conditions-only MLP | [results/baselines/NOTES.md](results/baselines/NOTES.md) |

`analyze.py` reads every method's harness output and predictions under `results/` and
writes the cross-method tables (`results/comparison_by_set.csv`, `comparison_by_slice.csv`,
`e1_core_compare.csv`, `k_ablation.csv`, `rate_diagnostic.csv`, `calibration.csv`,
`costs.csv`) and the figures (`results/fig_*.png`) that RESULTS.md quotes. It recomputes
each 'all' row from `per_run.csv` with the harness's own `summarise` and stops if it differs
from the harness's `slices.csv`. Run it locally:
`uv run --no-project --with numpy --with pandas --with matplotlib python analyze.py`.
`verify/` holds the independent re-checks of each method (scripts and their CSVs).

Sizes, ranges, masking, storage and render times of the sets are in
[results/data_summary.csv](results/data_summary.csv) (written by `summarize_data.py`); the
sea-seed disjointness check in [results/seed_check.csv](results/seed_check.csv); tile-size
masking in [results/tile_size.csv](results/tile_size.csv). A classical reference, masked
phase correlation between consecutive tiles (K = 2), scored by the harness, is in
`results/baselines/phasecorr/<set>/` (val, e1-core with `e1_compare.csv`, test-rate,
test-rate frozen runs only, and E1's frozen set): it is the frozen-sea check (`test-rate`
seaRate 0) and a picture of how much a motion estimator that does not know about waves is
pulled by them (its bias grows with the sea's rate; see `test-rate/slices.csv`).

## Files

| File | What it does |
|---|---|
| `make_jobs.mjs` | draws every set's runs (`randomRun` + E2's redraws) and writes `jobs/<set>.json` (`runs` jobs) and `jobs/seeds.csv`; checks ranges, the held-out sea regime and seed disjointness |
| `render_all.sh` | renders one set on the box and rectifies it as it renders (`e2.rectify stream`) |
| `e2/geom.py` | the tile: per-run orientation θ and centre c_run, pixel maps, linear luminance, rectification |
| `e2/rectify.py` | runs → `tiles/<run_id>.npy` + tables; verify-then-delete PNGs; `finalize` builds `frames.csv`, `labels.csv`, `runs.csv`, `set.json` |
| `e2/data.py` | `TileSet`, `WindowDataset` (K-frame windows, exact augmentations), `prediction_frame` |
| `e2/evaluate.py` | prediction CSV → `per_run.csv`, `slices.csv`, `summary.md` |
| `e2/baselines.py` | masked phase correlation (a classical reference, and the frozen-sea check) |
| `summarize_data.py` | `results/data_summary.csv`, `results/seed_check.csv` |
| `e2e/`, `flowhead/`, `spectral/`, `baselines/` | the methods: training, prediction and their tables (each has a `NOTES.md` under `results/<method>/`) |
| `analyze.py` | the cross-method comparison: `results/comparison_*.csv`, `k_ablation.csv`, `rate_diagnostic.csv`, `calibration.csv`, `costs.csv`, `fig_*.png` |
| `verify/` | independent re-checks of every method: leak audit, re-inference from frozen checkpoints, recomputed headline numbers |
| `tile_size.py` | `results/tile_size.csv`: masked fraction for several tile sizes, from run poses |
| `tests/` | `test_geom.py` (synthetic frozen texture: the tile samples the right ground and the texture moves at exactly −v·dt in tile axes; it uses geom's own pose convention — the same check on the engine's rendered frozen-sea poses, every dihedral augmentation, is `verify/leak_audit/geom_check_summary.csv`), `test_data.py` (labels, inputs, augmentations), `test_evaluate.py` (oracle / constant offset / predict-zero), `test_box.py` (on the box: frozen-sea phase correlation, disjoint sets); per method `test_e2e.py`, `test_flowhead.py`, `test_spectral.py`, `test_baselines.py` (torch cases skip without torch) |

## Sets

All rendered at 720 × 1280 portrait, vfov 92 (fx = fy = 618.04), 10 fps, 8 s runs (81 frames)
after an 8 s warm-up, wind aloft `sea` (the aircraft crabs), turbulence, gimbal
stabilisation, start heading, pattern, sea and sky as `randomRun` draws them; then gimbal
pitch uniform in [−90°, −60°], altitude log-uniform in 30–200 m, airspeed uniform in
[max(15, 1.3 × wind aloft), 60] m/s.

| set | runs | what differs |
|---|---|---|
| `train` | the first 4500 of `jobs/train.json` | — |
| `val` | 150 | own seed |
| `test-iid` | 300 | own seed |
| `test-sea` | 150 | only the held-out sea: wind 16–20 m/s, or a swell of period 14–16 s |
| `test-alt` | 150 | altitude 220–400 m (log-uniform) |
| `test-rate` | 100 conditions × `render.seaRate` 0, 0.5, 1, 2 | paired (same seeds, same θ); rate r scales the sea's clock (ω = r·√(gk), like gravity r²·g), 0 freezes it |
| `e1-core` | E1's 432 core runs (121 frames, nadir, no wind aloft): 6 preset seas, all with one sea seed (0), × 72 flight and sun variants, so 6 independent wave fields, not 432 | rendered before the break-gate fix; θ random per run; run ids are E1's, so `e2.evaluate` writes `e1_compare.csv` beside E1's per-run b |
| `pilot` | 16 | the pilot (`jobs/pilot.json`), all PNGs kept |
| `e1-frozen` | E1's 24 frozen-sea runs | for the phase-correlation check only |

The held-out sea regime is excluded from every other E2 set. Run seeds (sea seed =
20240611 + seed) are unique across sets and never 0, E1's sea.

## On disk (`/workspace/data/e2/sets/<set>/`)

- `tiles/<run_id>.npy`: float16 `[F, 384, 384]`, linear-light luminance (Rec. 709 on decoded
  sRGB), `NaN` where the tile cell falls outside the image (the validity mask).
- `frames.csv`: `run_id, frame, t, gsd, valid_frac` — what a model may use besides the tiles
  (t only as differences: dt).
- `labels.csv`: `run_id, frame, t, cam_e, cam_n, cam_u, cam_x, cam_y, aircraft_heading,
  aircraft_bank` — labels only.
- `runs.csv`: per-run tile geometry (`n, gsd, side, theta_deg, c_run_e, c_run_n`), masking,
  speeds and the run's sea, sky and flight — labels / slicing only.
- `set.json`: counts, N, GSD range, bytes.

Render outputs other than the PNGs (run.json, frames.csv, trajectory.csv, imu.csv, meta.json,
index.csv) stay under `/workspace/data/e2/render/<set>/e2-<set>/`. The rendered PNGs were deleted once each run's tiles were verified (re-rectifying two frames
from the PNGs matches the stored tiles); the first 5 runs of each set keep theirs under
`/workspace/data/e2/render/<set>/e2-<set>/`. E1's PNGs are untouched.

## Tile geometry

Frame i samples ground points p = cam_pos_i + c_run + R(θ_run)·(x, y) on the plane up = 0, by
projecting camera-relative offsets d = c_run + R(θ)·(x, y) − (0, 0, cam_u_i) with the true
R_wc and K. c_run is the run-mean footprint centre (principal ray ∩ sea) relative to the
camera, constant within a run, so the tile translates exactly with the camera and the
water's content moves through it at −v (plus its own motion). θ_run is uniform in [0, 360)
from a hash of the set and run (test-rate: of the condition). Tile axes in ENU are
x̂ = (cos θ, sin θ), ŷ = (−sin θ, cos θ); arrays are `[row = y, col = x]`, both increasing
toward +y / +x.

N = 384 cells of gsd = run-mean cam_u / fx (the image's nadir GSD), so side = 0.62 × height
(derived: 384 / 618.04). At nadir the 720 px axis spans 1.165 × height (derived: 720 / 618.04),
so the square's circumscribed circle (0.88 × height across) fits with room left for attitude
changes, and at −60° pitch the footprint is wider still. The masked fraction for N = 256, 320,
384 and 512, from the val runs' poses (150 runs) and the pilot's (16), is in
[results/tile_size.csv](results/tile_size.csv); the masking of every set as built is in
`results/data_summary.csv`. A GSD equal to the nadir image GSD keeps the shortest waves E1
found most reliable (λ 0.4–0.5 m) wherever the image resolves them; the larger tile keeps
consecutive frames overlapping at the fastest, lowest runs (derived: 75 m/s at 30 m moves
~150 tile px per frame, 40 % of 384). Where a cell spans more than one image pixel (oblique
edge, camera lower than its run mean) the image is Gaussian-prefiltered (σ = ½√(r² − 1) px,
`aa_ratio_max` in runs.csv). float16 keeps linear light without a per-run scale (8-bit linear
light would leave a dusk sea a dozen grey levels); a frame is 295 kB (derived: 384² × 2 B).

## Loading windows

```python
import sys; sys.path.insert(0, "/workspace/mare-realtime/experiments/e2_learned_velocity")
from e2.data import TileSet, WindowDataset, prediction_frame, PRED_COLUMNS, unaugment_vector
ts = TileSet("/workspace/data/e2/sets/train", preload=False)   # preload=True: all tiles in RAM
ds = WindowDataset(ts, K=8, stride=1, step=1, augment=True, crop=256, bin=1, normalize="window")
x = ds[0]
# x["tiles"] [K, c, c] float32 (normalised over the window's valid pixels, 0 where masked)
# x["mask"] [K, c, c] bool, x["gsd"] m/px, x["dt"] s, x["target"] [2] m/s in tile axes
# (augmented with the tiles), x["aug"] the dihedral element, x["key"] (set, run_id, w_start, K, T)
```

`torch.utils.data.DataLoader(ds, batch_size=…, num_workers=…)` works directly (the key
collates to five lists). `ds.index` is the window table with the labels (`v_true_x/y`,
`v_true_e/n`, `T`, `valid_min`). Augmentations: the 8 rotations/flips of the square
(`augment_array`, `augment_vector`, `unaugment_vector`) and random crops; no time reversal.

## Predictions and evaluation

One row per window, in tile axes (ENU filled from θ by `prediction_frame`):
`set, run_id, w_start, K, T, v_true_x, v_true_y, v_est_x, v_est_y, sigma_x, sigma_y,
v_true_e, v_true_n, v_est_e, v_est_n`.

```python
df = prediction_frame(ts, keys, v_est_xy, sigma_xy)   # keys from ds[i]["key"] or ds.index
df[PRED_COLUMNS].to_csv("pred.csv", index=False)
```

```sh
cd /workspace/mare-realtime/experiments/e2_learned_velocity
/workspace/venv/bin/python -m e2.evaluate --pred pred.csv --sets /workspace/data/e2/sets --out results/<method>/<set> --name <method>
```

`v_true` is checked against the labels. Per run: b = |mean(v_est − v_true)|, b% = b / mean
|v_true|, window RMS, drift% over non-overlapping windows, b along/across the wind sea.
Slices: set, altitude, ground speed, sun elevation, wind speed, swell, gimbal pitch, K,
seaRate, pattern, masked fraction; median, p90 and the fractions with b ≤ 0.17 m/s,
b% ≤ 0.28 and b% ≤ 0.5.

Protocol: choose models and hyper-parameters on `val` only; predict every window
(`step = 1`, `min_valid = 0`) of a set; evaluate each test set once per frozen checkpoint
and record that no tuning used the test sets.

## Reproduce

```sh
node experiments/e2_learned_velocity/make_jobs.mjs        # jobs/*.json, jobs/seeds.csv
# on the box, GPU otherwise idle:
bash experiments/e2_learned_velocity/render_all.sh val
bash experiments/e2_learned_velocity/render_all.sh train --limit 4500
for s in test-iid test-sea test-alt test-rate; do bash experiments/e2_learned_velocity/render_all.sh $s; done
cd experiments/e2_learned_velocity
/workspace/venv/bin/python -m e2.rectify batch --render /workspace/data/e1/core/e1-core \
  --out /workspace/data/e2/sets/e1-core --set e1-core --keep all --workers 16
OMP_NUM_THREADS=1 /workspace/venv/bin/python -m pytest -q tests
/workspace/venv/bin/python summarize_data.py --out results
```

Locally: `uv run --no-project --with numpy --with scipy --with pandas --with opencv-python-headless --with pytest python -m pytest -q tests`.

The models are trained, predicted and scored by the commands in each method's NOTES.md
(`results/<method>/NOTES.md`, section "Reproduce"); predictions land in
`/workspace/data/e2/preds/<method>/` and harness output in `/workspace/data/e2/eval/<method>/`
on the box, with copies under `results/<method>/`. Then, locally:

```sh
uv run --no-project --with numpy --with pandas --with matplotlib python analyze.py   # tables + figures
```
