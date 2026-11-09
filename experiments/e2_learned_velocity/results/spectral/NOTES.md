# E2 model C: `spectral`, a learned dispersion-shell fit

**Question.** Does a network on the camera-fixed tile stack's 3-D spectrum remove the waves' own
motion better than E1's hand-made shell fit? Can it do so in low sun, and does it hold outside
its training conditions?

**Answer, in short.**
- The learned fit is about 4–6× better than the classical shell objective in the same pipeline.
- On E1's runs (e1-core) the median per-run bias is 0.171 m/s: 0.114 at midday and 0.254 in low
  sun. E1's dispersive shell fit had 0.237 and 1.461 on the same runs (T = 3.2), though E1 used a
  different per-run statistic; see "Against E1".
- On the E2 sets, with random seas, oblique gimbal, wind aloft and turbulence, the median is
  0.30–0.36 m/s. That is about twice the 0.17 m/s budget. 22–24 % of runs meet it, and 8–13 %
  meet b% ≤ 0.28.
- The model generalises to the held-out seas (0.356) and to 220–330 m altitude.
- It fails above 330 m (1.12 m/s), and it fails on any sea whose dispersion differs from real
  gravity (seaRate 2: 1.65 m/s), because √(g|k|) is built into the model.

Every number below is from a CSV under `results/` (path given per table). Numbers called
*derived* are computed from those CSVs. None of the models was tuned on a test set (see
"Protocol").

## Method

**Inputs (allowed only).**
- The K rectified tiles (float16 linear luminance; NaN marks the validity mask).
- The tile GSD (`frames.csv` / the window's `gsd`).
- The frame interval dt.
- A classical pattern velocity computed from the same tiles: masked phase correlation between
  consecutive tiles (`e2.baselines`, unchanged). For a window it is the median over the window's
  K − 1 pairs.

Nothing else reaches the model: no positions, speeds, heading, θ, sea, sky, run id, time or frame
index. The labels (`v_true`) are used only by the loss. During training, teacher forcing also
uses them to centre the second stage on some windows, decaying to none by step 3000; at
inference that never happens.

**Spectrum** (`spectral/spectrum.py`, E1's preparation on the GPU):
- 2 × 2 mask-aware binning: 192² cells of 2·gsd.
- Each frame divided by its valid mean.
- Each cell's temporal mean removed.
- Hann in t, x and y.
- A spatial FFT, then a temporal FFT zero-padded to 64 bins (×2 at K = 32, ×4 at K = 16).
- The k annulus is 3/192 ≤ |f| ≤ 0.4 cycles/cell (18 488 lattice points), full plane with the +
  branch.
- Whitening: each k's power is divided by its sum over ω'.

A candidate velocity V is worth J_k(V) = Sband_k(√(g|k|) − k·V). This is the whitened power within
±1 unpadded bin of the shell, read circularly (modulo 2π/dt) by linear interpolation.

Registration by the pattern velocity c is a per-k re-indexing of ω'. It is done inside these
gathers, so no image is shifted. Sign conventions are tested on a synthetic linear sea in
`tests/test_spectral.py`:
- the shell peaks at the truth;
- a frozen pattern is off the shell;
- the dihedral augmentation commutes;
- the fused backward matches autograd;
- the GPU augmentation equals `e2.data.augment_array`.

**Network** (`spectral/model.py`, 127 656 parameters at K = 32). The search runs in stages, each a
square grid of candidates c + o:

| run | stage 1 | stage 2 | stage 3 |
|---|---|---|---|
| `spectral`, `spectral-k16` | c = pattern velocity, ±6 m/s at 0.25 (49²) | ±0.8 at 0.05 (33²) around stage 1's argmax | — |
| `spectral-noreg` | c = 0, ±84 m/s at 1.0 (169²), on long waves only (\|f\| ≤ 0.1) | ±6 at 0.25 | ±0.8 at 0.05 |

Per stage:
1. A per-k MLP outputs a weight w_k ≥ 0. Its inputs are:
   - that k's whitened ω' profile, rolled to the stage centre's shell (64 values, log);
   - log|k|, |f|, the direction of k, the log power relative to the window median, and where the
     fixed-in-image line and the wrap fall;
   - a mean-pooled context vector over all k, plus log cell size, the dwell |c|·T/L and log T.
2. C(o) = Σ w_k J_k(c + o) / Σ w_k is computed by a fused, `torch.compile`d gather-and-sum.
   Backward recomputes J rather than storing it.
3. The cost volume is standardised, then a 5-layer dilated CNN adds to it, giving logits.
4. In coarse stages the argmax becomes the next centre. In the last stage the soft-argmax is δ,
   and v = c + δ.

σ (per tile axis) comes from a small head. Its inputs are the fine distribution's log variance,
stage 1's max probability and entropy, the log cell size and the dwell.

**Loss.**
- Cross-entropy on coarse stages, against a Gaussian (σ = step) around the truth.
- 0.3 × cross-entropy on the fine stage.
- Smooth-L1 (β = 0.05 m/s) on v.
- 0.1 × Gaussian NLL of the detached error, for σ.

**Training** (`spectral/train.py`):
- AdamW, lr 6e-4, weight decay 1e-4, 200 warm-up steps, then a cosine to 5 % that ends at the
  wall-time budget.
- Batch 16. Each epoch draws 20 000 random windows (stride 1, any start) with a random dihedral
  element; the vectors are transformed in the worker, the tiles on the GPU.
- Gradient clip 1.0.
- Validation every epoch on a fixed val subset: every 7th window start (1200 windows at K = 32,
  1500 at K = 16). best.pt is the checkpoint with the lowest val-subset median b.

## Runs, curves, cost

Sources: `training/<run>/curve.csv` and `report/curves.csv` (per epoch: minutes, train loss, val
median b, val window error, σ coverage, peak torch memory); figure `report/fig_curves.png`.

| run | K | wall (min) | epochs | windows seen (derived) | best epoch | val-subset median b at best | peak torch GPU memory (GiB) |
|---|---|---|---|---|---|---|---|
| `k32` (main, `spectral`) | 32 | 140.9 | 17 | 322 k | 16 (last) | 0.306 | 1.55 |
| `k16` (`spectral-k16`) | 16 | 55.7 | 7 | 139 k | 6 (last) | 0.316 | 1.48 |
| `k32-noreg` (`spectral-noreg`) | 32 | 85.4 | 7 | 129 k | 6 (last) | 0.455 | 2.04 |

**Main curve** (val subset median b, m/s):

| minutes | 6.5 | 21.7 | 47.5 | 85.2 | 115.7 | 140.9 |
|---|---|---|---|---|---|---|
| b | 0.518 | 0.439 | 0.404 | 0.341 | 0.318 | 0.306 |

- Train loss fell from 8.30 to 6.96, and the median val window error from 0.75 to 0.49 m/s.
- The curve was still falling when the budget ended, so all three runs are under-trained, and
  K = 16 and noreg had less training than K = 32.
- Throughput on the shared GPU was 38 windows/s for K = 32, 42 for K = 16 and 25 for noreg
  (derived: steps × 16 / minutes).
- The smoke test (24 train runs, scored on the same runs) is in `training/smoke/curve.csv`. Its
  loss fell from 10.8 to 6.9, and the median b fell from 0.61 to 0.27 in 600 steps, so the loss
  behaves.

## Protocol

- Models and the TTA choice were selected on val only.
- 8-fold dihedral test-time averaging was scored on val only (`spectral-tta/val`). It did not help:
  median b 0.340 against 0.338, so the frozen main configuration has no TTA.
- Each frozen checkpoint (`k32/best.pt`, `k16/best.pt`, `k32-noreg/best.pt`) was evaluated once on
  every test set, predicting every window (step 1, min_valid 0). That gives 89 k windows at K = 32
  and 112 k at K = 16. The harness ran with `--require-all`.
- No hyper-parameter, architecture or checkpoint choice used a test set.
- The pattern velocity was missing (all pairs failed) in 0 windows of any set. The truth lay inside
  stage 1's ±6 m/s grid in ≥ 99.85 % of windows (`preds/<set>.diag.csv`).

## Results

Source: `report/summary_all.csv` (the harness's `slices.csv`, 'all' rows). b is in m/s; fractions
are of runs.

| set | method | runs | b median | b p90 | b% median | b ≤ 0.17 | b% ≤ 0.28 | b% ≤ 0.5 | drift% median |
|---|---|---|---|---|---|---|---|---|---|
| val | spectral (K32) | 150 | 0.338 | 0.877 | 1.00 | 0.22 | 0.09 | 0.25 | 1.05 |
| test-iid | spectral | 300 | 0.300 | 0.901 | 0.81 | 0.24 | 0.10 | 0.24 | 0.99 |
| test-sea | spectral | 150 | 0.356 | 1.109 | 1.05 | 0.23 | 0.13 | 0.23 | 1.36 |
| test-alt | spectral | 150 | 0.438 | 2.245 | 1.32 | 0.22 | 0.10 | 0.25 | 1.39 |
| test-rate | spectral | 400 | 0.524 | 1.777 | 1.44 | 0.10 | 0.04 | 0.10 | 1.82 |
| e1-core | spectral | 432 | 0.171 | 0.515 | 0.60 | 0.50 | 0.27 | 0.45 | 0.60 |
| val | spectral-k16 | 150 | 0.323 | 0.840 | 0.93 | 0.24 | 0.13 | 0.29 | 0.98 |
| test-iid | spectral-k16 | 300 | 0.346 | 0.938 | 0.94 | 0.26 | 0.15 | 0.31 | 0.99 |
| test-sea | spectral-k16 | 150 | 0.331 | 1.120 | 1.10 | 0.18 | 0.10 | 0.21 | 1.30 |
| test-alt | spectral-k16 | 150 | 1.001 | 2.701 | 3.05 | 0.04 | 0.02 | 0.03 | 2.86 |
| test-rate | spectral-k16 | 400 | 0.530 | 1.453 | 1.51 | 0.10 | 0.04 | 0.12 | 1.58 |
| e1-core | spectral-k16 | 432 | 0.325 | 1.147 | 1.03 | 0.33 | 0.18 | 0.29 | 1.01 |
| val | spectral-noreg | 150 | 0.477 | 1.377 | 1.26 | 0.14 | 0.05 | 0.15 | 1.61 |
| test-iid | spectral-noreg | 300 | 0.469 | 1.368 | 1.38 | 0.15 | 0.06 | 0.18 | 1.58 |
| test-sea | spectral-noreg | 150 | 0.585 | 1.492 | 1.52 | 0.13 | 0.07 | 0.13 | 2.13 |
| test-alt | spectral-noreg | 150 | 1.243 | 7.422 | 3.87 | 0.04 | 0.01 | 0.06 | 4.25 |
| test-rate | spectral-noreg | 400 | 0.741 | 2.044 | 2.21 | 0.06 | 0.03 | 0.06 | 2.85 |
| e1-core | spectral-noreg | 432 | 0.260 | 1.129 | 0.83 | 0.31 | 0.10 | 0.29 | 0.89 |

**References.**
- Phase correlation, K = 2 (`results/baselines/phasecorr/*`): val 1.381, e1-core 1.439.
- The unlearned shell objective in this pipeline (`spectral-classical/val`: uniform whitened
  weights, argmax per stage, same grids and pattern centre): val 1.372 (p90 2.169).

The learned fit is about 4× better than the classical shell objective on val (derived). The
classical objective is barely better than its own centre.

### Slices (main, `report/slices_main.csv`)

**Sun elevation** (median b):

| set | ≤ 5° | 5–15° | 15–35° | > 35° |
|---|---|---|---|---|
| test-iid | 0.431 | 0.350 | 0.328 | 0.250 |
| val | 0.563 | 0.373 | 0.414 | 0.250 |

Low sun is still the hardest condition, but no longer by 6×.

**Altitude** (test-iid; test-alt is at 220–400 m):

| 30–50 m | 50–80 | 80–130 | 130–200 | 200–260 | 260–330 | 330–400 |
|---|---|---|---|---|---|---|
| 0.345 | 0.295 | 0.287 | 0.307 | 0.295 | 0.358 | 1.123 |

Up to 330 m it extrapolates (GSD 0.35–0.5 m/px against at most 0.33 in train). Above 330 m it
breaks, with p90 2.84.

**Ground speed.** On test-iid, 0.26–0.38 m/s up to 65 m/s and 0.51 above (8 runs). On e1-core,
60 m/s gives 0.255 against 0.13–0.14 at 15 and 30 m/s.

**Held-out sea** (test-sea).
- Wind 16–18 m/s: 0.394; wind 18–20 m/s: 0.523. Train wind is ≤ 16.
- Swell to 16 s: 0.334. Runs without swell (high wind only): 0.529.

**Gimbal pitch** (test-iid): −90 to −80°: 0.278; −80 to −70°: 0.367; −70 to −60°: 0.301.

**Sea rate** (test-rate, 100 paired conditions each):

| seaRate | 0 (frozen) | 0.5 | 1 | 2 |
|---|---|---|---|---|
| spectral | 0.356 | 0.503 | 0.317 | 1.647 |
| spectral-k16 | 0.416 | 0.502 | 0.344 | 1.258 |
| spectral-noreg | 0.583 | 0.609 | 0.467 | 1.840 |
| phasecorr | 0.076 | 0.669 | 1.341 | 2.501 |

The model has real-gravity dispersion built in, so it does not recognise a frozen sea. There it
adds a correction to a pattern velocity that was already right: 0.356 against phase correlation's
0.076. At seaRate 2 (equivalent to 4 g) it fails.

### Against E1 (e1-core, `report/e1_compare_by_sun.csv`)

Median per-run b (m/s). The E1 columns are E1's own per-run values for the same 432 runs.

| sun | spectral | spectral-k16 | spectral-noreg | E1 shell dispersive T = 3.2 | T = 6.4 | E1 SEA-RAFT g = 1 |
|---|---|---|---|---|---|---|
| midday (216) | 0.114 | 0.150 | 0.183 | 0.237 | 0.206 | 1.814 |
| low sun (216) | 0.254 | 0.622 | 0.406 | 1.461 | 1.366 | 1.556 |

**Main, by altitude:**

| sun | 30 m | 60 m | 120 m | 250 m |
|---|---|---|---|---|
| midday | 0.209 | 0.098 | 0.073 | 0.162 |
| low sun | 0.324 | 0.291 | 0.263 | 0.175 |

At midday, 67 % of runs meet 0.17 m/s; in low sun, 33 %.

Caveats:
- E1's per-run value was a median over its windows (with a truth-centred prior); E2's b is |mean|
  over every window. The two are not identical statistics.
- e1-core was rendered before the break-gate fix.
- E1's sea presets lie inside the training distribution in most respects (nadir, no wind aloft).

The unlearned objective in this pipeline, on every 10th e1-core window
(`spectral-classical/e1-core-every10th/by_sun.csv`), gives 0.424 at midday and 1.455 in low sun.
That reproduces E1's low-sun failure: 1.461 in the same file.

### Ablations

**K = 16 against K = 32.**
- On val, K = 16 is slightly better: 0.323 against 0.338. So val alone would not have chosen K = 32.
- On the tests, K = 32 is clearly better out of distribution:
  - e1-core: 0.171 against 0.325;
  - e1-core low sun: 0.254 against 0.622;
  - test-alt: 0.438 against 1.001.
- K = 16 also had 2.5× less training (derived from the minutes), so the comparison mixes window
  length and training.
- K = 64 (T = 6.4 s) was not trained.

**Registration (`spectral-noreg`).** Without the classical pattern velocity, the model's own coarse
global search over ±84 m/s replaces it. The fine stages are the same.
- It works: e1-core 0.260 and val 0.477, against phase correlation's 1.38–1.44.
- It is worse on every set than registration: val +41 % (derived).
- Its tail is far worse: test-alt p90 7.42 m/s, because the global argmax on long waves locks
  onto wrong solutions at altitude.
- It also had 60 % of the main run's training time.

### σ calibration (`<method>/calibration.csv`)

The table gives the fraction of per-axis |error|/σ within 1 and within 2 (Gaussian: 0.683 / 0.954).

| set | spectral | spectral-k16 | spectral-noreg |
|---|---|---|---|
| val | 0.731 / 0.936 | 0.774 / 0.944 | 0.814 / 0.945 |
| test-iid | 0.731 / 0.934 | 0.776 / 0.947 | 0.818 / 0.941 |
| test-sea | 0.671 / 0.909 | 0.702 / 0.915 | 0.757 / 0.928 |
| test-alt | 0.773 / 0.919 | 0.700 / 0.887 | 0.688 / 0.859 |
| test-rate, seaRate 2 | 0.347 / 0.612 | 0.457 / 0.771 | 0.427 / 0.774 |
| e1-core | 0.840 / 0.972 | 0.830 / 0.957 | 0.900 / 0.969 |

- Per window, σ ranks the error well. On val the deciles of σ run from 0.21 to 1.21 m/s, while the
  RMS error per decile runs from 0.23 to 1.35 m/s.
- In distribution, σ is slightly wide in the core and slightly short in the tails.
- On the out-of-physics seaRate 2 it is over-confident.
- σ is per window. It says nothing about the per-run bias, which is what the budget is about.

## Limitations (honest)

- **Not inside the budget.** The median run is at 0.30–0.36 m/s on the E2 sets, about twice
  0.17 m/s, and only 8–13 % meet b% ≤ 0.28. e1-core is easier (nadir, preset seas, no wind aloft,
  no turbulence) and should not be read as the headline.
- **Under-trained.** The curves were still falling at the budget, so more training is the cheapest
  next step, and K = 64 is untested.
- **Dispersion is hard-coded.**
  - √(g|k|) is in the cost volume, so the model cannot learn a different dispersion (seaRate test)
    and cannot recognise a frozen pattern.
  - On a real sea, currents, nonlinear waves and bound harmonics would shift the shell. They are
    not in the sim (E1's simulator-artefact table).
- **Window length.** The model needs T = 1.5–3.1 s of frames. Over shorter dwell (fast and low) the
  spectrum is smeared. At 60 m/s on e1-core it is 2× worse than at 15–30 m/s.
- **Above train altitude.** The model degrades above 330 m (GSD at least twice the train maximum).
- **Pattern velocity.** It comes from phase correlation on CPU: about 1 min per 300 runs with 16
  processes. It is an extra classical step at inference.
- **Gate.** The E2 sets use the fixed break gate; e1-core does not.
- **Low sun.** Low sun remains the hardest slice. Whether real low-sun imagery behaves like the
  render is unknown.
- **Statistic.** The per-run b here (|mean| over every window) differs from E1's median over
  windows.

## Files

**Code** (`experiments/e2_learned_velocity/spectral/`):
- `spectrum.py`, `model.py`, `data.py`, `pattern.py` — the model and its inputs;
- `train.py`, `predict.py`, `run_eval.sh` — training, prediction and scoring;
- `classical.py` — the unlearned reference;
- `calibration.py`, `report.py` — the tables in this file;
- tests in `tests/test_spectral.py`.

**Results** (`results/<method>/`, for spectral, spectral-k16, spectral-noreg, spectral-tta (val)
and spectral-classical):
- `<set>/{per_run,slices}.csv`, `summary.md` and `e1_compare.csv`;
- `preds/<set>.csv`, with `.diag.csv` beside it;
- `calibration.csv`.

**Main run only:**
- `report/` — the tables above and `fig_curves.png`;
- `training/<run>/{curve.csv,train_log.csv,args.json}`;
- `training/logs/`.

**On the box:**
- checkpoints in `/workspace/data/e2/spectral/{k32,k16,k32-noreg}/{best,last}.pt`;
- pattern velocities in `/workspace/data/e2/spectral/pattern/`;
- predictions in `/workspace/data/e2/preds/<method>/`;
- harness output in `/workspace/data/e2/eval/<method>/`.

## Reproduce (on the box, from `/workspace/mare-realtime/experiments/e2_learned_velocity`)

```sh
PY=/workspace/venv/bin/python
# 1. classical pattern velocities for the sets the shared baseline does not cover (CPU, ~20 min for train)
for s in test-iid test-sea test-alt train; do
  OMP_NUM_THREADS=1 $PY -m e2.baselines --set-root /workspace/data/e2/sets/$s --out /workspace/data/e2/spectral/pattern/$s.csv --workers 16
done   # val, e1-core, test-rate are read from /workspace/data/e2/eval/phasecorr/<set>.csv
# 2. tests
$PY -m pytest -q tests/test_spectral.py
# 3. training (main, then the ablations)
$PY -m spectral.train --out /workspace/data/e2/spectral/k32 --K 32 --stages reg --bs 16 --samples 20000 --epochs 30 --max-minutes 140 --workers 6 --lr 6e-4
$PY -m spectral.train --out /workspace/data/e2/spectral/k16 --K 16 --stages reg --bs 16 --samples 20000 --epochs 20 --max-minutes 55 --workers 4 --lr 6e-4
$PY -m spectral.train --out /workspace/data/e2/spectral/k32-noreg --K 32 --stages noreg --bs 16 --samples 20000 --epochs 14 --max-minutes 85 --workers 4 --lr 6e-4
# 4. predictions + harness (val first; tests once per frozen checkpoint)
bash spectral/run_eval.sh spectral /workspace/data/e2/spectral/k32/best.pt val test-iid test-sea test-alt test-rate e1-core
bash spectral/run_eval.sh spectral-k16 /workspace/data/e2/spectral/k16/best.pt val test-iid test-sea test-alt test-rate e1-core
bash spectral/run_eval.sh spectral-noreg /workspace/data/e2/spectral/k32-noreg/best.pt val test-iid test-sea test-alt test-rate e1-core
$PY -m spectral.predict --ckpt /workspace/data/e2/spectral/k32/best.pt --set val --tta --out /workspace/data/e2/preds/spectral-tta/val.csv
$PY -m spectral.classical --set val --out /workspace/data/e2/preds/spectral-classical/val.csv
# 5. tables
for m in spectral spectral-k16 spectral-noreg; do
  $PY -m spectral.calibration --pred /workspace/data/e2/preds/$m/{val,test-iid,test-sea,test-alt,test-rate,e1-core}.csv --out /workspace/data/e2/eval/$m/calibration.csv
done
$PY -m spectral.report --out /workspace/data/e2/spectral/report
```

The smoke test was run with the pairs of the first 24 train runs in
`SPECTRAL_PATTERN_DIRS=/workspace/data/e2/spectral/pattern-smoke`:
`python -m spectral.train --out …/smoke --K 32 --train-runs 24 --val-on-train --samples 720 --epochs 10 --teacher-steps 300 --lr 1e-3`.
The trainings shared the GPU with three other agents; K = 32 and noreg ran concurrently, and
K = 16 ran after noreg.
