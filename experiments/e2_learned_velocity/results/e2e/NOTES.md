# Model A (`e2e`): an end-to-end video network

E2 asks whether a network trained only on simulated sequences can learn to remove the
waves' own motion and output the aircraft's velocity over the water. Model A is the most
direct test of that. It takes K rectified tiles and returns the 2-D velocity in tile axes
with a σ. Nothing tells it what a wave is.

**Short answer, all numbers from the harness CSVs listed below:**

- **In distribution it removes most of the wave bias, but not enough.** On test-iid the
  median per-run bias is 0.191 m/s. Phase correlation (K = 2) gets 1.407 m/s on the same
  runs. 43 % of runs are within the 0.17 m/s budget, and the p90 is 0.445 m/s.
- **Held-out seas and altitudes cost a little.** Median b is 0.236 m/s on test-sea and
  0.276 m/s on test-alt. On E1's 432 core runs it is 0.199 m/s. E1's dispersion-shell fit
  gets 0.614 m/s on those same runs, and SEA-RAFT 1.676 m/s.
- **It does not measure the water's motion. It subtracts what it learned to expect.** On
  test-rate the same conditions are rendered with the sea's clock at 0, 0.5, 1 and 2. At
  rate 1 the median b is 0.182 m/s. At the other rates it is 1.24 (frozen sea), 1.06
  and 2.12 m/s. The error lies along the wind sea, with opposite signs at 0.5 and 2. A
  real sea has one gravity, so rates other than 1 are not physical. Even so, the result
  means the network applies the dispersion of the training seas as a prior. It does not
  recover the water's motion from the frames.
- **K matters, mostly for generalisation.** On val, K = 2 gets 0.205 m/s, K = 4
  0.203 m/s and K = 8 0.180 m/s. On test-iid they get 0.241, 0.246 and 0.191 m/s. The gap
  is largest on held-out conditions: on test-alt they get 0.594, 0.685 and 0.276 m/s,
  and on test-sea 0.317, 0.340 and 0.236 m/s. K = 4 is not between the other two. Its
  checkpoint was picked early (step 2800) on a noisy val subset.

## Architecture (`e2e/model.py`, 1.84 M parameters)

1. **Input preparation** (`prep`). Log luminance, normalised per frame over its valid
   cells. The mask is kept.
2. **Encoder.** One 2-D CNN is shared by every frame. Its widths are 32/64/96/128 at
   1/2, 1/4, 1/8 and 1/16 of the tile, with GroupNorm and residual blocks. FiLM on log gsd
   and dt scales every block, so the features can depend on metres and not only pixels.
   A 1 × 1 projection gives 8/16/32/32 correlation channels at each scale, unit-normalised
   per cell. Validity is carried to each scale. A cell counts as valid only if all its
   pixels are, and it is eroded by 2 cells so that features made from zero padding do
   not correlate.
3. **Cost volumes.** For every scale and every frame gap g = 1 … K − 1, the network
   computes the masked global cross-correlation of the features of frames k and k + g,
   summed over k, at every lag. It is done by zero-padded FFT, using one complex matmul
   of the K × K cross-spectral matrix per frequency, and normalised by the overlap at
   each lag. Camera motion puts every peak at g·u. A wave train moves its own scale's
   peak further, by g·c(λ)·dt, so the dispersion is spread across scales and gaps.
4. **Start.** The start is not learned and carries no gradient. It is the peak of the
   summed, softly whitened cross-power spectra of consecutive raw frames, u₀ in px/frame.
   This is the phase-correlation baseline's estimate, pulled by the waves in the same way.
5. **Read-out.** Four iterations share their weights, RAFT-style, and u is detached
   between them. Each iteration samples a 9 × 9 window of every (scale, gap) surface
   around the lag g·u, bilinearly at sub-cell positions. The window holds correlation
   and overlap plus the sampling phase, and becomes one token with a scale and gap
   embedding. That makes 28 tokens at K = 8 and 4 at K = 2. A 3-layer transformer
   (d = 128) reads these tokens plus a conditioning token built from log gsd, dt, u, |u|
   and the iteration index. The output is Δu and log σ per axis.
6. **Output.** v = −u·gsd/dt and σ_v = σ_u·gsd/dt, both in tile axes. Content moves
   through the tile at −v.

**Inputs actually used.** The model reads the K tiles with their NaN mask, gsd (m/px)
and dt (s), and nothing else. These come from `tiles/*.npy` and `frames.csv`, through
`e2e/loader.py`. The loader returns tiles, gsd, dt, the label, the augmentation id and
the row index; the row index is used only to look up the key afterwards. The network's
own u feeds back into the read-out. It never sees positions, airspeed, ground speed,
heading, sea or sky parameters, wind, the run id, the seed, the time, or the frame index.
`labels.csv` and `runs.csv` are used only for the label and by the harness.

**Run-level inference.** `E2E.forward_run` encodes each frame once and forms every
frame pair's spectrum product once. Window sums are differences of cumulative sums.
`tests/test_e2e.py::test_run_inference_matches_window_inference` checks that this gives
the same v, σ and u₀ as the window-by-window forward, to 1e-3.

**What did not work.** The first design was a (2+1)-D CNN on a stack aligned to the
phase-correlation start, with masked global pooling. In the smoke test it did not
learn at all in 600 steps on 16 runs: its training L1 stayed at the start's. Its log
was not kept. It was replaced by the cost-volume design above, before any full
training.

## Training

| | |
|---|---|
| data | train set (4500 runs, 333 000 K = 8 windows), memory-mapped float16 tiles, sampled uniformly with replacement |
| augmentation | the 8 rotations / flips of the tile, with the target turned the same way; 2 × 2 binning in 25 % of batches (gsd × 2, so train covers 0.085–0.66 m/px, derived from data_summary.csv) |
| loss | Σ over iterations 0.8^(3−i) · smooth-L1(v_i − v_true), β = 0.05 m/s, plus 0.2 × Gaussian NLL of the last iteration's σ with the error detached |
| optimiser | AdamW, lr 3e-4, wd 1e-4, 500-step warm-up, then a cosine over max(step/7000, hours/2.5) |
| other | batch 16, bf16 autocast (FFTs in fp32), gradient clip 1.0, EMA 0.999 (the EMA weights are scored and saved) |
| selection | every 700 steps on val (150 runs, windows every 4th start, 2850 windows); best.pt = lowest median per-run b |

The 2 × 2 binning stands in for a camera twice as high, since a tile cell is about one
image pixel at the tile's height. I chose it knowing, from the data summary, that
test-alt's gsd (0.354–0.642 m/px) lies outside train's (0.043–0.328). That knowledge
came from the set's design. No test result went into it, but test-alt is less of an
extrapolation in gsd than it would be without binning.

**Wall time and memory.** These figures come from `train/train.log` and `train/log.csv`.
The GPU was shared with three other agents throughout.

| run | steps | hours | best step (val) | peak GPU memory (training) |
|---|---|---|---|---|
| main, K = 8 | 6412 (stopped by the 2.5 h limit) | 2.51 | 6300 | 15.2 GB |
| ablation K = 2 | 7000 | 1.64 | 4200 | 3.7 GB |
| ablation K = 4 | 6412 | 0.96 | 2800 | 7.5 GB |

Inference peaks at 3.4–5.2 GB (`e2e.predict`). The K = 2 ablation ran alongside the main
run, so both shared the GPU with each other and with the other agents.

**Training curve.** See `train/log.csv` and `training_curves.png`. On the val subset,
the main run's median b fell from 0.286 m/s at step 700 to 0.265, 0.243, 0.225, 0.202,
0.177, 0.180, 0.180 and 0.170 m/s (steps 1400–6300). The training L1 of the last
iteration fell from 0.641 to 0.258 m/s, while the phase-correlation start stays at
about 0.92 m/s. **The run was still improving when the time limit stopped it.**

At equal steps the two were close up to step 2800 (K = 2 0.235 m/s, K = 8 0.225 m/s).
After that K = 2 plateaued: its subset val b sat at 0.220–0.235 m/s from step 2800 to
step 7000 (`../e2e-K2/train/log.csv`), while K = 8 kept improving.

**Smoke test** (`train/smoke_log.csv`; 16 train runs, scored on those same runs). The
training L1 went from 0.82 m/s at the start to 0.37 m/s for the best iteration
(iteration 2) and 0.48 m/s for the last, in 600 steps. The median b on the
same runs went from 1.080 m/s for the start to 0.569 m/s. The loss behaved. The model
was not fully overfitted in 600 steps. Two changes followed the smoke test: an
iteration embedding, and per-frame instead of per-window normalisation. Neither was
re-smoke-tested.

## Protocol

- No choice used a test set. The architecture was changed once, after the first
  design's smoke test on train runs. The hyper-parameters were set once and not tuned.
  The checkpoint (best.pt, on the val subset) and the number of read-out iterations
  were chosen on val.
- On the full val set, the median b after iterations 1–4 is 0.202, 0.167, 0.173 and
  0.180 m/s (`eval/iterations.csv`). The p90 is lowest after iteration 4, and σ is
  trained only at iteration 4. **I kept all 4 iterations, as trained.**
- **Each test set (test-iid, test-sea, test-alt, test-rate, e1-core) was predicted once
  per frozen checkpoint, and nothing was tuned on them.** No test-time augmentation was
  used. Every window was predicted (stride 1, step 1, min_valid 0), and the shared
  harness scored each set with `--require-all`.

## Results

All numbers come from the shared harness (`e2.evaluate`). Sources:
`summary_by_set.csv` (its `slice == all` rows, per method and set),
`eval/<set>/{per_run,slices,summary}.csv|md`, and the same files for
`../e2e-K2/` and `../e2e-K4/`. b is the run's mean error vector, in m/s. b% is b as a
percentage of ground speed. drift% is measured over non-overlapping windows.

### By set (main, K = 8)

| set | runs | b median | b p90 | b% median | drift% median | b ≤ 0.17 | b% ≤ 0.28 | b% ≤ 0.5 |
|---|---|---|---|---|---|---|---|---|
| val | 150 | 0.180 | 0.438 | 0.508 | 0.496 | 0.47 | 0.21 | 0.49 |
| test-iid | 300 | 0.191 | 0.445 | 0.527 | 0.533 | 0.43 | 0.20 | 0.48 |
| test-sea | 150 | 0.236 | 0.687 | 0.712 | 0.751 | 0.30 | 0.09 | 0.29 |
| test-alt | 150 | 0.276 | 0.786 | 0.818 | 0.819 | 0.26 | 0.13 | 0.31 |
| test-rate (all rates) | 400 | 1.077 | 2.687 | 2.817 | 2.772 | 0.12 | 0.05 | 0.13 |
| e1-core | 432 | 0.199 | 0.453 | 0.667 | 0.668 | 0.43 | 0.15 | 0.34 |

**Reference: phase correlation (K = 2)** on the same sets: val 1.381, test-iid 1.407,
test-sea 1.252, test-alt 2.668 and e1-core 1.439 m/s median b. These come from
`../baselines/phasecorr/val/slices.csv` and the baselines agent's `b0-phasecorr`
harness output, copied into `summary_by_set.csv`.

**E1 comparison** (`eval/e1-core/e1_compare.csv`, medians in
`e1_core_compare_medians.csv`). The rows are the same 432 runs, scored as E1 scored
them.

| method | median b (m/s) | fraction ≤ 0.17 |
|---|---|---|
| this model, K = 8 | 0.199 | 0.43 |
| this model, K = 4 | 0.241 | 0.34 |
| this model, K = 2 | 0.207 | 0.41 |
| E1 shell, dispersive T = 3.2 s | 0.614 | 0.24 |
| E1 shell, dispersive T = 6.4 s | 0.645 | 0.25 |
| E1 SEA-RAFT, g = 1 | 1.676 | 0.002 |
| E1 UFM-refine, g = 1 | 2.030 | 0.00 |

e1-core was rendered before the break-gate fix, so its whitecap gate travels at about
239° in every sea, a gate this model never saw in training. It is also nadir, with no
wind aloft, preset seas, 121 frames, and gsd up to 0.40.

### Slices (`eval/<set>/slices.csv`)

- **Altitude.** On test-iid, b is flat with altitude (0.177–0.199 median). On test-sea
  it is worst low down: 0.477 at 0–50 m against 0.188 at 130–200 m. On test-alt it is
  0.232 at 200–260 m and 0.375 at 330–400 m.
- **Ground speed.** b grows with speed: on test-iid, 0.115 below 20 m/s against 0.285 at
  50–65 m/s. b% does not grow with speed: 0.71 % below 20 m/s against 0.49 % at
  50–65 m/s. Fast windows have little overlap between frames.
- **Sun.** On test-iid there is no clear sun trend: 0.254 below 5°, 0.143 at 5–15° and
  0.190 above 35°. E1's shell fit failed in low sun; this model does not.
- **Wind and swell.** b rises with wind: on test-iid, 0.158 at 6–10 m/s against 0.244
  at 14–16 m/s. On test-sea it is 0.304 at 18–20 m/s, wind the model never saw in
  training. Swell runs are worse than swell-free ones: 0.204 against 0.143 on test-iid.
- **Gimbal pitch.** On test-iid, b is 0.179–0.203 across the pitch bins, with no trend.

### Sea rate (`sea_rate.csv`, from `eval/test-rate/per_run.csv`)

`b_along_wind` is signed. It is negative when the estimate is pulled the way phase
correlation is pulled.

| method | sea rate | runs | b median | b ≤ 0.17 | b along wind (median) |
|---|---|---|---|---|---|
| phase correlation | 0 | 100 | 0.076 | 0.82 | +0.002 |
| phase correlation | 0.5 | 100 | 0.669 | 0.00 | −0.649 |
| phase correlation | 1 | 100 | 1.341 | 0.00 | −1.273 |
| phase correlation | 2 | 100 | 2.501 | 0.00 | −2.367 |
| e2e K = 8 | 0 | 100 | 1.242 | 0.02 | −0.097 |
| e2e K = 8 | 0.5 | 100 | 1.063 | 0.01 | +1.009 |
| e2e K = 8 | 1 | 100 | 0.182 | 0.46 | −0.008 |
| e2e K = 8 | 2 | 100 | 2.123 | 0.00 | −2.093 |
| e2e K = 4 | 0 | 100 | 0.611 | 0.07 | −0.079 |
| e2e K = 4 | 0.5 | 100 | 0.663 | 0.00 | +0.614 |
| e2e K = 4 | 1 | 100 | 0.248 | 0.28 | −0.001 |
| e2e K = 4 | 2 | 100 | 1.100 | 0.03 | −1.083 |
| e2e K = 2 | 0 | 100 | 0.605 | 0.14 | −0.038 |
| e2e K = 2 | 0.5 | 100 | 0.720 | 0.02 | +0.646 |
| e2e K = 2 | 1 | 100 | 0.233 | 0.36 | −0.029 |
| e2e K = 2 | 2 | 100 | 1.087 | 0.02 | −1.061 |

**How to read this.** The K = 2 network, and the K = 4 one with it, behave almost
exactly like phase correlation minus a fixed correction of about the rate-1 wave pull. At rate 0.5 it is
+0.646 m/s; phase correlation's −0.649 plus a 1.27 m/s correction would give +0.62.
At rate 2 it is −1.061 m/s, where the same arithmetic gives −1.09 (derived from the
table).

The K = 8 network is further off at rates 0.5 and 2 than a fixed correction would be.
At rate 2 it keeps almost all of phase correlation's pull. On the frozen sea its error
is large (1.24 m/s) but not along the wind.

So K = 8 is not a fixed correction. It responds to the motion in the stack, but only
in the form the training seas take. When the waves move at speeds the training
dispersion does not allow, its estimate gets worse instead of following them. That fits the user's hypothesis, that a network
learns to remove wave motion implicitly, only in the narrow sense: it learned this
simulator's dispersion as a prior. It does not separate arbitrary water motion from
camera motion.

A real sea has one gravity, so this is not by itself a failure on real water. But it
means the model is only as right as the simulated dispersion and wave field are: their
spectrum, directions, the discrete k lattice, and the foam and gate drift. **That
dependence would carry straight into any sim-to-real gap.**

### K ablation (val; all sets in `summary_by_set.csv`)

| K | steps | val b median | val b p90 | test-iid | test-sea | test-alt | e1-core | test-rate rate 1 |
|---|---|---|---|---|---|---|---|---|
| 2 | 7000 (best 4200) | 0.205 | 0.484 | 0.241 | 0.317 | 0.594 | 0.207 | 0.233 |
| 4 | 6412 (best 2800) | 0.203 | 0.521 | 0.246 | 0.340 | 0.685 | 0.241 | 0.248 |
| 8 | 6412 (best 6300) | 0.180 | 0.438 | 0.191 | 0.236 | 0.276 | 0.199 | 0.182 |

**On the budget.** The budget is the number of training windows seen, with the same
batch, schedule, augmentation and selection rule for every K:

- **K = 2** had a 7000-step target and a 2.5 h cap, and finished 7000 steps.
- **K = 8** had the same target and cap. The cap stopped it at 6412 steps.
- **K = 4** was given 6412 steps to match K = 8, with a 1.5 h cap. It finished all of
  them in 0.96 h.

So K = 8 and K = 4 saw the same number of windows, and K = 2 saw 9 % more (derived). The
GPU was shared differently during each run, so wall time is not comparable.

Every run picks its checkpoint on the 2850-window val subset. The val curves are flat
within about ±0.02 m/s after step 2800 for K = 2 and K = 4
(`../e2e-K{2,4}/train/log.csv`), so their best step is partly noise.

### σ calibration (`eval/calibration.csv`, window level)

| set | within 1σ per axis (Gaussian 68.3 %) | within 2σ per axis (95.4 %) | median σ (2-D, m/s) | median error (2-D, m/s) |
|---|---|---|---|---|
| val | 74.8 % | 95.7 % | 0.390 | 0.283 |
| test-iid | 75.6 % | 95.7 % | 0.395 | 0.279 |
| test-sea | 69.5 % | 91.8 % | 0.588 | 0.447 |
| test-alt | 75.1 % | 94.5 % | 0.480 | 0.310 |
| test-rate | 36.8 % | 54.5 % | 0.428 | 1.398 |
| e1-core | 80.5 % | 98.3 % | 0.383 | 0.249 |

In distribution σ is about right, slightly wide at 1σ and right at 2σ; the errors are
heavier-tailed than a Gaussian. σ grows on test-sea and test-alt, where the errors grow.
It hardly grows on test-rate (median σ 0.428 m/s), where the median error is
1.398 m/s. **σ does not
detect a sea that moves unlike the training seas.**

σ is a per-window figure. A run's error is mostly a bias that is common to its
windows, so σ/√n says nothing about b.

## Honest limitations

- **The budget is not met.** About 43 % of runs are within 0.17 m/s in distribution, and
  the p90 is 2.6× the budget. b% ≤ 0.28 holds for about 20 % of in-distribution runs.
- **The training was cut by time and was still improving.** The final b may be lower
  with more steps. Nothing here measures how much lower.
- **The network learned the simulator's wave motion as a prior** (sea-rate test). Its
  accuracy on real water depends on how close the simulated dispersion, spectra, foam
  and whitecap drift are to real water. Those artefacts are listed in E2's data notes:
  the preset directions, the gate speed, the k lattice.
- **σ is not a reliable out-of-distribution alarm.**
- **The phase-correlation start is fixed and not learned.** If it is off by more than the
  read-out's window (±4 cells at each scale per iteration, ±12 px/frame per step), the
  read-out cannot recover. On the val subset the window RMS error is 0.61 m/s
  (`train/log.csv`, last row) against a median 2-D error of 0.28 m/s on full val
  (`eval/calibration.csv`), which suggests such outliers exist.
- **Only one seed and one main run.** No error bars come from repeated training.
- **e1-core** differs from training in the break-gate direction and in its presets. It
  does about as well as test-iid.
- **The smoke test predates two small changes** (iteration embedding, per-frame
  normalisation). The full runs used the final code.

## Reproduce

On the box, from `/workspace/mare-realtime/experiments/e2_learned_velocity`:

```sh
# tests (synthetic: start, alignment, cost volume peak, run- vs window-level inference)
/workspace/venv/bin/python -m pytest -q tests/test_e2e.py
# smoke test: 16 train runs, scored on themselves
bash e2e/run_train.sh /workspace/data/e2/e2e/smoke --train-runs 16 --val-on-train --steps 1200 --val-every 300 --val-step 8 --warmup 100 --p-bin 0
# main (K = 8) and the K ablations
bash e2e/run_train.sh /workspace/data/e2/e2e/main   --K 8 --steps 7000 --max-hours 2.5 --val-every 700 --workers 7 --gpu-mem-gb 16
bash e2e/run_train.sh /workspace/data/e2/e2e/abl-K2 --K 2 --steps 7000 --max-hours 2.5 --val-every 700 --workers 3 --gpu-mem-gb 5
bash e2e/run_train.sh /workspace/data/e2/e2e/abl-K4 --K 4 --steps 6412 --max-hours 1.5 --val-every 700 --workers 5 --gpu-mem-gb 9   # steps = what K = 8 reached
# predictions + harness + calibration, val first (selection), then every test set once
GPU_MEM_GB=8 bash e2e/run_eval.sh /workspace/data/e2/e2e/main/best.pt e2e val
GPU_MEM_GB=8 bash e2e/run_eval.sh /workspace/data/e2/e2e/main/best.pt e2e test-iid test-sea test-alt test-rate e1-core
GPU_MEM_GB=4 bash e2e/run_eval.sh /workspace/data/e2/e2e/abl-K2/best.pt e2e-K2 val test-iid test-sea test-alt test-rate e1-core
GPU_MEM_GB=8 bash e2e/run_eval.sh /workspace/data/e2/e2e/abl-K4/best.pt e2e-K4 val test-iid test-sea test-alt test-rate e1-core
# tables and the figure
/workspace/venv/bin/python -m e2e.collect --methods e2e e2e-K4 e2e-K2 b0-phasecorr --out /workspace/data/e2/eval/e2e/summary_by_set.csv
/workspace/venv/bin/python -m e2e.analyse --methods e2e e2e-K4 e2e-K2 phasecorr --out /workspace/data/e2/eval/e2e/sea_rate.csv
/workspace/venv/bin/python -m e2e.plots --runs K8=/workspace/data/e2/e2e/main K4=/workspace/data/e2/e2e/abl-K4 K2=/workspace/data/e2/e2e/abl-K2 --out /workspace/data/e2/eval/e2e/training_curves.png
```

Code lives in `experiments/e2_learned_velocity/e2e/`: `model.py`, `loader.py`,
`train.py`, `predict.py`, `calib.py`, `collect.py`, `analyse.py`, `plots.py`, `bench.py`,
`run_train.sh` and `run_eval.sh`. Its tests are in `tests/test_e2e.py`. On the box,
checkpoints are in `/workspace/data/e2/e2e/{main,abl-K2,abl-K4}/{best,last}.pt`.
Predictions are in `/workspace/data/e2/preds/<method>/` and the harness output in
`/workspace/data/e2/eval/<method>/`.

**Files in this directory.**

- `summary_by_set.csv` and `e1_core_compare_medians.csv`
- `sea_rate.csv`, `training_curves.png`
- `eval/` (per set: per_run, slices, summary, e1_compare; plus calibration.csv and
  iterations.csv)
- `preds/` (prediction CSVs; the .npz files hold the start and every iteration)
- `train/` (args.json, log.csv, train.log, smoke_log.csv)
