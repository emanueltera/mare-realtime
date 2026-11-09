# E2 baselines (B0–B3)

The reference points for E2's learned models, run on the same tiles, windows and runs and
scored by the shared harness (`e2.evaluate`). Code: `experiments/e2_learned_velocity/baselines/`.
Every number below is copied from a CSV in this directory: `tables/overview.csv` (the
harness's `all` row per method × set × K), `tables/slices.csv` (its slices),
`e1_crosscheck/summary.csv`, `b3/selection.csv`, `b3/curves.csv` and the logs in `logs/`.
Numbers marked *derived* are arithmetic on those.

## What was run

| id | method | windows scored | uses the label? |
|---|---|---|---|
| B0 `b0-phasecorr` | masked phase correlation between consecutive tiles (`e2.baselines`, unchanged); window estimate = component-wise median of its K − 1 pair velocities | K = 2, 8, 16, every window (step 1) | no |
| B1 `b1-searaft` | pretrained SEA-RAFT-M on consecutive tiles; per pair v = −median(flow)·gsd/dt over valid cells; window = median over pairs | K = 2, 8, 16, every window | no |
| B2 `b2-shell-dispersive`, `b2-shell-static` | E1's 3-D FFT shell fit (`experiments/e1_wave_bias/shell_fit.py`, imported unchanged) on the E2 tiles | K = 32 and 64 (E1's T = 3.2 / 6.4 s), windows every 16 frames (E1's 1.6 s), **not** every window | **yes: the prior** |
| B3 `b3-mlp` | B1's K = 8 velocity + an MLP correction from (log gsd, \|v_pattern\|, pattern direction) | K = 8, every window | no |

Ablations: `b3-const` (B1 + the train-mean correction, no parameters), `b3-mlp-full` (best
4-feature MLP on val), `b3-mlp-radial` (best rotation-equivariant MLP on val; this is also
the selected `b3-mlp`, so its predictions are the same and are not duplicated here),
`b2-shell-*-kmax0475` (the shell restricted to E1's physical wavelength range, e1-core only).

### Inputs (confirming only allowed inputs)

- B0, B1: the tiles, their validity mask (NaN), the run's tile gsd and dt from `frames.csv`.
  Nothing else; v_true / ENU are filled afterwards by `prediction_frame` for scoring.
- B3: B1's window velocity (from the above) and gsd. No tile pixels, no position, speed,
  heading, θ, sea/sky parameters or run id. Labels only as training targets.
- B2: the tiles, gsd and dt, **plus E1's truth-centred prior**: the search is centred on the
  true window velocity + a uniform ±5 m/s offset per component and spans ±8 m/s. This uses
  the label and is generous to the baseline (a deployable fit would have to resolve the
  shell's ambiguities without it). B2 is therefore an optimistic reference, not an
  estimator a learned model has to be fair to.

## Methods in more detail

**B1.** Tiles are linear luminance; SEA-RAFT gets them re-encoded with the sRGB curve (the
tiles were decoded from sRGB frames), ×255, copied to 3 channels, masked cells 0. Model and
settings as E1: checkpoint `MemorySlices/Tartan-C-T-TSKH-spring540x960-M`, config spring-M,
4 iterations, full resolution (384²). TF32 matmuls/convolutions (≈3× faster than strict
fp32 here; E1 had TF32 convolutions only). A cell counts if it is valid in tile i and its
match lands ≥ 16 px inside tile i + 1 on a valid cell; < 1 % such cells → NaN.

**B2.** E1's `prepare` / `fit_velocity` / `shell_stats` and E1's defaults (whiten weight,
band 1 bin, kmin 3 bins, kmax 0.8 × Nyquist, pad 4, coarse 0.2 / fine 0.02 m/s). E2's tile
indexes [row = +y, col = +x] like E1's [north, east], so E1's (E, N) are E2's tile (x, y)
with no sign change. Differences from E1: E2's tile is 384 cells of the nadir image GSD
(0.62 × height) at a random θ; E1's was 256 cells over ~0.70 × height, ENU-aligned. With
kmax 0.8 × Nyquist the finer E2 cells reach ~1.7× shorter waves than E1 did; the
`kmax0475` ablation restores E1's physical range. The GPU objective runs 256 velocities per
batch (E1: 2048) to fit the shared GPU; that changes speed only.

**B3.** MLP, SiLU, last layer zero-initialised (starts at "no correction"). Features:
`full` = (log(gsd/0.12), |v_p|/30, cos φ, sin φ) → (Δx, Δy); `radial` = (log(gsd/0.12),
|v_p|/30) → (Δ along v_p, Δ across v_p), equivariant by construction. Training data: the
K = 8 windows of the first 1500 train runs (111 000 windows; B1 was run on those only, for
GPU time; 4 inputs need no more), with the 8 dihedral augmentations applied to v_p and the
target together. AdamW (wd 1e-4), one-cycle lr 2e-3, batch 2048, 40 epochs, CPU.
Grid (16 configs): features {full, radial} × hidden {32, 128} × depth {2, 3} × loss {MSE,
Huber δ = 1 m/s}. Selection, by epoch and by config: lowest val median per-run b (ties: val
p90). Selected: **radial, hidden 128, depth 2, Huber, epoch 6** (`b3/selection.csv`, 17 154
parameters).

## Smoke test and training curves (B3)

`b3/smoke_n64.csv`, `b3/smoke_n4096.csv`, `logs/b3_smoke.log` (hidden 256, depth 3, MSE; the
"val" columns there are the same subset): 64 windows — train loss 23.7 → 1.16 over 3000
epochs (still falling); 4096 windows — 42.0 → 36.7 over 300 epochs and flat: the four
features leave most of the error unexplained, as expected for inputs that cannot see the
waves. (`val_mse` sums the two components, `train_loss` averages them.)

Grid (`b3/curves.csv`, `b3/selection.csv`): every config's best val median b lies in
1.60–1.81 m/s against B1's 1.83 (`b1`) and B1 + train-mean's 1.83 (`const`). Huber beats MSE
in every pair (MSE chases B1's flow failures, see below). The selected curve: val median b
1.82 (epoch 1) → 1.61 (epoch 6) → 1.67–1.68 (epochs 10–40) while the train loss keeps
falling (1.53 → 1.40), i.e. the best epoch is an early, partly lucky one; picking per epoch
over 16 × 40 val evaluations of 150 runs overstates B3 on val somewhat (its test numbers
below are the honest ones).

## Wall time and GPU memory

From `logs/` (GPU shared with three other agents' trainings throughout):

- B0: pair velocities on CPU (16 processes): 41–142 s per set, 7.7 min for all six
  (`logs/b0_pairs.log`); windows + harness 7 min. No GPU.
- B1: 129 pairs/s on the idle GPU (val); 13–53 pairs/s once the GPU was shared. Per set:
  val 138 s, train (1500 runs) 4394 s, e1-core 3099 s, test-iid 1608 s, test-rate 2151 s,
  test-sea 837 s, test-alt 898 s (`logs/b1_*.log`); 07:50–10:11 wall. Torch peak 4.03 GiB at
  40 pairs per batch (2.07 GiB at 20; e1-core and test-iid ran in a second process at 20,
  same results up to TF32 rounding); nvidia-smi showed 6.0 GB / 3.4 GB per process.
- B2: 6 processes, ~1.0–2.3 GB GPU each (nvidia-smi); per set 401–2304 s; all sets plus the
  e1-core ablation 6701 s (*derived*, sum of the per-set wall times from the log files'
  timestamps; per-run times are in `b2-shell-logs/`).
- B3: 134 s for the 16-config grid on CPU; prediction seconds. No GPU.

My GPU use was ≈ 15 GB for most of the run (B1 6 GB + 6 × 1.5 GB shell); briefly ≈ 25 GB
(*derived*: 6.0 + 8 × 2.34) in the first two minutes of B2 (8 processes at E1's batch of
2048), which I stopped and restarted at 6 processes × 256.

## Results

b = |run mean of (v_est − v_true)| (m/s), per run; cells are **median / p90 / fraction of
runs with b ≤ 0.17 m/s** (`tables/overview.csv`).

| method (K) | val | test-iid | test-sea | test-alt | test-rate | e1-core |
|---|---|---|---|---|---|---|
| B0 b0-phasecorr (K=8) | 1.26 / 1.98 / 0.00 | 1.36 / 2.07 / 0.00 | 1.21 / 1.99 / 0.00 | 2.68 / 3.21 / 0.00 | 0.95 / 2.64 / 0.20 | 1.39 / 2.43 / 0.00 |
| B1 b1-searaft (K=8) | 1.83 / 3.13 / 0.00 | 1.84 / 3.02 / 0.00 | 1.64 / 3.01 / 0.00 | 3.27 / 4.28 / 0.00 | 1.16 / 4.31 / 0.18 | 2.08 / 3.92 / 0.00 |
| B2 b2-shell-dispersive (K=32) | 1.29 / 2.12 / 0.06 | 1.22 / 2.10 / 0.07 | 1.19 / 2.21 / 0.02 | 1.88 / 2.86 / 0.04 | 0.89 / 2.64 / 0.06 | 0.44 / 1.78 / 0.34 |
| B2 b2-shell-dispersive (K=64) | 1.56 / 3.65 / 0.05 | 1.40 / 3.15 / 0.05 | 1.42 / 3.27 / 0.03 | 1.62 / 2.98 / 0.02 | 1.35 / 3.63 / 0.03 | 0.47 / 1.80 / 0.36 |
| B2 b2-shell-static (K=32) | 1.37 / 2.29 / 0.01 | 1.38 / 2.25 / 0.00 | 1.32 / 2.05 / 0.00 | 2.62 / 3.46 / 0.00 | 0.94 / 2.40 / 0.11 | 1.34 / 2.21 / 0.01 |
| B3 b3-mlp (K=8) | 1.60 / 3.08 / 0.01 | 1.67 / 3.29 / 0.00 | 1.43 / 3.16 / 0.01 | 2.96 / 4.79 / 0.00 | 1.32 / 4.11 / 0.04 | 2.10 / 5.65 / 0.00 |
| b3-const (K=8) | 1.83 / 3.13 / 0.00 | 1.85 / 3.02 / 0.00 | 1.63 / 3.02 / 0.00 | 3.27 / 4.29 / 0.00 | 1.17 / 4.30 / 0.18 | 2.08 / 3.93 / 0.00 |

Same rows, **b% median / drift% median / fraction b% ≤ 0.28 / fraction b% ≤ 0.5**:

| method (K) | val | test-iid | test-sea | test-alt | test-rate | e1-core |
|---|---|---|---|---|---|---|
| B0 (K=8) | 3.49 / 3.48 / 0.00 / 0.00 | 3.73 / 3.76 / 0.00 / 0.00 | 3.13 / 3.18 / 0.00 / 0.00 | 7.15 / 7.11 / 0.00 / 0.00 | 2.59 / 2.58 / 0.13 / 0.21 | 4.48 / 4.46 / 0.00 / 0.00 |
| B1 (K=8) | 5.00 / 4.98 / 0.00 / 0.00 | 5.40 / 5.41 / 0.00 / 0.00 | 4.32 / 4.29 / 0.00 / 0.00 | 9.34 / 9.31 / 0.00 / 0.00 | 3.55 / 3.45 / 0.14 / 0.19 | 7.06 / 7.11 / 0.00 / 0.00 |
| B2 dispersive (K=32) | 3.36 / 4.03 / 0.02 / 0.05 | 3.14 / 3.38 / 0.04 / 0.08 | 2.83 / 3.39 / 0.01 / 0.03 | 4.17 / 4.31 / 0.01 / 0.04 | 2.66 / 3.03 / 0.03 / 0.05 | 1.12 / 1.29 / 0.15 / 0.29 |
| B2 static (K=32) | 3.93 / 4.11 / 0.01 / 0.01 | 3.91 / 4.01 / 0.00 / 0.00 | 3.69 / 4.07 / 0.00 / 0.01 | 7.38 / 7.17 / 0.00 / 0.00 | 2.53 / 3.06 / 0.05 / 0.10 | 4.22 / 4.30 / 0.00 / 0.01 |
| B3 (K=8) | 4.62 / 4.61 / 0.01 / 0.02 | 5.08 / 5.09 / 0.00 / 0.00 | 3.99 / 4.02 / 0.01 / 0.01 | 8.31 / 8.31 / 0.00 / 0.00 | 3.71 / 3.66 / 0.02 / 0.05 | 7.04 / 7.07 / 0.00 / 0.01 |
| b3-const (K=8) | 4.97 / 4.96 / 0.00 / 0.00 | 5.38 / 5.40 / 0.00 / 0.00 | 4.30 / 4.30 / 0.00 / 0.00 | 9.33 / 9.29 / 0.00 / 0.00 | 3.51 / 3.46 / 0.13 / 0.19 | 7.02 / 7.10 / 0.00 / 0.00 |

K barely matters for B0 / B1 (val median b: B0 1.38 / 1.26 / 1.28 and B1 1.85 / 1.83 / 1.82
at K = 2 / 8 / 16): the bias is systematic, not noise that a longer median averages away.

### Sea rate (test-rate, paired runs; `tables/slices.csv`, slice `sea_rate`), median b / fraction ≤ 0.17

| method (K) | rate 0 (frozen) | 0.5 | 1 | 2 |
|---|---|---|---|---|
| B0 (8) | 0.085 / 0.80 | 0.654 / 0.00 | 1.324 / 0.00 | 2.342 / 0.00 |
| B1 (8) | 0.088 / 0.73 | 0.859 / 0.00 | 1.772 / 0.00 | 3.801 / 0.00 |
| B2 dispersive (32) | 0.528 / 0.11 | 0.656 / 0.04 | 1.188 / 0.09 | 2.173 / 0.00 |
| B2 static (32) | 0.277 / 0.37 | 0.760 / 0.04 | 1.323 / 0.02 | 1.805 / 0.01 |
| B3 (8) | 0.541 / 0.15 | 0.775 / 0.02 | 1.584 / 0.00 | 3.488 / 0.00 |
| b3-const (8) | 0.090 / 0.73 | 0.859 / 0.00 | 1.779 / 0.00 | 3.795 / 0.00 |

### Other slices worth knowing (`tables/slices.csv`)

- e1-core by sun elevation, B2 dispersive K = 32: midday (35–90°) median 0.145, p90 0.649,
  57 % of runs ≤ 0.17; low sun (5–15°) 1.275, 11 %. B0 1.32 / 1.49 and B1 2.14 / 2.02 in the
  same bins. This is the only place any baseline reaches the budget for a majority of runs.
- Altitude: B0 on test-iid grows 0.94 → 1.84 m/s from ≤ 50 m to 130–200 m, and 2.15 → 2.98
  on test-alt (200–400 m); B1 1.29 → 2.38 and 2.96 → 3.39. The phase-speed bias grows with
  altitude as E1 found.
- B2 dispersive on test-iid by flight pattern (K = 32): straight 0.96, random 0.90,
  lawnmower 1.16, racetrack 1.16, orbit 1.46, figure-8 1.52.

## Cross-check against E1 on e1-core (`e1_crosscheck/summary.csv`)

Same 432 renders; E1 measured on full frames (flow) or its own ENU tile (shell), with
medians over pairs/windows; E2 on its θ-rotated tile with the harness's means.

| pairing | runs | median b E2 | median b E1 | median (E2 − E1) | median \|b_E2 − b_E1\| (vector) | Spearman |
|---|---|---|---|---|---|---|
| B1 K=2 vs flow:searaft g=1 | 414 | 2.026 | 1.676 | +0.138 | 0.179 | 0.80 |
| B2 dispersive K=32 vs shell T=3.2 | 432 | 0.439 | 0.614 | −0.091 | 0.222 | 0.94 |
| B2 dispersive K=64 vs shell T=6.4 | 432 | 0.469 | 0.645 | −0.054 | 0.225 | 0.91 |
| B2 static K=32 vs shell:static T=3.2 | 432 | 1.336 | 1.526 | −0.174 | 0.302 | 0.89 |
| B2 dispersive kmax0475 K=32 vs T=3.2 | 432 | 0.633 | 0.614 | +0.005 | 0.107 | 0.96 |
| B2 dispersive kmax0475 K=64 vs T=6.4 | 432 | 0.700 | 0.645 | −0.005 | 0.073 | 0.96 |
| B2 static kmax0475 K=32 vs T=3.2 | 432 | 1.454 | 1.526 | −0.010 | 0.064 | 0.96 |

- The shell adaptation reproduces E1 when given E1's wavelength range: median vector
  difference 0.06–0.11 m/s, Spearman 0.96, median of b equal within 0.06 m/s. With E1's
  default kmax on the finer E2 tile it does *better* than E1 at midday (0.145 vs 0.237 at
  K = 32; light slice) because it also uses waves shorter than E1's tile resolved.
- B1 on tiles is biased somewhat more than E1's full-frame flow (median +0.14 m/s per run,
  Spearman 0.80; midday Spearman 0.98, low sun 0.68). Likely causes, not separated: the
  384² tile gives SEA-RAFT less context than 720 × 1280, the grey sRGB re-encoding differs
  from the rendered colour frame, and E1 used 23 pairs per run, B1 all 120. The Pearson r
  of b in `summary.csv` is ≈ 0 because of 12 runs (K = 2) where B1 collapses (b 12.6–52.3
  m/s) and E1 does not; without them r = 0.82 (*derived* from `e1_crosscheck/per_run.csv`).
- 18 e1-core runs (30 m altitude, 60 m/s) have no E1 flow:searaft g=1 value; on the tiles B1
  returns b = 47–67 m/s there: at ~124 px per frame (*derived*: 60 × 0.1 / 0.0485) the flow
  collapses toward zero. 26 e1-core runs (all 30 m, 60 m/s), 11 test-iid runs (30–52 m,
  53–74 m/s) and 8 val runs have B1 b > 10 m/s. B0 handles these (its integer stage searches
  the whole tile).

## What this says for the learned models

- No baseline meets the budget on a live sea: B0/B1/B3 have at most 1.3 % of runs at b ≤ 0.17 on
  val, test-iid, test-sea, test-alt and e1-core; B2 with the truth-centred prior 2–7 % on the
  E2 sets and 34–36 % on e1-core.
- The tile pipeline is not the limit: on the frozen sea B0 gets 0.085 m/s median (80 % ≤ 0.17)
  and B1 0.088 (73 %), and their bias scales with the sea's clock (B0 0.65 / 1.32 / 2.34 at
  rate 0.5 / 1 / 2). What remains on a live sea is the waves' own motion.
- B3 (conditions-only correction) removes 0.17–0.31 m/s of the median on val and the E2 test
  sets with the waves at rate 1 (*derived*: B1 − B3 medians, e.g. test-iid 1.84 − 1.67), not
  more, and makes things worse where the conditions differ from training: frozen sea 0.088 →
  0.541 (it adds a correction for waves that are not there), e1-core p90 3.92 → 5.65.
  `b3-const` equals B1 within 0.02 m/s everywhere: with θ random per run there is no average
  vector to subtract. **A network that only reaches B3's level is not reading the waves.**
  Beating B3 clearly — and, ideally, B2 dispersive's 0.89–1.88 m/s on the E2 sets without
  its truth-centred prior — while keeping the frozen-sea runs near B0's 0.085 is the bar.

## Limitations (honest)

- B2 uses the label to place its search (E1's design); it is optimistic, and still poor on
  the E2 sets. On E2's flights the shell fit is also much worse than on E1's straight nadir
  lines even on a frozen sea (static model, rate 0: median 0.277 at K = 32 against B0's
  0.085 on the same runs), and K = 64 is worse than K = 32 nearly everywhere. A constant
  velocity over 3–6 s is assumed by the shell; manoeuvres, turbulence, oblique gimbal and
  wind aloft (crab) violate that or the ENU-agnostic geometry, but the pattern slice
  (straight runs 0.96 vs random 0.90) shows turns are not the whole story. I did not isolate
  the cause.
- B2 scores windows every 16 frames, not every window (step 1) as the protocol asks for the
  other methods; its per-run b averages 4 (K = 32) or 2 (K = 64) windows on 81-frame runs,
  6 or 4 on e1-core. drift% for B2 therefore rests on 2–4 non-overlapping windows.
- B1 and B3 use SEA-RAFT with TF32 and at 384² tiles; E1 ran fp32-matmul on full frames.
  B3 was trained on the first 1500 of 4500 train runs.
- B3's per-epoch selection on 150 val runs flatters its val numbers (best epoch 6 of 40;
  1.605 vs ~1.68 later); test numbers are the fair ones.
- e1-core was rendered before the break-gate fix (the gate travels toward ~239° there); the
  E2 sets use the fixed gate. Low-altitude, fast runs make B1 collapse (above); medians are
  robust to it, means and p90s are not.

## Protocol

B0 and B1 have no tuned parameters (settings fixed from E1 and the data agent's
`e2.baselines` before any scoring). B2's settings are E1's defaults, fixed before scoring;
the `kmax0475` variant is an ablation for the E1 cross-check, not a selection. B3 was
selected on val only (`b3/selection.csv`); its checkpoint `b3/best.pt` (and the ablation
checkpoints `b3/best-full.pt`, `b3/best-radial.pt`, also chosen on val) were frozen before
any test set was predicted, and each test set was evaluated once per frozen checkpoint.
**Nothing was tuned on test-iid, test-sea, test-alt, test-rate or e1-core.**

## Reproduce

On the box (`/workspace/mare-realtime`, `/workspace/venv`); `bash
experiments/e2_learned_velocity/baselines/run_all.sh` runs all of it in order:

```sh
cd /workspace/mare-realtime/experiments/e2_learned_velocity
SETS="val test-iid test-sea test-alt test-rate e1-core"
bash baselines/b0_pairs.sh                                  # B0 pairs (CPU)
bash baselines/eval_windows.sh b0-phasecorr phasecorr $SETS # windows K=2/8/16 + harness
bash baselines/b1_pairs.sh                                  # B1 pairs (GPU), incl. first 1500 train runs
bash baselines/eval_windows.sh b1-searaft searaft $SETS
bash baselines/b2_shell.sh                                  # B2 (6 processes) + e1-core kmax ablation
# B2 harness, B3 smoke/grid/predict/harness, E1 cross-check, tables: see run_all.sh
/workspace/venv/bin/python -m baselines.correction smoke --preds /workspace/data/e2/preds/b1-searaft --out /workspace/data/e2/baselines/b3
/workspace/venv/bin/python -m baselines.correction train --preds /workspace/data/e2/preds/b1-searaft --out /workspace/data/e2/baselines/b3
/workspace/venv/bin/python -m baselines.compare_e1 --eval /workspace/data/e2/eval --out /workspace/data/e2/baselines/e1_crosscheck
/workspace/venv/bin/python -m baselines.tables --eval /workspace/data/e2/eval --out /workspace/data/e2/baselines/tables
```

Tests: `tests/test_baselines.py` (window medians, labels carried through, B3's dihedral
augmentation equals `e2.data.augment_vector`), with the package's other tests.

## Files here

- `<method>/<set>/` — the harness's `per_run.csv`, `slices.csv`, `summary.md` (and
  `e1_compare.csv` for e1-core); `<method>/preds/` — prediction CSVs (gzip'd above 1 MB;
  `b1-searaft/preds/train.csv.gz` is B3's training input). `b3-mlp-radial` has no preds:
  they are `b3-mlp`'s.
- `b3/` — `selection.csv`, `curves.csv`, smoke curves, checkpoints.
- `b2-shell-diag/<set>.csv.gz` — per window and model: prior, J at the estimate and at the
  truth, second peak, at_edge, quad_ok, λ_c, nk.
- `e1_crosscheck/`, `tables/`, `logs/`, `b2-shell-logs/`.
- `phasecorr/` — the data agent's K = 2 phase-correlation reference (not part of B0–B3).

On the box: pairs and B2/B3 working files in `/workspace/data/e2/baselines/`, predictions in
`/workspace/data/e2/preds/<method>/`, harness outputs in `/workspace/data/e2/eval/<method>/`.
