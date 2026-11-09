# E3 — relative pose between frames, end to end, without GNSS: results

**Everything here is simulated: the aircraft, the ocean and the sensors. Nothing in E3 has been
run on real water.** The setup is in [`../README.md`](../README.md). Every number below names the
file it comes from. Paths are relative to this folder (`results/`). `eval/<set>/` is written by
`e3/evaluate.py`; `analysis/*.csv` by the scripts in `../analysis/`; the figures by
[`../analyze.py`](../analyze.py). Numbers marked *derived* are arithmetic on those files.
Bracketed ranges [lo, hi] are 95 % bootstrap intervals over runs (or over test-rate conditions).
Runs are the independent unit, not pairs. **There is one training run with one seed**, so no
interval covers training-run variation.

**Terms.** *drift* = horizontal end-point error / horizontal distance flown, in %. *b* = that end
error / run duration, in m/s. Steps are chained from the true start and placed in the world with
one of three attitudes: `true` (isolates the translation), `ahrs` (the AHRS attitude, with its
yaw error; what a real system gets) and `chain` (the method's own rotations from the AHRS start).
`net` is PoseNet (images + sensors). `blind` is the same head with sensors only. `airspeed` is
dead reckoning from airspeed along the nose and the gyro, with no wind knowledge.

## 1. The answer

On simulated 60 s flights with no GNSS, the network's chained steps drift a median **1.18 %
[1.07, 1.39]** of the distance flown with the true attitude. With the AHRS attitude, which a real
system would have, it is **1.76 % [1.40, 2.08]**. With its own rotations it is **2.08 %**. The p90 is
4.6 to 5.3 %, and 7.5 to 14 % of flights stay at or below 0.5 %. The 60 s median flatters closed
routes (orbits, figure-8s), where a constant error partly cancels by the end: straight flights
drift a median 2.65 %, and any 5 to 30 s window about 2 % (§3.3). The sensor-only baselines drift a
median 33 to 36 %, so the images cut drift about 16 to 30 times, depending on the attitude (*derived*). But the airspeed
baseline knows nothing about wind, so it is a weak reference for magnitude. The main limits,
with the assumed 2 m baro bias and 2° AHRS yaw bias (their sizes move with these values; §4):
(1) the network takes its metric scale mostly from the barometer. Pooled over flights, about 73 %
of a run's baro scale error reaches its output, but that figure is weighted toward low flights.
The share passed on rises with altitude, to about all of it above 120 m (§3.2). (2) Heading
error, from the AHRS yaw and from integrating the gyro, adds 50 to 75 % on top at that yaw bias,
and the learned rotation correction adds nothing over the simulated gyro. (3) The estimate
shifts with the speed of the simulated waves: drift is 1.6 to 1.8 times higher on a frozen or
double-speed sea than at the trained rate. That is a weaker form of the wave-speed dependence E2
found. The mechanism is not identified (§3.6).

Sources: `eval/test-long/summary.md`; CIs from `analysis/heading_sensitivity.csv` (yaw bias 2°, the
default); `analysis/scale_source_summary.csv`; `analysis/sea_rate.csv`.

## 2. Tables

### Table 1. test-long drift (120 runs × 60 s)

| method | attitude | median drift % | p90 % | runs ≤ 0.5 % | median b (m/s) |
|---|---|---|---|---|---|
| **net** | true | **1.18** [1.07, 1.39] | 4.63 | 14.2 % | 0.43 |
| **net** | ahrs | **1.76** [1.40, 2.08] | 5.31 | 8.3 % | 0.60 |
| **net** | chain | **2.08** [1.53, 2.36] | 5.34 | 7.5 % | 0.61 |
| blind | true | 32.6 | 63.6 | 0 % | 11.6 |
| blind | ahrs | 32.8 | 64.0 | 0 % | 11.7 |
| blind | chain | 33.5 | 64.5 | 0 % | 11.5 |
| airspeed | true | 34.9 | 76.9 | 0 % | 12.2 |
| airspeed | ahrs | 35.5 | 76.3 | 0 % | 12.3 |
| airspeed | chain | 35.6 | 76.2 | 0 % | 12.1 |

Values from `eval/test-long/summary.md`. CIs from `analysis/heading_sensitivity.csv` (row yaw bias
2.0; computed for the network only). For comparison, val (100 × 12 s, the checkpoint-selection
set) gives net 2.52 / 4.15 / 4.17 % median drift for true / ahrs / chain (`eval/val/summary.md`).

### Table 2. Heading sensitivity (test-long, net, AHRS yaw-bias σ varied)

Only `yaw_bias` was changed (`python -m e3.evaluate --noise yaw_bias=Xd`); yaw white noise stays at
0.3°. The network never sees yaw, so the existing predictions are valid. The true-attitude median
is 1.18 % at every row.

| yaw-bias σ | ahrs median % | ahrs p90 % | ahrs runs ≤ 0.5 % | ahrs / true (medians) | runs with AHRS drift ≤ 1.1 × own true drift | chain median % | airspeed ahrs median % |
|---|---|---|---|---|---|---|---|
| 0° | 1.20 [1.08, 1.38] | 4.64 | 15.8 % | 1.01 | 98 % | 1.75 | 34.9 |
| 0.3° | 1.24 [1.05, 1.45] | 4.46 | 12.5 % | 1.05 | 84 % | 1.70 | 35.0 |
| 0.5° | 1.34 [1.04, 1.51] | 4.32 | 12.5 % | 1.14 | 79 % | 1.74 | 35.0 |
| 1° | 1.45 [1.13, 1.68] | 4.26 | 10.0 % | 1.23 | 61 % | 1.85 | 35.1 |
| **2° (default)** | **1.76** [1.40, 2.08] | 5.31 | 8.3 % | **1.49** | 45 % | 2.08 | 35.5 |
| 3° | 2.14 [1.60, 2.47] | 6.15 | 5.8 % | 1.81 | 38 % | 2.16 | 36.1 |
| 5° | 2.68 [2.18, 3.40] | 7.86 | 3.3 % | 2.27 | 23 % | 2.88 | 36.9 |

Source: `analysis/heading_sensitivity.csv` (also holds 0.1, 0.2, 0.75 and 1.5°). The X = 2° row
reproduces `eval/test-long` exactly. The ≤ 1.1 × column is one-sided: it also counts runs where
the AHRS drift is lower than the true-attitude drift (column `runs_within_10pct_of_true_pct`). The
two-sided share (|ahrs / true − 1| ≤ 0.1) is 98 / 76 / 70 / 47 / 29 / 25 / 14 % for the rows above
(`analysis/verify_findings.csv`).

### Table 3. Sea rate (test-rate: 30 conditions × seaRate 0, 1, 2; 20 s each; paired)

| seaRate | net true median % | net true p90 % | net median b (m/s) | net ahrs median % | net chain median % | blind true % | airspeed true % |
|---|---|---|---|---|---|---|---|
| 0 (frozen) | 2.91 [2.21, 3.60] | 8.52 | 1.27 | 3.77 | 3.99 | 26.6 | 26.8 |
| **1 (trained)** | **1.83** [1.35, 2.38] | 4.05 | 0.75 | 3.53 | 3.18 | 26.6 | 26.8 |
| 2 (double) | 3.36 [2.56, 3.57] | 5.65 | 1.18 | 4.24 | 4.13 | 26.6 | 26.8 |

Paired drift ratio, true attitude (geometric mean of per-condition ratios): rate 0 / rate 1 =
**1.69 [1.31, 2.20]**, Wilcoxon p = 1.1e-5, 77 % of conditions worse. Rate 2 / rate 1 = **1.78
[1.40, 2.27]**, p = 2.3e-4, 83 % worse. Rate 0 against rate 2: 0.95 [0.75, 1.19], p = 0.78. The
baselines use no images and are bit-identical across rates (max difference 0). Sources:
`eval/test-rate/summary.md`, `analysis/sea_rate.csv`.

## 3. Evidence

### 3.1 Overview figures

- `fig_drift_cdf.png`: share of the 120 test-long flights at or below each drift, one panel per
  attitude. The network's curve sits more than a decade left of both baselines in every panel.
- `fig_drift_vs_conditions.png`: drift (true attitude) against altitude, ground speed and sea
  wind, one dot per flight. The baselines get worse with wind and at low ground speed, as dead
  reckoning without wind knowledge should. The network shows no clear trend by eye (fitted
  correlations are in §3.4).
- `fig_tracks.png`: six flights, picked in `analyze.py` (per-flight numbers in
  `analysis/tracks_selected.csv`). Best: an orbit (run 109), 0.08 % at the end but 13.2 m
  (0.74 %) at its worst, because the loop brings the error back. Median: run 059, 1.20 % true,
  3.07 % AHRS. Worst: run 071, flown at 31 m, 7.69 %. The lawnmower (run 017) shows the AHRS yaw
  error bending the track: 54 m end error with the true attitude, 209 m with the AHRS.
- `fig_training.png`: validation curves by epoch (`net_log.csv`, `blind_log.csv`). The network
  ends at 0.136 m and 3.6 % median per-step error; blind ends at 1.199 m and 32.2 %. The network's
  curve was roughly flat over the last three epochs as the learning rate went to zero (0.1359,
  0.1371, 0.1356 m at epochs 13 to 15). The rotation panel shows the learned correction never beats the gyro
  (network 9.055, gyro alone 9.053 thousandths of a degree per step).

### 3.2 Where the scale comes from: mostly the baro

Script `../analysis/scale_source.py`; outputs `analysis/scale_source.csv`,
`analysis/scale_source_summary.csv`. Slopes by altitude bin, leverage, interaction and the
airspeed gap: `analysis/verify_claims.csv`. Bias against altitude, bin SDs and R², the training-
range drop and the altitude effects outside the baro: `analysis/verify_round3.csv`.

- **The reproduction is exact.** Re-chaining with the true attitude matches `evaluate.py`'s drift
  for every run, method and set (max difference 3.6e-14 %). Within a run, the blind model's step
  length follows the re-simulated airspeed white noise with slope 1.00 [0.81, 1.23] (test-long).
  So the predictions were made with the same seeded sensor draw that the analyses re-simulate.
- **Per run, the network's along-track scale error follows the baro's error.** The baro bias is
  drawn independently of altitude (Spearman −0.04 on test-long), so the slope fairly measures how
  much of it passes through. Regressing the scale error on the baro scale error gives slope
  **0.73 [0.64, 0.83]**, R² 0.68 on test-long; 0.85 [0.74, 0.95], R² 0.74 on val; 0.87
  [0.69, 0.98] on test-rate at rate 1. Full inheritance would be 1. The scale scatter not
  explained by the baro has an SD of 1.7 to 1.8 % per run.
- **The pooled slope is not one constant.** By altitude bin (30 to 50, 50 to 80, 80 to 120,
  120 to 200 m) it is 0.66 / 0.82 / 0.88 / 1.13 on test-long (39 / 24 / 22 / 35 runs; top bin
  [0.67, 1.54], R² 0.45) and 0.76 / 0.84 / 0.85 / 1.15 on val. The pooled fit is weighted toward
  low flights, where the baro scale error is largest (SD 5.1 % at 30 to 50 m against 1.3 % at 120
  to 200 m). Runs at or below 50 m supply 73 % of the leverage on test-long and 56 % on val. An
  interaction with log altitude is 0.31 [0.10, 0.52] on test-long but 0.06 [−0.16, 0.30] on val,
  so the trend is clear on one set only. Pooled, the network removes about 15 to 27 % of the
  baro's error, almost all of it low down; above 120 m it passes on about all of it.
- **Which input does that removing was not tested.** The network also sees airspeed and the baro
  height itself. The airspeed scale error adds nothing detectable (slope 0.06 [−0.09, 0.22] on
  test-long). On test-long it is too small to fill the gap to full inheritance (air slope minus
  (1 − baro slope): −0.21 [−0.37, −0.03]); on val it is not (0.03 [−0.19, 0.25]). Dropping the 2
  runs whose baro height lies outside the 30 to 200 m training range moves the test-long slope
  only to 0.76. No run separates the images from the sensors (next step 1 would).
- **Step by step it also reads the baro.** The within-run slope of step length on the baro's white
  noise is 0.48 [0.24, 0.73] on test-long and 0.75 [0.63, 0.86] on val. The two sets disagree,
  and no analysis explained why.
- **Blind is wind-dominated.** Its per-run scale error has no usable baro slope (R² 0.002 on
  test-long). Its median |scale error| is 7.1 % (test-long) and 15.5 % (val), because it cannot
  know the wind.
- **The route decides how much scale error turns into drift.** On closed routes a constant scale
  error mostly cancels at the end point. Median displacement / path is 0.39 on test-long against
  1.00 on val. Drift implied by scale alone is 0.58 % against 2.40 % (Spearman with actual drift
  0.71 and 0.98). Straight runs on test-long drift a median 2.65 %.
- **How much a per-run correction would remove** (test-long, net, true attitude; paired median
  ratios). Removing each run's own scale error: 1.18 % → 0.66 % [0.53, 0.87], ratio 0.77
  [0.60, 0.88]. Removing scale and the mean cross-track angle: → 0.49 % [0.39, 0.57], ratio
  0.59 [0.37, 0.73]. So about 40 % of the 60 s drift is a constant per-run scale and heading
  offset, and the rest varies within the run. On val's 12 s runs the same removal leaves almost
  nothing (2.52 % → 0.039 %, ratio 0.013 [0.007, 0.030]). These are oracle corrections, not
  something a real system could do.
- **Altitude.** Median drift falls from 1.66 % at 30 to 50 m to 0.90 % at 120 to 200 m (test-long;
  39 and 35 runs). Median |baro scale error| falls with it (3.5 % → 0.75 %), since a 2 m bias is a
  bigger fraction low down. With each run's scale removed, drift is 0.90 / 0.47 / 0.50 / 0.71 %
  across four bins: no steady fall, but the lowest bin is still the worst. Its Spearman with
  altitude is −0.08 (p = 0.41) on test-long and +0.22 (p = 0.025) on val. Two altitude effects
  that do not run through the baro partly cancel. The scale error left after the baro fit shrinks
  with altitude (|residual| Spearman −0.24, p = 0.008 on test-long; −0.25, p = 0.013 on val). The
  per-run mean cross-track angle grows with it (|cross_h| +0.23, p = 0.010; +0.24, p = 0.015).
  A floor of about 1 % scale error not from the baro remains at high altitude. The test-long bins hold 22 to 39 runs (39 / 24 / 22 / 35), so
  neighbouring CIs overlap.

### 3.3 How the drift builds up over a minute

Script `../analysis/drift_structure.py`; outputs `analysis/drift_per_run_structure.csv`,
`analysis/drift_by_pattern.csv`, `analysis/error_growth.png`. Window lengths from
`../analysis/sea_rate.py` (`analysis/sea_rate_horizons.csv`).

- `analysis/error_growth.png`: median and RMS error against time (log-log) and median drift so
  far. The network's error grows close to linearly; blind's grows exactly linearly.
- **Growth is close to linear, so the error is mostly bias-like (correlated over many seconds),
  not independent step noise.** Growth alone does not say whether that bias is constant over the
  run; the split-half test below does. The
  per-run growth exponent over 1 to 60 s has median 0.84 [0.76, 0.90]; blind's is 0.99
  [0.97, 1.01]. By pattern: straight 0.96, figure-8 0.55. Below-linear growth comes from closed
  paths turning a body-frame bias back on itself. From 20 to 60 s alone, the per-run median is
  0.72 [0.52, 0.82] (`sea_rate.py`).
- **The 60 s headline depends on the window.** Median drift is about 2 % for windows of 5 to 30 s
  (5 s: 2.01 % [1.79, 2.41]; 30 s: 1.81 % [1.46, 2.02]) and 1.18 % at 60 s, because closed
  patterns partly cancel. The 60 s figure therefore understates the per-distance error of open
  flight. p90 is 4.6 to 5.5 % at every window.
- **Random step noise is a small part.** The end error is a median 9.2 [7.6, 11.5] times what
  independent steps with the observed per-step scatter would give (29.5 on straight runs, 5.8 on
  figure-8s). By that measure, random noise is about 1 % of the error energy (*derived*: 1/9.2²).
  Per-step errors are correlated (median lag-1 autocorrelation of the raw step error 0.77
  [0.76, 0.79]; 0.71 for the residual after an affine fit), so this comparison is indicative.
  Against the affine-residual scatter instead, the ratio is 12.5 [10.0, 14.5]
  (`analysis/verify_findings.csv`).
- **About half of the bias is stable over the minute; the rest wanders.** A constant
  camera-frame offset fitted on 0 to 30 s explains 51 % [31, 68] of the error energy over 30 to
  60 s. The oracle per-run scale and heading removal in §3.2 cuts about 40 % of the 60 s drift. The along-track scale error in the
  first half correlates with the second half at Spearman 0.86 [0.77, 0.92].
- **Along-track error dominates.** 97.2 % [95.3, 98.3] of the path-integrated error energy is
  along-track. Median |along| is 1.69 % of distance against 0.32 % cross-track. It has no
  systematic sign (signed medians about 0).
- **The sets agree at equal length.** test-rate at rate 1 (20 s) gives 1.83 % [1.35, 2.39];
  test-long cut at 20 s gives 1.88 % [1.61, 2.51] (Mann-Whitney p = 0.79;
  `analysis/sea_rate.csv`). val does not fit as well: its 12 s median is 2.52 %
  (`eval/val/summary.md`), against 1.81 % [1.63, 2.33] for test-long cut at 12 s. No analysis
  computed a CI for val or explained the gap.
- `fig_error_growth.png`: median drift so far with IQR band, true and AHRS attitude. The ratio
  falls with time (about 2 to 2.5 % early on, 1.18 % at 60 s) and the AHRS curve stays above the
  true one throughout (1.76 % at 60 s; `analysis/error_growth.csv`).

### 3.4 What the drift depends on

Source: `analysis/drift_vs_conditions.csv` (test-long, net, true attitude; Spearman with
bootstrap CIs and permutation p).

- **Altitude** −0.29 [−0.46, −0.10], p = 0.002. Holding the relative baro bias fixed, it drops to
  −0.11 [−0.30, 0.11]. Much of the altitude effect is consistent with the baro bias being a
  larger fraction of height low down. The drop is partly built in: |baro bias / height| is itself
  a function of altitude (Spearman −0.57 with it; `analysis/verify_claims.csv`). Altitude effects
  that do not go through the baro also exist (§3.2), and the causal test (predictions with
  `baro_bias=0`) has not been run.
- **|baro bias / height|** +0.36 [0.18, 0.52]; still +0.25 [0.07, 0.43] with altitude held fixed.
- **Ground speed** −0.25 [−0.42, −0.06]; −0.29 [−0.46, −0.11] with the baro bias held fixed.
- **Displacement / path** +0.43 [0.27, 0.56] (open routes drift more at the end point).
- **Mean |bank|** −0.37 [−0.53, −0.20], p < 0.0005, the strongest single correlate. With
  displacement / path held fixed it drops to −0.09 [−0.28, 0.07]: banked flights fly closed
  routes, whose end-point error partly cancels.
- **Airspeed** −0.25 [−0.42, −0.07], p = 0.004, about the same as ground speed.
- **No detectable relation** with sea wind (+0.01), wind aloft (−0.03), crab (0.00), sun
  elevation (+0.02) or mount pitch (+0.08); all CIs include 0.
- **The bad tail** (drift above the p90 of 4.63 %, 12 runs) has a median |baro bias / height| of
  4.8 % against 1.6 % in the rest (p = 0.006). No figure-8 or racetrack run is in it.
- 17 conditions were tested at n = 120, so a single p near 0.05 means little.

### 3.5 Heading: the AHRS yaw and the gyro

Sources: `analysis/heading_sensitivity.csv`; yaw sweep evaluations written by `e3/evaluate.py` to
the session scratchpad (`e3data/eval_yaw/`).

- Drift rises steadily with the AHRS yaw-bias σ (Table 2). At the default 2°, the AHRS median is
  1.49 times the true-attitude median.
- Keeping the median within 10 % of the true-attitude median takes a yaw-bias σ of about **0.4°
  (95 % CI 0.23 to 1.50)**. The threshold is poorly constrained because the curve is flat below
  0.5°. On a per-run criterion (median of each run's AHRS / true ratio) it is 1.8° [1.1, 2.4].
  These two crossings were printed by `drift_structure.py`, not saved to a CSV.
- **The chained attitude never comes within 10 % of true**, even with a perfect AHRS yaw: 1.75 %
  at 0°, 1.48 times true. Its heading error comes from integrating the gyro. At 0° the median
  |yaw error| at the end is 1.88° for the network's own rotations and 1.87° for the gyro alone.
  The mean per-step rotation error is 0.0096° for both (`eval/test-long/summary.md`).

### 3.6 Sea rate: the estimate depends on the waves' speed

Script `../analysis/sea_rate.py`; outputs `analysis/sea_rate.csv`,
`analysis/sea_rate_per_condition.csv`. Figure `fig_sea_rate.png`. Phase-speed medians, the
wind-sea against swell split and the heading check: `analysis/verify_round3.csv`.

- `fig_sea_rate.png`: test-rate drift (true attitude), median with IQR bars, at wave speeds 0, 1
  and 2. The network is best at 1; the baselines are flat because they use no images.
- **Drift is worse off the trained rate** (Table 3). Within a condition the path and the sensor
  draw are identical across rates, so the difference comes from the images alone.
- **Part of the extra error lies along the wind-sea direction, with opposite signs at the two
  rates.** Signed along-wave velocity error (+ toward the direction the wind sea travels): +0.86
  [0.53, 1.12] m/s at rate 0, −0.08 [−0.41, −0.003] at rate 1, −0.84 [−1.02, −0.66] at rate 2.
  The paired change is +0.81 [0.60, 1.13] m/s at rate 0 (87 % of conditions positive) and −0.65
  [−0.86, −0.38] at rate 2 (80 % negative). The cross-wave change is not significant (p = 0.70,
  0.90). The change points along the waves (Rayleigh p = 1.5e-7 at rate 0, 5e-5 at rate 2). Per
  condition, the rate-0 and rate-2 changes are anticorrelated (Spearman −0.76). A network that
  measured motion independently of the waves would keep its rate-1 error at every rate.
- **Its size does not scale with the waves' speed.** Against the peak phase speed (median
  17.1 m/s, range 8.0 to 23.2), the signed change has Spearman +0.06 (p = 0.75) at rate 0 and
  −0.23 (p = 0.23) at rate 2; |Δ along| has 0.07 (p = 0.71) and 0.30 (p = 0.10). The median
  change is 4 to 5 % of the median phase speed. Half the conditions (15) have a swell peak. Yet
  where wind sea and swell differ by more than 60° (8 conditions), the change follows the wind
  sea (median +0.59 m/s at rate 0, −0.47 at rate 2) and is about zero along the swell (−0.03,
  0.00). So the dominant wave system is not what moves it. The sign pattern fits an estimate
  that depends on the waves' motion. It does not identify the mechanism, such as following the
  pattern and subtracting a learned rate-1 correction. The peak phase speed is not the speed of
  the texture the camera sees, so the null relation does not rule that mechanism out either.
- **There is also an along-track part, mainly at rate 2.** Flight headings are unrelated to the
  wave direction (mean cos of track to wind sea −0.02 [−0.23, 0.18]), so a purely along-wave
  change would roughly average out along-track. It does not: the paired along-track change is
  −0.66 [−0.79, −0.33] m/s at rate 2 (p = 2.8e-4, 80 % of conditions negative), as large as the
  along-wave change, and +0.39 [0.10, 0.65] at rate 0 (p = 0.014, 73 % positive). On faster
  waves the network also under-reads its own speed. The cross-track change is not significant
  (p = 0.66, 0.69).
- **Rate 0 adds bias only; rate 2 adds bias and noise.** Step scatter about the run mean is
  unchanged at rate 0 (+0.003 m, p = 0.33) and grows at rate 2 (+0.038 [0.026, 0.042] m).
- **Against E2.** The effect is much weaker. E3's ratio of median b is 1.70 [1.29, 2.18] (rate 0 /
  rate 1) and 1.58 [1.16, 2.12] (rate 2 / rate 1). E2's model A at K = 8 had 6.8× and 11.7×, also
  ratios of medians (copied from E2's RESULTS.md §2.4 into `sea_rate.csv`). The median of E3's
  per-condition ratios is 1.63 [1.24, 1.94] and 1.66 [1.27, 2.35] (`analysis/verify_findings.csv`). The metrics differ (E2: 8 s velocity
  bias on stabilised nadir tiles; E3: chained 20 s drift from hard-mounted frames with noisy
  sensors), and E3's rate-1 floor is about 4 times higher, which shrinks a ratio for the same
  added error. Only the direction and rough size are comparable.
- **If the network does follow the pattern, how much cannot be read from these data.** The along-wave
  error changes by about −0.65 to −0.8 m/s per unit of sea rate. Full reliance on a fixed prior
  would give minus the apparent pattern pull of these frames, and a true measurement would give
  0. No pattern-tracking baseline exists on E3 frames to give that pull.
- Off the trained rate, the AHRS and chain attitudes dilute the effect, since their yaw error is
  the same at every rate (geometric-mean ratios 1.13 to 1.31; the rate-0 CIs include 1).

## 4. What this does not show

- **Simulated only.** Aircraft, ocean, camera and sensors all come from the same simulator family
  the network was trained on. Nothing has been checked against real recordings. "Depends on the
  waves" refers to this renderer's wave field.
- **One training run, one seed.** No interval covers training-run variation.
- **Two frames, 0.1 s apart.** The network has no memory. It cannot build up height, wind or
  scale over time. Its per-run scale follows the baro bias; step to step, between about half
  (test-long, 0.48) and three quarters (val, 0.75) of the baro's noise reaches the output (§3.2).
- **The sensor noise model is assumed** (`NOISE` in `e3/data.py`): a per-run bias plus per-frame
  white noise, nothing else. AHRS tilt bias σ 0.3°, noise 0.1°; AHRS yaw bias σ 2°, noise 0.3°;
  gyro bias σ 0.05°/s per axis, noise 0.003° per step; baro bias σ 2 m, noise 0.3 m; airspeed
  scale error σ 2 %, noise 0.3 m/s, measured along the nose. No baro drift within a run, no sensor
  lag, no airspeed direction. Real sensors may be better or worse; the results move with these
  values, the baro bias and yaw bias most.
- **The camera is ideal.** Pinhole with exact, known intrinsics, no lens distortion, rolling
  shutter, motion blur, exposure changes, lens contamination or vibration, and an exactly known
  mount (the nose direction in camera axes is taken from the simulator). The frames are rendered,
  shrunk to 288 × 512 luma, and the network trains and tests on the same renderer.
- **The gyro bias is moderate.** 0.05°/s per axis is a calibrated MEMS gyro; an uncalibrated one
  can be several times worse, which would grow the `chain` heading error in proportion.
- **The baro bias sets much of the result.** Pooled, about 73 % of it reaches the output on
  test-long, but the share rises with altitude, to about all of it above 120 m (§3.2). A 2 m bias
  is 6.7 % of height at 30 m (*derived*). Which input removes the rest is untested. The
  counterfactual "drift with zero baro bias" was not run either: the baro is a network input, so
  it needs new predictions on the GPU box.
  The "baro undone" figures in `scale_source.csv` (`drift_nobaro`) assume full inheritance, so
  they overcorrect, mostly on low flights.
- **The airspeed baseline knows no wind.** It is a strawman for magnitude: its 35 % drift mostly
  measures the wind. The 16 to 30 times improvement is against that, not against a dead reckoner
  with a wind estimate. Blind has the same handicap.
- **Rotation: the network adds nothing over the simulated gyro** (§3.1, §3.5). All of the
  improvement is in translation. The simulated gyro's bias (0.05°/s) still drives the `chain`
  heading error.
- **The 60 s median flatters closed routes.** Straight flights drift a median 2.65 %, and windows
  of 5 to 30 s about 2 % (§3.3). Peak error during orbits and figure-8s is larger than the end
  error.
- **val and test-long disagree at 12 s** (2.52 % against 1.81 %), and the baro slopes differ
  (0.85 against 0.73, CIs just overlapping). No analysis found why. val was also the
  checkpoint-selection set.
- **Small samples where noted.** test-rate has 30 conditions; test-long altitude bins hold 22 to 39
  runs (val's 18 to 28).
- **Do not quote** the in-sample bias/random split in `analysis/drift_per_run_structure.csv`
  (columns `E_sys_m`, `E_res_m`, `E_offset_m`, about 99.7 % "systematic"). The fit forces the
  residual sum to zero by construction. Use the split-half and random-walk figures in §3.3.
- **Assumptions in the analyses.** The prediction files carry no noise key. The analyses assume
  the default `NOISE`, which the exact reproduction of `per_run.csv` and the blind airspeed slope
  of 1 support. Bootstrap seeds differ between scripts, so the same median can carry slightly
  different CIs (60 s true-attitude median: [1.07, 1.39] in `heading_sensitivity.csv`,
  [1.08, 1.39] in `error_growth.csv`).
- **One file was overwritten.** `analysis/error_growth.csv` is written by both
  `drift_structure.py` and `analyze.py`. The copy on disk is `analyze.py`'s (net, true and AHRS,
  median / IQR / CI). The growth exponents quoted in §3.3 come from
  `drift_per_run_structure.csv`, not from that file.

## 5. Setup, cost and next steps

**Setup.** See [`../README.md`](../README.md). In short: PoseNet (7.76 M parameters, counted from
`e3/model.py`; blind head 0.17 M) takes two consecutive 10 Hz frames from a hard-mounted camera
plus a 14-value sensor vector (AHRS gravity, gyro rotation, baro height, airspeed). It predicts
translation in camera axes, in tenths of the baro height, and a correction to the gyro rotation.
Trained for 16 epochs on 360 k pairs (3000 × 12 s runs), 3.4 h (`net_log.csv`). Flights at
30 to 200 m and 15 to 60 m/s airspeed, wind aloft from the sea's wind, so the aircraft crabs.
Scored by `e3/evaluate.py` on val (100 × 12 s), test-long (120 × 60 s) and test-rate (30 × 3 ×
20 s).

**Cost.** On one RTX PRO 6000 (96 GB): rendering 3.0 h (18:54 to 21:54 UTC, 30 Oct;
`logs/chain.log`); a failed first training attempt and its fix, about 0.45 h (`logs/run-failed-1.log`;
the data loader hit the 1024 open-file limit); training 3.4 h, then prediction and scoring
0.04 h (22:21 to 01:49 UTC; `logs/run.log`, `net_log.csv`). About 6.9 h of box time in all, about
$37 at the instance's $5.33/h (the rate is vast.ai's billing, not recorded in this folder;
*derived*). The box bills whether busy or not, so idle time around the run is not counted here.
Blind trained in 0.049 h (`blind_log.csv`), alongside the network.

**Next steps.**

1. **Measure the baro dependence directly.** Run `e3.predict` and `e3.evaluate` on test-long with
   `--noise baro_bias=0` and with a larger bias (e.g. `baro_bias=4`), on the GPU box. That gives
   the causal inheritance slope by altitude, the drift with a perfect baro, and whether the
   altitude effect on drift runs through the baro.
2. **Train on varied sea rates.** Draw seaRate from, say, 0 to 2 in training and rerun test-rate.
   If the along-wave and along-track changes with opposite signs at rates 0 and 2 go away,
   training on one rate was the cause.
3. **Add a pattern-tracking baseline on E3 frames** (phase correlation, like E2's B0), so the
   apparent wave pull is known and the −0.65 to −0.8 m/s per unit rate slope can be read as prior
   or measurement.
4. **Give the network more than two frames** (or a longer baseline), so it can estimate height and
   scale from the images over time instead of per step from the baro, and train a second seed to
   measure training-run variation.

## Verification

Seven findings from two independent number checks were re-checked against the data with
`../analysis/verify_findings.py` (output `analysis/verify_findings.csv`). All seven held and were
applied.

- **Table 2, "within 10 %" column.** `drift_structure.py` counts AHRS drift ≤ 1.1 × true drift
  (one-sided); re-computed from the yaw-sweep `per_run.csv` files: 98.3 / 84.2 / 79.2 / 60.8 / 45.0 /
  37.5 / 22.5 %. Column renamed to "≤ 1.1 × own true drift"; two-sided values (98.3 / 75.8 / 70.0 /
  46.7 / 29.2 / 25.0 / 14.2 %) added under the table.
- **Altitude bin sizes (§3.2, §4).** test-long bins hold 39 / 24 / 22 / 35 runs
  (`scale_source_summary.csv`); the 18 was val's 80 to 120 m bin. Changed to 22 to 39.
- **§3.6 b ratios.** 1.70 and 1.58 are ratios of median b (1.270 / 0.748, 1.179 / 0.748 m/s);
  medians of per-condition ratios are 1.63 [1.24, 1.94] and 1.66 [1.27, 2.35]. Reworded as "ratio of
  median b", noted that E2's figures are the same kind, and added the per-condition medians.
- **§3.1 training curve.** `net_log.csv` val_t_err_m at epochs 12 to 15: 0.1395, 0.1359, 0.1371,
  0.1356 m. "Still falling" replaced by "roughly flat over the last three epochs".
- **§3.3 autocorrelation.** The 0.71 is the affine-fit residual's lag-1 autocorrelation (`acf1`);
  the raw step error, whose scatter the 9.2 ratio uses, has 0.77 [0.76, 0.79]. Re-computed ratio
  9.18 [7.77, 11.23] (CI differs from the quoted one only by bootstrap seed); against the residual
  scatter 12.5 [10.0, 14.5]. Both now stated.
- **§3.4 omitted correlates.** `drift_vs_conditions.csv` holds mean |bank| −0.37 [−0.53, −0.20]
  (−0.09 [−0.28, 0.07] with displacement / path held fixed) and airspeed −0.25 [−0.42, −0.07],
  p = 0.004. Both added.
- **§5 cost.** No file in the experiment records the 3 h rendering time or the $5.33/h rate (only
  the 3.4 h in `net_log.csv` and 0.049 h in `blind_log.csv`). The total is now marked as an estimate
  using figures from outside the experiment's files, not *derived*.

`../README.md` was checked and needed no change: its "1.6 to 1.8 times" sea-rate summary still
covers every version of the ratio above.

### Round 3

Eight further findings were re-checked against the data with `../analysis/verify_round3.py`
(output `analysis/verify_round3.csv`) and the earlier `../analysis/verify_claims.py` (output
`analysis/verify_claims.csv`). All eight held and were applied. Two figures in them differed
slightly from what the data give; the data's values are used.

- **Baro slope by altitude (§1, §3.2, §4).** The bias is unrelated to altitude (Spearman −0.04,
  p = 0.68). Slopes by bin are 0.66 / 0.82 / 0.88 / 1.13 on test-long and 0.76 / 0.84 / 0.85 /
  1.15 on val; top-bin R² 0.45, CI [0.67, 1.54]. Baro scale SD 5.1 % low, 1.3 % high. Added the
  bins, the leverage share (73 % / 56 % from runs ≤ 50 m) and the interaction (clear on test-long
  only). "73 %" is now called pooled, and "15 to 27 % removed" is no longer a property of the
  network.
- **Who removes the rest (§3.2).** Nothing separates images from sensors. Reworded to "the network
  removes"; added the airspeed gap test and the training-range drop. The finding's "4 runs, slope
  0.77" did not reproduce exactly: 2 runs lie outside 30 to 200 m (slope 0.76); a 33 to 180 m
  window drops 15 (0.78). Same conclusion.
- **Altitude through the baro (§3.2, §3.4).** |b|/h correlates −0.57 with altitude, so the partial
  drop is partly built in. Residual scale against altitude −0.24 (p = 0.008) and −0.25 (p = 0.013);
  |cross_h| +0.23 (p = 0.010) and +0.24 (p = 0.015); scale-removed drift +0.22 (p = 0.025) on val.
  The cross-track figure holds for the signed mean angle (|cross_h|), not for the summed |cross|
  (`cross_abs`, +0.14, p = 0.14). "Runs through the baro" weakened; the other channels added.
- **Sea-rate mechanism (§1, §3.6, README).** Phase speed median 17.1 m/s (8.0 to 23.2); the
  correlations with it are null (rows already in `sea_rate.csv`, now reported). With wind sea and
  swell more than 60° apart (n 8) the change follows the wind sea. "This is what a network does
  when ..." dropped; section retitled; "wave-motion prior" changed to "wave-speed dependence",
  which is also how E2 states its own finding.
- **Along-track part (§3.6).** Mean cos of track to wind sea −0.02 [−0.23, 0.18]. Paired
  along-track change −0.66 [−0.79, −0.33] m/s at rate 2 (p = 2.8e-4, 80 % negative) and +0.39
  [0.10, 0.65] at rate 0 (p = 0.014) from `sea_rate.csv`. Added as its own bullet.
- **"Largely from the baro reading of that step" (§4).** The step-level slopes are 0.48 and 0.75.
  Replaced with those.
- **"Mostly a consistent bias" (§3.3).** Close-to-linear growth rules out independent step noise
  but not noise correlated over many seconds; the split-half test explains 51 % [31, 68]. Now
  "bias-like", with "about half stable" in the split-half bullet.
- **Ranking of limits (§1).** Table 2 takes the heading penalty from 1.01× to 2.27× with the yaw
  bias, and the baro share scales with the assumed 2 m. §1 now states both assumptions.

`../README.md` was changed in its result paragraph: "inherits about 73 %" is now pooled and
altitude-dependent, and "wave-motion prior" is now "wave-speed dependence".

### Round 4 (scope)

The scope reviewer did not finish (the machine slept twice), so this pass was done by hand
against §1 and §4:
- §1 now says the 60 s median flatters closed routes, with the straight-flight (2.65 %) and
  short-window (about 2 %) figures from §3.3.
- §4 gained two caveats: the ideal camera (exact intrinsics and mount, no distortion, rolling
  shutter, blur or exposure changes) and the moderate gyro bias.
- §5's cost now uses the logged times (`logs/chain.log`, `logs/run.log`, `logs/run-failed-1.log`).
