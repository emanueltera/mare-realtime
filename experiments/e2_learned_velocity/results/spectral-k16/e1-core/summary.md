# E2 evaluation: spectral-k16

Predictions: `/workspace/data/e2/preds/spectral-k16/e1-core.csv`. Every number below is derived from `per_run.csv` and `slices.csv` in this directory (computed by `e2.evaluate`). b = |run-mean (v_est − v_true)| (m/s); b% = b / mean ground speed; drift% over non-overlapping windows. Budget: b ≤ 0.17 m/s, b% ≤ 0.28; drift budget 0.5 %.

- e1-core: 432 of 432 runs predicted.

e1-core, the same runs as E1 (`e1_compare.csv`): median per-run b, m/s — this method 0.325; flow_searaft_g1 1.676; flow_ufmrefine980_g1 2.030; shell_dispersive_T3.2 0.614; shell_dispersive_T6.4 0.645.

## By set

| set | K | runs | b median | b p90 | b% median | b% p90 | rms median | drift% median | drift% p90 | b≤0.17 | b%≤0.28 | b%≤0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e1-core | 16 | 432 | 0.325 | 1.147 | 1.028 | 4.140 | 0.633 | 1.011 | 3.912 | 0.33 | 0.18 | 0.29 |

## By altitude

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 16 | (0.0, 50.0] | 108 | 0.255 | 0.761 | 0.813 | 0.953 | 0.35 | 0.12 |
| e1-core | 16 | (50.0, 80.0] | 108 | 0.161 | 0.812 | 0.533 | 0.596 | 0.52 | 0.32 |
| e1-core | 16 | (80.0, 130.0] | 108 | 0.236 | 0.893 | 0.905 | 0.858 | 0.38 | 0.23 |
| e1-core | 16 | (200.0, 260.0] | 108 | 0.719 | 1.571 | 2.007 | 1.917 | 0.06 | 0.03 |

## By ground_speed

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 16 | (0, 20] | 144 | 0.265 | 1.191 | 1.769 | 1.579 | 0.34 | 0.09 |
| e1-core | 16 | (20, 35] | 144 | 0.244 | 1.022 | 0.813 | 0.990 | 0.42 | 0.23 |
| e1-core | 16 | (50, 65] | 144 | 0.454 | 1.154 | 0.757 | 0.752 | 0.22 | 0.21 |

## By sun_elevation

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 16 | (5, 15] | 216 | 0.622 | 1.350 | 1.921 | 1.825 | 0.11 | 0.03 |
| e1-core | 16 | (35, 90] | 216 | 0.150 | 0.590 | 0.544 | 0.557 | 0.54 | 0.32 |

## By wind_speed

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 16 | (0.0, 6.0] | 72 | 0.156 | 0.528 | 0.521 | 0.496 | 0.51 | 0.33 |
| e1-core | 16 | (6.0, 10.0] | 288 | 0.370 | 1.201 | 1.199 | 1.232 | 0.33 | 0.17 |
| e1-core | 16 | (10.0, 14.0] | 72 | 0.404 | 1.035 | 1.490 | 1.286 | 0.14 | 0.06 |

## By gimbal_pitch

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 16 | (-90.01, -80.0] | 432 | 0.325 | 1.147 | 1.028 | 1.011 | 0.33 | 0.18 |

## By masked_frac

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 16 | (-0.01, 0.001] | 432 | 0.325 | 1.147 | 1.028 | 1.011 | 0.33 | 0.18 |

## By swell

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 16 | 0 | 288 | 0.301 | 1.010 | 0.955 | 0.953 | 0.32 | 0.18 |
| e1-core | 16 | 1 | 144 | 0.378 | 1.328 | 1.199 | 1.238 | 0.34 | 0.17 |

## By sea_rate

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 16 | 1 | 432 | 0.325 | 1.147 | 1.028 | 1.011 | 0.33 | 0.18 |

## By pattern

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 16 | straight | 432 | 0.325 | 1.147 | 1.028 | 1.011 | 0.33 | 0.18 |
