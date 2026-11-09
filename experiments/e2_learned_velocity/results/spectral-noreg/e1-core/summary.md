# E2 evaluation: spectral-noreg

Predictions: `/workspace/data/e2/preds/spectral-noreg/e1-core.csv`. Every number below is derived from `per_run.csv` and `slices.csv` in this directory (computed by `e2.evaluate`). b = |run-mean (v_est − v_true)| (m/s); b% = b / mean ground speed; drift% over non-overlapping windows. Budget: b ≤ 0.17 m/s, b% ≤ 0.28; drift budget 0.5 %.

- e1-core: 432 of 432 runs predicted.

e1-core, the same runs as E1 (`e1_compare.csv`): median per-run b, m/s — this method 0.260; flow_searaft_g1 1.676; flow_ufmrefine980_g1 2.030; shell_dispersive_T3.2 0.614; shell_dispersive_T6.4 0.645.

## By set

| set | K | runs | b median | b p90 | b% median | b% p90 | rms median | drift% median | drift% p90 | b≤0.17 | b%≤0.28 | b%≤0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e1-core | 32 | 432 | 0.260 | 1.129 | 0.834 | 4.829 | 0.583 | 0.888 | 5.341 | 0.31 | 0.10 | 0.29 |

## By altitude

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 32 | (0.0, 50.0] | 108 | 0.426 | 1.399 | 1.164 | 1.607 | 0.23 | 0.03 |
| e1-core | 32 | (50.0, 80.0] | 108 | 0.253 | 0.885 | 0.819 | 0.841 | 0.34 | 0.07 |
| e1-core | 32 | (80.0, 130.0] | 108 | 0.170 | 0.698 | 0.589 | 0.637 | 0.50 | 0.22 |
| e1-core | 32 | (200.0, 260.0] | 108 | 0.298 | 1.680 | 0.950 | 1.101 | 0.17 | 0.08 |

## By ground_speed

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 32 | (0, 20] | 144 | 0.195 | 1.225 | 1.297 | 1.759 | 0.41 | 0.03 |
| e1-core | 32 | (20, 35] | 144 | 0.233 | 0.882 | 0.777 | 0.772 | 0.35 | 0.10 |
| e1-core | 32 | (50, 65] | 144 | 0.369 | 1.108 | 0.615 | 0.654 | 0.17 | 0.17 |

## By sun_elevation

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 32 | (5, 15] | 216 | 0.406 | 1.404 | 1.148 | 1.444 | 0.18 | 0.06 |
| e1-core | 32 | (35, 90] | 216 | 0.183 | 0.798 | 0.668 | 0.713 | 0.44 | 0.14 |

## By wind_speed

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 32 | (0.0, 6.0] | 72 | 0.166 | 0.744 | 0.530 | 0.531 | 0.51 | 0.21 |
| e1-core | 32 | (6.0, 10.0] | 288 | 0.262 | 1.134 | 0.892 | 0.939 | 0.29 | 0.09 |
| e1-core | 32 | (10.0, 14.0] | 72 | 0.371 | 1.240 | 1.290 | 1.588 | 0.18 | 0.04 |

## By gimbal_pitch

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 32 | (-90.01, -80.0] | 432 | 0.260 | 1.129 | 0.834 | 0.888 | 0.31 | 0.10 |

## By masked_frac

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 32 | (-0.01, 0.001] | 432 | 0.260 | 1.129 | 0.834 | 0.888 | 0.31 | 0.10 |

## By swell

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 32 | 0 | 288 | 0.253 | 1.127 | 0.813 | 0.895 | 0.31 | 0.11 |
| e1-core | 32 | 1 | 144 | 0.272 | 1.139 | 0.879 | 0.841 | 0.31 | 0.08 |

## By sea_rate

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 32 | 1 | 432 | 0.260 | 1.129 | 0.834 | 0.888 | 0.31 | 0.10 |

## By pattern

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 32 | straight | 432 | 0.260 | 1.129 | 0.834 | 0.888 | 0.31 | 0.10 |
