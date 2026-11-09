# E2 evaluation: spectral

Predictions: `/workspace/data/e2/preds/spectral/e1-core.csv`. Every number below is derived from `per_run.csv` and `slices.csv` in this directory (computed by `e2.evaluate`). b = |run-mean (v_est − v_true)| (m/s); b% = b / mean ground speed; drift% over non-overlapping windows. Budget: b ≤ 0.17 m/s, b% ≤ 0.28; drift budget 0.5 %.

- e1-core: 432 of 432 runs predicted.

e1-core, the same runs as E1 (`e1_compare.csv`): median per-run b, m/s — this method 0.171; flow_searaft_g1 1.676; flow_ufmrefine980_g1 2.030; shell_dispersive_T3.2 0.614; shell_dispersive_T6.4 0.645.

## By set

| set | K | runs | b median | b p90 | b% median | b% p90 | rms median | drift% median | drift% p90 | b≤0.17 | b%≤0.28 | b%≤0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e1-core | 32 | 432 | 0.171 | 0.515 | 0.599 | 1.888 | 0.366 | 0.603 | 2.171 | 0.50 | 0.27 | 0.45 |

## By altitude

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 32 | (0.0, 50.0] | 108 | 0.267 | 0.864 | 0.888 | 0.980 | 0.38 | 0.15 |
| e1-core | 32 | (50.0, 80.0] | 108 | 0.144 | 0.496 | 0.539 | 0.491 | 0.55 | 0.33 |
| e1-core | 32 | (80.0, 130.0] | 108 | 0.145 | 0.400 | 0.438 | 0.533 | 0.55 | 0.33 |
| e1-core | 32 | (200.0, 260.0] | 108 | 0.166 | 0.356 | 0.570 | 0.546 | 0.52 | 0.27 |

## By ground_speed

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 32 | (0, 20] | 144 | 0.142 | 0.426 | 0.949 | 1.121 | 0.55 | 0.11 |
| e1-core | 32 | (20, 35] | 144 | 0.128 | 0.382 | 0.428 | 0.498 | 0.56 | 0.31 |
| e1-core | 32 | (50, 65] | 144 | 0.255 | 0.826 | 0.426 | 0.424 | 0.39 | 0.39 |

## By sun_elevation

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 32 | (5, 15] | 216 | 0.254 | 0.619 | 0.837 | 0.900 | 0.33 | 0.17 |
| e1-core | 32 | (35, 90] | 216 | 0.114 | 0.411 | 0.367 | 0.430 | 0.67 | 0.37 |

## By wind_speed

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 32 | (0.0, 6.0] | 72 | 0.099 | 0.403 | 0.327 | 0.322 | 0.74 | 0.42 |
| e1-core | 32 | (6.0, 10.0] | 288 | 0.178 | 0.569 | 0.674 | 0.721 | 0.49 | 0.26 |
| e1-core | 32 | (10.0, 14.0] | 72 | 0.252 | 0.490 | 0.820 | 0.902 | 0.31 | 0.18 |

## By gimbal_pitch

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 32 | (-90.01, -80.0] | 432 | 0.171 | 0.515 | 0.599 | 0.603 | 0.50 | 0.27 |

## By masked_frac

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 32 | (-0.01, 0.001] | 432 | 0.171 | 0.515 | 0.599 | 0.603 | 0.50 | 0.27 |

## By swell

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 32 | 0 | 288 | 0.162 | 0.497 | 0.553 | 0.558 | 0.52 | 0.28 |
| e1-core | 32 | 1 | 144 | 0.198 | 0.660 | 0.751 | 0.766 | 0.45 | 0.24 |

## By sea_rate

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 32 | 1 | 432 | 0.171 | 0.515 | 0.599 | 0.603 | 0.50 | 0.27 |

## By pattern

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 32 | straight | 432 | 0.171 | 0.515 | 0.599 | 0.603 | 0.50 | 0.27 |
