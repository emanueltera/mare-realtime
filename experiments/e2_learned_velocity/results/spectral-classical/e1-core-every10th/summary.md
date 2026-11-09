# E2 evaluation: classical

Predictions: `/workspace/data/e2/spectral/scratch/classical-e1core-step10.csv`. Every number below is derived from `per_run.csv` and `slices.csv` in this directory (computed by `e2.evaluate`). b = |run-mean (v_est − v_true)| (m/s); b% = b / mean ground speed; drift% over non-overlapping windows. Budget: b ≤ 0.17 m/s, b% ≤ 0.28; drift budget 0.5 %.

- e1-core: 432 of 432 runs predicted.

e1-core, the same runs as E1 (`e1_compare.csv`): median per-run b, m/s — this method 0.788; flow_searaft_g1 1.676; flow_ufmrefine980_g1 2.030; shell_dispersive_T3.2 0.614; shell_dispersive_T6.4 0.645.

## By set

| set | K | runs | b median | b p90 | b% median | b% p90 | rms median | drift% median | drift% p90 | b≤0.17 | b%≤0.28 | b%≤0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e1-core | 32 | 432 | 0.788 | 1.957 | 1.900 | 7.769 | 0.986 | 2.137 | 8.508 | 0.19 | 0.04 | 0.11 |

## By altitude

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 32 | (0.0, 50.0] | 108 | 0.685 | 1.379 | 1.545 | 1.764 | 0.23 | 0.05 |
| e1-core | 32 | (50.0, 80.0] | 108 | 0.907 | 1.576 | 1.746 | 1.951 | 0.29 | 0.06 |
| e1-core | 32 | (80.0, 130.0] | 108 | 0.526 | 1.968 | 1.372 | 1.492 | 0.24 | 0.05 |
| e1-core | 32 | (200.0, 260.0] | 108 | 0.846 | 2.171 | 3.094 | 3.120 | 0.00 | 0.00 |

## By ground_speed

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 32 | (0, 20] | 144 | 0.428 | 1.880 | 2.854 | 3.024 | 0.38 | 0.03 |
| e1-core | 32 | (20, 35] | 144 | 0.624 | 1.926 | 2.082 | 2.272 | 0.18 | 0.08 |
| e1-core | 32 | (50, 65] | 144 | 1.006 | 2.002 | 1.676 | 1.711 | 0.01 | 0.01 |

## By sun_elevation

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 32 | (5, 15] | 216 | 1.455 | 2.067 | 3.849 | 4.020 | 0.08 | 0.01 |
| e1-core | 32 | (35, 90] | 216 | 0.424 | 1.048 | 1.281 | 1.418 | 0.30 | 0.06 |

## By wind_speed

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 32 | (0.0, 6.0] | 72 | 0.220 | 1.007 | 0.763 | 0.799 | 0.42 | 0.04 |
| e1-core | 32 | (6.0, 10.0] | 288 | 0.868 | 1.997 | 2.134 | 2.319 | 0.16 | 0.04 |
| e1-core | 32 | (10.0, 14.0] | 72 | 0.954 | 1.954 | 2.532 | 3.229 | 0.10 | 0.01 |

## By gimbal_pitch

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 32 | (-90.01, -80.0] | 432 | 0.788 | 1.957 | 1.900 | 2.137 | 0.19 | 0.04 |

## By masked_frac

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 32 | (-0.01, 0.001] | 432 | 0.788 | 1.957 | 1.900 | 2.137 | 0.19 | 0.04 |

## By swell

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 32 | 0 | 288 | 0.723 | 1.830 | 1.771 | 2.034 | 0.20 | 0.04 |
| e1-core | 32 | 1 | 144 | 0.870 | 2.012 | 2.175 | 2.319 | 0.16 | 0.03 |

## By sea_rate

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 32 | 1 | 432 | 0.788 | 1.957 | 1.900 | 2.137 | 0.19 | 0.04 |

## By pattern

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 32 | straight | 432 | 0.788 | 1.957 | 1.900 | 2.137 | 0.19 | 0.04 |
