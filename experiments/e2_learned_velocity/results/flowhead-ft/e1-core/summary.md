# E2 evaluation: flowhead-ft

Predictions: `/workspace/data/e2/preds/flowhead-ft/e1-core.csv`. Every number below is derived from `per_run.csv` and `slices.csv` in this directory (computed by `e2.evaluate`). b = |run-mean (v_est − v_true)| (m/s); b% = b / mean ground speed; drift% over non-overlapping windows. Budget: b ≤ 0.17 m/s, b% ≤ 0.28; drift budget 0.5 %.

- e1-core: 432 of 432 runs predicted.

e1-core, the same runs as E1 (`e1_compare.csv`): median per-run b, m/s — this method 1.341; flow_searaft_g1 1.676; flow_ufmrefine980_g1 2.030; shell_dispersive_T3.2 0.614; shell_dispersive_T6.4 0.645.

## By set

| set | K | runs | b median | b p90 | b% median | b% p90 | rms median | drift% median | drift% p90 | b≤0.17 | b%≤0.28 | b%≤0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | 432 | 1.341 | 3.254 | 4.370 | 15.911 | 1.576 | 4.476 | 16.308 | 0.00 | 0.00 | 0.00 |

## By altitude

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | (0.0, 50.0] | 108 | 1.229 | 22.886 | 5.262 | 5.362 | 0.00 | 0.00 |
| e1-core | 8 | (50.0, 80.0] | 108 | 1.093 | 1.976 | 3.679 | 3.860 | 0.00 | 0.00 |
| e1-core | 8 | (80.0, 130.0] | 108 | 1.334 | 2.389 | 3.946 | 4.140 | 0.00 | 0.00 |
| e1-core | 8 | (200.0, 260.0] | 108 | 1.900 | 3.144 | 6.113 | 5.780 | 0.02 | 0.01 |

## By ground_speed

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | (0, 20] | 144 | 1.129 | 2.376 | 7.525 | 7.646 | 0.01 | 0.01 |
| e1-core | 8 | (20, 35] | 144 | 1.195 | 2.249 | 3.985 | 4.145 | 0.00 | 0.00 |
| e1-core | 8 | (50, 65] | 144 | 2.111 | 20.181 | 3.519 | 3.802 | 0.00 | 0.00 |

## By sun_elevation

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | (5, 15] | 216 | 1.165 | 3.239 | 3.823 | 4.024 | 0.01 | 0.00 |
| e1-core | 8 | (35, 90] | 216 | 1.507 | 3.244 | 4.897 | 5.060 | 0.00 | 0.00 |

## By wind_speed

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | (0.0, 6.0] | 72 | 1.815 | 3.699 | 6.190 | 6.237 | 0.00 | 0.00 |
| e1-core | 8 | (6.0, 10.0] | 288 | 1.323 | 3.218 | 4.291 | 4.426 | 0.00 | 0.00 |
| e1-core | 8 | (10.0, 14.0] | 72 | 0.996 | 2.710 | 3.398 | 3.418 | 0.01 | 0.00 |

## By gimbal_pitch

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | (-90.01, -80.0] | 432 | 1.341 | 3.254 | 4.370 | 4.476 | 0.00 | 0.00 |

## By masked_frac

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | (-0.01, 0.001] | 432 | 1.341 | 3.254 | 4.370 | 4.476 | 0.00 | 0.00 |

## By swell

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | 0 | 288 | 1.295 | 3.192 | 4.208 | 4.340 | 0.01 | 0.00 |
| e1-core | 8 | 1 | 144 | 1.410 | 3.374 | 4.680 | 4.792 | 0.00 | 0.00 |

## By sea_rate

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | 1 | 432 | 1.341 | 3.254 | 4.370 | 4.476 | 0.00 | 0.00 |

## By pattern

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | straight | 432 | 1.341 | 3.254 | 4.370 | 4.476 | 0.00 | 0.00 |
