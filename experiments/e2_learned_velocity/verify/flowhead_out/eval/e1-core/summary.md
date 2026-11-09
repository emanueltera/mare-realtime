# E2 evaluation: flowhead-verify

Predictions: `/workspace/data/e2/verify_flowhead/preds/e1-core.csv`. Every number below is derived from `per_run.csv` and `slices.csv` in this directory (computed by `e2.evaluate`). b = |run-mean (v_est − v_true)| (m/s); b% = b / mean ground speed; drift% over non-overlapping windows. Budget: b ≤ 0.17 m/s, b% ≤ 0.28; drift budget 0.5 %.

- e1-core: 432 of 432 runs predicted.

e1-core, the same runs as E1 (`e1_compare.csv`): median per-run b, m/s — this method 1.688; flow_searaft_g1 1.676; flow_ufmrefine980_g1 2.030; shell_dispersive_T3.2 0.614; shell_dispersive_T6.4 0.645.

## By set

| set | K | runs | b median | b p90 | b% median | b% p90 | rms median | drift% median | drift% p90 | b≤0.17 | b%≤0.28 | b%≤0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | 432 | 1.688 | 3.775 | 5.773 | 20.702 | 1.864 | 5.812 | 19.460 | 0.00 | 0.00 | 0.00 |

## By altitude

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | (0.0, 50.0] | 108 | 1.254 | 26.034 | 6.282 | 6.265 | 0.00 | 0.00 |
| e1-core | 8 | (50.0, 80.0] | 108 | 1.345 | 2.092 | 4.395 | 4.362 | 0.00 | 0.00 |
| e1-core | 8 | (80.0, 130.0] | 108 | 1.907 | 2.785 | 5.532 | 5.590 | 0.00 | 0.00 |
| e1-core | 8 | (200.0, 260.0] | 108 | 2.571 | 3.684 | 8.305 | 7.932 | 0.00 | 0.00 |

## By ground_speed

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | (0, 20] | 144 | 1.349 | 3.025 | 8.992 | 9.216 | 0.00 | 0.00 |
| e1-core | 8 | (20, 35] | 144 | 1.469 | 2.595 | 4.897 | 4.961 | 0.00 | 0.00 |
| e1-core | 8 | (50, 65] | 144 | 2.440 | 21.617 | 4.066 | 4.151 | 0.00 | 0.00 |

## By sun_elevation

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | (5, 15] | 216 | 1.499 | 3.664 | 5.217 | 5.299 | 0.00 | 0.00 |
| e1-core | 8 | (35, 90] | 216 | 1.828 | 3.791 | 6.119 | 6.255 | 0.00 | 0.00 |

## By wind_speed

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | (0.0, 6.0] | 72 | 2.054 | 3.848 | 7.441 | 7.782 | 0.00 | 0.00 |
| e1-core | 8 | (6.0, 10.0] | 288 | 1.619 | 3.732 | 5.689 | 5.832 | 0.00 | 0.00 |
| e1-core | 8 | (10.0, 14.0] | 72 | 1.271 | 3.295 | 4.451 | 4.470 | 0.00 | 0.00 |

## By gimbal_pitch

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | (-90.01, -80.0] | 432 | 1.688 | 3.775 | 5.773 | 5.812 | 0.00 | 0.00 |

## By masked_frac

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | (-0.01, 0.001] | 432 | 1.688 | 3.775 | 5.773 | 5.812 | 0.00 | 0.00 |

## By swell

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | 0 | 288 | 1.669 | 3.708 | 5.702 | 5.785 | 0.00 | 0.00 |
| e1-core | 8 | 1 | 144 | 1.777 | 3.829 | 5.920 | 5.976 | 0.00 | 0.00 |

## By sea_rate

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | 1 | 432 | 1.688 | 3.775 | 5.773 | 5.812 | 0.00 | 0.00 |

## By pattern

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | straight | 432 | 1.688 | 3.775 | 5.773 | 5.812 | 0.00 | 0.00 |
