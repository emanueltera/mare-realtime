# E2 evaluation: b3-mlp-radial

Predictions: `/workspace/data/e2/preds/b3-mlp-radial/e1-core.csv`. Every number below is derived from `per_run.csv` and `slices.csv` in this directory (computed by `e2.evaluate`). b = |run-mean (v_est − v_true)| (m/s); b% = b / mean ground speed; drift% over non-overlapping windows. Budget: b ≤ 0.17 m/s, b% ≤ 0.28; drift budget 0.5 %.

- e1-core: 432 of 432 runs predicted.

e1-core, the same runs as E1 (`e1_compare.csv`): median per-run b, m/s — this method 2.102; flow_searaft_g1 1.676; flow_ufmrefine980_g1 2.030; shell_dispersive_T3.2 0.614; shell_dispersive_T6.4 0.645.

## By set

| set | K | runs | b median | b p90 | b% median | b% p90 | rms median | drift% median | drift% p90 | b≤0.17 | b%≤0.28 | b%≤0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | 432 | 2.102 | 5.650 | 7.045 | 25.925 | 2.150 | 7.067 | 25.955 | 0.00 | 0.00 | 0.01 |

## By altitude

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | (0.0, 50.0] | 108 | 1.228 | 58.198 | 4.844 | 5.314 | 0.01 | 0.00 |
| e1-core | 8 | (50.0, 80.0] | 108 | 1.563 | 2.428 | 5.112 | 5.131 | 0.00 | 0.00 |
| e1-core | 8 | (80.0, 130.0] | 108 | 2.467 | 3.819 | 7.595 | 7.586 | 0.00 | 0.00 |
| e1-core | 8 | (200.0, 260.0] | 108 | 3.691 | 5.861 | 11.068 | 11.054 | 0.00 | 0.00 |

## By ground_speed

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | (0, 20] | 144 | 1.598 | 4.060 | 10.654 | 10.689 | 0.01 | 0.00 |
| e1-core | 8 | (20, 35] | 144 | 1.687 | 4.102 | 5.623 | 5.608 | 0.00 | 0.00 |
| e1-core | 8 | (50, 65] | 144 | 2.692 | 55.135 | 4.486 | 4.489 | 0.00 | 0.00 |

## By sun_elevation

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | (5, 15] | 216 | 2.173 | 5.138 | 7.304 | 7.319 | 0.00 | 0.00 |
| e1-core | 8 | (35, 90] | 216 | 2.042 | 5.744 | 6.120 | 6.123 | 0.00 | 0.00 |

## By wind_speed

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | (0.0, 6.0] | 72 | 2.178 | 4.993 | 7.167 | 7.165 | 0.00 | 0.00 |
| e1-core | 8 | (6.0, 10.0] | 288 | 2.076 | 5.477 | 7.157 | 7.205 | 0.00 | 0.00 |
| e1-core | 8 | (10.0, 14.0] | 72 | 1.909 | 5.686 | 5.668 | 5.770 | 0.00 | 0.00 |

## By gimbal_pitch

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | (-90.01, -80.0] | 432 | 2.102 | 5.650 | 7.045 | 7.067 | 0.00 | 0.00 |

## By masked_frac

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | (-0.01, 0.001] | 432 | 2.102 | 5.650 | 7.045 | 7.067 | 0.00 | 0.00 |

## By swell

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | 0 | 288 | 2.102 | 5.105 | 6.470 | 6.470 | 0.00 | 0.00 |
| e1-core | 8 | 1 | 144 | 2.112 | 5.842 | 7.424 | 7.403 | 0.00 | 0.00 |

## By sea_rate

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | 1 | 432 | 2.102 | 5.650 | 7.045 | 7.067 | 0.00 | 0.00 |

## By pattern

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | straight | 432 | 2.102 | 5.650 | 7.045 | 7.067 | 0.00 | 0.00 |
