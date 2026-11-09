# E2 evaluation: b3-const

Predictions: `/workspace/data/e2/preds/b3-const/e1-core.csv`. Every number below is derived from `per_run.csv` and `slices.csv` in this directory (computed by `e2.evaluate`). b = |run-mean (v_est − v_true)| (m/s); b% = b / mean ground speed; drift% over non-overlapping windows. Budget: b ≤ 0.17 m/s, b% ≤ 0.28; drift budget 0.5 %.

- e1-core: 432 of 432 runs predicted.

e1-core, the same runs as E1 (`e1_compare.csv`): median per-run b, m/s — this method 2.084; flow_searaft_g1 1.676; flow_ufmrefine980_g1 2.030; shell_dispersive_T3.2 0.614; shell_dispersive_T6.4 0.645.

## By set

| set | K | runs | b median | b p90 | b% median | b% p90 | rms median | drift% median | drift% p90 | b≤0.17 | b%≤0.28 | b%≤0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | 432 | 2.084 | 3.928 | 7.016 | 21.115 | 2.105 | 7.098 | 21.065 | 0.00 | 0.00 | 0.00 |

## By altitude

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | (0.0, 50.0] | 108 | 1.152 | 57.715 | 6.283 | 6.387 | 0.00 | 0.00 |
| e1-core | 8 | (50.0, 80.0] | 108 | 1.497 | 1.836 | 5.036 | 5.094 | 0.00 | 0.00 |
| e1-core | 8 | (80.0, 130.0] | 108 | 2.235 | 2.808 | 7.519 | 7.519 | 0.00 | 0.00 |
| e1-core | 8 | (200.0, 260.0] | 108 | 3.138 | 4.220 | 10.521 | 10.524 | 0.00 | 0.00 |

## By ground_speed

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | (0, 20] | 144 | 1.980 | 3.221 | 13.202 | 13.234 | 0.00 | 0.00 |
| e1-core | 8 | (20, 35] | 144 | 1.822 | 3.364 | 6.073 | 6.076 | 0.00 | 0.00 |
| e1-core | 8 | (50, 65] | 144 | 2.372 | 56.673 | 3.953 | 3.966 | 0.00 | 0.00 |

## By sun_elevation

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | (5, 15] | 216 | 2.034 | 4.173 | 6.701 | 6.842 | 0.00 | 0.00 |
| e1-core | 8 | (35, 90] | 216 | 2.143 | 3.268 | 7.200 | 7.243 | 0.00 | 0.00 |

## By wind_speed

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | (0.0, 6.0] | 72 | 2.285 | 3.215 | 7.802 | 7.802 | 0.00 | 0.00 |
| e1-core | 8 | (6.0, 10.0] | 288 | 2.051 | 4.220 | 6.981 | 7.098 | 0.00 | 0.00 |
| e1-core | 8 | (10.0, 14.0] | 72 | 1.892 | 3.638 | 6.499 | 6.584 | 0.00 | 0.00 |

## By gimbal_pitch

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | (-90.01, -80.0] | 432 | 2.084 | 3.928 | 7.016 | 7.098 | 0.00 | 0.00 |

## By masked_frac

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | (-0.01, 0.001] | 432 | 2.084 | 3.928 | 7.016 | 7.098 | 0.00 | 0.00 |

## By swell

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | 0 | 288 | 2.095 | 3.758 | 7.004 | 7.065 | 0.00 | 0.00 |
| e1-core | 8 | 1 | 144 | 2.049 | 4.437 | 7.278 | 7.289 | 0.00 | 0.00 |

## By sea_rate

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | 1 | 432 | 2.084 | 3.928 | 7.016 | 7.098 | 0.00 | 0.00 |

## By pattern

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | straight | 432 | 2.084 | 3.928 | 7.016 | 7.098 | 0.00 | 0.00 |
