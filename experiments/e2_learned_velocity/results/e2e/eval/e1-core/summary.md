# E2 evaluation: e2e

Predictions: `/workspace/data/e2/preds/e2e/e1-core.csv`. Every number below is derived from `per_run.csv` and `slices.csv` in this directory (computed by `e2.evaluate`). b = |run-mean (v_est − v_true)| (m/s); b% = b / mean ground speed; drift% over non-overlapping windows. Budget: b ≤ 0.17 m/s, b% ≤ 0.28; drift budget 0.5 %.

- e1-core: 432 of 432 runs predicted.

e1-core, the same runs as E1 (`e1_compare.csv`): median per-run b, m/s — this method 0.199; flow_searaft_g1 1.676; flow_ufmrefine980_g1 2.030; shell_dispersive_T3.2 0.614; shell_dispersive_T6.4 0.645.

## By set

| set | K | runs | b median | b p90 | b% median | b% p90 | rms median | drift% median | drift% p90 | b≤0.17 | b%≤0.28 | b%≤0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | 432 | 0.199 | 0.453 | 0.667 | 1.385 | 0.287 | 0.668 | 1.356 | 0.43 | 0.15 | 0.34 |

## By altitude

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | (0.0, 50.0] | 108 | 0.255 | 0.521 | 0.838 | 0.847 | 0.36 | 0.06 |
| e1-core | 8 | (50.0, 80.0] | 108 | 0.165 | 0.359 | 0.556 | 0.570 | 0.55 | 0.24 |
| e1-core | 8 | (80.0, 130.0] | 108 | 0.184 | 0.383 | 0.618 | 0.611 | 0.44 | 0.14 |
| e1-core | 8 | (200.0, 260.0] | 108 | 0.222 | 0.445 | 0.708 | 0.668 | 0.36 | 0.14 |

## By ground_speed

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | (0, 20] | 144 | 0.125 | 0.256 | 0.834 | 0.856 | 0.69 | 0.07 |
| e1-core | 8 | (20, 35] | 144 | 0.204 | 0.380 | 0.679 | 0.694 | 0.36 | 0.14 |
| e1-core | 8 | (50, 65] | 144 | 0.333 | 0.536 | 0.556 | 0.546 | 0.23 | 0.23 |

## By sun_elevation

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | (5, 15] | 216 | 0.165 | 0.392 | 0.576 | 0.573 | 0.51 | 0.16 |
| e1-core | 8 | (35, 90] | 216 | 0.230 | 0.475 | 0.775 | 0.771 | 0.34 | 0.13 |

## By wind_speed

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | (0.0, 6.0] | 72 | 0.165 | 0.396 | 0.551 | 0.546 | 0.50 | 0.24 |
| e1-core | 8 | (6.0, 10.0] | 288 | 0.201 | 0.449 | 0.676 | 0.675 | 0.42 | 0.12 |
| e1-core | 8 | (10.0, 14.0] | 72 | 0.238 | 0.476 | 0.784 | 0.775 | 0.36 | 0.14 |

## By gimbal_pitch

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | (-90.01, -80.0] | 432 | 0.199 | 0.453 | 0.667 | 0.668 | 0.43 | 0.15 |

## By masked_frac

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | (-0.01, 0.001] | 432 | 0.199 | 0.453 | 0.667 | 0.668 | 0.43 | 0.15 |

## By swell

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | 0 | 288 | 0.198 | 0.454 | 0.661 | 0.656 | 0.43 | 0.16 |
| e1-core | 8 | 1 | 144 | 0.202 | 0.451 | 0.682 | 0.684 | 0.42 | 0.12 |

## By sea_rate

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | 1 | 432 | 0.199 | 0.453 | 0.667 | 0.668 | 0.43 | 0.15 |

## By pattern

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | straight | 432 | 0.199 | 0.453 | 0.667 | 0.668 | 0.43 | 0.15 |
