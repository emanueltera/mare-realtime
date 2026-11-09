# E2 evaluation: e2e-K4

Predictions: `/workspace/data/e2/preds/e2e-K4/e1-core.csv`. Every number below is derived from `per_run.csv` and `slices.csv` in this directory (computed by `e2.evaluate`). b = |run-mean (v_est − v_true)| (m/s); b% = b / mean ground speed; drift% over non-overlapping windows. Budget: b ≤ 0.17 m/s, b% ≤ 0.28; drift budget 0.5 %.

- e1-core: 432 of 432 runs predicted.

e1-core, the same runs as E1 (`e1_compare.csv`): median per-run b, m/s — this method 0.241; flow_searaft_g1 1.676; flow_ufmrefine980_g1 2.030; shell_dispersive_T3.2 0.614; shell_dispersive_T6.4 0.645.

## By set

| set | K | runs | b median | b p90 | b% median | b% p90 | rms median | drift% median | drift% p90 | b≤0.17 | b%≤0.28 | b%≤0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e1-core | 4 | 432 | 0.241 | 0.633 | 0.795 | 2.300 | 0.571 | 0.794 | 2.285 | 0.34 | 0.14 | 0.28 |

## By altitude

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 4 | (0.0, 50.0] | 108 | 0.200 | 0.587 | 0.764 | 0.781 | 0.38 | 0.11 |
| e1-core | 4 | (50.0, 80.0] | 108 | 0.164 | 0.403 | 0.555 | 0.575 | 0.54 | 0.25 |
| e1-core | 4 | (80.0, 130.0] | 108 | 0.228 | 0.403 | 0.732 | 0.720 | 0.35 | 0.17 |
| e1-core | 4 | (200.0, 260.0] | 108 | 0.427 | 0.831 | 1.419 | 1.397 | 0.07 | 0.05 |

## By ground_speed

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 4 | (0, 20] | 144 | 0.212 | 0.492 | 1.415 | 1.401 | 0.38 | 0.04 |
| e1-core | 4 | (20, 35] | 144 | 0.220 | 0.539 | 0.733 | 0.730 | 0.35 | 0.12 |
| e1-core | 4 | (50, 65] | 144 | 0.314 | 0.775 | 0.523 | 0.532 | 0.28 | 0.27 |

## By sun_elevation

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 4 | (5, 15] | 216 | 0.208 | 0.562 | 0.696 | 0.700 | 0.39 | 0.16 |
| e1-core | 4 | (35, 90] | 216 | 0.273 | 0.658 | 0.909 | 0.882 | 0.28 | 0.12 |

## By wind_speed

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 4 | (0.0, 6.0] | 72 | 0.252 | 0.776 | 0.926 | 0.941 | 0.39 | 0.17 |
| e1-core | 4 | (6.0, 10.0] | 288 | 0.216 | 0.563 | 0.742 | 0.736 | 0.36 | 0.15 |
| e1-core | 4 | (10.0, 14.0] | 72 | 0.349 | 0.631 | 0.934 | 0.956 | 0.19 | 0.08 |

## By gimbal_pitch

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 4 | (-90.01, -80.0] | 432 | 0.241 | 0.633 | 0.795 | 0.794 | 0.34 | 0.14 |

## By masked_frac

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 4 | (-0.01, 0.001] | 432 | 0.241 | 0.633 | 0.795 | 0.794 | 0.34 | 0.14 |

## By swell

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 4 | 0 | 288 | 0.254 | 0.636 | 0.827 | 0.834 | 0.33 | 0.14 |
| e1-core | 4 | 1 | 144 | 0.218 | 0.620 | 0.732 | 0.724 | 0.34 | 0.15 |

## By sea_rate

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 4 | 1 | 432 | 0.241 | 0.633 | 0.795 | 0.794 | 0.34 | 0.14 |

## By pattern

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 4 | straight | 432 | 0.241 | 0.633 | 0.795 | 0.794 | 0.34 | 0.14 |
