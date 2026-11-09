# E2 evaluation: b3-mlp-radial

Predictions: `/workspace/data/e2/preds/b3-mlp-radial/test-alt.csv`. Every number below is derived from `per_run.csv` and `slices.csv` in this directory (computed by `e2.evaluate`). b = |run-mean (v_est − v_true)| (m/s); b% = b / mean ground speed; drift% over non-overlapping windows. Budget: b ≤ 0.17 m/s, b% ≤ 0.28; drift budget 0.5 %.

- test-alt: 150 of 150 runs predicted.

## By set

| set | K | runs | b median | b p90 | b% median | b% p90 | rms median | drift% median | drift% p90 | b≤0.17 | b%≤0.28 | b%≤0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test-alt | 8 | 150 | 2.958 | 4.793 | 8.306 | 20.623 | 3.015 | 8.307 | 20.715 | 0.00 | 0.00 | 0.00 |

## By altitude

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| test-alt | 8 | (200.0, 260.0] | 37 | 2.702 | 4.206 | 8.319 | 8.286 | 0.00 | 0.00 |
| test-alt | 8 | (260.0, 330.0] | 57 | 2.845 | 4.702 | 8.584 | 8.559 | 0.00 | 0.00 |
| test-alt | 8 | (330.0, 400.01] | 56 | 3.340 | 4.999 | 8.182 | 8.265 | 0.00 | 0.00 |

## By ground_speed

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| test-alt | 8 | (0, 20] | 23 | 2.433 | 4.169 | 16.994 | 17.105 | 0.00 | 0.00 |
| test-alt | 8 | (20, 35] | 46 | 3.424 | 5.355 | 12.589 | 12.712 | 0.00 | 0.00 |
| test-alt | 8 | (35, 50] | 40 | 3.325 | 4.651 | 7.945 | 7.900 | 0.00 | 0.00 |
| test-alt | 8 | (50, 65] | 31 | 2.886 | 4.089 | 4.992 | 5.024 | 0.00 | 0.00 |
| test-alt | 8 | (65, 200] | 10 | 1.135 | 2.318 | 1.594 | 1.566 | 0.00 | 0.00 |

## By sun_elevation

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| test-alt | 8 | (-90, 5] | 24 | 2.809 | 4.205 | 7.857 | 7.882 | 0.00 | 0.00 |
| test-alt | 8 | (5, 15] | 16 | 3.009 | 4.669 | 7.677 | 7.739 | 0.00 | 0.00 |
| test-alt | 8 | (15, 35] | 43 | 3.266 | 4.902 | 10.129 | 10.140 | 0.00 | 0.00 |
| test-alt | 8 | (35, 90] | 67 | 2.925 | 4.706 | 7.825 | 7.823 | 0.00 | 0.00 |

## By wind_speed

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| test-alt | 8 | (0.0, 6.0] | 53 | 3.347 | 5.172 | 9.850 | 9.866 | 0.00 | 0.00 |
| test-alt | 8 | (6.0, 10.0] | 35 | 3.495 | 5.229 | 8.199 | 8.327 | 0.00 | 0.00 |
| test-alt | 8 | (10.0, 14.0] | 35 | 2.845 | 4.059 | 7.426 | 7.424 | 0.00 | 0.00 |
| test-alt | 8 | (14.0, 16.0] | 27 | 2.343 | 3.322 | 5.157 | 5.133 | 0.00 | 0.00 |

## By gimbal_pitch

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| test-alt | 8 | (-90.01, -80.0] | 49 | 2.925 | 4.157 | 7.825 | 7.823 | 0.00 | 0.00 |
| test-alt | 8 | (-80.0, -70.0] | 56 | 3.092 | 4.764 | 8.139 | 8.120 | 0.00 | 0.00 |
| test-alt | 8 | (-70.0, -59.99] | 45 | 2.886 | 5.337 | 9.452 | 9.463 | 0.00 | 0.00 |

## By masked_frac

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| test-alt | 8 | (-0.01, 0.001] | 146 | 3.013 | 4.809 | 8.396 | 8.340 | 0.00 | 0.00 |
| test-alt | 8 | (0.001, 0.01] | 3 | 2.881 | 3.145 | 5.531 | 5.563 | 0.00 | 0.00 |
| test-alt | 8 | (0.01, 0.05] | 1 | 2.760 | 2.760 | 5.157 | 5.133 | 0.00 | 0.00 |

## By swell

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| test-alt | 8 | 0 | 55 | 3.306 | 4.538 | 8.612 | 8.621 | 0.00 | 0.00 |
| test-alt | 8 | 1 | 95 | 2.785 | 4.896 | 8.165 | 8.203 | 0.00 | 0.00 |

## By sea_rate

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| test-alt | 8 | 1 | 150 | 2.958 | 4.793 | 8.306 | 8.307 | 0.00 | 0.00 |

## By pattern

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| test-alt | 8 | figure8 | 23 | 2.576 | 4.479 | 8.584 | 8.559 | 0.00 | 0.00 |
| test-alt | 8 | lawnmower | 25 | 2.888 | 4.813 | 8.292 | 8.276 | 0.00 | 0.00 |
| test-alt | 8 | orbit | 21 | 2.925 | 4.437 | 9.019 | 8.959 | 0.00 | 0.00 |
| test-alt | 8 | racetrack | 27 | 3.214 | 4.199 | 7.151 | 7.150 | 0.00 | 0.00 |
| test-alt | 8 | random | 24 | 3.348 | 5.410 | 11.965 | 11.954 | 0.00 | 0.00 |
| test-alt | 8 | straight | 30 | 2.886 | 4.881 | 8.017 | 8.010 | 0.00 | 0.00 |
