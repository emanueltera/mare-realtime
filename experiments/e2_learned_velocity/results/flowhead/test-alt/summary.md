# E2 evaluation: flowhead

Predictions: `/workspace/data/e2/preds/flowhead/test-alt.csv`. Every number below is derived from `per_run.csv` and `slices.csv` in this directory (computed by `e2.evaluate`). b = |run-mean (v_est − v_true)| (m/s); b% = b / mean ground speed; drift% over non-overlapping windows. Budget: b ≤ 0.17 m/s, b% ≤ 0.28; drift budget 0.5 %.

- test-alt: 150 of 150 runs predicted.

## By set

| set | K | runs | b median | b p90 | b% median | b% p90 | rms median | drift% median | drift% p90 | b≤0.17 | b%≤0.28 | b%≤0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test-alt | 8 | 150 | 2.642 | 3.569 | 7.069 | 16.091 | 2.741 | 7.019 | 15.928 | 0.00 | 0.00 | 0.00 |

## By altitude

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| test-alt | 8 | (200.0, 260.0] | 37 | 2.010 | 3.163 | 4.829 | 4.796 | 0.00 | 0.00 |
| test-alt | 8 | (260.0, 330.0] | 57 | 2.681 | 3.622 | 7.386 | 7.203 | 0.00 | 0.00 |
| test-alt | 8 | (330.0, 400.01] | 56 | 2.804 | 3.669 | 7.123 | 7.101 | 0.00 | 0.00 |

## By ground_speed

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| test-alt | 8 | (0, 20] | 23 | 2.317 | 2.989 | 15.333 | 15.417 | 0.00 | 0.00 |
| test-alt | 8 | (20, 35] | 46 | 2.835 | 3.716 | 10.769 | 10.405 | 0.00 | 0.00 |
| test-alt | 8 | (35, 50] | 40 | 2.723 | 3.583 | 6.256 | 6.406 | 0.00 | 0.00 |
| test-alt | 8 | (50, 65] | 31 | 2.681 | 3.626 | 4.707 | 4.801 | 0.00 | 0.00 |
| test-alt | 8 | (65, 200] | 10 | 1.625 | 2.308 | 2.356 | 2.257 | 0.00 | 0.00 |

## By sun_elevation

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| test-alt | 8 | (-90, 5] | 24 | 2.622 | 3.361 | 6.259 | 6.092 | 0.00 | 0.00 |
| test-alt | 8 | (5, 15] | 16 | 2.638 | 3.452 | 5.942 | 5.946 | 0.00 | 0.00 |
| test-alt | 8 | (15, 35] | 43 | 2.732 | 3.614 | 8.679 | 8.701 | 0.00 | 0.00 |
| test-alt | 8 | (35, 90] | 67 | 2.628 | 3.641 | 7.130 | 7.061 | 0.00 | 0.00 |

## By wind_speed

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| test-alt | 8 | (0.0, 6.0] | 53 | 3.045 | 3.723 | 8.813 | 9.070 | 0.00 | 0.00 |
| test-alt | 8 | (6.0, 10.0] | 35 | 2.700 | 3.780 | 7.033 | 6.993 | 0.00 | 0.00 |
| test-alt | 8 | (10.0, 14.0] | 35 | 2.137 | 2.830 | 5.444 | 5.504 | 0.00 | 0.00 |
| test-alt | 8 | (14.0, 16.0] | 27 | 2.003 | 2.881 | 4.899 | 4.898 | 0.00 | 0.00 |

## By gimbal_pitch

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| test-alt | 8 | (-90.01, -80.0] | 49 | 2.436 | 3.016 | 5.697 | 5.721 | 0.00 | 0.00 |
| test-alt | 8 | (-80.0, -70.0] | 56 | 2.670 | 3.561 | 7.307 | 7.163 | 0.00 | 0.00 |
| test-alt | 8 | (-70.0, -59.99] | 45 | 2.776 | 3.863 | 8.471 | 8.526 | 0.00 | 0.00 |

## By masked_frac

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| test-alt | 8 | (-0.01, 0.001] | 146 | 2.667 | 3.582 | 7.157 | 7.061 | 0.00 | 0.00 |
| test-alt | 8 | (0.001, 0.01] | 3 | 2.353 | 2.413 | 4.348 | 4.306 | 0.00 | 0.00 |
| test-alt | 8 | (0.01, 0.05] | 1 | 2.608 | 2.608 | 4.873 | 4.805 | 0.00 | 0.00 |

## By swell

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| test-alt | 8 | 0 | 55 | 2.681 | 3.560 | 7.096 | 7.061 | 0.00 | 0.00 |
| test-alt | 8 | 1 | 95 | 2.608 | 3.581 | 7.041 | 6.993 | 0.00 | 0.00 |

## By sea_rate

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| test-alt | 8 | 1 | 150 | 2.642 | 3.569 | 7.069 | 7.019 | 0.00 | 0.00 |

## By pattern

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| test-alt | 8 | figure8 | 23 | 2.247 | 3.136 | 7.183 | 7.124 | 0.00 | 0.00 |
| test-alt | 8 | lawnmower | 25 | 2.650 | 3.544 | 6.574 | 6.527 | 0.00 | 0.00 |
| test-alt | 8 | orbit | 21 | 2.558 | 3.555 | 7.340 | 7.364 | 0.00 | 0.00 |
| test-alt | 8 | racetrack | 27 | 2.763 | 3.767 | 6.325 | 5.822 | 0.00 | 0.00 |
| test-alt | 8 | random | 24 | 2.859 | 3.688 | 8.804 | 9.139 | 0.00 | 0.00 |
| test-alt | 8 | straight | 30 | 2.376 | 3.561 | 6.831 | 7.037 | 0.00 | 0.00 |
