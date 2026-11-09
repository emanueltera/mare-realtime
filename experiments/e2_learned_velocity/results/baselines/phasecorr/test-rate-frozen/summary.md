# E2 evaluation: phasecorr-K2-frozen

Predictions: `/workspace/data/e2/eval/phasecorr/test-rate-frozen.csv`. Every number below is derived from `per_run.csv` and `slices.csv` in this directory (computed by `e2.evaluate`). b = |run-mean (v_est − v_true)| (m/s); b% = b / mean ground speed; drift% over non-overlapping windows. Budget: b ≤ 0.17 m/s, b% ≤ 0.28; drift budget 0.5 %.

- test-rate: 100 of 400 runs predicted (300 missing).

## By set

| set | K | runs | b median | b p90 | b% median | b% p90 | rms median | drift% median | drift% p90 | b≤0.17 | b%≤0.28 | b%≤0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test-rate | 2 | 100 | 0.076 | 0.270 | 0.236 | 0.589 | 0.257 | 0.236 | 0.589 | 0.82 | 0.59 | 0.84 |

## By altitude

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| test-rate | 2 | (0.0, 50.0] | 24 | 0.082 | 0.873 | 0.236 | 0.236 | 0.75 | 0.54 |
| test-rate | 2 | (50.0, 80.0] | 25 | 0.096 | 0.218 | 0.254 | 0.254 | 0.80 | 0.56 |
| test-rate | 2 | (80.0, 130.0] | 22 | 0.104 | 0.232 | 0.269 | 0.269 | 0.86 | 0.55 |
| test-rate | 2 | (130.0, 200.0] | 29 | 0.046 | 0.229 | 0.158 | 0.158 | 0.86 | 0.69 |

## By ground_speed

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| test-rate | 2 | (0, 20] | 12 | 0.039 | 0.097 | 0.239 | 0.239 | 0.92 | 0.58 |
| test-rate | 2 | (20, 35] | 28 | 0.056 | 0.145 | 0.197 | 0.197 | 0.93 | 0.64 |
| test-rate | 2 | (35, 50] | 37 | 0.100 | 0.299 | 0.241 | 0.241 | 0.84 | 0.57 |
| test-rate | 2 | (50, 65] | 21 | 0.127 | 0.308 | 0.248 | 0.248 | 0.62 | 0.57 |
| test-rate | 2 | (65, 200] | 2 | 0.142 | 0.244 | 0.206 | 0.206 | 0.50 | 0.50 |

## By sun_elevation

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| test-rate | 2 | (-90, 5] | 16 | 0.187 | 0.316 | 0.447 | 0.447 | 0.50 | 0.31 |
| test-rate | 2 | (5, 15] | 13 | 0.135 | 0.248 | 0.316 | 0.316 | 0.85 | 0.46 |
| test-rate | 2 | (15, 35] | 18 | 0.056 | 0.190 | 0.222 | 0.222 | 0.83 | 0.61 |
| test-rate | 2 | (35, 90] | 53 | 0.046 | 0.152 | 0.157 | 0.157 | 0.91 | 0.70 |

## By wind_speed

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| test-rate | 2 | (0.0, 6.0] | 29 | 0.059 | 0.242 | 0.158 | 0.158 | 0.83 | 0.69 |
| test-rate | 2 | (6.0, 10.0] | 36 | 0.082 | 0.240 | 0.247 | 0.247 | 0.89 | 0.58 |
| test-rate | 2 | (10.0, 14.0] | 24 | 0.103 | 0.262 | 0.274 | 0.274 | 0.75 | 0.50 |
| test-rate | 2 | (14.0, 16.0] | 11 | 0.063 | 0.417 | 0.209 | 0.209 | 0.73 | 0.55 |

## By gimbal_pitch

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| test-rate | 2 | (-90.01, -80.0] | 44 | 0.080 | 0.234 | 0.226 | 0.226 | 0.84 | 0.55 |
| test-rate | 2 | (-80.0, -70.0] | 28 | 0.061 | 0.263 | 0.189 | 0.189 | 0.86 | 0.71 |
| test-rate | 2 | (-70.0, -59.99] | 28 | 0.088 | 0.313 | 0.262 | 0.262 | 0.75 | 0.54 |

## By masked_frac

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| test-rate | 2 | (-0.01, 0.001] | 92 | 0.076 | 0.255 | 0.223 | 0.223 | 0.84 | 0.60 |
| test-rate | 2 | (0.001, 0.01] | 7 | 0.131 | 0.328 | 0.287 | 0.287 | 0.57 | 0.43 |
| test-rate | 2 | (0.01, 0.05] | 1 | 0.033 | 0.033 | 0.088 | 0.088 | 1.00 | 1.00 |

## By swell

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| test-rate | 2 | 0 | 25 | 0.071 | 0.164 | 0.197 | 0.197 | 0.92 | 0.72 |
| test-rate | 2 | 1 | 75 | 0.080 | 0.277 | 0.258 | 0.258 | 0.79 | 0.55 |

## By sea_rate

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| test-rate | 2 | 0.0 | 100 | 0.076 | 0.270 | 0.236 | 0.236 | 0.82 | 0.59 |

## By pattern

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| test-rate | 2 | figure8 | 20 | 0.091 | 0.156 | 0.263 | 0.263 | 0.90 | 0.55 |
| test-rate | 2 | lawnmower | 7 | 0.064 | 0.296 | 0.114 | 0.114 | 0.71 | 0.57 |
| test-rate | 2 | orbit | 19 | 0.118 | 0.384 | 0.390 | 0.390 | 0.63 | 0.37 |
| test-rate | 2 | racetrack | 19 | 0.063 | 0.260 | 0.188 | 0.188 | 0.84 | 0.68 |
| test-rate | 2 | random | 18 | 0.084 | 0.164 | 0.206 | 0.206 | 0.94 | 0.67 |
| test-rate | 2 | straight | 17 | 0.045 | 0.245 | 0.155 | 0.155 | 0.82 | 0.71 |
