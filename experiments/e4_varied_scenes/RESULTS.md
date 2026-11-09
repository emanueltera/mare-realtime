# E4: varied scenes, the onboard camera's format, and a first recorded flight

E4 retrains E3's network ([`../e3_relative_pose/`](../e3_relative_pose/)) on more varied
simulated scenes, then in the onboard camera's format. It then runs the camera-format model on
recorded flight 0087. Sections 1–2 are simulated only. Section 3 uses real recorded frames and
sensors, scored against the GNSS from the same flight.

Paths are relative to this folder. `results/eval/<set>/` is written by `e3/evaluate.py`, and
`results/eval/real-*` by `e3/real.py`. `results/*_log.csv` are the training logs, and
`results/logs/` holds the run logs. The checkpoints and prediction files are not in the repo. They
are in `vast-backup/data/e4/` at the repo root, copied from the GPU box and checksum-verified.
*drift* is the horizontal end-point error divided by the horizontal distance flown, in %. Values
are medians over runs unless stated otherwise.

## 1. More varied scenes (portrait camera, as in E3)

**Setup.**
- `make_jobs.mjs` draws, for each flight:
  - wind 0.7–28 m/s, fetch and spread, and optional swell;
  - sun from −8° to 88°, overcast, haze and exposure;
  - one of six water colours, foam and cloud placement.
- Flights are as in E3: altitude 30–200 m, airspeed 15–60 m/s, the 720 × 1280 portrait camera.
- Training adds camera effects (`e3/fx.py`): defocus, smear, vignette, dirt, gain and gamma,
  flicker, noise and JPEG.
- Models trained:
  - `var3k`: 3000 flights;
  - `var3k-s1`: the same with a second seed;
  - `var6k`: 6000 flights.
- All are 16 epochs (`run_e4.sh`). `e3` is E3's network, which was trained on E3's narrower scenes.

**Table 1. Median drift with the true attitude (%, 120 × 60 s flights; test-rate 30 × 20 s per rate)**

| set | e3 | var3k | var3k-s1 | var6k | blind | airspeed |
|---|---|---|---|---|---|---|
| test-long (E3's scenes) | 1.18 | 1.33 | 1.34 | **1.04** | 30.3 | 34.9 |
| test-varied | 3.48 | 2.41 | 2.21 | **1.94** | 35.2 | 39.9 |
| test-varied-fx (with camera effects) | 3.99 | 2.54 | 2.38 | **2.05** | 35.2 | 39.9 |
| test-rate, sea rate 0 / 1 / 2 | 2.91 / 1.83 / 3.36 | 4.19 / 1.96 / 4.23 | 3.91 / 2.24 / 4.25 | 3.51 / 2.06 / 3.67 | 28.0 | 26.8 |

Sources: `results/eval/{test-long,test-varied,test-varied-fx,test-rate}/summary.md`.

- **Variety helps on varied scenes.** E3's network drifts 3.48 % on them. The varied models drift
  1.94–2.41 %.
- **Twice the data helps.** `var6k` is the best model on test-long, test-varied and
  test-varied-fx. On test-rate, E3's network is as good or better at every rate.
- **The variety costs little on E3's own scenes.** The two 3000-flight seeds are 1.33 % and 1.34 %
  against E3's 1.18 %. The two seeds land 0.01–0.3 points apart on each set, which gives a rough
  size for run-to-run noise.
- **The sea-rate dependence got worse, not better.** Every E4 model drifts 3.5–4.3 % on a frozen
  or double-speed sea, against about 2 % at the trained rate. E3 drifted 2.9 % and 3.4 % at
  those rates. Training on one sea rate remains the likely cause, as E3 suggested.
- **With the simulated AHRS attitude**, test-long drift is 1.76 % (e3) and 1.52 % (var6k). On
  test-varied it is 3.74 % (e3) and 2.15 % (var6k).
- **None is near 0.5 %.** At most 21 % of flights reach it (var6k, test-long, true attitude).
- **Validation per-step error** (`results/*_log.csv`, last epoch): var3k 0.181 m (4.2 %), var3k-s1
  0.179 m, var6k 0.159 m (3.7 %).

## 2. The onboard camera's format

**Setup.**
- **Frame:** 1280 × 720 landscape, focal 610.7 px, no distortion. That gives 61.0° along the
  track and 92.7° across it. The flight direction is up the frame, as in the recordings. Frames
  are shrunk to 512 × 288 for the network, as before.
- **Range:** altitude 30–400 m and airspeed 15–75 m/s, wider than E3 to cover the recordings. The
  scene variety is as in section 1, with new seeds.
- **Not simulated:** the camera is hard-mounted, with no image stabilisation.
- **Models,** all trained on 3000 flights for 16 epochs with camera effects (`run_cam.sh`):
  - `cam3k`;
  - `cam3k-noair`, the same with the airspeed input set to zero, because flight 0087 recorded no
    airspeed;
  - `blind-cam`, which sees no images.
- **Code changes:** the camera now comes from each set and is stored in the checkpoint, and both
  are checked when a model is loaded. On the box, the old and new code gave identical draws,
  samples and outputs on the portrait sets. A frozen-sea warp test passes in the landscape
  format.

**Table 2. test-cam (120 × 60 s flights in the camera format): median drift %**

| model | true attitude | AHRS | own rotations | p90 (true) | flights ≤ 0.5 % (true) |
|---|---|---|---|---|---|
| cam3k | 2.28 | 2.69 | 2.92 | 6.03 | 7.5 % |
| cam3k-noair | 2.18 | 3.17 | 3.14 | 7.67 | 10.0 % |
| blind-cam | 26.8 | 27.2 | 27.5 | 72.2 | 0 % |
| airspeed dead reckoning | 26.6 | 26.7 | 27.7 | 86.8 | 0 % |

With camera effects (test-cam-fx), the true-attitude medians are 2.41 % (cam3k) and 2.37 %
(cam3k-noair). On test-cam the median per-step error is 4.2 % (cam3k) and 3.9 % (cam3k-noair)
of the step; with camera effects it is 4.5 % and 4.4 %. Sources:
`results/eval/test-cam{,-fx}/summary.md`, `results/cam3k*_log.csv`.

- **The camera format costs little, if anything.** Drift is 2.2–2.3 % here. The 3000-flight var3k
  seeds drift 2.2–2.4 % on test-varied, and var6k 1.9 % on twice the data. That holds even though
  these flights reach 400 m and 75 m/s and the view along the track is narrower. Each model is
  scored on its own test set.
- **Airspeed adds little.** Without it, the median with the true attitude is about the same; the
  p90 and the AHRS figure are worse, by about 1.6 and 0.5 points.

## 3. Recorded flight 0087

**Data** (`extract_real.py`, then `e3/real.py`, run by `score_real.sh`):
- the flight's 9 over-water stretches: 26,421 frame pairs (26,430 frames), 156.8 km, 44.1 min,
  at a median height of 105–130 m per stretch (56–174 m overall);
- the onboard camera's frames, turned to the landscape format;
- the recording's attitude, whose heading uses GNSS;
- either the camera's own gyro (with a bias taken from each stretch's first 10 s) or the
  attitude's rotations;
- the recording's height.

The model is `cam3k-noair`. Scoring is against the same flight's GNSS track.

**Checks before scoring:**
- **Image up against the track:** image-up agrees with the GNSS track to within the crab angle.
  The median cosine per stretch is at least 0.9936, and the median crab per stretch is −3.0° to
  +4.3°.
- **Gyro against attitude:** the camera's gyro tracks the attitude. Per-axis correlation is about
  0.99 or better over the flight, and 0.98 on the worst stretch and axis. The median difference
  is 0.02–0.03° per frame.
- **Sea speed:** phase correlation shows the sea moving down the frame at the speed GNSS
  predicts, within about 10 %. The prediction uses the recording's height, itself derived with
  GNSS speed, so this checks the frame geometry and timing, not the height.
- **Scoring code:** on simulated flights written in the same format, `e3/real.py` reproduces
  `e3/predict.py` + `e3/evaluate.py` to within 6e-6.

**Table 3. Flight 0087, distance-weighted over the 9 stretches (attitude rotations / camera gyro)**

| measure | attitude | camera gyro |
|---|---|---|
| drift, attitude heading | **3.15 %** | 3.56 % |
| drift after one best-fit scale per stretch | 2.49 % | 2.57 % |
| drift, heading from chaining the rotations | 3.18 % | 66 % |
| best-fit scale applied to the network (GNSS / network)¹ | 0.989 | 0.983 |

¹ Not independent: the height that sets the network's scale was derived using GNSS speed.

Per stretch, with the attitude's rotations:
- drift 0.59–6.58 %, median 3.53 %;
- median |direction error| 1.8–3.5°;
- the 10th–90th percentile of the per-step scale is about 0.90–1.10 in a typical stretch. Across
  stretches, p10 is 0.81–0.96 and p90 1.03–1.16; pooled over all steps it is 0.91–1.11.

Sources: `results/eval/real-flight-0087-{att,gyro}/summary.md` and `per_stretch.csv`.

- **The simulator-trained model works on real frames at about the drift it has in simulation.**
  It drifts 3.2 % over 157 km of real flight against 2.2 % (true attitude) to 3.2 % (AHRS) on
  test-cam, with stretches 2 to 11 minutes long against 1-minute test flights. It was trained
  only on simulated frames. The real figure uses a height calibrated with GNSS speed, so its
  scale is not GNSS-free. With the scale fitted out, it is 2.49 % (see "What this flight cannot
  show").
- **Direction, not scale, is most of the error.** A best-fit scale per stretch removes only
  about a fifth of the drift. A median direction error of about 2° is about 3.5 % sideways on
  its own.
- **The direction error is biased, and changes during the flight.** The predicted step points
  1.4° to the right of the GNSS track on average (`real_crab.py`, `results/eval/*/crab.txt`). On
  stretches 4–7, the middle third of the flight, it is 1.8–3.3° right. On stretches 0, 2 and 8 it
  is within 0.5°. It is not crab: the prediction follows the 4.3° crab of stretch 1 to within
  0.2°, and the bias does not correlate with crab (r = 0.15). Neither a fixed water drift nor a
  fixed camera yaw offset fits the steps much better than scale alone (`real_bias.py`, rms
  residual 5.96 → 5.90 m/s). Candidates, none tested:
  - the recording's GNSS-aided heading being off by 2–3° in mid-flight;
  - surface drift or current that changes over the flight;
  - the camera's stabilisation.
- **The camera's gyro cannot hold the heading on its own.** Chaining its rotations drifts 66 %
  overall, and 76–127 % on stretches 3, 6, 7 and 8. Its bias, taken from each stretch's first
  10 s, does not stay put over several minutes. With the attitude's rotations, chaining gives
  3.18 %. Long flights without GNSS need a heading reference besides this gyro.

**What this flight cannot show:**
- **Absolute scale.** The recording's height was derived from the video using GNSS speed
  (confirmed by the user, 8 November 2026), with a stated uncertainty of ±10–15 %. The network
  measures steps in units of that height, so its metric scale on this flight comes partly from
  GNSS. The best-fit scale of 0.98–0.99 is not evidence about the network's scale. The drift with
  the attitude's heading (3.15 %) is optimistic as a no-GNSS figure for the same reason. The
  independent measures are the direction error and the drift after fitting one scale per
  stretch (2.49 %), which removes the scale entirely.
- **Heading without GNSS.** The attitude used to place the steps in the world has a heading that
  uses GNSS.
- **Stabilisation.** It was on in the recording. The simulator does not model it, and the model
  was not trained with it.
- **Generality.** This is one flight, at median heights of 105–130 m (56–174 m overall), on one
  day's sea.

## Figures

Made by `plots.py` from the files above:
- `results/fig_sim_drift.png`: median drift by model on each simulated test set, with flight 0087
  for reference.
- `results/fig_real_tracks.png`: flight 0087, the network's track against GNSS for each stretch.
- `results/fig_real_along_flight.png`: the step-direction error and the crab angle along the
  flight (30 s medians), and the step length against GNSS.
- `results/fig_real_drift_by_stretch.png`: drift per stretch, with the heading from the attitude
  and from chaining the camera's gyro.

## 4. Cost and next steps

**Cost.** The box billed about $1.52/h.
- E4 training: var3k and var3k-s1 took about 10.6 h each, var6k 14.1 h; they ran side by side.
- Camera-format work: 05:12–16:59 UTC on 8 November, about 11.8 h and about $18. That covers
  rendering 3220 flights (about 4.5 h) and training (7.1 h, all three models side by side).

**Next steps.**
1. **Find the mid-flight direction bias.** Get an independent heading for 0087, or a flight with
   stabilisation off, and check the bias against it.
2. **Get a real height** (barometer or a laser altimeter) on the next recording, so scale can be
   tested.
3. **Train on varied sea rates.** Every E4 model still loses accuracy away from the trained rate.
4. **Use more data:** var6k beat both var3k seeds on every set except test-rate at the trained
   rate. Train a camera-format model on 6000 or more flights.
5. **Record the next flight with the camera's stabilisation off,** so it matches what was
   simulated.
