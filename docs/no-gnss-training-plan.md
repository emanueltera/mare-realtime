# No-GNSS positioning over water: training plan

Goals, inputs and candidate approaches for training a model that tracks an aircraft's
position over open water without GNSS, using mare-realtime as the main source of
training data and the few real recordings to fine-tune and, above all, to judge it. It
sets the direction and the decisions to make; it is not a full schedule.

Numbers are tagged: **[REAL]** measured on a recorded flight, **[SIM]** measured in a
simulator, **[TARGET]** a goal, **[DERIVED]** arithmetic or general physics done for this
plan. Sources are given as `repo/path:line`.

## 1. Goal

**Horizontal position error stays below 0.5 % of the distance flown, over water, with no
GNSS input at any point after take-off.**

The metric needs pinning down before any number is compared (decision D1):

- *Error*: horizontal distance between the estimate and the GNSS reference at the same
  instant, the estimate started from the reference's start point, with no rotation, scale
  or shift fitted afterwards (as `dragonsurf/tools/evaluate_gps.py:59-99` does).
- *Distance*: length of the GNSS reference path over the scored window, sampled at 1 Hz
  so that GNSS jitter does not inflate it (as `mayfly-demo/scripts/extract-flight.py:221-262`).
- *Which error*: the **final** error ÷ distance flatters a flight whose errors cancel on the
  way back. Report final, **maximum over the flight of error(t) ÷ distance-so-far** once
  the distance passes a floor (say 1 km), and p95. Proposed acceptance: the maximum
  stays under 0.5 % on every held-out over-water flight, not the average.

Where things stand:

| Reference point | Error ÷ distance | Status |
|---|---|---|
| Classical over-water RGB odometry (`dragonsurf`), quad flights 07-22 / 07-23 | 2.58 % best, 8.64 % distance-weighted over 22.55 km, 23.27 % worst (`dragonsurf/docs/*-07-23-2026-results.md:66-124`) | [REAL] |
| Same method, lake flights 07-15 | ≈ 5–18 % (`dragonsurf/docs/mcap-07-15-2026-results.md:52-59`, derived) | [REAL] |
| dragonsurf's end-to-end roadmap, gate G4 | "blue-water segment < 3 % of distance" (`dragonsurf/e2e/README.md:118-125`) | [TARGET] |
| Existing estimate on recorded flight 0087 (fixed-wing, 158.4 km, 45 min; GNSS for the first 10 s only) | final 0.137 %; max error 704.3 m ≈ 0.44 % of the whole distance (`mayfly-demo/src/lib/data/flight-0087.js:4`) | [REAL], one flight; the code that produced it is not in the local repos |

So 0.5 % is six times tighter than the over-water roadmap's own target. The published
and in-house results are the **bar to beat**, not a forecast. The point of this work is
to find out, in a simulator where every error source can be controlled and measured,
whether a model trained mostly on simulated data can hold each term of the budget in §2,
and then whether that carries over to real flights.

## 2. Error budget: what 0.5 % demands

All [DERIVED]. Biases are what matter: noise averages out over kilometres, a bias does not.

| Error source | How it turns into drift | Size allowed for the whole 0.5 % |
|---|---|---|
| Heading bias δψ | cross-track error ≈ sin δψ × distance | **δψ ≤ 0.29°**. A magnetometer AHRS is typically off by degrees; the converter's fit on 0087 reports "3.5° MAD" yaw [REAL] (`dragonfly/data/<cam>/flight0087/README.md:102-111`). |
| Scale: height above the water | a downward camera's speed ∝ height | 0.6 m at 120 m. A barometer reads pressure altitude, not height above the sea: ~8 m per hPa of sea-level pressure change, ~0.35 % per K of temperature error, plus waves. Whether imagery can fix the scale is experiment E3. |
| Water moving under the camera | vision measures motion relative to the surface pattern | 0.5 % of ground speed: **0.30 m/s at 60 m/s, 0.12 m/s at 25 m/s, 0.06 m/s at 12 m/s**. The apparent water velocity measured on quad hovers was 0.74–1.63 m/s at 12.6–110 m [REAL] (`dragonsurf/docs/*-07-23-2026-results.md:151-167`). |
| Camera clock | a frame interval read wrong is a speed error | The real camera runs at 0.1001 s a frame, not 0.1 s [REAL] (`extract-flight.py:248-249`): a 0.1 % speed error, a fifth of the budget, if ignored. |
| Dropped or repeated frames | a gap booked as no motion is lost distance | Measured at ≈ 0.6 % of the path on two lake flights [REAL] (`dragonsurf/docs/mcap-07-15-2026-results.md:114-118`): more than the whole budget. |
| Intrinsics, mount misalignment | a focal-length error is a scale error; a mount yaw is a heading error | Focal length to ~0.1 %; mount yaw well under 0.29°. |

### What the simulation experiments must answer

Each budget term becomes an experiment: a question, the cue a sim-trained model could
learn, and what to randomise and measure in mare-realtime. The bar to beat comes from
[visual-frontend-research.md](visual-frontend-research.md) and the in-house results above.
The figures in the table are each term's share if it had the whole 0.5 %. Since the terms
add, the research doc splits the budget evenly in root-sum-square for pass thresholds:
**≤ 0.17° heading bias, ≤ 0.29 % scale bias, ≤ 0.17 m/s water-motion bias at 60 m/s**
[DERIVED].

| # | Question | Cue a model could learn | Experiment in sim | Bar to beat |
|---|---|---|---|---|
| E1 | Can a model remove the waves' own motion and give velocity relative to the water with a bias under 0.3 m/s (0.12 m/s at 25 m/s)? | Dispersion: each wavelength moves at its own phase speed (ω = √(gk) + k·U), while ego-motion moves every scale together. Needs multi-scale texture over a stretch of time, plus height and attitude. | Sweep sea state, swell, altitude, speed and fps; score velocity-relative-to-water bias against truth; vary the temporal window to find how much history it needs. | Classical tracker: 0.74–1.63 m/s apparent water velocity [REAL]; dispersion fits from hovering drones: 0.06–0.09 m/s RMS over ~30 s |
| E2 | Can a model hold heading to 0.29° from the image? | The sun (position from time and approximate place, seen directly or as glint), sky brightness and colour gradients, glint geometry; waves as a relative cue. | Randomise the AHRS heading error; give the model time of day and approximate location; score heading after correction across sun elevation, overcast and haze. Requires the sim's sky and glint to be physically right, which is itself a check. | Best published flight results: about 0.51° (polarisation compass, upward camera); magnetometer AHRS: degrees |
| E3 | Can a model hold metric scale to 0.5 % when the baro drifts? | Wave dispersion fixes metric scale (λ = gT²/2π); texture statistics against the spectrum. | Randomise sea-level pressure drift, temperature and baro bias; score the height or speed scale error by sea state and altitude. | Baro alone: ~8 m per hPa; radar altimeter claims ~0.4–1 m |
| E4 | How much drift does an unknown surface current cause, and can anything reduce it? | A spatially uniform current gives exactly the same images as ego-motion (it is a change of frame), so in principle it can't be seen. Non-uniform currents refract and steepen waves, which might leave learnable traces. | **Control experiment.** Fly with uniform currents of 0–1 m/s and measure the drift floor, then with current gradients, then with a noisy current prior given to the filter. Needs a current model in the sim (§5). | — (sets how good a current prior must be) |
| E5 | Can the filter keep drift bounded when vision is weak? | Airspeed + IMU + attitude with wind as a slowly varying state, vision correcting it whenever texture allows. | Glassy water, heavy glare and long textureless stretches; airspeed and baro dropouts; score the drift through and after the gap. | — |
| E6 | Does it transfer? | — | The same model on the real bench (§6), compared with the sim run closest to each real flight. | The classical results in the table above |

Two more points shape the targets:

- **Speed helps.** At fixed-wing speeds (≈ 60 m/s mean ground speed on 0087 [REAL]), a few
  tenths of a m/s of water motion fits the budget. At quad speeds it does not. State the
  target per platform and speed class (decision D2).
- **Timing and data plumbing are part of accuracy.** Clock skew and frame gaps each use
  up a measurable share of the budget before any model runs.

The 0087 navigation output already carries a `solar_heading_correction_rad` column
(`dragonfly/data/<cam>/flight0087/over-water-<cam>_frame_navigation.csv`, header): a
sun-based heading correction has been tried before on real data.

## 3. Inputs

The model is causal: at time t it sees only data recorded at or before t. GNSS is never
an input. It may be a training **label** and the evaluation **reference**.

| Input | Rate (target) | In mare-realtime today | In the real recordings checked so far |
|---|---|---|---|
| RGB frame (one new frame per step) | 10 Hz | `frames/*.png`, ideal pinhole, no lens or sensor effects | `/camera/image`, 720×1280 portrait JPEG, 10 Hz [REAL] |
| IMU: specific force + body rates | 100–200 Hz | `imu.csv`, **perfect** (no noise or bias) | raw accelerometer and gyro at ≈ 202 Hz, not bias-corrected [REAL] |
| Attitude (roll, pitch, heading) | 50 Hz | true attitude, no AHRS error | the camera's own orientation stream is GNSS-free. **The fused attitude was fitted to GNSS** (yaw to GPS velocity), so it must not be used as an input |
| Airspeed | 10–50 Hz | true airspeed, no pitot model | **not present** in 0087; check the other platforms' logs |
| Barometric altitude | 10–50 Hz | **no barometer channel at all** | **not present** in 0087 ("no barometer"); the height channel was fitted with GNSS |

Sources for the real column: `dragonfly/data/<cam>/flight0087/README.md:48-118` and the
converter's `fuse_pose.py:8-21` on branch `dev/<cam>-c152`.

Two consequences:

- **The real data as recorded can only supply RGB, raw IMU and the camera's own
  orientation as model inputs.** Airspeed and barometric altitude exist only in
  simulation unless other aircraft logs turn up. So train with **sensor dropout**:
  airspeed and baro can each be missing. Report real-data results with the inputs that
  were actually present.
- **Remove ground-truth leaks from the sim inputs.** `trajectory.csv` holds `x, y, vE,
  vN, course, groundSpeed, windE/N/U` next to the inputs. The data loader takes inputs
  only from the sensor-model outputs (§5), never from these columns.

**Labels** (sim: exact; real: GNSS, post-flight only): horizontal velocity over the
ground, and displacement over windows of 1 s, 10 s and 60 s. In sim also: water-surface
velocity (current), wind, true heading and true height above the water, as auxiliary
targets.

## 4. Approaches

Over open water no single frame says where the aircraft is: there are no landmarks.
Position comes from integrating velocity, so the job is a **velocity estimate whose bias
is tiny**, plus a heading reference. "Single RGB frame" means one new frame per step; the
model keeps state, so motion comes from comparing it with what it has already seen.

Three candidates, from most structured to least. The plan is to build A first, then try
C, and keep B as the comparison.

**A. Learned measurements + classical filter.**
- A visual network outputs velocity relative to the water (wave motion removed) and a
  heading correction, each with an uncertainty.
- An EKF fuses those with IMU propagation, attitude, airspeed (with wind as a slowly
  varying state), baro (with its offset as a state) and a surface-current state seeded
  from a prior.
- Strengths: interpretable, data-efficient. Each part can be tested alone on sim truth,
  and dropout is handled by the filter.
- This extends the dragonsurf roadmap's S1–S3: a dense-flow front end
  (`dragonsurf/e2e/estimator/README.md`) plus a learned water-motion head behind
  `bias_velocity_hook` (`dragonsurf/src/water_vo_rgb/estimator.py:209`).

**B. End-to-end sequence model.**
- Per-frame image embeddings and the sensor streams go into a temporal transformer or
  state-space model. It outputs velocity and covariance, and integration gives position.
- Strengths: can learn cues nobody designed, such as dispersion and sun geometry.
- Risks: it needs the most data, and it can learn sim shortcuts (see the tiling problem
  in §5).

**C. Differentiable filter (A trained end to end).**
- The networks in A are trained through the filter. The loss is on displacement
  integrated over long windows, so the training signal is the bias that drives drift,
  not frame-by-frame error.
- This is the likely end state.

Shared design points:

- **Loss** on displacement over several horizons (1 s → minutes) plus heading error, not
  only per-frame velocity. Per-frame losses don't punish a small constant bias, and that
  bias is the drift.
- **Uncertainty and health output.** Glassy water, glare and land in view should raise
  the uncertainty. The filter then falls back on airspeed + wind + IMU.
- **Dense-flow backbone**: the team's UFM checkpoints are the obvious start. Faster
  alternatives with per-pixel uncertainty, their Jetson speed, and why UFM may not be the
  right front end are compared in [visual-frontend-research.md](visual-frontend-research.md).
- **Runtime**: the onboard target is Jetson-class at 10 Hz (dragonsurf gate G4). Keep a
  smaller student model in view from the start.

### Spectral wave models (FFT, wavelets)

The sea is close to a sum of waves obeying a known law, so a spectral view of it is a
measuring instrument, not just a feature. Numbers here are [DERIVED].

- **What it measures.** Rectify a short stack of frames onto the sea plane (attitude,
  height, and the motion predicted from IMU and airspeed), take its 3-D FFT over
  (kx, ky, ω), and the wave energy lies on the dispersion shell ω = √(g|k|) + k·v, where
  v is the error in the predicted motion relative to the water. Fitting the shell's
  Doppler tilt gives **velocity relative to the water with the waves' own motion already
  removed** (E1). Its curvature gives **metric scale**: with k in pixels, metres per pixel
  = g·k_px/ω², so δs/s = δk/k + 2·δω/ω (E3). One fit attacks both terms.
- **What limits it.** One frequency bin is worth λ/T of velocity: 0.67 m/s for 2 m waves
  over a 3 s window, 0.07 m/s over 30 s. Fitting many bins gives sub-bin precision, but
  the window is set by speed: at 60 m/s and 120 m, a patch stays in a ~90° field of view
  for about 4 s. So a fixed-wing must lean on short waves (λ ≈ 1–3 m), which the ground
  resolution (≈ 0.2 m/px at 120 m) still resolves. The window-versus-accuracy curve is
  the first thing to measure.
- **Wavelets and windowed transforms** give spectra local in space and time. They follow
  a moving footprint, mask glint and foam, and map how the Doppler shift changes across
  the image: a current gradient shows up there, which is the non-uniform current test
  in E4. Directional wavelets give the local wave vector, a relative heading cue.
- **How it enters the model.**
  1. A classical 3-D FFT dispersion fit on rendered onboard frames, with no training, as
     the first baseline for E1 and E3: it sets the bar the learned model must beat.
  2. Spectral input features or layers (the FFT of the rectified stack, or learned
     space–time filters), so the network learns a robust shell fit that copes with glint,
     foam and nonlinear waves.
  3. A dispersion-consistency loss on the flow network's output.
- **Simulator caveat.** mare-realtime is itself an FFT ocean: its wave vectors lie on each
  cascade's discrete grid (patches ≈ 1 km) and its frequencies are quantised to
  multiples of 2π/1200 s (`src/ocean/engine/waves.js`). A spectral estimator could lock
  onto that lattice, a shortcut no real sea offers. The waves are also linear, with no
  bound harmonics (which ride with their parent wave, off the linear shell). Real spectra
  carry such non-dispersive energy, and so do foam, glint and cloud shadows, which the
  fit must reject. Both go on the simulator work list (§5, item 6).

## 5. Simulated data (mare-realtime)

What exists: per-frame RGB with exact camera pose and intrinsics, trajectory truth, an
ideal IMU, domain-randomised sea / light / pattern sweeps, and seeded, resumable batch
generation (`README.md`, `docs/ARCHITECTURE.md`).

What it needs for this model, in order of expected effect on transfer to real flights:

1. **Attitude/AHRS error model**: heading bias (declination residual, hard- and
   soft-iron error that depends on heading), slow drift, lag, and roll/pitch error in
   sustained turns. Randomised and seeded per run.
2. **Barometer model**: pressure altitude, sea-level pressure that drifts slowly and has
   an unknown reference, temperature deviation, noise, bias walk, and static-port error
   with airspeed. Keep height above water as a label only.
3. **Surface current** that carries the waves, foam and texture along, with the
   dispersion relation shifted by it, and randomised per run. Plus **pitot model** (noise,
   bias, lag, missing sensor) and **wind that varies slowly** in time and space (today's
   is effectively constant over a run).
4. **Flight envelope matching the real platforms**: ground speeds up to ≈ 70 m/s (sim
   airspeed tops out at 32 m/s; 0087 averaged 59 m/s [REAL]). Runs of many kilometres
   (sample jobs are 10–60 s, while 0087 is 45 min).
5. **Camera model matching the real onboard camera**: wide FOV with fisheye distortion
   (render wide and remap, or port `../mare`'s distortion model), hard mount and
   in-camera stabilisation, mount misalignment and jitter, rolling shutter, motion blur,
   noise, auto-exposure and white balance, JPEG at the real quality, 720×1280 portrait.
6. **Break the sea's tiling**: the largest wave patch repeats every ≈ 1 km and every
   1200 s. At 60 m/s that is a repeat every 17 s, a fake landmark a sequence model can
   exploit. Randomise each cascade's offset per run, make the largest one non-periodic, or
   keep sequences shorter than the period. The same lattice shows up in the spectrum
   (discrete wave vectors, frequencies quantised to 2π/1200 s): randomise each cascade's
   patch size and rotation per run, and drop the frequency quantisation where a run
   doesn't need to loop. Add second-order (bound) waves so the spectrum carries the
   non-dispersive energy a real sea does.
7. **Timing**: camera-to-IMU offset, clock skew, jitter and dropped frames, all
   randomised.
8. **IMU errors**: noise, bias instability and random walk, scale factor, misalignment,
   vibration and a lever arm. `../mare/apps/batch/imu.ts:16-32` has a white-noise +
   random-walk start.
9. **Distractors and failure conditions**: glassy and glare-washed water, low cloud and
   fog, boats and wakes, water drops on the lens, cloud shadows moving at wind speed (these
   exist; their speed must stay realistic so they don't teach a false motion cue).

Every sensor model sits **downstream of the render**, as a seeded per-run pass over the
truth. Inputs can then be re-noised without re-rendering, and the same frames serve many
sensor draws.

Throughput and the cloud:

- Today [SIM, Apple M5]: ≈ 4 ms to render and ≈ 30 ms to encode PNG per 1280×720 frame,
  ≈ 2 MB per frame (`README.md:90-93`). At 1 M frames that is ≈ 2 TB of PNG [DERIVED].
  Write JPEG at the real camera's quality (also closer to the real input), or shards
  (WebDataset/TFRecord) straight from raw readback.
- **Headless rendering on Linux GPUs is unverified.** Only macOS/Metal has been tested;
  the Linux flags (ANGLE on Vulkan) have not been run (`scripts/browser.mjs:15-27`). This
  is the first infrastructure task: a container that renders `jobs/quick.json` on a cloud
  GPU and matches a reference frame within tolerance.
- The dragonsurf roadmap planned an Unreal Engine simulator for the same purpose. It has
  not been released (`dragonsurf/e2e/data/README.md:18-19`). Decide whether mare-realtime
  replaces it or both feed training (decision D4). Having two simulators is useful: train
  on one, test on the other.

## 6. Real data

Source: `gs://tera-earth-dev-users/aerial_image/` (onboard-camera files from all runs).
None of the local repos reference this bucket, so its contents are not known here.

1. **Inventory.** Per file: platform, date, duration, resolution and frame rate, which
   telemetry is embedded or logged separately (raw IMU, orientation, any baro or
   airspeed), whether GNSS is present to serve as the reference, and how the video and
   telemetry clocks relate. Reuse the converter on `dragonfly` branch `dev/<cam>-c152`
   (convert, telemetry extraction, chapter merge, MCAP writer) and
   `dragonsurf/tools/extract_*`.
2. **Over-water only.** Nothing masks land today. Use two signals:
   - **Curation**: GNSS track against a coastline polygon, with a buffer, plus a margin
     in time. GNSS is used here only to choose data, never as an input.
   - **Runtime**: a small water / land / horizon classifier on the frame. It also becomes
     a health input, so the model knows when it is not looking at open water. Train it on
     the curated split.
   - A segment counts as over water only if the full camera footprint is over water, not
     just the point below the aircraft.
3. **Keep GNSS-fitted channels out.** The converter fitted attitude bias, yaw and height
   to GNSS. Use the raw IMU and the camera's own orientation stream as real inputs. Use
   the fused channels only for analysis.
4. **Use the real data mainly to evaluate.** Freeze a held-out bench of whole flights,
   split by flight and by day, before any training. Score it freeze-then-score as
   dragonsurf does: predictions are hash-frozen before the reference is opened
   (`dragonsurf/tools/evaluate_gps.py:122-134`).
5. **Fine-tuning on what remains**:
   - self-supervised photometric / forward-backward losses on frame pairs (no labels
     needed);
   - GNSS velocity as a training **label** on training flights;
   - the real images' statistics to tune the sim's camera model;
   - fit the IMU noise model to the real IMU's Allan variance.

## 7. The loop: generate ⇄ train ⇄ evaluate

```
        ┌───────────────────────────────────────────────────────────────┐
        ▼                                                               │
  generate (mare-realtime)  →  sensor pass (seeded noise, dropout)  →  shards
        │                                                               │
        ▼                                                               │
  train (sim + real fine-tune mix)  →  checkpoint                        │
        │                                                               │
        ▼                                                               │
  evaluate: frozen sim benches + frozen real bench, per-condition slices │
        │                                                               │
        └──► failure slices → next generation's sampling weights ───────┘
```

- **Benches, frozen at the start.**
  - `sim-heldout`: conditions, seeds and sea fields never trained on.
  - `sim-long`: runs of ≥ 50 km.
  - `real-heldout`: whole over-water flights.
  - Never report a bench that was trained on as held out (`dragonsurf/e2e/README.md:99`).
- **Per-slice report** for every checkpoint: drift (final / max / p95) by sea state,
  sun elevation, altitude, ground speed, current, wind, heading-sensor quality and
  missing sensors. The aggregate hides the failures.
- **Targeted generation**: the next round of data oversamples the slices that are
  failing, within the domain bounds, rather than sampling uniformly. Keep a fixed share
  of uniform samples so other slices don't regress.
- **Sim-to-real gap tracking**: the same model on the sim's closest match to each real
  flight (same speed, height, sun and estimated sea) against the real result. A growing
  gap points at the simulator, not the model.
- **Gates**: a real-bench regression blocks promotion. A sim improvement that does not
  carry over to real is reported as such.
- **Reproducibility**: every checkpoint records the generator commit, job files, sensor
  seeds, data shards, the training config and the bench hashes.

## 8. Milestones (outline)

1. **M0 — Ground truth and metric.** Real-data inventory, over-water curation, the frozen
   real bench, metric code (D1), and the Linux GPU rendering smoke test.
2. **M1 — Sensor realism.** AHRS, baro, pitot, IMU and timing models. Surface current,
   envelope to ≈ 70 m/s, long runs, no tiling. Camera model matched to the real camera.
3. **M2 — Baseline.** Approach A with a pre-trained flow backbone and the classical
   filter, on sim benches then the real bench. This is the number everything is compared
   with.
4. **M3 — Loop running.** Automated generate → train → evaluate with per-slice reports
   and targeted sampling. Run the current control experiment E4 early: it sets how good a
   current prior must be.
5. **M4 — The budget experiments.** E1 (wave motion removed: velocity relative to the
   water), E2 (heading from sun, sky and glint), E3 (scale from wave dispersion), E5
   (bounded drift when vision is weak). Expect the largest gains here. Candidate methods
   are in [visual-frontend-research.md](visual-frontend-research.md) and
   [sim-native-ideas.md](sim-native-ideas.md).
6. **M5 — Transfer and deployment.** E6 on the real bench, end-to-end training (C), and
   compression for onboard runtime.

## 9. Decisions to make

- **D1** The exact metric and acceptance rule (§1): final vs maximum vs p95, minimum
  distance, scored windows.
- **D2** Platforms and speed classes the 0.5 % target applies to. At quad speeds the
  water-motion allowance is ≈ 0.06–0.12 m/s.
- **D3** What is known at the start: initial position and heading from the launch point,
  or heading from the AHRS only? With an AHRS-only start, the 0.29° budget falls entirely
  on the in-flight heading cues.
- **D4** mare-realtime alone, or alongside the planned Unreal Engine simulator.
- **D5** Which real platforms can carry airspeed and baro in future campaigns. Without
  them in real data, those inputs are trained only in simulation.
- **D6** Front end: UFM, or a smaller flow network with per-pixel uncertainty (see
  [visual-frontend-research.md](visual-frontend-research.md)). Decide on runtime and on
  experiment E1, not licences: this is a personal experiment.
- **D7** Fallback sensors, if experiments E2–E4 show the five inputs can't hold a term
  of the budget: a radar altimeter (scale), a navigation-grade gyro or an upward-looking
  polarisation compass (heading), a surface-current forecast (current prior). Decide
  after the experiments, not before.
