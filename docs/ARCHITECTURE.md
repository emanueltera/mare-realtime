# Architecture

mare-realtime generates simulated UAV flights over a real-time spectral ocean: a
trajectory (with its ground truth and IMU), the onboard camera's view of the sea along
it, and the camera pose for every frame, across many ocean and light conditions.

```
src/ocean/        the real-time ocean engine (three.js, WebGL2, GPU FFT) and the UAV
src/trajectory/   flight trajectories: pure JS, deterministic, runs in node and the browser
src/conditions/   sea / light presets, job files and sweep expansion: pure JS
src/batch/        the headless page the batch generator drives (batch.html)
src/app/          the interactive app (index.html): design a run, preview it, export it
scripts/          generate.mjs — the batch generator (Playwright → files on disk)
jobs/             example job files
tests/            node:test unit tests (`pnpm test`, or `node --test tests/`)
e2e/              Playwright tests (the engine in a real browser)
```

## Conventions (everywhere)

- Positions are metres in a local **east / north / up** frame on the mean sea surface:
  `x` east, `y` north, `altitude` (or `z`) up above mean sea level.
- Directions are **compass degrees**: 0 = north, 90 = east, clockwise.
- Sea and wind directions are the way the waves / air **travel toward**, not the
  meteorological "wind from". A sea with `windDirection: 90` runs east, and so does the
  wind that raised it.
- Aircraft attitude: `heading` compass degrees of the nose, `pitch` degrees nose-up
  positive, `bank` degrees right-wing-down positive (yaw–pitch–roll, Z-Y-X).
- Body frame for the IMU: **FRD** (x forward, y right wing, z down). Specific force
  `f = a − g` in body axes (level, unaccelerated flight reads `fz ≈ −9.81`); body rates
  `p, q, r` in rad/s about x, y, z.
- Camera frame for exported poses: **OpenCV** (x right, y down, z along the optical
  axis). `R_wc` maps camera-frame vectors to ENU; a world point `P` projects as
  `p_c = R_wcᵀ (P − t_wc)`, `u = fx·p_c.x/p_c.z + cx`, `v = fy·p_c.y/p_c.z + cy`, with
  pixel (0, 0) at the top-left corner of the top-left pixel.
- Time: `t` is seconds since the start of the trajectory. The sea has its own clock
  (`seaTime`), offset from `t` by the run's warm-up and start offset.
- Gravity `g = 9.81` m/s². Air density is not modelled.

## Trajectories (`src/trajectory/`)

Pure, deterministic functions of `(spec, t)`: no three.js, no DOM, no `Math.random`
(seeded PRNG only). The same spec gives bit-identical samples on the same JS engine
build; node and a browser agree to about 1e-12 (identical at the CSVs' 6 decimals) but
not to the bit, since their `Math.sin`, `exp`, `atan2`… may differ in the last place.

```js
import { createTrajectory } from './src/trajectory/index.js';
const traj = createTrajectory(spec, { sea });  // sea: SeaState, only for wind: 'sea'
traj.duration;          // s
traj.stateAt(12.5);     // TrajectorySample at any t in [0, duration] (interpolated)
traj.sample(200);       // TrajectorySample[] at 200 Hz from t = 0 to duration inclusive
traj.warnings;          // string[]: e.g. a mean wind at cruise ≥ 0.9 × airspeed (WIND_WARNING)
```

The flight is simulated, not drawn: a guidance law flies the pattern's waypoints with a
coordinated-turn autopilot (bank-limited, roll-rate-limited, the lift along the wings'
normal: ψ̇ = (V·γ̇ + g·cos γ)·tan φ / (V·cos γ), g·tan φ / V when level), the air
moves with the wind (ground velocity = air velocity + wind, so the nose crabs into a
cross wind), and turbulence is a seeded, band-limited gust field added to the wind and
the attitude. It is integrated on a fixed internal step from t = 0 and stored, so
`stateAt` is a lookup, never a re-integration.

```js
/**
 * @typedef {object} TrajectorySpec   JSON-serialisable
 * @property {number} [seed]            default 1; drives every random choice in the trajectory
 * @property {number} duration          seconds
 * @property {PatternSpec} pattern
 * @property {number} altitude          cruise altitude, m above mean sea level
 * @property {AltitudeProfile} [altitudeProfile]  default { type: 'constant' }
 * @property {number} [airspeed]        m/s; default 24
 * @property {number} [airspeedVariation]  m/s of slow seeded variation; default 0
 * @property {{ x?: number, y?: number, heading?: number }} [start]  default 0, 0, heading 0;
 *   `heading` orients the pattern (see PatternSpec) and is the initial ground course (the
 *   flight starts trimmed, see below). Most patterns' first leg runs along it; waypoints,
 *   or an orbit centre not abeam the start, lie anywhere, and guidance turns onto them
 * @property {WindSpec} [wind]          default 'sea'
 * @property {number} [turbulence]      0 (still air) – 1 (strong); default 0.15
 * @property {GimbalSpec} [gimbal]
 *
 * @typedef {
 *   | { type: 'straight' }
 *   | { type: 'orbit', radius?: number, direction?: 1 | -1, center?: [number, number] }
 *   | { type: 'racetrack', length?: number, radius?: number, direction?: 1 | -1 }
 *   | { type: 'figure8', radius?: number }
 *   | { type: 'lawnmower', legLength?: number, spacing?: number, legs?: number }
 *   | { type: 'waypoints', points: Array<[number, number] | [number, number, number]>, loop?: boolean }
 *   | { type: 'random', legLength?: [number, number], turn?: [number, number] }
 * } PatternSpec   lengths and radii in metres, relative to `start` and its heading: a pair
 *   (orbit `center`, waypoint `points`) is [right, ahead] of the start heading, i.e. [east,
 *   north] when it is 0; waypoint altitudes are m above mean sea level; `direction` 1 is
 *   clockwise seen from above; a random pattern's `turn` is signed, positive right
 *
 * @typedef {
 *   | { type: 'constant' }
 *   | { type: 'sine', amplitude: number, period: number }
 *   | { type: 'steps', amplitude: number, interval: number }   seeded climbs/descents
 * } AltitudeProfile
 *
 * @typedef {'sea' | 'none' | { speed: number, direction: number }} WindSpec
 *   'sea': the sea state's U10 and direction carried up to altitude by a neutral log
 *   profile over open water; { speed, direction }: wind at altitude, m/s, compass toward
 *
 * @typedef {object} GimbalSpec
 * @property {number} [pitch]       degrees, negative looks down; default -35
 * @property {number} [yaw]         degrees from the nose, positive right; default 0
 * @property {number} [fov]         vertical field of view, degrees; default 48
 * @property {number} [stabilized]  0 hard-mounted – 1 fully stabilised; default 0.55
 *
 * @typedef {object} TrajectorySample
 * @property {number} t             s since the start
 * @property {number} x             m east
 * @property {number} y             m north
 * @property {number} altitude      m above mean sea level
 * @property {number} vE            ground velocity, m/s east
 * @property {number} vN            m/s north
 * @property {number} vU            m/s up
 * @property {number} heading       compass degrees of the nose
 * @property {number} course        compass degrees of the ground track
 * @property {number} pitch         degrees nose-up
 * @property {number} bank          degrees right-wing-down
 * @property {number} turnRate      deg/s of heading, positive right
 * @property {number} airspeed      m/s true airspeed
 * @property {number} groundSpeed   m/s horizontal
 * @property {number} windE         wind (incl. gusts) at the aircraft, m/s east
 * @property {number} windN
 * @property {number} windU
 * @property {number} fx            specific force, body FRD, m/s²
 * @property {number} fy
 * @property {number} fz
 * @property {number} p             body rates, FRD, rad/s
 * @property {number} q
 * @property {number} r
 * @property {number} gimbalPitch   degrees (the spec's, unless a pattern scans it)
 * @property {number} gimbalYaw
 * @property {number} gimbalFov
 */
```

Choices the flight model makes (documented in the code too):

- **Trimmed start.** At t = 0 the aircraft is at `start`, level, at the profile's
  altitude and airspeed, its ground track along `start.heading` (where a waypoint
  pattern's first point lies elsewhere, it then turns toward it), its nose crabbed into
  the mean cross wind, and banked as the guidance asks there (wings level on a straight
  leg, the circle's bank on an orbit), so a windy run does not open with a correction
  turn. With no wind, heading = course = `start.heading`.
- Guidance is an L1 variant: the reference point is L1 further *along* the path than the
  closest point (it slides round corners rather than jumping), and the law divides by
  the chord to it, so a = V²/R holds exactly on circles. Commands are sampled at the
  200 Hz internal step and held; the plant is integrated with RK4.
- Lawnmower half turns are semicircles when spacing / 2 fits the turn radius (sized for
  airspeed plus mean wind at 30° bank), Omega turns otherwise. Orbit, racetrack and
  figure-8 radii are fixed: in strong wind they are carried downwind (the 35° bank limit
  cannot hold them at high ground speed).
- `steps` draws its levels uniformly within cruise ± amplitude and ramps between them at
  2 m/s; commanded altitude is floored at 10 m. The flight path follows its command
  through second-order dynamics (a lagged, rate-limited path rate reached at ≤ 20°/s²
  of pitch acceleration), so q and the load factor ramp at the profile's corners and at a
  new waypoint altitude rather than stepping. `airspeedVariation` sets the size of a slow
  seeded wander: its RMS is `airspeedVariation / √2` (that of a sine of that amplitude),
  so it is not a bound — over minutes the peaks reach 1.5–2× it (floored at 10 m/s).
  With `wind: 'none'`, turbulence still gusts at a base intensity. Gusts are a frozen
  function of time; their intensity and length scales are set at cruise altitude.
- `sample(rate)` has no extra sample at `duration` when duration·rate is not whole.
  `createTrajectory` also returns `spec` (resolved) and `rate` (the internal 200 Hz).
- **Wind it cannot outfly.** Nothing stops a spec with a mean wind at cruise near or
  above the airspeed; the flight then drifts downwind and its guidance hunts. Such a
  trajectory carries a warning in `warnings` (wind ≥ `WIND_WARNING` = 0.9 × airspeed),
  which the generator logs and writes to `meta.json`; `randomRun` never draws one.
- Pattern geometry that cannot be flown (a radius, length, spacing or leg count that is
  not positive, a direction other than ±1, a non-finite waypoint) throws a RangeError.
- Memory: about 42 kB per second of flight (26 doubles per internal step at 200 Hz).

The trajectory module also exports `toAircraftState(sample)` (the engine's
`AircraftState`: `{ x, y, altitude, heading, pitch, bank, speed: airspeed, turnRate,
time: t, visible: 1 }`), CSV / JSON writers for samples, `WIND_WARNING`, and
`enuToGeodetic(origin) → (east, north) → [lat, lon]` (geodesy.js: the point on the
local tangent plane at the origin, through ECEF to WGS-84; what the generator writes
for a run's `origin`).

## The engine (`src/ocean/`)

`new OceanEngine(host, options)`; `host` is `{ canvas, hud, root, onStatus }` as before.

Interactive use (the app) is unchanged: it runs its own `requestAnimationFrame` loop,
adapts quality to the frame rate, and blends between scenes.

**Manual mode** (`options.manual: true`) is for generating data. No animation-frame loop,
no adaptive quality, no blending, no reduced-motion time scale, pixel ratio 1:

```js
const engine = new OceanEngine(host, {
  manual: true,
  quality: 'high',          // a fixed preset from quality.js; never changes during the run
  seed: 20240611,           // the sea's Gaussian field (waves.js); same seed → same sea
  size: [1280, 720],        // exact drawing-buffer pixels, independent of the page layout
  preserveDrawingBuffer: true
});
engine.setScene(spec);      // with manual: true the scene is entered at once (no blend)
engine.setSeaTime(seconds); // put the sea's clock at a given time (e.g. after warm-up)
engine.advance(dt);         // step everything (sea, foam, flight, rigs, exposure) by dt seconds,
                            //   in sub-steps of at most 0.1 s (so the frame rate does not change the sea)
engine.renderFrame({ mode: 'shaded' });   // render now, synchronously; mode switches are instant
engine.frameTruth();        // FrameTruth for what was just rendered
await engine.capture();     // the frame just rendered as a PNG Blob
```

A run, frame 0 at trajectory t = 0 after the sea has run for the warm-up:

```js
engine.setScene(spec);                    // aircraft: { path: 'trajectory', trajectory }
engine.setSeaTime(render.seaTime);
for (let i = 0; i < warmup * fps; i++) engine.advance(1 / fps, { flight: false });
engine.seek(0);                           // the flight at t = 0, the camera rigs restarted settled
for (let i = 0; i < frames; i++) {
  if (i > 0) engine.advance(1 / fps);
  for (const mode of modes) {
    engine.renderFrame({ mode });
    truth = engine.frameTruth();          // the same for every mode of a frame
    png = await engine.capture();         // straight after renderFrame (see below)
  }
}
```

Details of the manual engine:

- `advance(dt)` sub-steps: a dt over 0.1 s is taken as ⌈dt / 0.1⌉ equal steps (the foam's
  decay, drift and spread are first-order steps, and each whitecap belongs to its own
  instant's waves), so the same seaTime gives the same sea at any fps. Both clocks end
  exactly dt on; a flight's t is a float sum of the sub-steps.
- `advance(dt, { flight: false })` holds the flights and recorded tracks while the sea,
  foam and camera rigs run (a warm-up). `seek(t)` puts the current scene's flight at t
  (any path; in the app it also works, and playback continues from there); in manual
  mode it also restarts the camera rigs there, settled: a chase rig takes the sample's
  `vE / vN / vU` as the motion it has been following, so it holds its framing from the
  first frame. Without them it eases in over about a second.
- `renderFrame` brings the GPU sea to the current sea time if `advance` has not (after
  `setSeaTime`, or with no `advance` yet), but foam only builds up over `advance` steps.
  The exposure meter steps by the time advanced since the previous frame drawn (the
  first frame meters itself fully). Label modes are drawn without the shaded image's
  ±1 LSB dither, so their PNGs carry the ramp values.
- `frameTruth()` reads the sea under the camera back off the GPU cascades the frame was
  drawn from (one texel, synchronously, once per frame drawn): call it before the next
  `advance`, which moves them on (it throws otherwise).
- `capture()` reads the canvas: without `preserveDrawingBuffer`, call it in the same
  task as `renderFrame` (no `await` in between), before the browser composites.
- The preset's render scale is not applied: the buffer is exactly `size`. If the browser
  gives a smaller drawing buffer than `size` (Chromium caps it near 2²⁵ pixels: 8192×4096
  does not fit, 7680×4320 does), the constructor throws rather than render a crop that
  the truth's K would not describe. The app's live
  overrides (`control.js`) are ignored; frames depend only on the calls made, the seed
  and the GPU (same GPU and driver → byte-identical PNGs; verified on Apple M5 / ANGLE
  Metal).
- Every scene of a manual engine, and every trajectory scene, is placed unshifted: its
  positions are absolute ENU metres.
- `scene.aircraft.altitude` may be left out with `path: 'trajectory'`. The engine reads
  only `duration` and, from `stateAt(t)`, `x, y, altitude, heading, pitch, bank,
  airspeed, turnRate` and optionally `vE, vN, vU, gimbalPitch, gimbalYaw, gimbalFov`
  (`TrajectoryState` in `src/ocean/scene.js`). The interactive app loops it over
  `duration` when nothing seeks it; `seek(t)` holds t within [0, duration] (a seek to the
  end shows the end), and a frame after a seek shows the flight at that t rather than
  stepping it on (the app seeks every frame: it owns the clock). The manual engine holds
  it at its ends.
- `FrameTruth.aircraft` is null when the scene has no aircraft. `camera.K` is read off the
  projection matrix actually used (a view offset would move the principal point; no rig
  uses one, so it is the image centre and fx = fy); `camera.fovY` is the K's.
  `src/ocean/engine/truth.js` exports `cameraTruth(camera, origin, w, h)` and
  `projectTruth(truth, enuPoint) → [u, v, Z]`, and `tests/engine-truth.test.js` checks
  them against three.js's own projection (< 0.01 px).
- `window.__mare` carries the readout (`{ readout, stats }`); in dev, `window.__mareEngine`
  is the engine.

Scene additions:

- `aircraft.path: 'trajectory'` with `aircraft.trajectory` — any object with
  `stateAt(t) → TrajectorySample` and `duration`. The aircraft is placed at exactly
  `toAircraftState(trajectory.stateAt(t))` — no engine gusts, no smoothing, no scene
  shift — and the gimbal follows the sample's `gimbalPitch / gimbalYaw / gimbalFov`. In the
  interactive app the playback loops over `duration`; `engine.seek(t)` moves it.
- `hud: false` hides the HUD canvas drawing.

Scene removals (deck-only, nothing here set them): `randomize` (the engine's cycling
domain-randomised samples, `randomizer.js`; runs are randomised by `randomRun` instead),
`dim` and `scrim` (darkening for slide copy), and the readout's `sample` index.

```js
/**
 * @typedef {object} FrameTruth
 * @property {number} t             trajectory seconds of this frame
 * @property {number} seaTime       the sea's clock, s
 * @property {object} aircraft      the AircraftState rendered
 * @property {object} camera
 * @property {[number, number, number]} camera.position   ENU metres (t_wc)
 * @property {number[]} camera.rotation   3×3 row-major R_wc (OpenCV camera axes → ENU)
 * @property {[number, number, number, number]} camera.quaternion  the same rotation, [x, y, z, w]
 * @property {number} camera.fovY   degrees
 * @property {number} camera.width  pixels
 * @property {number} camera.height
 * @property {number[]} camera.K    3×3 row-major intrinsics [fx, 0, cx, 0, fy, cy, 0, 0, 1]
 * @property {number} seaHeightUnderCamera  m, the simulated surface straight below the camera:
 *   every cascade the frame was drawn from, summed at full resolution on the GPU and read
 *   back (probe.js), the choppy sideways displacement inverted so it is the water over
 *   (cam_e, cam_n)
 * @property {number} heightAboveSea        camera height above that surface, m
 * @property {object} sea           the resolved SeaState rendered
 * @property {{ significantHeight: number, peakPeriod: number, peakWavelength: number, beaufort: number }} seaStats
 * @property {object} light         the resolved Lighting rendered
 */
```

## Runs and jobs (`src/conditions/`, `src/batch/`, `scripts/generate.mjs`)

A **run** is one flight over one ocean, rendered:

```js
/**
 * @typedef {object} RunSpec   JSON-serialisable
 * @property {string} id
 * @property {import('../src/ocean/scene.js').SeaState} sea
 * @property {import('../src/ocean/scene.js').Lighting} light
 * @property {TrajectorySpec} trajectory
 * @property {object} render
 * @property {number} [render.width]     default 1280
 * @property {number} [render.height]    default 720
 * @property {number} [render.fps]       default 10
 * @property {string} [render.quality]   default 'high'
 * @property {string[]} [render.modes]   default ['shaded']; any of shaded, height, velocity,
 *   normals, foam, depth (the label channels, colourised as channels.js ramps them)
 * @property {'onboard' | 'chase'} [render.view]  default 'onboard'
 * @property {number} [render.warmup]    seconds the sea and foam run before frame 0; default 8
 * @property {number} [render.seaTime]   the sea's clock at the start of warm-up; default 0
 * @property {number} [render.seaSeed]   default 20240611
 * @property {boolean} [render.frames]   false writes the flight only (run.json, meta.json,
 *   trajectory.csv, imu.csv; no browser, so no frames.csv); default true
 * @property {number} [imuRate]          Hz of imu.csv; default 200
 * @property {{ lat: number, lon: number }} [origin]  if set, positions are also written as
 *   latitude / longitude (flat-earth about the origin)
 */
```

A **job** is a base run plus what varies:

```json
{
  "name": "sea-state-sweep",
  "base": { "trajectory": { "duration": 60, "altitude": 120, "pattern": { "type": "lawnmower" } } },
  "sweep": { "sea": ["calm", "moderate", "strong"], "light": ["midday", "low-sun"], "seed": [1, 2] },
  "random": { "count": 20, "seed": 7 }
}
```

`sweep` is a grid: each key is a dotted path into the RunSpec (`sea`, `light` and
`trajectory.pattern` also accept preset names), and every combination is a run. `seed`
sets `trajectory.seed` and `render.seaSeed` together. `random` instead draws `count`
runs with domain-randomised sea (including fetch and swell), light, altitude, airspeed
(at least `WIND_MARGIN` × the wind aloft),
start heading, turbulence, gimbal (including its stabilisation) and pattern (seeded); with
both blocks, the random runs follow the grid's in the same list. The expansion is pure
(`expandJob(job) → RunSpec[]`) and gives each run a stable id:
`<slug(name)>-<index>-<tokens>`, where only axes with more than one value contribute a
token.

`resolveRun` fills every field, so even `{}` is a flyable run: sea `moderate`, light
`morning`, pattern `transit`, duration 60 s, altitude 120 m, and the trajectory module's
own defaults for the rest (`trajectoryDefaults` is built from them). A sea or light given
as an object is filled from `scene.js`'s `seaDefaults` / `lightDefaults`, not from a
preset. Validation is strict: unknown keys, presets, modes or out-of-range numbers throw
`resolveRun: <field> …`.

`src/conditions/index.js` exports (pure, node and browser):

```js
seaPresets      // Record<string, { label: string, description: string, sea: SeaState }>
                //   at least: glassy, calm, light, moderate, fresh, strong, gale, plus
                //   swell variants (e.g. moderate-swell, cross-swell)
lightPresets    // Record<string, { label: string, description: string, light: Lighting }>
                //   at least: midday, morning, low-sun, overcast, hazy, dusk
patternPresets  // Record<string, { label: string, description: string, pattern: PatternSpec }>
                //   at least: transit, loiter, survey, racetrack, figure8, random
renderDefaults  // the RunSpec.render defaults above
resolveRun(partial)       // RunSpec with presets looked up and every default filled in
expandJob(job)            // RunSpec[] (resolved), stable ids
randomRun(seed, base?)    // one domain-randomised RunSpec
runSummary(run)           // short human label, e.g. "moderate · low-sun · survey · s1"
runScene(run, trajectory) // the SceneSpec the batch renders a run from (views.js)
chaseCamera(frame?)       // render.view 'chase': the batch's chase rig (the app's too)
onboardCamera(stabilized) // render.view 'onboard'
```

Also exported: `trajectoryDefaults`, `renderModes`, `seaSeedFor(seed)` (= 20240611 +
seed), `describeSea`, `WIND_MARGIN` (a random run's airspeed is at least 1.3 × its wind
aloft) and the `DEFAULT_*` names.

`scripts/generate.mjs` starts the app (or takes `--url`), opens `batch.html` in headless
Chromium on the real GPU, and for each run writes:

```
out/<job>/<run-id>/
  run.json          the RunSpec, fully resolved (every default filled in)
  meta.json         sea stats (Hs, Tp, wavelength, Beaufort), wind aloft, engine quality,
                    seeds, versions, frame count, timings
  trajectory.csv    TrajectorySample per rendered frame
  imu.csv           t, fx, fy, fz, p, q, r at imuRate
  frames.csv        FrameTruth per frame, flattened (camera pose, K, height above sea)
  frames/000000.png the onboard camera, one per frame (and frames/<mode>/ per label mode)
  video.mp4         if ffmpeg is on the PATH and --video is given
out/<job>/index.csv one line per run: id, conditions, Hs, frames, paths
```

### The batch page and the generator in detail

`src/batch/page.js` (batch.html) renders one run at a time and exposes:

```js
window.mare.ready                  // true once the module has loaded
await window.mare.load(run)        // → { frames, meta: { renderer, vendor, software, duration,
                                   //     warmupSteps, seaTimeAtFrame0, loadSeconds } }
await window.mare.frame(i)         // → { index, truth: FrameTruth, sample: TrajectorySample,
                                   //     images: { [mode]: 'data:image/png;base64,…' } }
window.mare.dispose()
```

- `load` resolves the run again, builds a manual engine at exactly `width × height` with
  `seed: render.seaSeed` and the run's quality, sets the sea's clock to `render.seaTime`,
  runs `render.warmup` seconds in steps of 1 / fps with the flight held
  (`advance(dt, { flight: false })`, the last step shorter if the warm-up is not a whole
  number of frames), then `seek(0)`. Frames are `floor(duration · fps) + 1`, at
  t = i / fps; frame i's sea time is `seaTime + warmup + t`.
- `frame(i)` goes forward only, one `advance(1 / fps)` per frame, renders every mode in
  `render.modes` (the truth is the first mode's; all modes draw the same camera) and
  captures each straight after drawing it. `sample` is `stateAt(truth.t)`: `truth.t` is a
  float sum of the steps (and sub-steps), within a few ulps of i / fps.
- The scene is `runScene(run, trajectory)` (src/conditions/views.js). `view: 'onboard'` is
  the gimbal camera with the trajectory's `gimbal.stabilized`; `view: 'chase'` is
  `chaseCamera()`, `{ distance: 24, azimuth: 160, elevation: 12, lag: 0.5, fov: 42,
  frame: [0, 0] }`: the aircraft centred, the camera behind its right quarter (azimuth is
  clockwise from the nose: 90 off the right wing, 180 dead astern) and a little
  above. The app's chase preview is the same rig with the aircraft moved off the panel.
- Label modes draw the sky as a flat dark mask and the aircraft (chase view) as a flat
  grey silhouette over the ramp.
- No WebGL2, no float render targets, a shader error or a lost context throw with a
  message; the generator records the run as failed.

`scripts/generate.mjs`:

- Input: a job file (it has `name`, `base`, `sweep` or `random`) or a single RunSpec (a
  job of one, named after the file). `--only <text>` and `--limit N` pick runs by id.
- Without `--url` it runs vite in its own process (`scripts/serve.mjs`) on `--port`
  (strict), **without hot reload**: vite reloads every open page when an HTML file under
  the root changes (a `vite build` into dist/), which would lose a run. A page lost under
  a run (reloaded, crashed) is retried once on a fresh page.
- Chromium: `scripts/browser.mjs` — the GPU flags per platform (`--use-angle=metal` on
  macOS), or SwiftShader with `--software`; Playwright's own build if installed, else
  the newest cached one, or `MARE_CHROMIUM`.
- `--concurrency N` renders N runs at once, each in its own page (own WebGL context).
- `trajectory.csv`: `frame` then the TrajectorySample columns in the order above, one row
  per frame. `frames.csv`: `frame, t, seaTime, cam_e, cam_n, cam_u, R00 … R22 (R_wc
  row-major), qx, qy, qz, qw, fx, fy, cx, cy, width, height, fovY, heightAboveSea,
  seaHeightUnderCamera, aircraft_x, aircraft_y, aircraft_altitude, aircraft_heading,
  aircraft_pitch, aircraft_bank`. With `origin`, trajectory.csv adds `lat, lon` and
  frames.csv `cam_lat, cam_lon, aircraft_lat, aircraft_lon` (WGS-84 radii of curvature at
  the origin: the point on the local tangent plane at the origin, through ECEF to WGS-84).
  Plain decimals, 6 places (9 for rotations, quaternions and lat/lon); index.csv, a
  summary, 4. Every trajectory.csv and imu.csv row is node's `stateAt` (with frames, at the
  t the frame drew); frames.csv's aircraft columns are the pose the browser drew.
- `meta.json` is written last, and `"status": "complete"` marks the run done: a run whose
  directory holds a complete meta.json and the same run.json is skipped (resume), unless
  `--force`. A run being redone starts from an empty directory. A failed run gets
  `"status": "failed"` and its `error`.
- `index.csv` lists every run of the job (`complete`, `failed` or `pending`) with
  `id, status, summary, sea_hs, sea_tp, beaufort, wind_speed, sun_elevation, pattern,
  altitude, frames, rendered, seconds, path, error`.
- `--video` encodes to `video.mp4.tmp` and renames it only when ffmpeg succeeds;
  `meta.video` is `{ file }` or `{ error }`. A run already complete is re-encoded when its
  `meta.video` is missing or an error, and its meta.json updated.
- `meta.warnings`: the trajectory's warnings (see Trajectories).
- The exit code is 1 if any run failed, 2 for bad arguments (unknown flags included);
  SIGINT / SIGTERM stop ffmpeg and close the browser, write index.csv and exit 130 (a
  second signal exits at once). A run cut short writes no meta.json: it stays `pending`
  and is redone next time. The vite the generator starts leaves SIGTERM to it
  (serve.mjs takes vite's own exit-on-SIGTERM listener off).
- `playwright.config.js` / `e2e/`: the batch page on the GPU (`pnpm test:e2e`; vite on
  `MARE_E2E_PORT`, default 5175, through serve.mjs).
