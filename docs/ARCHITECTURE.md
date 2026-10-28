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
tests/            node:test unit tests (pure modules)
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
(seeded PRNG only). The same spec gives bit-identical samples in node and the browser.

```js
import { createTrajectory } from './src/trajectory/index.js';
const traj = createTrajectory(spec, { sea });  // sea: SeaState, only for wind: 'sea'
traj.duration;          // s
traj.stateAt(12.5);     // TrajectorySample at any t in [0, duration] (interpolated)
traj.sample(200);       // TrajectorySample[] at 200 Hz from t = 0 to duration inclusive
```

The flight is simulated, not drawn: a guidance law flies the pattern's waypoints with a
coordinated-turn autopilot (bank-limited, roll-rate-limited, ψ̇ = g·tan φ / V), the air
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
 * @property {{ x?: number, y?: number, heading?: number }} [start]  default 0, 0, heading 0
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
 * } PatternSpec   lengths and radii in metres, relative to `start` and its heading
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

The trajectory module also exports `toAircraftState(sample)` (the engine's
`AircraftState`: `{ x, y, altitude, heading, pitch, bank, speed: airspeed, turnRate,
time: t, visible: 1 }`) and CSV / JSON writers for samples.

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
engine.advance(dt);         // step everything (sea, foam, flight, rigs, exposure) by dt seconds
engine.renderFrame({ mode: 'shaded' });   // render now, synchronously; mode switches are instant
engine.frameTruth();        // FrameTruth for what was just rendered
await engine.capture();     // the frame just rendered as a PNG Blob
```

Scene additions:

- `aircraft.path: 'trajectory'` with `aircraft.trajectory` — any object with
  `stateAt(t) → TrajectorySample` and `duration`. The aircraft is placed at exactly
  `toAircraftState(trajectory.stateAt(t))` — no engine gusts, no smoothing, no scene
  shift — and the gimbal follows the sample's `gimbalPitch / gimbalYaw / gimbalFov`. In the
  interactive app the playback loops over `duration`; `engine.seek(t)` moves it.
- `hud: false` hides the HUD canvas drawing.

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
 * @property {number} seaHeightUnderCamera  m, the sea surface below the camera (approximate:
 *   the CPU sampler's strongest components, see sampler.js)
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
 * @property {boolean} [render.frames]   false writes the trajectory and truth only; default true
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
  "sweep": { "sea": ["calm", "moderate", "rough"], "light": ["midday", "low-sun"], "seed": [1, 2] },
  "random": { "count": 20, "seed": 7 }
}
```

`sweep` is a grid: each key is a dotted path into the RunSpec (`sea`, `light` and
`trajectory.pattern` also accept preset names), and every combination is a run. `seed`
sets `trajectory.seed` and `render.seaSeed` together. `random` instead draws `count`
runs with domain-randomised sea, light, altitude, airspeed and gimbal (seeded). The
expansion is pure (`expandJob(job) → RunSpec[]`) and gives each run a stable id.

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
```

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
