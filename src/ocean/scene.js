/**
 * The contract between the deck and the ocean behind it. Every slide names a scene;
 * the stage blends from whatever it is showing to that scene over `transition`
 * seconds, so a slide change is a change of weather and camera rather than a cut.
 *
 * Mayfly's platform is a fixed-wing UAV flying over open water, so a scene can put
 * that aircraft in the sky (`aircraft`) and hang the camera off it: a chase camera
 * for the cinematic views, or the aircraft's own gimbal camera for what Mayfly
 * actually sees.
 *
 * Conventions, used everywhere a direction or position appears:
 * - Positions are metres in a local east/north frame on the mean sea surface:
 *   `x` east, `y` north. (The three.js world maps that to x = east, z = -north,
 *   with +y up; nothing outside `src/lib/ocean/` needs to know.)
 * - Directions are compass degrees: 0 = north, 90 = east.
 * - Sea directions are the way the waves *travel*, not the meteorological
 *   "wind from" — so `windDirection: 90` is a sea running eastward.
 */

/**
 * @typedef {object} SeaState
 * @property {number} windSpeed wind speed at 10 m (U10), m/s; drives the wind-sea spectrum
 * @property {number} windDirection compass degrees the wind sea travels toward
 * @property {number} [fetch] km of open water the wind has blown over; default 400
 * @property {number} [spread] directional spreading exponent (cos^2s); higher is more
 *   long-crested; default 8
 * @property {number} [swellHeight] significant height of a separate swell system, m; default 0
 * @property {number} [swellDirection] compass degrees the swell travels toward
 * @property {number} [swellPeriod] swell peak period, s; default 11
 * @property {number} [choppiness] horizontal displacement scale, 0 (rolling) – 1.6 (sharp
 *   crests); default 1
 */

/**
 * @typedef {object} Lighting
 * @property {number} sunElevation degrees above the horizon (-8 for blue hour … 90)
 * @property {number} sunAzimuth compass degrees toward the sun
 * @property {number} [overcast] 0 clear – 1 fully overcast; default 0
 * @property {number} [haze] 0 crisp horizon – 1 thick marine haze; default 0.2
 * @property {number} [exposure] multiplier on the tone-mapped image; default 1
 */

/**
 * A camera placed in the world, independent of the aircraft.
 * @typedef {object} OceanCamera
 * @property {'free'} [mode] the default when `mode` is absent
 * @property {number} height metres above mean sea level
 * @property {number} pitch degrees, negative looks down at the water
 * @property {number} heading compass degrees the camera looks toward
 * @property {number} [fov] vertical field of view, degrees; default 50
 * @property {number} [x] east position of the camera, m; default 0
 * @property {number} [y] north position of the camera, m; default 0
 * @property {number} [drift] m/s the camera travels along its heading, like a camera
 *   carried along; default 0
 * @property {number} [orbit] deg/s of slow yaw, for slides that circle the scene; default 0
 * @property {number} [sway] 0–1, how much the camera rides the sea like one on a buoy:
 *   heave, pitch and roll coupled to the current sea state; default 0
 */

/**
 * A camera flying with the aircraft, looking at it — the cinematic view. Placed
 * relative to the aircraft's nose and its horizontal plane (not its bank, so the
 * horizon stays level while the aircraft rolls), and eased toward that spot so it
 * trails the aircraft's motion slightly, like a real chase plane.
 * @typedef {object} ChaseRig
 * @property {'chase'} mode
 * @property {number} distance metres from the aircraft
 * @property {number} azimuth degrees around the aircraft from its nose, clockwise seen
 *   from above: 180 is directly behind, 90 off the right wing
 * @property {number} elevation degrees above (+) or below (−) the aircraft
 * @property {number} [lag] seconds of follow smoothing; default 0.5
 * @property {number} [fov] vertical field of view, degrees; default 42
 * @property {number} [orbit] deg/s the rig drifts around the aircraft; default 0
 * @property {[number, number]} [frame] where the aircraft sits in the picture, as
 *   fractions of the half-width and half-height from the centre (+x right, +y up), so
 *   it can clear the copy; default [0.3, -0.05]
 * @property {number} [heading] compass degrees to measure `azimuth` from instead of the
 *   aircraft's nose: the rig then keeps its bearing while the aircraft turns under it (a
 *   replay's long turns, or keeping the sun behind the camera); default the nose
 * @property {string} [include] id of a timed track to keep in the picture with the
 *   aircraft (an estimate flown beside the reference): the rig aims at the middle of the
 *   aircraft and that track's position at the scene's `trackClock`, and pulls back beyond
 *   `distance` as far as it needs to hold both, easing in and out as the gap changes.
 *   `frame` then places that middle point
 */

/**
 * The aircraft's own camera, on its gimbal (see `AircraftSpec.gimbal`) — what Mayfly
 * sees. Requires `aircraft`. The aircraft model is not drawn from this view.
 * @typedef {object} OnboardRig
 * @property {'onboard'} mode
 * @property {number} [stabilized] 0 hard-mounted (rolls and pitches with the airframe)
 *   – 1 fully gimbal-stabilised; default 0.55
 */

/**
 * @typedef {OceanCamera | ChaseRig | OnboardRig} CameraRig
 */

/**
 * A fixed-wing UAV, the platform Mayfly flies on. Drawn at true size (about a 3 m
 * span) unless `scale` says otherwise, flying a coordinated path: it banks into turns
 * (tan φ = V·ω/g), its control surfaces and propeller move, its navigation lights
 * show at low sun, and `turbulence` adds gust response.
 * @typedef {object} AircraftSpec
 * @property {number} altitude metres above mean sea level
 * @property {number} [speed] airspeed, m/s; default 24
 * @property {number} [heading] compass degrees the nose points at the start; default 0
 * @property {number} [x] metres east of the origin at the start; default 0
 * @property {number} [y] metres north of the origin at the start; default 0
 * @property {'straight' | 'orbit' | 'weave' | 'track'} [path] default 'straight'.
 *   `orbit` loiters on a circle round (x, y); `weave` flies gentle S-turns about its
 *   heading; `track` flies along `tracks[track]` at the track's altitudes
 * @property {number} [orbitRadius] metres, for 'orbit'; default 180
 * @property {1 | -1} [orbitDirection] 1 clockwise seen from above, −1 anticlockwise; default 1
 * @property {string} [track] id of a TrackOverlay to fly, for 'track'; the aircraft replaces
 *   that track's marker and loops over its `loopSeconds`
 * @property {number} [turbulence] 0–1 gust-driven roll/pitch/yaw; default 0.15
 * @property {{ pitch: number, yaw?: number, fov?: number }} [gimbal] the camera on the
 *   aircraft: pitch in degrees (negative looks down), yaw from the nose, vertical fov;
 *   default { pitch: -35, yaw: 0, fov: 48 }
 * @property {boolean} [footprint] draw the gimbal camera's view frustum and its footprint
 *   on the sea — the patch of water the model is reading; default false
 * @property {number} [scale] size multiplier for overhead views where a true-size
 *   aircraft would be a speck; default 1
 */

/**
 * What the frame shows. `shaded` is the photoreal render; the rest are the
 * ground-truth channels the simulator emits for training, coloured by
 * `channels.js` so the slide legends and the shader agree.
 * @typedef {'shaded' | 'height' | 'velocity' | 'normals' | 'foam' | 'depth'} RenderMode
 */

/**
 * A vertical wipe: the right of the divider shows `mode`, the left the shaded render.
 * @typedef {object} SplitView
 * @property {Exclude<RenderMode, 'shaded'>} mode
 * @property {number} [position] 0–1 of the frame width where the divider rests; default 0.5
 * @property {boolean} [sweep] animate the divider slowly back and forth; default false
 */

/**
 * A path drawn on the water, in the local east/north frame (metres).
 * @typedef {object} TrackOverlay
 * @property {string} id
 * @property {Array<[number, number]>} points [east, north] pairs in metres, in time order
 * @property {number[]} [altitudes] metres above mean sea level, one per point; absent means
 *   on the water. A track in the air is drawn at altitude with its shadow on the sea.
 * @property {string} color any CSS colour
 * @property {boolean} [dashed]
 * @property {number} [width] pixels; default 3
 * @property {boolean} [marker] run a position marker along the path on a loop
 * @property {number} [loopSeconds] seconds for the marker to traverse the path; default 18
 * @property {number[]} [times] seconds, one per point: a recorded track. A timed track is
 *   not looped at constant speed over `loopSeconds`; its marker, and the aircraft when it
 *   flies it, are placed by interpolating `times` at the scene's `trackClock`, so every
 *   timed track in a scene shows the same instant of the flight.
 * @property {string} [gapTo] id of another (timed) track: draw a thin connector from this
 *   track's marker to the other track's position at the same instant — the live gap between
 *   an estimate and the reference it is scored against
 */

/**
 * The playback clock for timed tracks (recorded flights). Starts at `start` seconds into
 * the recording when the scene opens and runs at `rate` × real time; when it reaches
 * `loop[1]` it returns to `loop[0]` (the aircraft and markers fade out and back in, as on
 * a looped track), so a slide can replay one interesting stretch of a long flight.
 * @typedef {object} TrackClock
 * @property {number} [start] seconds into the recording; default the first track time
 * @property {number} [rate] playback speed, 1 = real time; default 1
 * @property {[number, number]} [loop] seconds into the recording to loop between; default
 *   the whole recording
 * @property {[number, number]} [window] seconds before and after the clock to draw the timed
 *   tracks over, so a chase view of a long flight shows the stretch just flown rather than
 *   every pass at once: the older half of the time before the clock fades out, the last
 *   quarter after it; default the whole of each track
 * @property {number} [clear] seconds just before the clock to leave undrawn (fading in
 *   behind them), so the aircraft flying a timed track has a silhouette against the sea
 *   rather than ending a ribbon of its own colour; default 0
 *
 * With a `scrim`, a replay's timed tracks also fade toward that side of the frame, so a
 * pass that crosses behind the slide's copy is faint there.
 */

/**
 * @typedef {'flow' | 'grid' | 'horizon'} OverlayName
 * - `flow`: animated streaks on the surface following the simulated orbital velocity
 *   (the ground truth the model is trained to recover)
 * - `grid`: a metric grid floating on the sea, for scale on the overhead views
 * - `horizon`: the true horizon line and the camera's attitude, drawn as a HUD
 */

/**
 * @typedef {object} SceneSpec
 * @property {SeaState} sea
 * @property {Lighting} light
 * @property {CameraRig} camera
 * @property {AircraftSpec} [aircraft] the UAV, when it is in the scene
 * @property {RenderMode} [mode] what the whole frame shows; default 'shaded'
 * @property {SplitView} [split]
 * @property {OverlayName[]} [overlays]
 * @property {TrackOverlay[]} [tracks]
 * @property {TrackClock} [trackClock] playback of timed tracks; ignored without them
 * @property {boolean} [randomize] cycle domain-randomised samples (sea, light, camera and,
 *   with an onboard rig, the aircraft's altitude, speed, gimbal and turbulence) every few
 *   seconds, ignoring the scene's values except as the starting point
 * @property {number} [dim] 0–1 uniform darkening so copy reads over the water; default 0
 * @property {'left' | 'right' | 'bottom' | 'none'} [scrim] a soft gradient behind the
 *   side of the frame the copy sits on; default 'none'
 * @property {number} [transition] seconds to blend in from the previous scene; default 2.4
 */

/** @type {Required<Pick<SeaState, 'fetch' | 'spread' | 'swellHeight' | 'swellDirection' | 'swellPeriod' | 'choppiness'>>} */
export const seaDefaults = {
	fetch: 400,
	spread: 8,
	swellHeight: 0,
	swellDirection: 0,
	swellPeriod: 11,
	choppiness: 1
};

/** @type {Required<Pick<Lighting, 'overcast' | 'haze' | 'exposure'>>} */
export const lightDefaults = { overcast: 0, haze: 0.2, exposure: 1 };

/** @type {Required<Pick<OceanCamera, 'fov' | 'x' | 'y' | 'drift' | 'orbit' | 'sway'>>} */
export const cameraDefaults = { fov: 50, x: 0, y: 0, drift: 0, orbit: 0, sway: 0 };

/** @type {Required<Omit<ChaseRig, 'mode' | 'distance' | 'azimuth' | 'elevation' | 'heading' | 'include'>>} */
export const chaseDefaults = { lag: 0.5, fov: 42, orbit: 0, frame: [0.3, -0.05] };

/** @type {Required<Omit<OnboardRig, 'mode'>>} */
export const onboardDefaults = { stabilized: 0.55 };

/** @type {Required<Omit<AircraftSpec, 'altitude' | 'track'>>} */
export const aircraftDefaults = {
	speed: 24,
	heading: 0,
	x: 0,
	y: 0,
	path: 'straight',
	orbitRadius: 180,
	orbitDirection: 1,
	turbulence: 0.15,
	gimbal: { pitch: -35, yaw: 0, fov: 48 },
	footprint: false,
	scale: 1
};

export const DEFAULT_TRANSITION = 2.4;
