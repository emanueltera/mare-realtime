/**
 * The fixed-wing UAV Mayfly flies on: its 3D model and its flight. This is the entry
 * point the ocean engine codes against; the implementation lives in `./aircraft/`:
 * - `attitude.js` — empty state, state blending, the attitude quaternion;
 * - `flight.js` — the paths (straight, orbit, weave, track), coordinated turns and gusts;
 * - `model.js` — the procedural airframe, its moving parts and its lights;
 * - `geometry.js`, `textures.js`, `turbulence.js` — what those are built from.
 * This module imports three, so only the client-side engine may load it.
 *
 * Model frame: metres, origin at the centre of gravity, nose toward −Z, right wing
 * toward +X, up +Y. World placement goes through the engine's `toWorld`, so the
 * engine is free to keep a floating origin.
 *
 * Attitude conventions: heading in compass degrees (0 north, 90 east), pitch in
 * degrees nose-up positive, bank in degrees right-wing-down positive.
 */

/**
 * @typedef {object} AircraftState
 * @property {number} x metres east
 * @property {number} y metres north
 * @property {number} altitude metres above mean sea level
 * @property {number} heading compass degrees the nose points toward
 * @property {number} pitch degrees, nose up positive
 * @property {number} bank degrees, right wing down positive
 * @property {number} speed airspeed, m/s
 * @property {number} turnRate deg/s, positive turning right
 * @property {number} time seconds since the flight began (propeller, strobes)
 * @property {number} visible 0–1, for a looping track that fades out at its end and back in
 */

/**
 * @typedef {object} AircraftLighting
 * @property {import('three').Texture | null} environment prefiltered sky for reflections, or null
 * @property {import('three').Vector3} sunDirection world-space unit vector toward the sun
 * @property {import('three').Color} sunColor linear radiance of the sun — applied as the sun's
 *   irradiance on a surface facing it (the engine's `sunIrradiance`), in the sea's linear units
 * @property {import('three').Color} ambient linear sky ambient — applied as the sky's irradiance on
 *   a horizontal surface (the engine's `skyIrradiance`); with an `environment` it is only a fill
 * @property {number} night 0 daylight – 1 dark enough that nav lights and strobes read
 * @property {number} [exposure] multiplier on sun, ambient and environment, for an engine that
 *   exposes in its own shaders rather than through `renderer.toneMappingExposure`; default 1
 */

/**
 * @typedef {object} AircraftModel
 * @property {import('three').Group} object add to the scene once; `pose` moves it
 * @property {(state: AircraftState, toWorld: (east: number, north: number, up: number) => import('three').Vector3) => void} pose
 *   place and orient the model, and animate propeller, control surfaces and lights from the state
 * @property {(opacity: number) => void} setOpacity fade the whole model (0 hides it), e.g. as a camera closes in
 * @property {(lighting: AircraftLighting) => void} setLighting match the model to the sky and sun
 * @property {(scale: number) => void} setScale exaggerate the size for overhead map views
 * @property {import('three').Vector3} cameraMount the gimbal camera's position in the model frame
 *   (at the EO lens; `setGimbal` updates it in place)
 * @property {number} span wingspan, metres, at scale 1
 * @property {() => void} dispose free every geometry, material and texture
 * @property {(pitch: number, yaw?: number) => void} [setGimbal] point the gimbal ball in
 *   degrees, as `AircraftSpec.gimbal` gives it (pitch negative looks down; yaw from the
 *   nose, positive right); the ball starts at pitch −35
 */

/**
 * @typedef {object} Flight
 * @property {AircraftState} state the current state, updated in place by `step`
 * @property {(dt: number) => AircraftState} step advance by dt seconds and return `state`
 */

export { emptyState, blendStates, attitudeQuaternion } from './aircraft/attitude.js';
export { createFlight } from './aircraft/flight.js';
export { createAircraftModel } from './aircraft/model.js';
