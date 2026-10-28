/**
 * Live state shared between the app and the ocean engine, as plain objects (no
 * framework): the UI wraps them in its own reactivity.
 *
 * `oceanControl` is written by the UI and read by the engine every frame: anything
 * set here is layered over the current scene.
 *
 * `oceanReadout` is written by the engine a few times a second: what is actually on
 * screen right now.
 */

/**
 * @typedef {object} OceanControl
 * @property {Partial<import('./scene.js').SeaState>} sea
 * @property {Partial<import('./scene.js').Lighting>} light
 * @property {Exclude<import('./scene.js').RenderMode, 'shaded'> | null} splitMode overrides
 *   the scene's `split.mode` when set
 * @property {number | null} splitPosition overrides the divider (0–1) and stops any sweep
 */

/** @type {OceanControl} */
export const oceanControl = {
	sea: {},
	light: {},
	splitMode: null,
	splitPosition: null
};

export function resetOceanControl() {
	oceanControl.sea = {};
	oceanControl.light = {};
	oceanControl.splitMode = null;
	oceanControl.splitPosition = null;
}

/**
 * @typedef {object} OceanReadout
 * @property {'loading' | 'running' | 'fallback' | 'error'} status
 * @property {number} windSpeed U10 being rendered, m/s
 * @property {number} windDirection compass degrees the sea travels toward
 * @property {number} significantHeight Hs of the combined sea, m
 * @property {number} peakPeriod Tp of the combined sea (the peak of S(ω), wind sea or swell), s
 * @property {number} peakWavelength deep-water wavelength at Tp, m
 * @property {number} beaufort Beaufort force for the wind speed, 0–12
 * @property {number} sunElevation degrees
 * @property {number} haze 0–1
 * @property {number} overcast 0–1
 * @property {number} cameraHeight m above the instantaneous sea surface (for an onboard
 *   rig, the aircraft's height above it)
 * @property {number} fov degrees
 * @property {number} roll camera roll right now, degrees (sway, or the aircraft through its gimbal)
 * @property {number} pitch camera pitch right now, degrees, including sway or aircraft motion
 * @property {number} heave camera heave right now, m (free-camera sway only)
 * @property {boolean} aircraft whether the UAV is in the scene
 * @property {number} aircraftAltitude m above mean sea level
 * @property {number} aircraftSpeed airspeed, m/s
 * @property {number} aircraftHeading compass degrees
 * @property {number} aircraftBank degrees, right wing down positive
 * @property {number} aircraftPitch degrees, nose up positive
 * @property {'free' | 'chase' | 'onboard'} cameraMode
 * @property {number} trackTime seconds into the recording the scene's timed tracks show
 *   right now (see `TrackClock`); -1 when the scene has none
 * @property {number} trackGap metres between a `gapTo` track's marker and the track it
 *   points at, right now; -1 when there is none
 * @property {import('./scene.js').RenderMode} mode
 * @property {number} fps smoothed frames per second
 */

/** @type {OceanReadout} */
export const oceanReadout = {
	status: 'loading',
	windSpeed: 0,
	windDirection: 0,
	significantHeight: 0,
	peakPeriod: 0,
	peakWavelength: 0,
	beaufort: 0,
	sunElevation: 0,
	haze: 0,
	overcast: 0,
	cameraHeight: 0,
	fov: 0,
	roll: 0,
	pitch: 0,
	heave: 0,
	aircraft: false,
	aircraftAltitude: 0,
	aircraftSpeed: 0,
	aircraftHeading: 0,
	aircraftBank: 0,
	aircraftPitch: 0,
	cameraMode: 'free',
	trackTime: -1,
	trackGap: -1,
	mode: 'shaded',
	fps: 0
};
