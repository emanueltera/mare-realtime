/**
 * The preview's SceneSpec: the run's sea and sky, the aircraft flying the trajectory
 * exactly (`aircraft.path: 'trajectory'`), the trajectory drawn on the sea, and one of
 * three cameras — a chase plane, the aircraft's own gimbal camera, or an overhead view
 * of the whole flight. Pure: no DOM and no engine, only numbers in and a scene out.
 *
 * Conventions as everywhere: metres east / north / up of the origin on the mean sea
 * surface, compass degrees (0 north, 90 east, clockwise).
 *
 * The overlay and the cameras are framed for the part of the window the ocean shows
 * through (`Viewport`): the panel and the timeline cover the rest, so the chase plane
 * puts the aircraft in the middle of what is left and the overhead view fits the flight
 * into it.
 */

import { chaseCamera, onboardCamera } from '../conditions/index.js';

/** @typedef {import('../ocean/scene.js').SceneSpec} SceneSpec */
/** @typedef {import('../ocean/scene.js').TrackOverlay} TrackOverlay */
/** @typedef {import('../trajectory/index.js').Trajectory} Trajectory */
/** @typedef {import('./state.js').ViewState} ViewState */

/**
 * The window the ocean is drawn into, and how much of each side the interface covers.
 * @typedef {object} Viewport
 * @property {number} width px
 * @property {number} height px
 * @property {number} [top] px covered by the bar along the top; default 0
 * @property {number} right px covered by the panel
 * @property {number} bottom px covered by the timeline (and, narrow, the panel)
 */

/** Seconds between the drawn track's points: fine enough for a 35° turn at 24 m/s. */
const TRACK_STEP = 1;
/** The drawn trajectory: the interface's one warm colour, apart from the sea's cyan. */
const TRACK_COLOR = 'rgba(255, 164, 104, 0.85)';
/** Vertical field of view of the overhead camera, degrees. */
const OVERHEAD_FOV = 50;
/**
 * Degrees short of straight down the overhead camera looks (the free rig stops at −89°
 * so its heading stays defined); `overheadCamera` moves it to compensate.
 */
const OVERHEAD_TILT = 1;
const DEG = Math.PI / 180;

/**
 * The trajectory as a track in the air, one point a second plus the last, with its
 * shadow on the sea. Untimed: the engine draws it whole, and the replay clock (which a
 * timed track would start) is not involved.
 * @param {Trajectory} trajectory
 * @returns {TrackOverlay}
 */
export function trackOverlay(trajectory) {
	const samples = trajectory.sample(1 / TRACK_STEP);
	const last = samples[samples.length - 1];
	if (!last || last.t < trajectory.duration - 1e-6) samples.push(trajectory.stateAt(trajectory.duration));
	return {
		id: 'trajectory',
		points: samples.map((s) => /** @type {[number, number]} */ ([s.x, s.y])),
		altitudes: samples.map((s) => s.altitude),
		color: TRACK_COLOR,
		width: 2.5
	};
}

/**
 * The trajectory's horizontal extent, m.
 * @param {TrackOverlay} track
 */
export function bounds(track) {
	let minE = Infinity;
	let maxE = -Infinity;
	let minN = Infinity;
	let maxN = -Infinity;
	for (const [e, n] of track.points) {
		if (e < minE) minE = e;
		if (e > maxE) maxE = e;
		if (n < minN) minN = n;
		if (n > maxN) maxN = n;
	}
	return { minE, maxE, minN, maxN };
}

/**
 * The visible part of the window: its centre as a fraction of the half-width and
 * half-height from the window's centre (+x right, +y up, as a chase rig's `frame`), and
 * its size in px.
 * @param {Viewport} vp
 */
function visible(vp) {
	const top = vp.top ?? 0;
	const w = Math.max(1, vp.width - vp.right);
	const h = Math.max(1, vp.height - vp.bottom - top);
	return {
		w,
		h,
		fx: (w / 2 - vp.width / 2) / (vp.width / 2),
		fy: -(top + h / 2 - vp.height / 2) / (vp.height / 2)
	};
}

/**
 * A free camera high above the flight looking straight down, north up, high enough to
 * hold the whole trajectory in the visible part of the window with a margin, and placed
 * so the flight is centred in it.
 *
 * The track is drawn at the aircraft's altitude, which is a good fraction of the camera's
 * height for a small pattern: seen in perspective it looks larger than its shadow on the
 * sea. So the fit is made in the plane of its highest point, and the centring in the
 * plane of its mean altitude.
 * @param {TrackOverlay} track
 * @param {Viewport} vp
 * @returns {{ camera: import('../ocean/scene.js').OceanCamera, metresPerPixel: number }}
 */
export function overheadCamera(track, vp) {
	const b = bounds(track);
	const v = visible(vp);
	const alts = track.altitudes ?? [0];
	const top = Math.max(0, ...alts);
	const mean = alts.reduce((a, h) => a + h, 0) / alts.length;
	const margin = 1.25;
	const tan = Math.tan((OVERHEAD_FOV / 2) * DEG);
	// Metres per pixel needed for the extent to fit across and down; at a distance d below
	// the camera, the H px of the image span 2·d·tan(fov/2) m.
	const span = Math.max(80, (b.maxE - b.minE) * margin);
	const depth = Math.max(80, (b.maxN - b.minN) * margin);
	const mpp = Math.max(span / v.w, depth / v.h);
	const height = Math.max(top + 150, top + (mpp * vp.height) / (2 * tan));
	// Ground metres per pixel, for sizing the aircraft, and at the track's mean altitude,
	// for centring it.
	const metresPerPixel = (2 * height * tan) / vp.height;
	const atTrack = (2 * (height - mean) * tan) / vp.height;
	const ce = (b.minE + b.maxE) / 2;
	const cn = (b.minN + b.maxN) / 2;
	// Shift the camera so the flight's centre lands in the middle of the visible part (a
	// point fx half-widths right of the image centre is fx·W/2 px, east of it), and back
	// south for the degree it looks north of straight down.
	return {
		camera: {
			mode: 'free',
			x: ce - v.fx * (vp.width / 2) * atTrack,
			y: cn - v.fy * (vp.height / 2) * atTrack - (height - mean) * Math.tan(OVERHEAD_TILT * DEG),
			height,
			pitch: -90 + OVERHEAD_TILT,
			heading: 0,
			fov: OVERHEAD_FOV
		},
		metresPerPixel
	};
}

/**
 * The preview scene.
 * @param {object} args
 * @param {Trajectory} args.trajectory
 * @param {TrackOverlay} args.track
 * @param {import('../ocean/scene.js').SeaState} args.sea
 * @param {import('../ocean/scene.js').Lighting} args.light
 * @param {ViewState} args.view
 * @param {Viewport} args.viewport
 * @returns {SceneSpec}
 */
export function buildScene({ trajectory, track, sea, light, view, viewport }) {
	const g = trajectory.spec.gimbal;
	const v = visible(viewport);
	/** @type {SceneSpec['camera']} */
	let camera;
	let scale = 1;
	if (view.camera === 'onboard') {
		camera = onboardCamera(g.stabilized);
	} else if (view.camera === 'overhead') {
		const fit = overheadCamera(track, viewport);
		camera = fit.camera;
		// A 3 m aircraft is a pixel or two from kilometres up: draw it about 30 px across.
		scale = Math.min(120, Math.max(1, (30 * fit.metresPerPixel) / 3));
	} else {
		// The batch's chase shot (render.view 'chase'), with the aircraft moved into the
		// middle of the uncovered part of the window.
		camera = chaseCamera([clampFrame(v.fx), clampFrame(v.fy)]);
	}
	return {
		sea: { ...sea },
		light: { ...light },
		camera,
		aircraft: {
			path: 'trajectory',
			trajectory,
			gimbal: { pitch: g.pitch, yaw: g.yaw, fov: g.fov },
			footprint: view.footprint,
			scale
		},
		mode: view.mode,
		overlays: [...view.overlays],
		tracks: view.track ? [track] : [],
		transition: 0.8
	};
}

/** @param {number} f */
const clampFrame = (f) => Math.max(-0.8, Math.min(0.8, f));
