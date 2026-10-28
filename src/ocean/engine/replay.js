/**
 * Replaying recorded flights: a scene's timed tracks (TrackOverlay.times), its playback
 * clock (TrackClock) and what the overlay layer draws for them each frame — the
 * estimate's marker, a glowing ring with a short comet trail along the last seconds of
 * its track, and the connector from it to the reference at the same instant (gapTo).
 *
 * Everything here is a few dozen segments a frame; the long tracks themselves are drawn
 * once, when the scene is built (engine.buildTracks).
 */
import * as THREE from 'three';
import { TimedTrack, clockTime, clockVisibility, isTimed, resolveClock } from '../aircraft/track.js';
import { DEG, parseColor, srgbToLinear } from './util.js';

/** Seconds of flight the marker's trail covers, and its sampling step. */
const TRAIL_SECONDS = 12;
const TRAIL_STEP = 0.5;
/** Ring radius, in thousandths of the frame height. */
const RING = 11;
const RING_SEGMENTS = 28;

/**
 * @typedef {object} ReplayTrack
 * @property {string} id
 * @property {TimedTrack} timed
 * @property {boolean} marker whether to draw its marker (not when the aircraft flies it)
 * @property {[number, number, number]} color linear rgb
 * @property {string | undefined} gapTo
 */

/**
 * A scene's recorded tracks and the clock that plays them.
 * @typedef {object} Replay
 * @property {import('../aircraft/track.js').ResolvedClock} clock
 * @property {Map<string, ReplayTrack>} tracks
 * @property {number} time recording seconds shown now
 * @property {number} visible 0–1 around the loop's wrap
 * @property {number} gap horizontal metres of the first gapTo pair now; -1 if none
 */

/**
 * @param {import('../scene.js').SceneSpec} spec
 * @param {string | undefined} flown id of the track the aircraft flies
 * @returns {Replay | null}
 */
export function createReplay(spec, flown) {
	const all = spec.tracks ?? [];
	const clock = resolveClock(spec.trackClock, all);
	if (!clock) return null;
	/** @type {Map<string, ReplayTrack>} */
	const tracks = new Map();
	for (const t of all) {
		if (!isTimed(t)) continue;
		const c = parseColor(t.color);
		tracks.set(t.id, {
			id: t.id,
			timed: new TimedTrack(t),
			marker: Boolean(t.marker) && t.id !== flown,
			color: [srgbToLinear(c.r), srgbToLinear(c.g), srgbToLinear(c.b)],
			gapTo: t.gapTo
		});
	}
	const replay = { clock, tracks, time: clock.start, visible: 1, gap: -1 };
	advanceReplay(replay, 0);
	return replay;
}

const _a = { x: 0, y: 0, z: 0 };
const _b = { x: 0, y: 0, z: 0 };

/**
 * Put the replay at `elapsed` simulation seconds since its scene opened.
 * @param {Replay} replay
 * @param {number} elapsed
 */
export function advanceReplay(replay, elapsed) {
	replay.time = clockTime(replay.clock, elapsed);
	replay.visible = clockVisibility(replay.clock, replay.time);
	replay.gap = -1;
	for (const t of replay.tracks.values()) {
		const target = t.gapTo ? replay.tracks.get(t.gapTo) : undefined;
		if (!target) continue;
		t.timed.position(replay.time, _a);
		target.timed.position(replay.time, _b);
		replay.gap = Math.hypot(_a.x - _b.x, _a.y - _b.y);
		break;
	}
}

/**
 * Where a replay's track is now, in absolute east/north/up metres (the layer's shift
 * applied), or null when the replay has no such track.
 * @param {Replay} replay
 * @param {string} id
 * @param {{ east: number, north: number }} shift
 * @returns {{ east: number, north: number, up: number } | null}
 */
export function replayPosition(replay, id, shift) {
	const t = replay.tracks.get(id);
	if (!t) return null;
	t.timed.position(replay.time, _a);
	return { east: _a.x + shift.east, north: _a.y + shift.north, up: _a.z };
}

const WHITE = /** @type {[number, number, number]} */ ([1, 1, 1]);
const SHADOW = /** @type {[number, number, number]} */ ([0, 0, 0]);
const _right = new THREE.Vector3();
const _up = new THREE.Vector3();
const _ndcA = new THREE.Vector3();
const _ndcB = new THREE.Vector3();

/**
 * @typedef {object} ReplayView
 * @property {THREE.PerspectiveCamera} camera the frame's camera (world matrices current)
 * @property {(east: number, north: number, up: number) => THREE.Vector3} toWorld
 * @property {number} height frame height, px (the overlay layer's)
 * @property {number} width frame width, px
 */

/**
 * The overlay lines for a replay this frame: markers, their trails and drop lines, and
 * the gap connectors.
 * @param {Replay} replay
 * @param {{ east: number, north: number }} shift
 * @param {number} opacity the layer's weight in the scene blend
 * @param {ReplayView} view
 * @param {import('./lines.js').Polyline[]} out appended to
 */
export function replayLines(replay, shift, opacity, view, out) {
	const alpha = opacity * replay.visible;
	if (alpha < 0.002) return;
	const cam = view.camera;
	const unit = view.height / 1000;
	const tanHalf = Math.tan((cam.fov * DEG) / 2);
	_right.setFromMatrixColumn(cam.matrixWorld, 0).normalize();
	_up.setFromMatrixColumn(cam.matrixWorld, 1).normalize();
	/** Metres per pixel at a world point. */
	const mpp = (/** @type {THREE.Vector3} */ p) => (2 * p.distanceTo(cam.position) * tanHalf) / Math.max(1, view.height);
	/** @param {{ x: number, y: number, z: number }} p */
	const world = (p) => view.toWorld(p.x + shift.east, p.y + shift.north, p.z);

	for (const t of replay.tracks.values()) {
		const now = world(t.timed.position(replay.time, _a));
		const target = t.gapTo ? replay.tracks.get(t.gapTo) : undefined;
		const scale = mpp(now);
		const radius = RING * unit * scale;

		if (target) {
			const there = world(target.timed.position(replay.time, _b));
			// Start the connector at the ring's edge, measured on screen.
			_ndcA.copy(now).project(cam);
			_ndcB.copy(there).project(cam);
			const px = Math.hypot(((_ndcB.x - _ndcA.x) * view.width) / 2, ((_ndcB.y - _ndcA.y) * view.height) / 2);
			const trim = t.marker && px > 1 ? Math.min(0.9, (RING * unit * 1.25) / px) : 0;
			const from = now.clone().lerp(there, trim);
			if (px > RING * unit * 1.5 || !t.marker) {
				out.push({ points: [from, there], color: SHADOW, alpha: alpha * 0.4, width: 5 * unit });
				out.push({ points: [from, there], color: WHITE, alpha: alpha * 0.95, width: 2.2 * unit });
			}
			// The reference's end: a small white dot (under the aircraft when it flies it).
			out.push({ points: [there, there.clone()], color: WHITE, alpha: alpha * 0.95, width: 5 * unit });
		}

		if (!t.marker) continue;
		const color = t.color;
		// The comet: the last seconds of the track, fading toward the tail.
		const start = Math.max(t.timed.start, replay.time - TRAIL_SECONDS);
		if (replay.time - start > TRAIL_STEP) {
			/** @type {THREE.Vector3[]} */
			const trail = [];
			for (let s = replay.time; s > start; s -= TRAIL_STEP) trail.push(world(t.timed.position(s, _b)));
			trail.push(world(t.timed.position(start, _b)));
			const tail = (replay.time - start) / TRAIL_SECONDS;
			out.push({ points: trail, color, alpha: alpha * 0.3, alphaEnd: alpha * 0.3 * (1 - tail), width: 10 * unit });
			out.push({ points: trail, color, alpha: alpha * 0.95, alphaEnd: alpha * 0.95 * (1 - tail), width: 3.6 * unit });
		}
		// A drop line to the sea, so its height reads.
		if (t.timed.airborne) {
			const foot = world({ x: _a.x, y: _a.y, z: 2.5 });
			out.push({ points: [now, foot], color, alpha: alpha * 0.45, width: 1.2 * unit });
		}
		// The ring, facing the camera, with a glow under it and a bright centre.
		/** @type {THREE.Vector3[]} */
		const ring = [];
		for (let i = 0; i <= RING_SEGMENTS; i++) {
			const a = ((i % RING_SEGMENTS) / RING_SEGMENTS) * Math.PI * 2;
			ring.push(now.clone().addScaledVector(_right, Math.cos(a) * radius).addScaledVector(_up, Math.sin(a) * radius));
		}
		out.push({ points: ring, color: SHADOW, alpha: alpha * 0.35, width: 5.5 * unit });
		out.push({ points: ring, color, alpha: alpha * 0.35, width: 9 * unit });
		out.push({ points: ring, color, alpha, width: 2.4 * unit });
		out.push({ points: [now, now.clone()], color: WHITE, alpha, width: 5 * unit });
	}
}
