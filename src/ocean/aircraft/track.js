/**
 * Recorded tracks: a track with `times` is a flight log, not a shape to loop at constant
 * speed. This module interpolates one smoothly in time and runs the clock that replays
 * it (`TrackClock` in scene.js). No three.js here, so it can be checked in node.
 *
 * Interpolation is a cubic Hermite in time (Catmull–Rom tangents, allowing for uneven
 * sample spacing), so position and velocity are continuous across samples. The heading,
 * turn rate, climb and ground speed are taken from centred differences over a few
 * samples and interpolated the same way, so the airframe's heading and bank change
 * smoothly at any playback rate rather than stepping at every two-second sample.
 */

/** Samples either side for the heading and speed chords. */
const CHORD = 2;
/** Samples either side for the turn rate (a difference of the chord headings). */
const TURN = 2;
/** Real seconds the aircraft and markers take to fade out before the loop wraps, and in after. */
export const WRAP_FADE = 0.3;

/**
 * Where a timed track is at a moment.
 * @typedef {object} TrackSample
 * @property {number} x metres east
 * @property {number} y metres north
 * @property {number} z metres above mean sea level
 * @property {number} heading compass degrees of the (smoothed) direction of travel
 * @property {number} turnRate deg/s of flight time, positive right
 * @property {number} gamma flight-path angle, degrees, climbing positive
 * @property {number} speed ground speed, m/s of flight time
 */

/** @returns {TrackSample} */
export function emptySample() {
	return { x: 0, y: 0, z: 0, heading: 0, turnRate: 0, gamma: 0, speed: 0 };
}

/**
 * Whether a TrackOverlay is a recorded, timed track.
 * @param {import('../scene.js').TrackOverlay} track
 */
export function isTimed(track) {
	return Boolean(track.times && track.times.length === track.points.length && track.points.length > 1);
}

/** @param {number} deg */
const compass = (deg) => ((deg % 360) + 360) % 360;

/**
 * Cubic Hermite on [0, 1] with tangents already scaled by the interval.
 * @param {number} p0 @param {number} m0 @param {number} p1 @param {number} m1 @param {number} u
 */
function hermite(p0, m0, p1, m1, u) {
	const u2 = u * u;
	const u3 = u2 * u;
	return (2 * u3 - 3 * u2 + 1) * p0 + (u3 - 2 * u2 + u) * m0 + (-2 * u3 + 3 * u2) * p1 + (u3 - u2) * m1;
}

/**
 * Finite-difference (Catmull–Rom) tangents of `v` against `t`, per unit time.
 * @param {Float64Array} v
 * @param {Float64Array} t
 */
function tangents(v, t) {
	const n = v.length;
	const m = new Float64Array(n);
	for (let i = 0; i < n; i++) {
		const a = Math.max(0, i - 1);
		const b = Math.min(n - 1, i + 1);
		m[i] = (v[b] - v[a]) / Math.max(1e-6, t[b] - t[a]);
	}
	return m;
}

export class TimedTrack {
	/** @param {import('../scene.js').TrackOverlay} track a timed track (see isTimed) */
	constructor(track) {
		const n = track.points.length;
		const T = Float64Array.from(/** @type {number[]} */ (track.times));
		const X = Float64Array.from(track.points, (p) => p[0]);
		const Y = Float64Array.from(track.points, (p) => p[1]);
		const alts = track.altitudes && track.altitudes.length === n ? track.altitudes : null;
		const Z = Float64Array.from({ length: n }, (_, i) => (alts ? alts[i] : 0));
		/** Whether the track is in the air (has altitudes). */
		this.airborne = Boolean(alts);
		this.n = n;
		this.T = T;
		this.X = X;
		this.Y = Y;
		this.Z = Z;
		this.mX = tangents(X, T);
		this.mY = tangents(Y, T);
		this.mZ = tangents(Z, T);
		this.start = T[0];
		this.end = T[n - 1];

		// Smoothed heading (unwrapped, radians), ground speed and climb at each sample.
		const psi = new Float64Array(n);
		const speed = new Float64Array(n);
		const gamma = new Float64Array(n);
		for (let i = 0; i < n; i++) {
			const a = Math.max(0, i - CHORD);
			const b = Math.min(n - 1, i + CHORD);
			const de = X[b] - X[a];
			const dn = Y[b] - Y[a];
			const dist = Math.hypot(de, dn);
			const dt = Math.max(1e-6, T[b] - T[a]);
			psi[i] = Math.atan2(de, dn);
			if (i > 0) {
				while (psi[i] - psi[i - 1] > Math.PI) psi[i] -= 2 * Math.PI;
				while (psi[i] - psi[i - 1] < -Math.PI) psi[i] += 2 * Math.PI;
			}
			speed[i] = dist / dt;
			gamma[i] = Math.atan2(Z[b] - Z[a], Math.max(dist, 1e-3));
		}
		const turn = new Float64Array(n);
		for (let i = 0; i < n; i++) {
			const a = Math.max(0, i - TURN);
			const b = Math.min(n - 1, i + TURN);
			turn[i] = (psi[b] - psi[a]) / Math.max(1e-6, T[b] - T[a]);
		}
		this.psi = psi;
		this.turn = turn;
		this.speed = speed;
		this.gamma = gamma;
		this.mTurn = tangents(turn, T);
		this.hint = 0;
	}

	/**
	 * Index of the sample interval holding t (clamped to the track).
	 * @param {number} t
	 */
	segment(t) {
		const T = this.T;
		let i = Math.min(Math.max(this.hint, 0), this.n - 2);
		if (t >= T[i] && t <= T[i + 1]) return i;
		if (i + 2 < this.n && t >= T[i + 1] && t <= T[i + 2]) return (this.hint = i + 1);
		let lo = 0;
		let hi = this.n - 2;
		while (lo < hi) {
			const mid = (lo + hi + 1) >> 1;
			if (T[mid] <= t) lo = mid;
			else hi = mid - 1;
		}
		return (this.hint = lo);
	}

	/**
	 * Position only, at time t (clamped to the recording).
	 * @param {number} t seconds
	 * @param {{ x: number, y: number, z: number }} out
	 */
	position(t, out) {
		const tc = Math.min(this.end, Math.max(this.start, t));
		const i = this.segment(tc);
		const h = this.T[i + 1] - this.T[i];
		const u = h > 0 ? (tc - this.T[i]) / h : 0;
		out.x = hermite(this.X[i], this.mX[i] * h, this.X[i + 1], this.mX[i + 1] * h, u);
		out.y = hermite(this.Y[i], this.mY[i] * h, this.Y[i + 1], this.mY[i + 1] * h, u);
		out.z = hermite(this.Z[i], this.mZ[i] * h, this.Z[i + 1], this.mZ[i + 1] * h, u);
		return out;
	}

	/**
	 * Everything at time t (clamped to the recording).
	 * @param {number} t seconds
	 * @param {TrackSample} out
	 */
	sample(t, out) {
		this.position(t, out);
		const tc = Math.min(this.end, Math.max(this.start, t));
		const i = this.segment(tc);
		const h = this.T[i + 1] - this.T[i];
		const u = h > 0 ? (tc - this.T[i]) / h : 0;
		// Heading as a Hermite whose tangent is the turn rate, so the two agree.
		const psi = hermite(this.psi[i], this.turn[i] * h, this.psi[i + 1], this.turn[i + 1] * h, u);
		out.heading = compass((psi * 180) / Math.PI);
		const turn = hermite(this.turn[i], this.mTurn[i] * h, this.turn[i + 1], this.mTurn[i + 1] * h, u);
		out.turnRate = (turn * 180) / Math.PI;
		out.speed = this.speed[i] + (this.speed[i + 1] - this.speed[i]) * u;
		out.gamma = ((this.gamma[i] + (this.gamma[i + 1] - this.gamma[i]) * u) * 180) / Math.PI;
		return out;
	}
}

/**
 * A scene's playback clock, resolved: a pure function of the seconds since the scene
 * opened, so the flight and the markers of one scene always agree.
 * @typedef {object} ResolvedClock
 * @property {number} start seconds into the recording at scene time 0
 * @property {number} rate recording seconds per (simulation) second
 * @property {number} from loop start
 * @property {number} to loop end
 */

/**
 * The clock for a scene's timed tracks, or null when it has none.
 * @param {import('../scene.js').TrackClock | undefined} spec
 * @param {import('../scene.js').TrackOverlay[]} tracks
 * @returns {ResolvedClock | null}
 */
export function resolveClock(spec, tracks) {
	const timed = tracks.filter(isTimed);
	if (timed.length === 0) return null;
	let first = Infinity;
	let last = -Infinity;
	for (const t of timed) {
		const times = /** @type {number[]} */ (t.times);
		first = Math.min(first, times[0]);
		last = Math.max(last, times[times.length - 1]);
	}
	const from = Math.max(first, Math.min(spec?.loop?.[0] ?? first, last));
	const to = Math.min(last, Math.max(spec?.loop?.[1] ?? last, from));
	const start = Math.min(to, Math.max(from, spec?.start ?? from));
	const rate = Math.max(0, spec?.rate ?? 1);
	return { start, rate, from, to };
}

/**
 * Seconds into the recording `elapsed` simulation seconds after the scene opened.
 * @param {ResolvedClock} clock
 * @param {number} elapsed
 */
export function clockTime(clock, elapsed) {
	const span = clock.to - clock.from;
	const t = clock.start + clock.rate * Math.max(0, elapsed);
	if (span <= 1e-6) return Math.min(t, clock.to);
	return clock.from + ((((t - clock.from) % span) + span) % span);
}

/**
 * 0–1: how visible what the clock places is at recording time `t` — fading out over the
 * last WRAP_FADE real seconds before the loop wraps and back in over the first after.
 * @param {ResolvedClock} clock
 * @param {number} t
 */
export function clockVisibility(clock, t) {
	const span = clock.to - clock.from;
	if (span <= 1e-6 || clock.rate <= 0) return 1;
	// A loop over the whole recording still wraps, so it fades the same way.
	const fade = Math.min(span / 4, WRAP_FADE * clock.rate);
	const edge = Math.min(t - clock.from, clock.to - t);
	const x = Math.max(0, Math.min(1, edge / fade));
	return x * x * (3 - 2 * x);
}
