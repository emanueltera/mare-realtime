/**
 * The autopilot: lateral path following onto the pattern (an L1 guidance law), and the
 * limits and gains of the inner loops the trajectory integrator applies (bank and roll
 * rate, climb and flight-path angle, airspeed).
 *
 * L1 guidance (Park, Deyst & How, 2004; the law ArduPilot's fixed-wing navigation is
 * built on) steers toward a reference point ahead on the path with the lateral
 * acceleration
 *
 *   a = 2 · V² · sin η / L
 *
 * where V is the ground speed, η the angle from the ground track to the reference point
 * and L the distance to it. Here the reference point is L1 further along the path than
 * the aircraft's closest point (the original takes the L1 circle's crossing with the
 * path; the two agree on a straight leg the aircraft is on, and measuring along the path
 * lets the point slide round corners instead of jumping). On a circle of radius R the
 * point is at chord L = 2R·sin(L1/2R) and η = L1/2R, so a = V²/R exactly: the law flies
 * circles without a curvature feed-forward, and closes onto straight legs like a PD loop
 * with the damping and period set below. L1 scales with the ground speed, so the response
 * keeps its period downwind and up. The acceleration is flown as the coordinated-turn
 * bank tan φ = a / g, limited to ±autopilot.maxBank.
 *
 * Lengths are metres in ENU (x east, y north); bearings and courses are compass radians.
 */

const G = 9.81;
const DEG = Math.PI / 180;
const TAU = 2 * Math.PI;

/** Autopilot limits and gains. Angles in degrees, rates per second. */
export const autopilot = Object.freeze({
	/** Largest bank commanded, deg. */
	maxBank: 35,
	/** Roll-rate limit, deg/s. */
	rollRate: 45,
	/** First-order lag of the bank behind its command, s. */
	rollLag: 0.35,
	/** L1 period, s, and damping: L1 = damping · period · V / π (≈ 4 s of flight). */
	l1Period: 17,
	l1Damping: 0.75,
	/** Smallest L1 distance, m, so the law stays calm at a crawl in a head wind. */
	l1Min: 30,
	/** Climb and sink-rate limits, m/s. */
	maxClimb: 5,
	maxSink: 5,
	/** Altitude error → climb-rate gain, 1/s. */
	altitudeGain: 0.35,
	/** Largest flight-path angle commanded, deg. */
	maxGamma: 12,
	/** Lag of the flight-path angle behind its command, s, and its rate limit, deg/s. */
	gammaLag: 0.7,
	gammaRate: 8,
	/** Lag of the flight-path rate behind the rate that asks for, s, and its limit, deg/s². */
	gammaRateLag: 0.25,
	gammaAccel: 20,
	/** Lag of the airspeed behind its command, s, and the acceleration limit, m/s². */
	airspeedLag: 2.5,
	maxAcceleration: 1.5,
	/** Angle of attack at 1 g and the reference airspeed, deg (α ∝ n / V²). */
	cruiseAlpha: 2,
	referenceAirspeed: 24,
	/** Slowest airspeed commanded, m/s. */
	minAirspeed: 10,
	/** Lowest altitude commanded, m above mean sea level. */
	minAltitude: 10
});

/** @param {number} a radians @returns {number} the same angle in (−π, π] */
export function wrapPi(a) {
	return a - TAU * Math.ceil((a - Math.PI) / TAU);
}

/**
 * @param {number} v
 * @param {number} lo
 * @param {number} hi
 */
export const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

/**
 * @typedef {object} LateralCommand
 * @property {number} bank commanded bank, radians, right wing down positive
 * @property {number} acceleration lateral acceleration asked for, m/s², positive right
 * @property {number} targetX the L1 reference point, m east
 * @property {number} targetY m north
 * @property {number} segment index of the segment being flown
 * @property {number} progress m along it
 * @property {number | undefined} altitude the segment's altitude, if it sets one
 */

/** Segments looked ahead for the reference point past the end of the current one. */
const LOOKAHEAD_SEGMENTS = 6;

/**
 * Path following for one flight. Stateful (which segment, how far along it), and
 * stepped in time order by the integrator.
 */
export class LateralGuidance {
	/** @param {import('./patterns.js').Path} path */
	constructor(path) {
		this.path = path;
		/** Segment being flown, and the closest point's progress along it, m. */
		this.index = 0;
		this.progress = 0;
		/** Path length before the current segment, m. */
		this.base = 0;
		/**
		 * Correction to the aircraft's progress along the whole path, m. Switching to the
		 * next segment changes which point is "closest" (cutting a corner, the aircraft's
		 * projection on the next leg is not where it left the last one), so the progress
		 * jumps; the jump is taken up here and let go of gradually, over about L1 of
		 * travel, so the reference point (and the bank command) moves continuously
		 * through the switch.
		 */
		this.offset = 0;
		/** @type {number} L1 at the last update, m. */
		this.l1 = autopilot.l1Min;
		this.started = false;
		/** @type {LateralCommand} */
		this.out = { bank: 0, acceleration: 0, targetX: 0, targetY: 0, segment: 0, progress: 0, altitude: undefined };
		this._p = { x: 0, y: 0 };
	}

	/**
	 * @param {number} x m east
	 * @param {number} y m north
	 * @param {number} vE ground velocity, m/s east
	 * @param {number} vN m/s north
	 * @returns {LateralCommand}
	 */
	update(x, y, vE, vN) {
		const path = this.path;
		let seg = path.segment(this.index);
		const local = seg.closest(x, y, this.started ? this.progress : 0);
		if (this.started) this.offset *= Math.exp(-Math.abs(local - this.progress) / this.l1);
		this.started = true;
		this.progress = local;
		// Past the end of a leg (its finish line): on to the next.
		const before = this.base + this.progress + this.offset;
		for (let n = 0; n < 16 && this.progress >= seg.length; n++) {
			this.base += seg.length;
			this.index++;
			seg = path.segment(this.index);
			this.progress = seg.closest(x, y, 0);
			this.offset = before - this.base - this.progress;
		}

		const speed = Math.max(1, Math.hypot(vE, vN));
		let L1 = Math.max(autopilot.l1Min, (autopilot.l1Damping * autopilot.l1Period * speed) / Math.PI);
		// On a tight circle, keep the reference point within the near half of it.
		if (seg.radius !== undefined) L1 = Math.min(L1, 1.8 * seg.radius);

		// The reference point: L1 further along the path than the closest point, carried
		// over the end of this segment onto the next ones. Measured along the path rather
		// than as the L1 circle's crossing, it slides continuously round corners (the
		// crossing jumps when a sharp turn's next leg doubles back inside the circle), and
		// the law below divides by the chord to it, which keeps a = V²/R exact on a circle.
		const p = this._p;
		let j = 0;
		this.l1 = L1;
		let along = this.progress + this.offset + L1;
		let target = seg;
		while (along > target.length && j < LOOKAHEAD_SEGMENTS) {
			along -= target.length;
			j++;
			target = path.segment(this.index + j);
		}
		target.point(Math.min(along, target.length), p);
		// Cut a corner and settled on the next leg without crossing the finish line of this
		// one (the along-track distance can creep up on its end for ever): move on once the
		// aircraft is nearer the next segment than this one.
		if (j >= 1) {
			const next = path.segment(this.index + 1);
			if (next.distance(x, y) + 1 < seg.distance(x, y)) {
				const at = this.base + this.progress + this.offset;
				this.base += seg.length;
				this.index++;
				this.progress = next.closest(x, y, 0);
				this.offset = at - this.base - this.progress;
				seg = next;
			}
		}

		const course = Math.atan2(vE, vN);
		const bearing = Math.atan2(p.x - x, p.y - y);
		// η beyond ±90° (the point behind) is flown as a hard turn toward it.
		const eta = clamp(wrapPi(bearing - course), -Math.PI / 2, Math.PI / 2);
		// The chord to the reference point; no longer than L1, so far off the path the law
		// still turns the aircraft firmly toward it.
		const chord = clamp(Math.hypot(p.x - x, p.y - y), 1, L1);
		const a = (2 * speed * speed * Math.sin(eta)) / chord;
		const out = this.out;
		out.acceleration = a;
		out.bank = clamp(Math.atan(a / G), -autopilot.maxBank * DEG, autopilot.maxBank * DEG);
		out.targetX = p.x;
		out.targetY = p.y;
		out.segment = this.index;
		out.progress = this.progress;
		out.altitude = seg.altitude;
		return out;
	}
}
