/**
 * How the UAV flies: a kinematic path (straight, orbit, weave or a recorded track),
 * flown in coordinated turns, with gust response on top. No three.js here, so the
 * flight can be checked in node.
 *
 * Determinism: every quantity is a function of elapsed time. The paths and the gusts
 * are closed-form; the one piece of real dynamics — the bank rolling in toward the
 * coordinated-turn bank through a rate limit and a short lag — is integrated on a
 * fixed internal clock from t = 0, so the same spec and the same elapsed time give
 * the same state however the caller slices `step(dt)`.
 */
import { aircraftDefaults } from '../scene.js';
import { createTurbulence, hashSeed } from './turbulence.js';
import { emptyState } from './attitude.js';
import { TimedTrack, clockTime, clockVisibility, emptySample, isTimed, resolveClock } from './track.js';

const G = 9.81;
const DEG = Math.PI / 180;

/** Roll-rate limit, deg/s. */
const ROLL_RATE = 45;
/** First-order lag of the bank behind its command, s. The command is taken this far ahead. */
const ROLL_LAG = 0.35;
/** Fixed integration step for the roll dynamics, s. */
const H = 1 / 120;
/** Cruise angle of attack, degrees: the nose sits a little above the flight path. */
const CRUISE_ALPHA = 2.2;
/** Largest bank the autopilot commands, degrees. */
const MAX_BANK = 40;
/** Weave: heading swing either side of the base heading (deg) and its period (s). */
const WEAVE_AMPLITUDE = 17;
const WEAVE_PERIOD = 22;
/** Fraction of a track loop over which the aircraft fades out at the end and in at the start. */
const TRACK_FADE = 0.04;

/**
 * What a path prescribes at a moment: where the aircraft is, which way it points,
 * how fast it is turning, its flight-path angle, and the coordinated-turn bank.
 * @typedef {object} Kinematics
 * @property {number} x
 * @property {number} y
 * @property {number} altitude
 * @property {number} heading compass degrees
 * @property {number} turnRate deg/s, positive right
 * @property {number} gamma flight-path angle, degrees, climbing positive
 * @property {number} bank coordinated-turn bank for `turnRate` at the airspeed, degrees
 * @property {number} visible 0–1
 * @property {number} speed ground speed, m/s, for a recorded track (others fly `spec.speed`)
 */

/** @returns {Kinematics} */
function emptyKinematics() {
	return { x: 0, y: 0, altitude: 0, heading: 0, turnRate: 0, gamma: 0, bank: 0, visible: 1, speed: 0 };
}

/**
 * Coordinated-turn bank for a turn rate: tan φ = V·ω / g.
 * @param {number} speed m/s
 * @param {number} turnRate deg/s
 */
export function coordinatedBank(speed, turnRate) {
	const bank = Math.atan((speed * turnRate * DEG) / G) / DEG;
	return Math.max(-MAX_BANK, Math.min(MAX_BANK, bank));
}

/** @param {number} deg */
const compass = (deg) => ((deg % 360) + 360) % 360;

/** @param {number} x */
const smoothstep = (x) => {
	const t = Math.max(0, Math.min(1, x));
	return t * t * (3 - 2 * t);
};

/**
 * Bessel function of the first kind, integer order, by its power series (fine for
 * the small arguments the weave uses).
 * @param {number} n
 * @param {number} x
 */
function besselJ(n, x) {
	let sum = 0;
	let term = (x / 2) ** n;
	for (let k = 1; k <= n; k++) term /= k;
	for (let m = 0; m < 12; m++) {
		sum += term;
		term *= -((x / 2) ** 2) / ((m + 1) * (m + 1 + n));
	}
	return sum;
}

/**
 * @typedef {Required<Omit<import('../scene.js').AircraftSpec, 'track'>> & { track?: string }} FullSpec
 */

/**
 * @param {FullSpec} s
 * @returns {(t: number, out: Kinematics) => Kinematics}
 */
function straight(s) {
	const h = s.heading * DEG;
	return (t, out) => {
		out.x = s.x + s.speed * t * Math.sin(h);
		out.y = s.y + s.speed * t * Math.cos(h);
		out.altitude = s.altitude;
		out.heading = compass(s.heading);
		out.turnRate = 0;
		out.gamma = 0;
		out.bank = 0;
		out.visible = 1;
		return out;
	};
}

/**
 * Loiter on a circle round (x, y), starting due north of the centre.
 * @param {FullSpec} s
 * @returns {(t: number, out: Kinematics) => Kinematics}
 */
function orbit(s) {
	const dir = s.orbitDirection >= 0 ? 1 : -1;
	const radius = Math.max(1, s.orbitRadius);
	const omega = (dir * s.speed) / radius; // rad/s, positive clockwise from above
	const bank = coordinatedBank(s.speed, omega / DEG);
	return (t, out) => {
		const angle = omega * t;
		out.x = s.x + radius * Math.sin(angle);
		out.y = s.y + radius * Math.cos(angle);
		out.altitude = s.altitude;
		out.heading = compass(angle / DEG + 90 * dir);
		out.turnRate = omega / DEG;
		out.gamma = 0;
		out.bank = bank;
		out.visible = 1;
		return out;
	};
}

/**
 * Gentle S-turns about the base heading: ψ(t) = ψ₀ + A sin ωt. Position is the exact
 * integral of the velocity, through the Jacobi–Anger expansion of sin(A sin ωt) and
 * cos(A sin ωt), so it is closed-form in t.
 * @param {FullSpec} s
 * @returns {(t: number, out: Kinematics) => Kinematics}
 */
function weave(s) {
	const A = WEAVE_AMPLITUDE * DEG;
	const w = (2 * Math.PI) / WEAVE_PERIOD;
	const h0 = s.heading * DEG;
	const J = Array.from({ length: 8 }, (_, n) => besselJ(n, A));
	return (t, out) => {
		const theta = w * t;
		// ∫ cos(A sin ωt) dt and ∫ sin(A sin ωt) dt from 0 to t.
		let ic = J[0] * t;
		for (let k = 1; 2 * k < J.length; k++) ic += (2 * J[2 * k] * Math.sin(2 * k * theta)) / (2 * k * w);
		let is = 0;
		for (let k = 0; 2 * k + 1 < J.length; k++) {
			const n = 2 * k + 1;
			is += (2 * J[n] * (1 - Math.cos(n * theta))) / (n * w);
		}
		// sin(h0 + φ) = sin h0 cos φ + cos h0 sin φ;  cos(h0 + φ) = cos h0 cos φ − sin h0 sin φ
		out.x = s.x + s.speed * (Math.sin(h0) * ic + Math.cos(h0) * is);
		out.y = s.y + s.speed * (Math.cos(h0) * ic - Math.sin(h0) * is);
		out.altitude = s.altitude;
		out.heading = compass(s.heading + WEAVE_AMPLITUDE * Math.sin(theta));
		out.turnRate = WEAVE_AMPLITUDE * w * Math.cos(theta);
		out.gamma = 0;
		out.bank = coordinatedBank(s.speed, out.turnRate);
		out.visible = 1;
		return out;
	};
}

/**
 * Fly a TrackOverlay: arc-length parametrised at constant ground speed so one pass
 * takes `loopSeconds`, at the track's altitudes. Heading, turn and climb come from the
 * path's smoothed tangent, curvature and slope; the bank is the coordinated-turn bank
 * the aircraft would hold flying that curvature at its airspeed (the loop is a
 * time-lapse of the real flight, so the ground speed is much higher than airspeed).
 * @param {FullSpec} s
 * @param {import('../scene.js').TrackOverlay} track
 * @returns {{ evaluate: (t: number, out: Kinematics) => Kinematics, length: number, loopSeconds: number }}
 */
function trackPath(s, track) {
	const pts = track.points;
	const alts = track.altitudes && track.altitudes.length === pts.length ? track.altitudes : null;
	const cumulative = [0];
	for (let i = 1; i < pts.length; i++) {
		cumulative.push(cumulative[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
	}
	const length = Math.max(1e-6, cumulative[cumulative.length - 1]);
	const loopSeconds = Math.max(1, track.loopSeconds ?? 18);

	// Resample uniformly by arc length.
	const n = Math.max(64, Math.min(4096, Math.ceil(length / 2)));
	const ds = length / (n - 1);
	const X = new Float64Array(n);
	const Y = new Float64Array(n);
	const Z = new Float64Array(n);
	let seg = 0;
	for (let i = 0; i < n; i++) {
		const d = i * ds;
		while (seg < pts.length - 2 && cumulative[seg + 1] < d) seg++;
		const span = cumulative[seg + 1] - cumulative[seg];
		const k = span > 0 ? Math.max(0, Math.min(1, (d - cumulative[seg]) / span)) : 0;
		X[i] = pts[seg][0] + (pts[seg + 1][0] - pts[seg][0]) * k;
		Y[i] = pts[seg][1] + (pts[seg + 1][1] - pts[seg][1]) * k;
		Z[i] = alts ? alts[seg] + (alts[seg + 1] - alts[seg]) * k : s.altitude;
	}

	// Smoothed tangent (heading), slope (flight-path angle), then curvature.
	const window = Math.max(1, Math.round(Math.max(8, Math.min(60, length * 0.012)) / ds));
	const psi = new Float64Array(n);
	const gamma = new Float64Array(n);
	const kappa = new Float64Array(n);
	for (let i = 0; i < n; i++) {
		const a = Math.max(0, i - window);
		const b = Math.min(n - 1, i + window);
		psi[i] = Math.atan2(X[b] - X[a], Y[b] - Y[a]);
		if (i > 0) {
			// Unwrap so the heading is continuous along the path.
			while (psi[i] - psi[i - 1] > Math.PI) psi[i] -= 2 * Math.PI;
			while (psi[i] - psi[i - 1] < -Math.PI) psi[i] += 2 * Math.PI;
		}
		gamma[i] = Math.atan2(Z[b] - Z[a], (b - a) * ds);
	}
	for (let i = 0; i < n; i++) {
		const a = Math.max(0, i - window);
		const b = Math.min(n - 1, i + window);
		kappa[i] = (psi[b] - psi[a]) / ((b - a) * ds); // rad/m, positive turning right
	}

	/** @param {Float64Array} arr @param {number} f */
	const sample = (arr, f) => {
		const i = Math.min(n - 2, Math.floor(f));
		const k = f - i;
		return arr[i] + (arr[i + 1] - arr[i]) * k;
	};

	return {
		length,
		loopSeconds,
		evaluate(t, out) {
			const u = (((t % loopSeconds) + loopSeconds) % loopSeconds) / loopSeconds;
			const f = u * (n - 1);
			out.x = sample(X, f);
			out.y = sample(Y, f);
			out.altitude = sample(Z, f);
			out.heading = compass(sample(psi, f) / DEG);
			out.turnRate = (s.speed * sample(kappa, f)) / DEG;
			out.gamma = sample(gamma, f) / DEG;
			out.bank = coordinatedBank(s.speed, out.turnRate);
			out.visible = smoothstep(Math.min(u, 1 - u) / TRACK_FADE);
			return out;
		}
	};
}

/**
 * Fly a recorded (timed) track: placed by the scene's playback clock, a pure function
 * of the time since the scene opened, so the markers the engine draws from the same
 * clock show the same instant. Heading, turn rate and climb are the recording's own,
 * smoothed over a few samples (see track.js), and the bank is the coordinated-turn bank
 * for that turn rate at the recorded ground speed: the bank the aircraft flew, whatever
 * the playback rate.
 * @param {import('../scene.js').TrackOverlay} track
 * @param {import('./track.js').ResolvedClock} clock
 * @returns {(t: number, out: Kinematics) => Kinematics}
 */
function timedPath(track, clock) {
	const timed = new TimedTrack(track);
	const sample = emptySample();
	return (t, out) => {
		const time = clockTime(clock, t);
		timed.sample(time, sample);
		out.x = sample.x;
		out.y = sample.y;
		out.altitude = sample.z;
		out.heading = sample.heading;
		out.turnRate = sample.turnRate;
		out.gamma = sample.gamma;
		out.bank = coordinatedBank(Math.max(sample.speed, 1), sample.turnRate);
		out.visible = clockVisibility(clock, time);
		out.speed = sample.speed;
		return out;
	};
}

/**
 * @param {import('../scene.js').AircraftSpec} spec
 * @param {import('../scene.js').TrackOverlay[]} [tracks]
 * @param {import('../scene.js').TrackClock} [trackClock] playback of timed tracks
 * @returns {import('../aircraft.js').Flight}
 */
export function createFlight(spec, tracks = [], trackClock) {
	/** @type {FullSpec} */
	const s = { ...aircraftDefaults, ...spec };
	const track = s.path === 'track' ? tracks.find((t) => t.id === s.track && t.points.length > 1) : undefined;
	const clock = track && isTimed(track) ? resolveClock(trackClock, tracks) : null;

	/** @type {(t: number, out: Kinematics) => Kinematics} */
	let kinematics;
	if (track && clock) kinematics = timedPath(track, clock);
	else if (track) kinematics = trackPath(s, track).evaluate;
	else if (s.path === 'orbit') kinematics = orbit(s);
	else if (s.path === 'weave') kinematics = weave(s);
	else kinematics = straight(s);

	const gust = createTurbulence(hashSeed([s.path, s.heading, s.x, s.y, s.altitude, s.track ?? '', s.orbitRadius]));
	const gustOut = { bank: 0, pitch: 0, heading: 0, altitude: 0, climbRate: 0 };
	const kin = emptyKinematics();
	const ahead = emptyKinematics();

	/** The bank command, taken ROLL_LAG ahead so the lag does not make the bank trail the turn. */
	/** @param {number} t */
	const command = (t) => kinematics(t + ROLL_LAG, ahead).bank;

	/** @param {number} bank @param {number} t @param {number} dt */
	const advance = (bank, t, dt) => {
		const want = (command(t) - bank) / ROLL_LAG;
		const rate = Math.max(-ROLL_RATE, Math.min(ROLL_RATE, want));
		return bank + rate * dt;
	};

	// Committed roll state on the fixed clock.
	let committedTime = 0;
	let committedBank = command(0);

	/** Bank (without gusts) at time t, integrated on the fixed clock. */
	/** @param {number} t */
	const bankAt = (t) => {
		// A recording's bank is already smooth, and its loop jumps: no roll dynamics.
		if (clock) return kin.bank;
		if (t < committedTime) {
			committedTime = 0;
			committedBank = command(0);
		}
		while (committedTime + H <= t) {
			committedBank = advance(committedBank, committedTime, H);
			committedTime += H;
		}
		const rest = t - committedTime;
		return rest > 1e-9 ? advance(committedBank, committedTime, rest) : committedBank;
	};

	const state = emptyState();
	state.speed = s.speed;

	/** @param {number} t */
	const evaluate = (t) => {
		kinematics(t, kin);
		gust(t, s.turbulence, gustOut);
		const bank = bankAt(t) + gustOut.bank;
		const climb = Math.atan2(gustOut.climbRate, Math.max(1, s.speed)) / DEG;
		state.x = kin.x;
		state.y = kin.y;
		state.altitude = kin.altitude + gustOut.altitude;
		state.heading = compass(kin.heading + gustOut.heading);
		// Flight path plus the angle of attack. In a coordinated turn the wing needs
		// α = α0·n = α0 / cos φ, but that extra lift tilts the nose toward the turn centre,
		// not up: the Euler pitch it adds is α·cos φ = α0. So the pitch angle is the
		// flight path plus the cruise α, whatever the bank.
		state.pitch = kin.gamma + climb + CRUISE_ALPHA + gustOut.pitch;
		state.bank = bank;
		// A recording flies at its own ground speed (m/s of flight time); the rest at the spec's.
		state.speed = clock ? kin.speed : s.speed;
		state.turnRate = kin.turnRate;
		state.time = t;
		state.visible = kin.visible;
		return state;
	};

	evaluate(0);

	return {
		state,
		step(dt) {
			return evaluate(Math.max(0, state.time + dt));
		}
	};
}
