/**
 * Simulated UAV flights: a pattern flown by an autopilot through wind and turbulence,
 * integrated once on a fixed step and stored, with the ground truth and the IMU at every
 * step. Pure and deterministic — no three.js, no DOM, no Math.random or clock — so the
 * same spec gives bit-identical samples on the same JS engine build. Node and a browser
 * agree to about 1e-12 (identical at the CSVs' 6 decimals) but not to the bit: their
 * Math.sin, exp, atan2 and the like may differ in the last place, and the integration
 * carries that through.
 *
 * Frames and units (see docs/ARCHITECTURE.md): positions in metres, ENU (x east, y north,
 * altitude above mean sea level); directions in compass degrees (0 north, 90 east,
 * clockwise), wind toward, not from; attitude heading / pitch nose-up / bank
 * right-wing-down, yaw–pitch–roll; IMU in FRD body axes, specific force f = a − g.
 *
 * The flight model, per internal step (1/200 s):
 *
 * - Lateral: L1 guidance onto the pattern's current leg or circle (guidance.js) gives a
 *   bank command, limited to ±35°; the bank follows it through a 45°/s roll-rate limit
 *   and a 0.35 s lag, with the turbulence's roll disturbance on top. The turn is
 *   coordinated: the lift lies along the wings' normal, so the part of it that turns
 *   the path is tan φ times the part that holds it up and bends it, and
 *   ψ̇ = (V·γ̇ + g·cos γ)·tan φ / (V·cos γ) — g·tan φ / V when level. The small sideslip of
 *   a real coordinated turn is ignored, so the nose points along the air velocity.
 * - Vertical: the altitude profile (or a waypoint's altitude) with its climb-rate
 *   feed-forward, an altitude-error loop, climb / sink limits of 5 m/s and a flight-path
 *   angle limit of ±12°. The flight path follows its command through second-order
 *   dynamics — a lagged, rate-limited path rate reached at a limited pitch acceleration —
 *   so the pitch rate and the load factor ramp at a profile's corners instead of jumping.
 * - Speed: the airspeed follows its command through a lag and a 1.5 m/s² limit.
 * - Air: ground velocity = air velocity (airspeed along the heading and the flight-path
 *   angle) + wind (the mean wind at the aircraft's altitude + gusts), so in a cross wind
 *   the nose crabs into it and course and heading differ by the crab angle.
 * - Attitude: pitch = flight-path angle + angle of attack, α = 2° · n · (24 / V)² with
 *   the load factor n = 1 / cos φ, so the nose rises in a banked turn and at low speed.
 *
 * The autopilot runs at the step rate and its commands are held over each step; the
 * plant (position, heading, bank, flight-path angle, airspeed) is integrated over the
 * step with classical Runge–Kutta, the gusts evaluated where each stage falls. At every
 * step the state, its analytic rates and acceleration, and the IMU computed from them
 * are stored, so the IMU agrees with the derivative of the stored velocity and attitude
 * to the integration error.
 */
import { G, bodyRates, specificForce } from './imu.js';
import { LateralGuidance, autopilot, clamp } from './guidance.js';
import { createAirspeedCommand, createAltitudeProfile } from './profiles.js';
import { createPath, patternDefaults as P } from './patterns.js';
import { createWindField } from './wind.js';
import { createGusts, emptyGust } from './turbulence.js';

export { windAloft, seaRoughness } from './wind.js';
export { autopilot } from './guidance.js';
export { patternDefaults } from './patterns.js';
export { STEP_CLIMB_RATE } from './profiles.js';
export { lengthScales, gustIntensity } from './turbulence.js';
export { nedToBody, specificForce, bodyRates } from './imu.js';
export { sampleColumns, imuColumns, formatNumber, samplesToCsv, imuToCsv, samplesToJson, parseCsv } from './io.js';
export { enuToGeodetic } from './geodesy.js';

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
 */

/**
 * Lengths and radii in metres; points and centres are [right, ahead] of the start
 * (east, north when the start heading is 0). `direction` 1 is clockwise seen from above.
 * Waypoint altitudes are m above mean sea level.
 * @typedef {(
 *   | { type: 'straight' }
 *   | { type: 'orbit', radius?: number, direction?: 1 | -1, center?: [number, number] }
 *   | { type: 'racetrack', length?: number, radius?: number, direction?: 1 | -1 }
 *   | { type: 'figure8', radius?: number }
 *   | { type: 'lawnmower', legLength?: number, spacing?: number, legs?: number }
 *   | { type: 'waypoints', points: Array<[number, number] | [number, number, number]>, loop?: boolean }
 *   | { type: 'random', legLength?: [number, number], turn?: [number, number] }
 * )} PatternSpec
 */

/**
 * @typedef {(
 *   | { type: 'constant' }
 *   | { type: 'sine', amplitude: number, period: number }
 *   | { type: 'steps', amplitude: number, interval: number }
 * )} AltitudeProfile
 */

/**
 * 'sea': the sea state's U10 and direction carried up to altitude by a neutral log
 * profile over open water (wind.js); { speed, direction }: the wind at altitude, m/s,
 * compass toward; 'none': still air (turbulence, if any, still blows).
 * @typedef {'sea' | 'none' | { speed: number, direction: number }} WindSpec
 */

/**
 * @typedef {object} GimbalSpec
 * @property {number} [pitch]       degrees, negative looks down; default -35
 * @property {number} [yaw]         degrees from the nose, positive right; default 0
 * @property {number} [fov]         vertical field of view, degrees; default 48
 * @property {number} [stabilized]  0 hard-mounted – 1 fully stabilised; default 0.55
 */

/**
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

/**
 * @typedef {object} Trajectory
 * @property {ResolvedTrajectorySpec} spec   every default filled in
 * @property {number} duration               s
 * @property {number} rate                   Hz of the internal step
 * @property {(t: number) => TrajectorySample} stateAt  any t, clamped to [0, duration]
 * @property {(rate: number) => TrajectorySample[]} sample  at t = i / rate, i = 0 … ⌊duration·rate⌋
 * @property {string[]} warnings  what makes the flight unlike its pattern (a wind it
 *   cannot outfly); empty for an ordinary flight
 */

/**
 * Mean wind at cruise, as a fraction of the airspeed, from which a flight is flagged: a
 * pattern cannot be flown much above it (the aircraft crabs past 65°, its ground speed
 * into the wind falls toward nothing and the guidance hunts).
 */
export const WIND_WARNING = 0.9;

/**
 * @typedef {Required<Omit<TrajectorySpec, 'start' | 'gimbal'>> & {
 *   start: { x: number, y: number, heading: number },
 *   gimbal: Required<GimbalSpec>
 * }} ResolvedTrajectorySpec
 */

/** Gimbal defaults (the onboard camera's mount). */
export const gimbalDefaults = Object.freeze({ pitch: -35, yaw: 0, fov: 48, stabilized: 0.55 });

/** Defaults for every optional TrajectorySpec field. */
export const trajectoryDefaults = Object.freeze({
	seed: 1,
	altitudeProfile: Object.freeze({ type: 'constant' }),
	airspeed: 24,
	airspeedVariation: 0,
	start: Object.freeze({ x: 0, y: 0, heading: 0 }),
	wind: 'sea',
	turbulence: 0.15,
	gimbal: gimbalDefaults
});

/** Internal integration rate, Hz: the step the flight is simulated and stored at. */
export const INTERNAL_RATE = 200;

const DEG = Math.PI / 180;

/**
 * @param {readonly number[]} a
 * @returns {[number, number]}
 */
const pair = (a) => [a[0], a[1]];

/**
 * @param {string} name
 * @param {number} v
 * @returns {number} v, if it is a finite number > 0
 */
function positive(name, v) {
	if (!(Number.isFinite(v) && v > 0)) throw new RangeError(`trajectory: pattern.${name} must be > 0, got ${v}`);
	return v;
}

/**
 * @param {number} v
 * @returns {1 | -1}
 */
function rotation(v) {
	if (v !== 1 && v !== -1) throw new RangeError(`trajectory: pattern.direction must be 1 or -1, got ${v}`);
	return v;
}

/**
 * A pattern with its defaults filled in (orbit centre included), as plain JSON. Throws
 * on geometry that cannot be flown (a radius, length or spacing that is not positive).
 * @param {PatternSpec} p
 * @returns {PatternSpec}
 */
function resolvePattern(p) {
	if (!p || typeof p !== 'object' || typeof p.type !== 'string') throw new TypeError('trajectory: pattern.type is required');
	switch (p.type) {
		case 'straight':
			return { type: 'straight' };
		case 'orbit': {
			const radius = positive('radius', p.radius ?? P.orbit.radius);
			const direction = rotation(p.direction ?? P.orbit.direction);
			/** @type {[number, number]} */
			const center = p.center ? [p.center[0], p.center[1]] : [direction * radius, 0];
			if (!center.every(Number.isFinite)) throw new RangeError('trajectory: pattern.center must be two finite numbers');
			return { type: 'orbit', radius, direction, center };
		}
		case 'racetrack':
			return {
				type: 'racetrack',
				length: positive('length', p.length ?? P.racetrack.length),
				radius: positive('radius', p.radius ?? P.racetrack.radius),
				direction: rotation(p.direction ?? P.racetrack.direction)
			};
		case 'figure8':
			return { type: 'figure8', radius: positive('radius', p.radius ?? P.figure8.radius) };
		case 'lawnmower': {
			const legs = p.legs ?? P.lawnmower.legs;
			if (!(Number.isInteger(legs) && legs >= 1)) throw new RangeError(`trajectory: pattern.legs must be a whole number >= 1, got ${legs}`);
			return {
				type: 'lawnmower',
				legLength: positive('legLength', p.legLength ?? P.lawnmower.legLength),
				spacing: positive('spacing', p.spacing ?? P.lawnmower.spacing),
				legs
			};
		}
		case 'waypoints': {
			const points = (p.points ?? []).map((q) => /** @type {[number, number] | [number, number, number]} */ ([...q]));
			if (!points.every((q) => (q.length === 2 || q.length === 3) && q.every(Number.isFinite))) {
				throw new RangeError('trajectory: pattern.points must be [right, ahead] or [right, ahead, altitude] pairs of finite numbers');
			}
			return { type: 'waypoints', points, loop: p.loop ?? P.waypoints.loop };
		}
		case 'random': {
			const legLength = pair(p.legLength ?? P.random.legLength);
			const turn = pair(p.turn ?? P.random.turn);
			positive('legLength[0]', legLength[0]);
			if (!(legLength[1] >= legLength[0])) throw new RangeError('trajectory: pattern.legLength must be [min, max] with max >= min');
			if (!(turn.every(Number.isFinite) && turn[1] >= turn[0])) throw new RangeError('trajectory: pattern.turn must be [min, max] with max >= min');
			return { type: 'random', legLength, turn };
		}
		default:
			throw new TypeError(`trajectory: unknown pattern type '${/** @type {{ type: string }} */ (p).type}'`);
	}
}

/**
 * The spec with every default filled in: a fresh, JSON-serialisable object (the input
 * is not modified). Throws on what cannot be flown.
 * @param {TrajectorySpec} spec
 * @returns {ResolvedTrajectorySpec}
 */
export function resolveTrajectorySpec(spec) {
	if (!(Number.isFinite(spec.duration) && spec.duration > 0)) throw new RangeError('trajectory: duration must be > 0 s');
	if (!Number.isFinite(spec.altitude)) throw new RangeError('trajectory: altitude is required (m above mean sea level)');
	const d = trajectoryDefaults;
	const wind = spec.wind ?? d.wind;
	return {
		seed: spec.seed ?? d.seed,
		duration: spec.duration,
		pattern: resolvePattern(spec.pattern),
		altitude: spec.altitude,
		altitudeProfile: /** @type {AltitudeProfile} */ ({ ...(spec.altitudeProfile ?? d.altitudeProfile) }),
		airspeed: spec.airspeed ?? d.airspeed,
		airspeedVariation: spec.airspeedVariation ?? d.airspeedVariation,
		start: {
			x: spec.start?.x ?? d.start.x,
			y: spec.start?.y ?? d.start.y,
			heading: spec.start?.heading ?? d.start.heading
		},
		wind: typeof wind === 'string' ? wind : { speed: wind.speed, direction: wind.direction },
		turbulence: spec.turbulence ?? d.turbulence,
		gimbal: { ...gimbalDefaults, ...spec.gimbal }
	};
}

/** The stored channels, one Float64Array each, a value per internal step. */
const CHANNELS = /** @type {const} */ ([
	'x',
	'y',
	'h',
	'vE',
	'vN',
	'vU',
	'aE',
	'aN',
	'aU',
	'psi', // heading, degrees, unwrapped (continuous through 360)
	'psiDot', // deg/s
	'theta', // pitch, degrees
	'thetaDot',
	'phi', // bank, degrees
	'phiDot',
	'V', // airspeed
	'VDot',
	'wE', // wind incl. gusts
	'wN',
	'wU',
	'fx',
	'fy',
	'fz',
	'p',
	'q',
	'r'
]);

/**
 * Fly a trajectory. The whole flight is integrated here, once; `stateAt` and `sample`
 * only look it up. Memory is 26 doubles per internal step (≈ 42 kB per second of
 * flight).
 * @param {TrajectorySpec} input
 * @param {{ sea?: { windSpeed: number, windDirection: number } }} [options] the SeaState
 *   (only its wind is read), needed for wind: 'sea'
 * @returns {Trajectory}
 */
export function createTrajectory(input, { sea } = {}) {
	const spec = resolveTrajectorySpec(input);
	const { duration, seed } = spec;
	const RATE = INTERNAL_RATE;
	const H = 1 / RATE;
	// Nodes at k / RATE for k = 0 … N, the last at or just past `duration`.
	const N = Math.max(1, Math.ceil(duration * RATE - 1e-9));

	/** @type {Record<(typeof CHANNELS)[number], Float64Array>} */
	const ch = /** @type {any} */ ({});
	for (const name of CHANNELS) ch[name] = new Float64Array(N + 1);

	const windField = createWindField(spec.wind, sea);
	const cruiseWind = windField(spec.altitude, { east: 0, north: 0, dEast: 0, dNorth: 0 });
	const cruiseWindSpeed = Math.hypot(cruiseWind.east, cruiseWind.north);
	const gusts = createGusts(seed, {
		turbulence: spec.turbulence,
		windSpeed: cruiseWindSpeed,
		altitude: spec.altitude,
		airspeed: spec.airspeed
	});
	const altitudeCommand = createAltitudeProfile(spec.altitudeProfile, spec.altitude, seed, duration);
	const airspeedCommand = createAirspeedCommand(spec.airspeed, spec.airspeedVariation, seed);
	const path = createPath(
		spec.pattern,
		{ x: spec.start.x, y: spec.start.y, heading: spec.start.heading * DEG },
		{ seed, airspeed: spec.airspeed, windSpeed: cruiseWindSpeed, maxBank: autopilot.maxBank }
	);
	const guidance = new LateralGuidance(path);
	/** @type {string[]} */
	const warnings = [];
	if (cruiseWindSpeed >= WIND_WARNING * spec.airspeed) {
		warnings.push(
			`the mean wind at cruise (${cruiseWindSpeed.toFixed(1)} m/s) is ${cruiseWindSpeed >= spec.airspeed ? 'at or above' : 'close to'} the airspeed (${spec.airspeed} m/s): the pattern cannot be flown as laid out`
		);
	}

	const ap = autopilot;
	const rollRate = ap.rollRate * DEG;
	const gammaRate = ap.gammaRate * DEG;
	const maxGamma = ap.maxGamma * DEG;
	// Bank can run past the command in a gust, never this far.
	const bankStop = 60 * DEG;

	const gammaAccel = ap.gammaAccel * DEG;

	// The plant's state: position, heading, bank, flight-path angle and its rate, airspeed.
	const X = 0;
	const Y = 1;
	const Hh = 2;
	const PSI = 3;
	const PHI = 4;
	const GAMMA = 5;
	const GDOT = 6;
	const V = 7;
	const NS = 8;
	const S = new Float64Array(NS);
	const tmp = new Float64Array(NS);
	const k1 = new Float64Array(NS);
	const k2 = new Float64Array(NS);
	const k3 = new Float64Array(NS);
	const k4 = new Float64Array(NS);

	// The aircraft starts at the start point, level, at the profile's altitude and the
	// commanded airspeed, already trimmed for the first leg: its ground track along the
	// start heading (the direction the pattern is laid out from), the nose crabbed into
	// the mean cross wind, and banked as the guidance asks there (wings level on a
	// straight leg, the circle's bank on an orbit). Started on the bare start heading,
	// every windy flight would open with a correction turn of tens of degrees.
	const cmd = { altitude: 0, rate: 0 };
	altitudeCommand(0, cmd);
	const wind = { east: 0, north: 0, dEast: 0, dNorth: 0 };
	S[X] = spec.start.x;
	S[Y] = spec.start.y;
	S[Hh] = cmd.altitude;
	S[GAMMA] = 0;
	S[GDOT] = 0;
	S[V] = airspeedCommand(0);
	{
		const course = spec.start.heading * DEG;
		windField(S[Hh], wind);
		// The wind across the course, positive to the right; the air velocity cancels it.
		const cross = wind.east * Math.cos(course) - wind.north * Math.sin(course);
		S[PSI] = course - Math.asin(clamp(cross / S[V], -0.95, 0.95));
		const vE = S[V] * Math.sin(S[PSI]) + wind.east;
		const vN = S[V] * Math.cos(S[PSI]) + wind.north;
		// The guidance is stateful, but a second update at the same point and velocity (the
		// first step's, below) gives the same command, so asking it here changes nothing.
		S[PHI] = guidance.update(S[X], S[Y], vE, vN).bank;
	}

	/** What the autopilot asks of the plant, held over each step (it runs at the step rate). */
	const hold = { bank: 0, gamma: 0, airspeed: 0 };

	/**
	 * The plant's rates of change in `state`, with the gusts `g` and the held commands.
	 * @param {Float64Array} state
	 * @param {import('./turbulence.js').Gust} g
	 * @param {Float64Array} out
	 */
	function rates(state, g, out) {
		const v = state[V];
		const gamma = state[GAMMA];
		const psi = state[PSI];
		windField(state[Hh], wind);
		const cg = Math.cos(gamma);
		out[X] = v * cg * Math.sin(psi) + wind.east + g.east;
		out[Y] = v * cg * Math.cos(psi) + wind.north + g.north;
		out[Hh] = v * Math.sin(gamma) + g.up;
		// The flight path bends through a second-order response: the rate the lag asks
		// for (rate-limited) is reached at a limited pitch acceleration, so a step in the
		// command (a profile's corner, a new waypoint altitude) ramps the pitch rate and
		// the load factor instead of jumping them in one step.
		const gdWant = clamp((hold.gamma - gamma) / ap.gammaLag, -gammaRate, gammaRate);
		out[GDOT] = clamp((gdWant - state[GDOT]) / ap.gammaRateLag, -gammaAccel, gammaAccel);
		const gammaDot = state[GDOT] + g.pitch * DEG;
		out[GAMMA] = gammaDot;
		// Coordinated: the lift lies in the plane of the wings' normal, so the part of it
		// that bends the path up (V·γ̇ + g·cos γ, per unit mass) and the part that turns it
		// (V·cos γ·ψ̇) are in the ratio of cos φ to sin φ. Level, this is ψ̇ = g·tan φ / V;
		// climbing or descending in a turn, it keeps the side force at zero.
		out[PSI] = ((v * gammaDot + G * Math.cos(gamma)) * Math.tan(state[PHI])) / (v * Math.cos(gamma));
		out[PHI] = clamp((hold.bank - state[PHI]) / ap.rollLag, -rollRate, rollRate) + g.roll * DEG;
		out[V] = clamp((hold.airspeed - v) / ap.airspeedLag, -ap.maxAcceleration, ap.maxAcceleration);
		return out;
	}

	// Gusts at the start, middle and end of a step; the end of one is the start of the next.
	let gNode = gusts(0, emptyGust());
	let gEnd = emptyGust();
	const gMid = emptyGust();
	const f = { x: 0, y: 0, z: 0 };
	const w = { x: 0, y: 0, z: 0 };

	for (let k = 0; k <= N; k++) {
		const t = k / RATE;
		const psi = S[PSI];
		const phi = S[PHI];
		const gamma = S[GAMMA];
		const v = S[V];
		const h = S[Hh];

		// The autopilot, sampled at the node and held over the step: guidance on the ground
		// velocity, the altitude loop on the altitude.
		windField(h, wind);
		const cg = Math.cos(gamma);
		const sg = Math.sin(gamma);
		const sp = Math.sin(psi);
		const cp = Math.cos(psi);
		const lateral = guidance.update(S[X], S[Y], v * cg * sp + wind.east + gNode.east, v * cg * cp + wind.north + gNode.north);
		if (lateral.altitude !== undefined) {
			cmd.altitude = Math.max(ap.minAltitude, lateral.altitude);
			cmd.rate = 0;
		} else altitudeCommand(t, cmd);
		const climb = clamp(cmd.rate + ap.altitudeGain * (cmd.altitude - h), -ap.maxSink, ap.maxClimb);
		hold.bank = lateral.bank;
		hold.gamma = clamp(Math.asin(clamp(climb / v, -1, 1)), -maxGamma, maxGamma);
		hold.airspeed = airspeedCommand(t);

		// Rates at the node: the stored derivatives and the IMU come from these.
		rates(S, gNode, k1);
		const vE = k1[X];
		const vN = k1[Y];
		const vU = k1[Hh];
		const psiDot = k1[PSI];
		const phiDot = k1[PHI];
		const gammaDot = k1[GAMMA];
		const VDot = k1[V];
		windField(h, wind);

		// Acceleration over the ground: d/dt of V·(cos γ sin ψ, cos γ cos ψ, sin γ) + wind,
		// the wind changing with time (gusts) and with height (the mean profile's shear).
		const aE = VDot * cg * sp + v * (-sg * gammaDot * sp + cg * cp * psiDot) + gNode.dEast + wind.dEast * vU;
		const aN = VDot * cg * cp + v * (-sg * gammaDot * cp - cg * sp * psiDot) + gNode.dNorth + wind.dNorth * vU;
		const aU = VDot * sg + v * cg * gammaDot + gNode.dUp;

		// Attitude: the nose sits α above the flight path, more in a turn (load factor)
		// and at low speed.
		const n = 1 / Math.cos(phi);
		const ka = ap.cruiseAlpha * DEG * (ap.referenceAirspeed / v) ** 2;
		const alpha = ka * n;
		const alphaDot = ka * (n * Math.tan(phi) * phiDot - (2 * n * VDot) / v);
		const theta = gamma + alpha;
		const thetaDot = gammaDot + alphaDot;

		specificForce(aE, aN, aU, psi, theta, phi, f);
		bodyRates(psiDot, thetaDot, phiDot, theta, phi, w);

		ch.x[k] = S[X];
		ch.y[k] = S[Y];
		ch.h[k] = h;
		ch.vE[k] = vE;
		ch.vN[k] = vN;
		ch.vU[k] = vU;
		ch.aE[k] = aE;
		ch.aN[k] = aN;
		ch.aU[k] = aU;
		ch.psi[k] = psi / DEG;
		ch.psiDot[k] = psiDot / DEG;
		ch.theta[k] = theta / DEG;
		ch.thetaDot[k] = thetaDot / DEG;
		ch.phi[k] = phi / DEG;
		ch.phiDot[k] = phiDot / DEG;
		ch.V[k] = v;
		ch.VDot[k] = VDot;
		ch.wE[k] = wind.east + gNode.east;
		ch.wN[k] = wind.north + gNode.north;
		ch.wU[k] = gNode.up;
		ch.fx[k] = f.x;
		ch.fy[k] = f.y;
		ch.fz[k] = f.z;
		ch.p[k] = w.x;
		ch.q[k] = w.y;
		ch.r[k] = w.z;
		if (k === N) break;

		// Advance to the next node: classical Runge–Kutta on the plant, commands held.
		gusts((k + 0.5) / RATE, gMid);
		gusts((k + 1) / RATE, gEnd);
		for (let i = 0; i < NS; i++) tmp[i] = S[i] + 0.5 * H * k1[i];
		rates(tmp, gMid, k2);
		for (let i = 0; i < NS; i++) tmp[i] = S[i] + 0.5 * H * k2[i];
		rates(tmp, gMid, k3);
		for (let i = 0; i < NS; i++) tmp[i] = S[i] + H * k3[i];
		rates(tmp, gEnd, k4);
		for (let i = 0; i < NS; i++) S[i] += (H / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
		S[PHI] = clamp(S[PHI], -bankStop, bankStop);
		S[V] = Math.max(ap.minAirspeed * 0.5, S[V]);
		const swap = gNode;
		gNode = gEnd;
		gEnd = swap;
	}

	const gimbal = spec.gimbal;

	/**
	 * @param {number} time s
	 * @returns {TrajectorySample}
	 */
	function stateAt(time) {
		const tc = clamp(Number.isFinite(time) ? time : 0, 0, duration);
		const u = tc * RATE;
		const k = Math.min(N - 1, Math.floor(u));
		const s = u - k;
		// Cubic Hermite basis on [k, k + 1]; m0, m1 are derivatives per second, hence ·H.
		const s2 = s * s;
		const s3 = s2 * s;
		const h00 = 2 * s3 - 3 * s2 + 1;
		const h10 = (s3 - 2 * s2 + s) * H;
		const h01 = -2 * s3 + 3 * s2;
		const h11 = (s3 - s2) * H;
		/** @param {Float64Array} v @param {Float64Array} d */
		const hermite = (v, d) => h00 * v[k] + h10 * d[k] + h01 * v[k + 1] + h11 * d[k + 1];
		/** @param {Float64Array} v */
		const linear = (v) => v[k] + (v[k + 1] - v[k]) * s;

		const vE = hermite(ch.vE, ch.aE);
		const vN = hermite(ch.vN, ch.aN);
		const heading = hermite(ch.psi, ch.psiDot);
		return {
			t: tc,
			x: hermite(ch.x, ch.vE),
			y: hermite(ch.y, ch.vN),
			altitude: hermite(ch.h, ch.vU),
			vE,
			vN,
			vU: hermite(ch.vU, ch.aU),
			heading: compass(heading),
			course: compass(Math.atan2(vE, vN) / DEG),
			pitch: hermite(ch.theta, ch.thetaDot),
			bank: hermite(ch.phi, ch.phiDot),
			turnRate: linear(ch.psiDot),
			airspeed: hermite(ch.V, ch.VDot),
			groundSpeed: Math.hypot(vE, vN),
			windE: linear(ch.wE),
			windN: linear(ch.wN),
			windU: linear(ch.wU),
			fx: linear(ch.fx),
			fy: linear(ch.fy),
			fz: linear(ch.fz),
			p: linear(ch.p),
			q: linear(ch.q),
			r: linear(ch.r),
			gimbalPitch: gimbal.pitch,
			gimbalYaw: gimbal.yaw,
			gimbalFov: gimbal.fov
		};
	}

	return {
		spec,
		duration,
		rate: RATE,
		warnings,
		stateAt,
		sample(rate) {
			if (!(rate > 0)) throw new RangeError('trajectory: sample rate must be > 0 Hz');
			const count = Math.floor(duration * rate + 1e-9);
			/** @type {TrajectorySample[]} */
			const out = new Array(count + 1);
			for (let i = 0; i <= count; i++) out[i] = stateAt(i / rate);
			return out;
		}
	};
}

/** @param {number} deg */
function compass(deg) {
	const c = deg % 360;
	return c < 0 ? c + 360 : c === 0 ? 0 : c;
}

/**
 * The engine's AircraftState for a sample: where to draw the aircraft and how.
 * @param {TrajectorySample} s
 * @returns {{ x: number, y: number, altitude: number, heading: number, pitch: number, bank: number, speed: number, turnRate: number, time: number, visible: number }}
 */
export function toAircraftState(s) {
	return {
		x: s.x,
		y: s.y,
		altitude: s.altitude,
		heading: s.heading,
		pitch: s.pitch,
		bank: s.bank,
		speed: s.airspeed,
		turnRate: s.turnRate,
		time: s.t,
		visible: 1
	};
}
