/**
 * What the autopilot is asked to hold over time: the altitude profile and the airspeed
 * command. Both are closed-form (or precomputed) functions of t, so they are the same
 * whatever the integrator does.
 */
import { autopilot } from './guidance.js';
import { logSpaced, noiseValue, smoothNoise, stream, uniform } from './random.js';

/** Climb / sink rate of the `steps` profile's transitions, m/s (inside the autopilot's limit). */
export const STEP_CLIMB_RATE = 2;

/**
 * @typedef {{ altitude: number, rate: number }} AltitudeCommand
 *   altitude: m above mean sea level; rate: its time derivative, m/s (the climb-rate
 *   feed-forward)
 */

/**
 * @param {import('./index.js').AltitudeProfile} profile
 * @param {number} cruise m above mean sea level
 * @param {number} seed
 * @param {number} duration s; the steps profile is laid out this far
 * @returns {(t: number, out: AltitudeCommand) => AltitudeCommand}
 */
export function createAltitudeProfile(profile, cruise, seed, duration) {
	const floor = autopilot.minAltitude;
	switch (profile.type) {
		case 'constant': {
			const h = Math.max(floor, cruise);
			return (_, out) => {
				out.altitude = h;
				out.rate = 0;
				return out;
			};
		}
		case 'sine': {
			if (!(profile.period > 0)) throw new RangeError('trajectory: sine altitude profile needs period > 0');
			const A = profile.amplitude;
			const w = (2 * Math.PI) / profile.period;
			return (t, out) => {
				const h = cruise + A * Math.sin(w * t);
				out.altitude = Math.max(floor, h);
				out.rate = h > floor ? A * w * Math.cos(w * t) : 0;
				return out;
			};
		}
		case 'steps': {
			const interval = profile.interval;
			if (!(interval > 0)) throw new RangeError('trajectory: steps altitude profile needs interval > 0');
			const random = stream(seed, 'altitude');
			// Interval k holds level k, drawn uniformly within ±amplitude of cruise (so the
			// steps wander about the cruise altitude rather than away from it); the command
			// ramps from wherever it was at the interval's start toward that level at
			// STEP_CLIMB_RATE, and waits there.
			const count = Math.ceil(duration / interval) + 2;
			const from = new Float64Array(count);
			const level = new Float64Array(count);
			const start = Math.max(floor, cruise);
			from[0] = level[0] = start;
			for (let k = 1; k < count; k++) {
				level[k] = Math.max(floor, cruise + uniform(random, -profile.amplitude, profile.amplitude));
				from[k] = toward(from[k - 1], level[k - 1], STEP_CLIMB_RATE * interval);
			}
			return (t, out) => {
				const k = Math.min(count - 1, Math.max(0, Math.floor(t / interval)));
				const dt = t - k * interval;
				const gap = level[k] - from[k];
				const ramp = Math.abs(gap) / STEP_CLIMB_RATE;
				out.altitude = toward(from[k], level[k], STEP_CLIMB_RATE * dt);
				out.rate = dt < ramp ? Math.sign(gap) * STEP_CLIMB_RATE : 0;
				return out;
			};
		}
		default:
			throw new TypeError(`trajectory: unknown altitude profile '${/** @type {{ type: string }} */ (profile).type}'`);
	}
}

/**
 * @param {number} from
 * @param {number} to
 * @param {number} step largest move, ≥ 0
 */
function toward(from, to, step) {
	return to > from ? Math.min(to, from + step) : Math.max(to, from - step);
}

/**
 * The airspeed command: the cruise airspeed with a slow seeded wander (periods of 20 s to
 * a few minutes) whose amplitude is about `variation` — scaled so its RMS is that of a
 * sinusoid of amplitude `variation`.
 * @param {number} airspeed m/s
 * @param {number} variation m/s
 * @param {number} seed
 * @returns {(t: number) => number}
 */
export function createAirspeedCommand(airspeed, variation, seed) {
	const base = Math.max(autopilot.minAirspeed, airspeed);
	if (!(variation > 0)) return () => base;
	const noise = smoothNoise(stream(seed, 'airspeed'), logSpaced(1 / 240, 1 / 20, 6));
	const k = variation / Math.SQRT2;
	return (t) => Math.max(autopilot.minAirspeed, base + k * noiseValue(noise, t));
}
