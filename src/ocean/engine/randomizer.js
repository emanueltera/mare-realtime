/**
 * Domain randomisation for the `randomize` scenes: a deterministic sample per index,
 * so sample 17 is always the same sea under the same sky, seen through the same lens.
 * Ranges follow the deck's description of the training data; a few rules keep every
 * combination worth looking at (no black dusk under a closed deck, hazier overcast).
 */
import { lerp, mulberry32 } from './util.js';

/**
 * @typedef {object} Sample
 * @property {import('./state.js').ResolvedSea} sea
 * @property {import('./state.js').ResolvedLight} light
 * @property {{ altitude: number, speed: number, gimbalPitch: number, fov: number, turbulence: number, stabilized: number }} onboard
 * @property {{ height: number, pitch: number, fov: number }} free
 */

/**
 * @param {number} index 1-based sample number
 * @param {import('./state.js').ResolvedSea} baseSea for the fields the randomiser leaves alone
 * @param {import('./state.js').ResolvedLight} baseLight
 * @returns {Sample}
 */
export function randomSample(index, baseSea, baseLight) {
	const random = mulberry32(0x9e3779b1 ^ Math.imul(index, 2654435761));
	/** @param {number} a @param {number} b */
	const uniform = (a, b) => lerp(a, b, random());
	/** @param {number} a @param {number} b */
	const logUniform = (a, b) => Math.exp(lerp(Math.log(a), Math.log(b), random()));

	const windSpeed = uniform(2, 20);
	const windDirection = uniform(0, 360);
	const swellHeight = random() < 0.25 ? 0 : uniform(0.1, 2.5);
	const swellPeriod = uniform(8, 16);
	const swellDirection = windDirection + uniform(-70, 70);
	const choppiness = uniform(0.6, 1.4);

	let sunElevation = uniform(-4, 65);
	const sunAzimuth = uniform(0, 360);
	let overcast = random() < 0.45 ? uniform(0, 0.25) : uniform(0.25, 1);
	let haze = uniform(0, 0.8);
	// Low sun under a closed deck is just dark grey: keep dusk skies open.
	if (sunElevation < 6) overcast = Math.min(overcast, 0.4);
	if (sunElevation < 0) haze = Math.min(haze, 0.55);
	// A closed deck comes with thick air.
	if (overcast > 0.7) haze = Math.max(haze, 0.3);
	// Calm, overcast and low sun together reads as a blank: lift the sun a little.
	if (windSpeed < 4 && overcast > 0.6) sunElevation = Math.max(sunElevation, 15);

	const altitude = logUniform(25, 400);
	const fov = uniform(35, 75);
	// Any gimbal angle from just under the horizon to steeply down, at any altitude.
	const gimbalPitch = uniform(-8, -70);
	const turbulence = uniform(0, 0.8);
	const stabilized = uniform(0.3, 1);
	const speed = uniform(18, 32);

	return {
		sea: {
			...baseSea,
			windSpeed,
			windDirection,
			swellHeight,
			swellPeriod,
			swellDirection: ((swellDirection % 360) + 360) % 360,
			choppiness
		},
		light: { ...baseLight, sunElevation, sunAzimuth, overcast, haze },
		onboard: { altitude, speed, gimbalPitch, fov, turbulence, stabilized },
		free: { height: logUniform(3, 40), pitch: uniform(-2, -30), fov: uniform(35, 75) }
	};
}
