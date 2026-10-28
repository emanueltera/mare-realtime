/**
 * Turbulence: a seeded, band-limited gust field added to the mean wind, and the small
 * roll and pitch disturbances the same air gives the airframe.
 *
 * Each channel is smooth noise (a sum of sinusoids, see random.js) whose spectrum
 * follows the Dryden forms of the low-altitude turbulence model (MIL-F-8785C), with
 * its length scales from the cruise altitude and its corner frequencies from the
 * airspeed (Taylor's frozen turbulence: a spatial wavelength λ passes at f = V / λ):
 *
 *   horizontal  Φ(f) ∝ 1 / (1 + (2π f L_u / V)²)
 *   vertical    Φ(f) ∝ (1 + 3 (2π f L_w / V)²) / (1 + (2π f L_w / V)²)²
 *
 * then rolled off above ~1 Hz, the airframe's own response: the aircraft is carried
 * along by gusts it can follow and shrugs off the sharpest ones. The roll-off is what
 * makes the gust velocity differentiable, so the aircraft's acceleration (and its IMU)
 * stays finite and independent of the integration step.
 *
 * Intensity is the `turbulence` knob (0 still air – 1 strong) times a base that grows
 * with the mean wind, after the model's σ_w ∝ wind: at turbulence 1 in a 15 m/s wind
 * the vertical gust is about 2.4 m/s RMS, the model's "severe" at low level; the
 * default 0.15 in a moderate breeze is a light chop of a few tenths of a m/s. The
 * gust field is a function of time, not of position: the aircraft flies through one
 * frozen realisation along its own track.
 */
import { logSpaced, noiseRate, noiseValue, smoothNoise, stream } from './random.js';

const FT = 0.3048;
/** Components per channel: enough for a broad spectrum without visible periodicity. */
const COMPONENTS = 24;
/** Above this, Hz, the airframe no longer follows the gust. */
const AIRFRAME_RESPONSE = 1;

/**
 * Low-altitude Dryden length scales, m (MIL-F-8785C, the formula is in feet).
 * @param {number} altitude m
 */
export function lengthScales(altitude) {
	const h = Math.max(10, altitude);
	const hft = h / FT;
	return { horizontal: h / (0.177 + 0.000823 * Math.min(hft, 1000)) ** 1.2, vertical: h };
}

/**
 * RMS gust intensities, m/s, and the roll / pitch disturbance rates, deg/s.
 * @param {number} turbulence 0–1
 * @param {number} windSpeed mean wind at altitude, m/s
 * @param {number} altitude m
 */
export function gustIntensity(turbulence, windSpeed, altitude) {
	const k = Math.max(0, turbulence);
	const hft = Math.max(10, altitude) / FT;
	const vertical = k * (0.6 + 0.12 * Math.max(0, windSpeed));
	// The model's ratio of horizontal to vertical intensity near the surface.
	const horizontal = vertical / (0.177 + 0.000823 * Math.min(hft, 1000)) ** 0.4;
	return {
		horizontal,
		vertical,
		roll: k * (10 + 0.6 * Math.max(0, windSpeed)),
		pitch: k * (3 + 0.15 * Math.max(0, windSpeed))
	};
}

/**
 * @typedef {object} Gust
 * @property {number} east gust velocity, m/s
 * @property {number} north
 * @property {number} up
 * @property {number} dEast its time derivative, m/s²
 * @property {number} dNorth
 * @property {number} dUp
 * @property {number} roll roll-rate disturbance, deg/s (the autopilot works against it)
 * @property {number} pitch flight-path-rate disturbance, deg/s
 */

/** @returns {Gust} */
export function emptyGust() {
	return { east: 0, north: 0, up: 0, dEast: 0, dNorth: 0, dUp: 0, roll: 0, pitch: 0 };
}

/**
 * @param {number} seed the trajectory's seed
 * @param {{ turbulence: number, windSpeed: number, altitude: number, airspeed: number }} o
 * @returns {(t: number, out: Gust) => Gust}
 */
export function createGusts(seed, { turbulence, windSpeed, altitude, airspeed }) {
	const sigma = gustIntensity(turbulence, windSpeed, altitude);
	if (!(turbulence > 0)) {
		return (_, out) => {
			out.east = out.north = out.up = out.dEast = out.dNorth = out.dUp = out.roll = out.pitch = 0;
			return out;
		};
	}
	const random = stream(seed, 'gusts');
	const L = lengthScales(altitude);
	const V = Math.max(5, airspeed);
	/** @param {number} f */
	const airframe = (f) => 1 / (1 + (f / AIRFRAME_RESPONSE) ** 2) ** 2;
	/** @param {number} f */
	const horizontalPsd = (f) => airframe(f) / (1 + ((2 * Math.PI * f * L.horizontal) / V) ** 2);
	/** @param {number} f */
	const verticalPsd = (f) => {
		const x = ((2 * Math.PI * f * L.vertical) / V) ** 2;
		return (airframe(f) * (1 + 3 * x)) / (1 + x) ** 2;
	};
	// Rotational disturbances: the gust's variation across the span and along the
	// fuselage, which sits at shorter wavelengths than the gust itself.
	/** @param {number} f */
	const rotationPsd = (f) => airframe(f) / (1 + (f / 0.4) ** 2);
	const freqs = logSpaced(0.004, 2.5, COMPONENTS);
	const east = smoothNoise(random, freqs, horizontalPsd);
	const north = smoothNoise(random, freqs, horizontalPsd);
	const up = smoothNoise(random, freqs, verticalPsd);
	const roll = smoothNoise(random, freqs, rotationPsd);
	const pitch = smoothNoise(random, freqs, rotationPsd);
	return (t, out) => {
		out.east = sigma.horizontal * noiseValue(east, t);
		out.north = sigma.horizontal * noiseValue(north, t);
		out.up = sigma.vertical * noiseValue(up, t);
		out.dEast = sigma.horizontal * noiseRate(east, t);
		out.dNorth = sigma.horizontal * noiseRate(north, t);
		out.dUp = sigma.vertical * noiseRate(up, t);
		out.roll = sigma.roll * noiseValue(roll, t);
		out.pitch = sigma.pitch * noiseValue(pitch, t);
		return out;
	};
}
