/**
 * Seeded, deterministic gust response for the aircraft: a few octaves of sinusoids
 * per channel whose amplitudes follow a Dryden-like spectrum (flat below a corner
 * frequency, falling above it) shaped again by the airframe's own response, so the
 * motion is dominated by slow wallows with a little quicker chop on top. Being a sum
 * of sinusoids, every channel and its time derivative are closed-form functions of
 * time: the same seed and time always give the same gust.
 */

/** @param {number} seed */
function mulberry32(seed) {
	let a = seed >>> 0;
	return () => {
		a = (a + 0x6d2b79f5) >>> 0;
		let t = a;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

/**
 * Octave centre frequencies, Hz. Log-spaced so each component stands for a band
 * whose width is proportional to its frequency.
 */
const OCTAVES = [0.045, 0.08, 0.14, 0.25, 0.44, 0.78, 1.35];

/**
 * @typedef {object} Channel
 * @property {Float64Array} freq angular frequencies, rad/s
 * @property {Float64Array} amp
 * @property {Float64Array} phase
 */

/**
 * @param {() => number} random
 * @param {number} corner Hz where the gust spectrum turns down
 * @param {number} response Hz where the airframe stops following the gust
 * @returns {Channel}
 */
function channel(random, corner, response) {
	const n = OCTAVES.length;
	const freq = new Float64Array(n);
	const amp = new Float64Array(n);
	const phase = new Float64Array(n);
	let power = 0;
	for (let i = 0; i < n; i++) {
		const f = OCTAVES[i] * (0.85 + 0.3 * random());
		// Dryden-like PSD ∝ 1 / (1 + (f/fc)²); band power ∝ PSD · f for log spacing;
		// the airframe acts as a first-order low-pass at `response`.
		const band = f / (1 + (f / corner) ** 2) / (1 + (f / response) ** 2);
		freq[i] = 2 * Math.PI * f;
		amp[i] = Math.sqrt(band);
		phase[i] = 2 * Math.PI * random();
		power += amp[i] ** 2 / 2;
	}
	// Normalise to unit RMS.
	const rms = Math.sqrt(power);
	for (let i = 0; i < n; i++) amp[i] /= rms;
	return { freq, amp, phase };
}

/**
 * @param {Channel} c
 * @param {number} t
 */
function value(c, t) {
	let v = 0;
	for (let i = 0; i < c.freq.length; i++) v += c.amp[i] * Math.sin(c.freq[i] * t + c.phase[i]);
	return v;
}

/**
 * @param {Channel} c
 * @param {number} t
 */
function rate(c, t) {
	let v = 0;
	for (let i = 0; i < c.freq.length; i++) v += c.amp[i] * c.freq[i] * Math.cos(c.freq[i] * t + c.phase[i]);
	return v;
}

/**
 * Peak-ish excursions at turbulence 1. A unit-RMS channel times amplitude / 2.5 puts
 * the quoted amplitude at about 2.5 σ: reached now and then, rarely exceeded. The
 * flight model adds the climb angle of the altitude gust to the pitch, so the pitch
 * channel itself is kept below the ±3° the airframe shows in total.
 */
const AMPLITUDE = { bank: 6, pitch: 1.8, heading: 1.5, altitude: 1.5 };
const PEAK = 2.5;

/**
 * @typedef {object} Gust
 * @property {number} bank degrees
 * @property {number} pitch degrees
 * @property {number} heading degrees
 * @property {number} altitude metres
 * @property {number} climbRate m/s, the time derivative of `altitude`
 */

/**
 * @param {number} seed
 * @returns {(t: number, intensity: number, out?: Gust) => Gust}
 */
export function createTurbulence(seed) {
	const random = mulberry32(seed);
	const bank = channel(random, 0.25, 0.7);
	const pitch = channel(random, 0.3, 0.9);
	const heading = channel(random, 0.15, 0.4);
	const altitude = channel(random, 0.05, 0.16);
	return (t, intensity, out = { bank: 0, pitch: 0, heading: 0, altitude: 0, climbRate: 0 }) => {
		const k = Math.max(0, intensity) / PEAK;
		out.bank = k * AMPLITUDE.bank * value(bank, t);
		out.pitch = k * AMPLITUDE.pitch * value(pitch, t);
		out.heading = k * AMPLITUDE.heading * value(heading, t);
		out.altitude = k * AMPLITUDE.altitude * value(altitude, t);
		out.climbRate = k * AMPLITUDE.altitude * rate(altitude, t);
		return out;
	};
}

/**
 * A stable 32-bit hash of a few numbers and strings, to seed each flight.
 * @param {Array<number | string>} parts
 */
export function hashSeed(parts) {
	let h = 2166136261;
	for (const part of parts) {
		const s = String(part);
		for (let i = 0; i < s.length; i++) {
			h ^= s.charCodeAt(i);
			h = Math.imul(h, 16777619);
		}
		h ^= 0x7c;
		h = Math.imul(h, 16777619);
	}
	return h >>> 0;
}
