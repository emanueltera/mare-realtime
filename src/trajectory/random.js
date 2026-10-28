/**
 * Seeded randomness for trajectories: a small PRNG, a way to give every random choice
 * its own stream, and smooth noise as a sum of sinusoids.
 *
 * Every random quantity in a trajectory is drawn from a stream named after what it is
 * for (`stream(seed, 'gusts')`, `stream(seed, 'pattern')`, …), so adding a draw to one
 * part of the flight never shifts the numbers another part sees. No `Math.random`, no
 * clock: the same seed gives the same flight, bit for bit, in node and the browser.
 */

/**
 * A stable 32-bit FNV-1a hash of a few numbers and strings, to seed a stream.
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
		// A separator, so ['ab', 'c'] and ['a', 'bc'] hash apart.
		h ^= 0x7c;
		h = Math.imul(h, 16777619);
	}
	return h >>> 0;
}

/**
 * mulberry32: a tiny, fast PRNG with a 32-bit state; plenty for flight noise.
 * @param {number} seed
 * @returns {() => number} uniform in [0, 1)
 */
export function mulberry32(seed) {
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
 * The PRNG for one named purpose within a trajectory.
 * @param {number} seed the trajectory's seed
 * @param {string} name what the stream is for
 */
export function stream(seed, name) {
	return mulberry32(hashSeed(['mare-trajectory', seed, name]));
}

/**
 * @param {() => number} random
 * @param {number} lo
 * @param {number} hi
 */
export function uniform(random, lo, hi) {
	return lo + (hi - lo) * random();
}

/**
 * Smooth, band-limited noise: a sum of sinusoids at log-spaced, slightly jittered
 * frequencies with random phases, normalised to unit RMS. Being closed-form, the value
 * and its time derivative are exact at any t, which is what lets gusts enter the
 * aircraft's velocity without making its acceleration (and so the IMU) depend on the
 * integration step.
 * @typedef {object} SmoothNoise
 * @property {Float64Array} omega angular frequencies, rad/s
 * @property {Float64Array} amp
 * @property {Float64Array} phase
 */

/**
 * @param {() => number} random
 * @param {number[]} frequencies centre frequencies, Hz; each component stands for the band
 *   around it, so log spacing gives every octave its share
 * @param {(f: number) => number} [psd] one-sided power spectral density shape (any scale);
 *   default flat
 * @returns {SmoothNoise}
 */
export function smoothNoise(random, frequencies, psd = () => 1) {
	const n = frequencies.length;
	const omega = new Float64Array(n);
	const amp = new Float64Array(n);
	const phase = new Float64Array(n);
	let power = 0;
	for (let i = 0; i < n; i++) {
		const f = frequencies[i] * (0.88 + 0.24 * random());
		// Band power ∝ PSD · bandwidth, and log spacing makes the bandwidth ∝ f.
		const band = psd(f) * f;
		omega[i] = 2 * Math.PI * f;
		amp[i] = Math.sqrt(band);
		phase[i] = 2 * Math.PI * random();
		power += amp[i] ** 2 / 2;
	}
	const rms = Math.sqrt(power) || 1;
	for (let i = 0; i < n; i++) amp[i] /= rms;
	return { omega, amp, phase };
}

/**
 * @param {SmoothNoise} c
 * @param {number} t s
 */
export function noiseValue(c, t) {
	let v = 0;
	for (let i = 0; i < c.omega.length; i++) v += c.amp[i] * Math.sin(c.omega[i] * t + c.phase[i]);
	return v;
}

/**
 * The time derivative of `noiseValue`, per second.
 * @param {SmoothNoise} c
 * @param {number} t s
 */
export function noiseRate(c, t) {
	let v = 0;
	for (let i = 0; i < c.omega.length; i++) v += c.amp[i] * c.omega[i] * Math.cos(c.omega[i] * t + c.phase[i]);
	return v;
}

/**
 * `count` frequencies spaced evenly in log between `lo` and `hi`, Hz.
 * @param {number} lo
 * @param {number} hi
 * @param {number} count
 */
export function logSpaced(lo, hi, count) {
	const out = [];
	for (let i = 0; i < count; i++) out.push(lo * (hi / lo) ** (count === 1 ? 0 : i / (count - 1)));
	return out;
}
