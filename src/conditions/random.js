/**
 * Seeded randomness for the condition generators. Every random choice in a job goes
 * through these, never `Math.random`, so a job file expands to the same runs on any
 * machine, in node or the browser, today or next year.
 */

/**
 * Deterministic 32-bit PRNG (mulberry32), the same generator the engine seeds its
 * wave field with.
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
 * Mix two integers into a well-spread unsigned 32-bit seed (murmur3's finaliser), so
 * neighbouring inputs (seed 7, run 0, 1, 2 …) start unrelated streams.
 * @param {number} a
 * @param {number} b
 * @returns {number}
 */
export function mixSeed(a, b) {
	let h = (a >>> 0) ^ Math.imul(b >>> 0, 0x9e3779b1);
	h = Math.imul(h ^ (h >>> 16), 0x85ebca6b);
	h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35);
	return (h ^ (h >>> 16)) >>> 0;
}

/**
 * A short, stable hex digest of a JSON value (FNV-1a over its JSON text), for naming
 * a swept value that has no shorter name.
 * @param {unknown} value
 * @returns {string} six hex digits
 */
export function shortHash(value) {
	const text = JSON.stringify(value) ?? '';
	let h = 0x811c9dc5;
	for (let i = 0; i < text.length; i++) h = Math.imul(h ^ text.charCodeAt(i), 0x01000193);
	return (h >>> 0).toString(16).padStart(8, '0').slice(0, 6);
}

/**
 * The draws a domain randomiser needs, from one stream.
 * @param {number} seed
 */
export function sampler(seed) {
	const random = mulberry32(seed);
	return {
		random,
		/** @param {number} a @param {number} b */
		uniform: (a, b) => a + (b - a) * random(),
		/** Log-uniform: equal odds per octave, for scales like altitude and fetch. @param {number} a @param {number} b */
		logUniform: (a, b) => Math.exp(Math.log(a) + (Math.log(b) - Math.log(a)) * random()),
		/** @template T @param {readonly T[]} items @returns {T} */
		pick: (items) => items[Math.min(items.length - 1, Math.floor(random() * items.length))]
	};
}
