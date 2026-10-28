/**
 * Small numeric helpers shared by the ocean engine. Plain JavaScript, no three.js,
 * so the scene-blending and randomiser code can be unit-checked in Node.
 */

export const DEG = Math.PI / 180;

/** @param {number} x @param {number} lo @param {number} hi */
export const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

/** @param {number} x */
export const saturate = (x) => clamp(x, 0, 1);

/** @param {number} a @param {number} b @param {number} t */
export const lerp = (a, b, t) => a + (b - a) * t;

/** @param {number} e0 @param {number} e1 @param {number} x */
export function smoothstep(e0, e1, x) {
	const t = saturate((x - e0) / (e1 - e0));
	return t * t * (3 - 2 * t);
}

/** Quintic ease in-out: zero velocity and acceleration at both ends. @param {number} t */
export function smootherstep(t) {
	const x = saturate(t);
	return x * x * x * (x * (x * 6 - 15) + 10);
}

/** Wrap degrees into [0, 360). @param {number} deg */
export const wrap360 = (deg) => ((deg % 360) + 360) % 360;

/** Signed shortest difference b − a in degrees, in (−180, 180]. @param {number} a @param {number} b */
export function angleDelta(a, b) {
	let d = wrap360(b - a);
	if (d > 180) d -= 360;
	return d;
}

/** Interpolate compass degrees along the shortest arc. @param {number} a @param {number} b @param {number} t */
export const lerpAngle = (a, b, t) => wrap360(a + angleDelta(a, b) * t);

/**
 * Frame-rate independent exponential approach toward a target.
 * @param {number} current @param {number} target @param {number} tau seconds @param {number} dt
 */
export function damp(current, target, tau, dt) {
	if (tau <= 0) return target;
	return target + (current - target) * Math.exp(-dt / tau);
}

/**
 * Deterministic 32-bit PRNG (mulberry32).
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
 * Standard normal deviates from a uniform source (Box–Muller), two at a time.
 * @param {() => number} random
 * @returns {() => number}
 */
export function gaussian(random) {
	/** @type {number | null} */
	let spare = null;
	return () => {
		if (spare !== null) {
			const s = spare;
			spare = null;
			return s;
		}
		let u = 0;
		while (u <= 1e-12) u = random();
		const v = random();
		const r = Math.sqrt(-2 * Math.log(u));
		spare = r * Math.sin(2 * Math.PI * v);
		return r * Math.cos(2 * Math.PI * v);
	};
}

/**
 * Parse a CSS colour (#rgb, #rrggbb, #rrggbbaa, rgb(), rgba()) into sRGB 0–1 channels.
 * Anything unrecognised comes back as the HUD's accent colour.
 * @param {string} css
 * @returns {{ r: number, g: number, b: number, a: number }}
 */
export function parseColor(css) {
	const s = css.trim().toLowerCase();
	const hex = s.match(/^#([0-9a-f]{3,8})$/);
	if (hex) {
		let h = hex[1];
		if (h.length === 3 || h.length === 4) h = [...h].map((c) => c + c).join('');
		const n = (/** @type {number} */ i) => parseInt(h.slice(i, i + 2), 16) / 255;
		return { r: n(0), g: n(2), b: n(4), a: h.length === 8 ? n(6) : 1 };
	}
	const fn = s.match(/^rgba?\(([^)]+)\)$/);
	if (fn) {
		const parts = fn[1].split(/[\s,/]+/).filter(Boolean).map((p) => p.trim());
		const ch = (/** @type {string} */ p) => (p.endsWith('%') ? parseFloat(p) / 100 : parseFloat(p) / 255);
		const alpha = parts[3] === undefined ? 1 : parts[3].endsWith('%') ? parseFloat(parts[3]) / 100 : parseFloat(parts[3]);
		return { r: ch(parts[0]), g: ch(parts[1]), b: ch(parts[2]), a: alpha };
	}
	return { r: 79 / 255, g: 216 / 255, b: 230 / 255, a: 1 };
}

/** sRGB transfer to linear. @param {number} c */
export const srgbToLinear = (c) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
