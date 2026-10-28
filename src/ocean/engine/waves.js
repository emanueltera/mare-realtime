/**
 * The simulated sea: a Tessendorf FFT ocean on the GPU, in cascades.
 *
 * Every frame:
 *  1. `evolve` builds h0(k) from the sea state on the fly (fetch-limited JONSWAP wind
 *     sea + a swell band, cos^2s spreading, deep-water dispersion — the formulas of
 *     `../spectrum.js`, ported to GLSL), times a fixed seeded Gaussian field, and
 *     advances it to time t. It writes six complex spectra per bin, packed in pairs of
 *     real fields: height + vertical velocity, horizontal displacement, slopes,
 *     Jacobian terms and horizontal orbital velocity.
 *  2. A radix-4 Stockham inverse FFT runs over rows then columns (four passes each at
 *     N = 256, butterflies from a precomputed texture), all cascades at once: they are
 *     stacked vertically in one texture.
 *  3. `assemble` unpacks each cascade into four mip-mapped, tiling half-float textures:
 *     displacement + vertical velocity; slopes and their second moments, for LEAN-style
 *     roughness; horizontal orbital velocity (for the flow overlay); and the cascade's
 *     share of the surface Jacobian (from which the surface and the foam field,
 *     `foam.js`, find the crests that break) with the slope cross moment.
 *
 * Frames: the simulation works in the local east/north frame of `scene.js`. Texel
 * (i, j) of a cascade of patch L sits at (i·L/N east, j·L/N north). Wave travel is
 * h̃(k,t) = h0(k)e^{−iωt} + conj(h0(−k))e^{iωt}, so energy at k runs toward +k, and the
 * choppy displacement is D̃ = i·k̂·h̃, which sharpens crests (Jacobian < 1 on them).
 */
import * as THREE from 'three';
import { G, swellParams, windSeaParams } from '../spectrum.js';
import { FullscreenPass, floatTarget } from './gpu.js';
import { DEG, gaussian, mulberry32, smoothstep } from './util.js';

/** Largest patch, metres; the others step down by RATIO so no two tile together. */
export const BASE_PATCH = 1013;
const RATIO = 6.71;
/** A band edge sits at this many wavelengths per patch of the finer cascade. */
const EDGE_HARMONIC = 6;
/** The animation loops seamlessly after this many seconds (ω is quantised to it). */
export const LOOP_SECONDS = 1200;
export const OMEGA_0 = (2 * Math.PI) / LOOP_SECONDS;
/** Bins of cascade 0 nearer the origin than this are integrated over their cell. */
const SUPERSAMPLE_RADIUS = 18;
/** e-folding life of foam on the water once its crest has passed, s. */
const FOAM_LIFETIME = 2.4;
/** Along-wind diffusivity of foam at 10 m/s, m²/s: how fast patches draw out into streaks. */
const FOAM_DIFFUSIVITY = 0.9;
/**
 * Whitecaps come from the dominant waves breaking, not from ripples: each cascade's share
 * in the Jacobian that is thresholded for breaking crests, where the shorter cascades
 * only roughen the edge of a breaking crest.
 */
export const JACOBIAN_WEIGHTS = /** @type {[number, number, number, number]} */ ([1, 0.55, 0.2, 0]);

/**
 * Breaking threshold, in standard deviations of the weighted Jacobian below 1, by wind
 * speed (m/s). Calibrated by measuring the rendered whitecap channel from above so the
 * foam covers about what Monahan & O'Muircheartaigh (1980) give for whitecap coverage,
 * W ≈ 3.84·10⁻⁶·U^3.41: a fraction of a percent at force 4, a few percent at force 7.
 */
const BREAK_SIGMAS = [
	[4, 3.0],
	[6, 2.05],
	[9, 1.55],
	[12, 1.18],
	[15, 0.85],
	[18, 0.5],
	[22, 0.3]
];

/** @param {number} windSpeed m/s */
export function breakThreshold(windSpeed) {
	const t = BREAK_SIGMAS;
	if (windSpeed <= t[0][0]) return t[0][1];
	for (let i = 1; i < t.length; i++) {
		if (windSpeed <= t[i][0]) {
			const f = (windSpeed - t[i - 1][0]) / (t[i][0] - t[i - 1][0]);
			return t[i - 1][1] + (t[i][1] - t[i - 1][1]) * f;
		}
	}
	return t[t.length - 1][1];
}

/**
 * Patch sizes and wavenumber bands for a cascade count.
 * @param {number} count
 * @param {number} size FFT resolution
 */
export function cascadeLayout(count, size) {
	/** @type {number[]} */
	const patches = [];
	for (let i = 0; i < count; i++) patches.push(BASE_PATCH / Math.pow(RATIO, i));
	/** @type {Array<[number, number]>} */
	const bands = [];
	for (let i = 0; i < count; i++) {
		const lo = i === 0 ? 0 : (EDGE_HARMONIC * 2 * Math.PI) / patches[i];
		const hi = i === count - 1 ? Infinity : (EDGE_HARMONIC * 2 * Math.PI) / patches[i + 1];
		bands.push([lo, hi]);
	}
	const nyquist = (Math.PI * size) / patches[count - 1];
	return { patches, bands, nyquist };
}

/**
 * The resolved sea state the simulation runs: every field filled.
 * @typedef {Required<import('../scene.js').SeaState>} ResolvedSea
 */

/**
 * JONSWAP S(ω), mirrored from spectrum.js so the CPU-side band moments match the GPU.
 * @param {number} w
 * @param {{ alpha: number, omegaPeak: number, gamma: number }} band
 */
function jonswap(w, { alpha, omegaPeak, gamma }) {
	if (w <= 0 || omegaPeak / w > 3.5) return 0;
	const sigma = w <= omegaPeak ? 0.07 : 0.09;
	const o = w - omegaPeak;
	const r = Math.exp(-(o * o) / (2 * sigma * sigma * omegaPeak * omegaPeak));
	return ((alpha * G * G) / Math.pow(w, 5)) * Math.exp(-1.25 * Math.pow(omegaPeak / w, 4)) * Math.pow(gamma, r);
}

/**
 * ∫ k^n S(k) dk over a wavenumber band, for the 1D (omnidirectional) spectrum.
 * @param {(w: number) => number} S S(ω)
 * @param {number} k0
 * @param {number} k1
 * @param {number} n
 */
function bandMoment(S, k0, k1, n) {
	const lo = Math.max(k0, 1e-4);
	const hi = Math.min(k1, 2000);
	if (hi <= lo) return 0;
	const steps = 160;
	const a = Math.log(lo);
	const b = Math.log(hi);
	let sum = 0;
	for (let i = 0; i <= steps; i++) {
		const k = Math.exp(a + ((b - a) * i) / steps);
		const w = Math.sqrt(G * k);
		// S(k) dk = S(ω) dω/dk dk, and dk = k d(ln k)
		const v = S(w) * (G / (2 * w)) * Math.pow(k, n) * k;
		sum += v * (i === 0 || i === steps ? 0.5 : 1);
	}
	return (sum * (b - a)) / steps;
}

const EVOLVE = /* glsl */ `
uniform sampler2D uNoise;
uniform int uSize;
uniform float uPatch[4];
uniform float uKMin[4];
uniform float uKMax[4];
uniform vec4 uWind;          // alpha, omegaPeak, gamma, spread exponent s
uniform float uWindDir;      // radians, compass bearing the wind sea travels toward
uniform float uSpreadTaper;  // s falls as (ω/ωp)^-taper above the peak
uniform vec4 uSwell;         // alpha, omegaPeak, gamma, active (0/1)
uniform float uSwellDir;
uniform float uSwellSpread;
uniform float uDamp;         // small-wave suppression length, m
uniform float uTime;         // seconds, wrapped to the loop period
uniform float uOmega0;       // dispersion is quantised to multiples of this
uniform int uSuper;
layout(location = 0) out vec4 out0;
layout(location = 1) out vec4 out1;
layout(location = 2) out vec4 out2;

const float G = 9.81;
const float PI = 3.141592653589793;

vec2 cmul(vec2 a, vec2 b) { return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }

float jonswap(float w, vec3 band) {
	if (band.x <= 0.0 || w <= 0.0) return 0.0;
	float q = band.y / w;
	if (q > 3.5) return 0.0;
	float sigma = w <= band.y ? 0.07 : 0.09;
	float o = w - band.y;
	float r = exp(-(o * o) / (2.0 * sigma * sigma * band.y * band.y));
	float q2 = q * q;
	return band.x * G * G / (w * w * w * w * w) * exp(-1.25 * q2 * q2) * pow(band.z, r);
}

// Normalisation of cos^2s(θ/2) over a full turn: Γ(s+1)/(2√π Γ(s+½)), with
// Γ(s+1)/Γ(s+½) ≈ √(s + ¼ + 1/(32(s+¼))) — within 0.4% of spectrum.js's exact form.
float spreadNorm(float s) {
	return sqrt(s + 0.25 + 1.0 / (32.0 * (s + 0.25))) / (2.0 * sqrt(PI));
}

float spreading(float theta, float mean, float s) {
	float c = abs(cos(0.5 * (theta - mean)));
	return spreadNorm(s) * pow(c, 2.0 * s);
}

// Directional wavenumber spectrum S(k⃗), m^4: ∫∫ S dk_e dk_n = m0.
float density(vec2 k) {
	float kl = length(k);
	if (kl < 1e-6) return 0.0;
	float w = sqrt(G * kl);
	float theta = atan(k.x, k.y);
	float s = uWind.w;
	if (w > uWind.y) s = max(min(s, 2.0), s * pow(w / uWind.y, -uSpreadTaper));
	float S = jonswap(w, uWind.xyz) * spreading(theta, uWindDir, s);
	if (uSwell.w > 0.5) S += jonswap(w, uSwell.xyz) * spreading(theta, uSwellDir, uSwellSpread);
	// S(ω)·dω/dk over k for the polar Jacobian
	return S * (G / (2.0 * w)) / kl;
}

vec2 amplitude(ivec2 idx, int c) {
	int N = uSize;
	if (abs(idx.x) >= N / 2 || abs(idx.y) >= N / 2 || (idx.x == 0 && idx.y == 0)) return vec2(0.0);
	float dk = 2.0 * PI / uPatch[c];
	vec2 k = vec2(idx) * dk;
	float kl = length(k);
	if (kl < uKMin[c] || kl >= uKMax[c]) return vec2(0.0);
	float S = 0.0;
	if (c == 0 && max(abs(idx.x), abs(idx.y)) < uSuper) {
		// Narrow peaks (swell, a strong wind sea) are integrated over the cell so the
		// energy is right however the peak falls between bins.
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				vec2 kk = (vec2(idx) + (vec2(float(i), float(j)) + 0.5) * 0.25 - 0.5) * dk;
				S += density(kk);
			}
		}
		S *= 1.0 / 16.0;
	} else {
		S = density(k);
	}
	S *= exp(-kl * kl * uDamp * uDamp);
	ivec2 t = ivec2((idx.x + N) % N, (idx.y + N) % N + c * N);
	vec2 xi = texelFetch(uNoise, t, 0).xy;
	// E|h0|² = S Δk² / 2, so h(x) has variance m0 once h0(k) and h0(−k) both evolve.
	return xi * (0.5 * sqrt(S) * dk);
}

void main() {
	ivec2 p = ivec2(gl_FragCoord.xy);
	int N = uSize;
	int c = p.y / N;
	ivec2 local = ivec2(p.x, p.y - c * N);
	ivec2 idx = ivec2(local.x < N / 2 ? local.x : local.x - N, local.y < N / 2 ? local.y : local.y - N);
	float dk = 2.0 * PI / uPatch[c];
	vec2 k = vec2(idx) * dk;
	float kl = length(k);

	vec2 h0k = amplitude(idx, c);
	vec2 h0m = amplitude(-idx, c);

	float w = sqrt(G * kl);
	w = max(uOmega0, floor(w / uOmega0 + 0.5) * uOmega0);
	float ph = w * uTime;
	vec2 e = vec2(cos(ph), -sin(ph));                    // e^{-iωt}
	vec2 a = cmul(h0k, e);                               // h0(k) e^{-iωt}
	vec2 b = cmul(vec2(h0m.x, -h0m.y), vec2(e.x, -e.y)); // conj(h0(-k)) e^{iωt}
	vec2 h = a + b;
	vec2 v = cmul(vec2(0.0, -w), a - b);                 // dh/dt

	vec2 kh = kl > 0.0 ? k / kl : vec2(0.0);
	float ik = kl > 0.0 ? 1.0 / kl : 0.0;
	// Two real fields per complex IFFT: c = A + iB with A, B Hermitian.
	vec2 c0 = h + vec2(-v.y, v.x);                       // height + i·vertical velocity
	vec2 c1 = cmul(h, vec2(-kh.y, kh.x));                // D_e + i D_n,   D = i k̂ h
	vec2 c2 = cmul(h, vec2(-k.y, k.x));                  // ∂h/∂e + i ∂h/∂n
	vec2 c3 = -cmul(h, vec2(k.x * k.x, k.y * k.y)) * ik; // ∂D_e/∂e + i ∂D_n/∂n
	vec2 c4 = -h * (k.x * k.y * ik);                     // ∂D_e/∂n
	vec2 c5 = cmul(v, vec2(-kh.y, kh.x));                // u_e + i u_n (orbital velocity)
	out0 = vec4(c0, c1);
	out1 = vec4(c2, c3);
	out2 = vec4(c4, c5);
}`;

const FFT = /* glsl */ `
uniform sampler2D uIn0;
uniform sampler2D uIn1;
uniform sampler2D uIn2;
uniform sampler2D uButterfly; // per (output, stage): twiddle cos, sin, first input, radix
uniform int uSize;
uniform int uStage;
uniform int uHorizontal;
layout(location = 0) out vec4 out0;
layout(location = 1) out vec4 out1;
layout(location = 2) out vec4 out2;

vec2 cmul(vec2 a, vec2 b) { return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }

// Two complex values packed in a vec4, each times its own twiddle... here the same one.
vec4 tw(vec4 v, vec2 w) { return vec4(cmul(w, v.xy), cmul(w, v.zw)); }

void main() {
	ivec2 p = ivec2(gl_FragCoord.xy);
	int N = uSize;
	bool horizontal = uHorizontal == 1;
	int idx = horizontal ? p.x : p.y % N;
	int base = horizontal ? 0 : p.y - idx;
	// Mixed radix-2/4 Stockham autosort, inverse transform: output o of a pass of radix R
	// over sub-transforms of size p sums inputs i + m·N/R (m < R), i = ⌊o/Rp⌋·p + o mod p,
	// with twiddles ω^m, ω = e^{+2πi·(o mod Rp)/(Rp)}.
	vec4 bf = texelFetch(uButterfly, ivec2(idx, uStage), 0);
	vec2 w1 = bf.xy;
	int first = int(bf.z + 0.5);
	int radix = int(bf.w + 0.5);
	int step = N / radix;
	ivec2 dir = horizontal ? ivec2(1, 0) : ivec2(0, 1);
	ivec2 p0 = horizontal ? ivec2(first, p.y) : ivec2(p.x, base + first);
	vec4 a0 = texelFetch(uIn0, p0, 0);
	vec4 b0 = texelFetch(uIn1, p0, 0);
	vec4 c0 = texelFetch(uIn2, p0, 0);
	vec4 a1 = texelFetch(uIn0, p0 + dir * step, 0);
	vec4 b1 = texelFetch(uIn1, p0 + dir * step, 0);
	vec4 c1 = texelFetch(uIn2, p0 + dir * step, 0);
	vec4 A = a0 + tw(a1, w1);
	vec4 B = b0 + tw(b1, w1);
	vec4 C = c0 + tw(c1, w1);
	if (radix == 4) {
		vec2 w2 = cmul(w1, w1);
		vec2 w3 = cmul(w2, w1);
		ivec2 p2 = p0 + dir * (2 * step);
		ivec2 p3 = p0 + dir * (3 * step);
		A += tw(texelFetch(uIn0, p2, 0), w2) + tw(texelFetch(uIn0, p3, 0), w3);
		B += tw(texelFetch(uIn1, p2, 0), w2) + tw(texelFetch(uIn1, p3, 0), w3);
		C += tw(texelFetch(uIn2, p2, 0), w2) + tw(texelFetch(uIn2, p3, 0), w3);
	}
	out0 = A;
	out1 = B;
	out2 = C;
}`;

const ASSEMBLE = /* glsl */ `
uniform sampler2D uFft0;
uniform sampler2D uFft1;
uniform sampler2D uFft2;
uniform int uCascade;
uniform int uSize;
uniform float uChop;
layout(location = 0) out vec4 outDisp;
layout(location = 1) out vec4 outSlope;
layout(location = 2) out vec4 outMotion;
layout(location = 3) out vec4 outJac;

void main() {
	ivec2 p = ivec2(gl_FragCoord.xy);
	ivec2 q = ivec2(p.x, p.y + uCascade * uSize);
	vec4 f0 = texelFetch(uFft0, q, 0); // h, v, D_e, D_n
	vec4 f1 = texelFetch(uFft1, q, 0); // S_e, S_n, D_ee, D_nn
	vec4 f2 = texelFetch(uFft2, q, 0); // D_en, 0, u_e, u_n
	float jee = 1.0 + uChop * f1.z;
	float jnn = 1.0 + uChop * f1.w;
	float jen = uChop * f2.x;
	// Slopes of the displaced surface: steeper where the choppy displacement bunches crests.
	vec2 slope = f1.xy / max(vec2(jee, jnn), vec2(0.3));
	outDisp = vec4(uChop * f0.z, f0.x, uChop * f0.w, f0.y);
	outSlope = vec4(slope, slope * slope);
	outMotion = vec4(f2.z, f2.w, 0.0, 0.0);
	// This cascade's share of the Jacobian, so every cascade can fold into one J, and the
	// slope cross moment for the roughness covariance.
	outJac = vec4(uChop * f1.z, uChop * f1.w, jen, slope.x * slope.y);
}`;

/**
 * @typedef {object} CascadeTextures
 * @property {THREE.Texture} disp (λD_e, h, λD_n, ∂h/∂t)
 * @property {THREE.Texture} slope (s_e, s_n, s_e², s_n²)
 * @property {THREE.Texture} motion (u_e, u_n, 0, 0): horizontal orbital velocity
 * @property {THREE.Texture} jac (λ∂D_e/∂e, λ∂D_n/∂n, λ∂D_e/∂n, s_e·s_n): the cascade's Jacobian
 *   terms and the slope cross moment
 */

/**
 * Radices of an N-point transform: radix 4 throughout, with one radix-2 pass first when
 * log2 N is odd.
 * @param {number} N a power of two
 */
export function fftPlan(N) {
	/** @type {number[]} */
	const radices = [];
	let n = Math.round(Math.log2(N));
	if (n % 2 === 1) {
		radices.push(2);
		n -= 1;
	}
	for (; n > 0; n -= 2) radices.push(4);
	return radices;
}

/**
 * The mixed radix-2/4 Stockham butterflies of an N-point inverse transform, one row per
 * pass: for output o, the twiddle ω = e^{+2πi·(o mod Rp)/(Rp)} and the first of its R
 * inputs, ⌊o/Rp⌋·p + o mod p (the others follow every N/R), and R.
 * @param {number} N
 */
function butterflyTexture(N) {
	const radices = fftPlan(N);
	const data = new Float32Array(N * radices.length * 4);
	let p = 1;
	radices.forEach((R, stage) => {
		for (let o = 0; o < N; o++) {
			const ang = (2 * Math.PI * (o % (R * p))) / (R * p);
			const first = Math.floor(o / (R * p)) * p + (o % p);
			const at = (stage * N + o) * 4;
			data[at] = Math.cos(ang);
			data[at + 1] = Math.sin(ang);
			data[at + 2] = first;
			data[at + 3] = R;
		}
		p *= R;
	});
	const texture = new THREE.DataTexture(data, N, radices.length, THREE.RGBAFormat, THREE.FloatType);
	texture.minFilter = THREE.NearestFilter;
	texture.magFilter = THREE.NearestFilter;
	texture.needsUpdate = true;
	return texture;
}

export class WaveSimulation {
	/**
	 * @param {THREE.WebGLRenderer} renderer
	 * @param {{ size?: number, cascades?: number, seed?: number }} [options]
	 */
	constructor(renderer, options = {}) {
		this.renderer = renderer;
		this.size = options.size ?? 256;
		this.count = options.cascades ?? 3;
		this.seed = options.seed ?? 20240611;
		const layout = cascadeLayout(this.count, this.size);
		this.patches = layout.patches;
		this.bands = layout.bands;
		this.nyquist = layout.nyquist;
		this.passCount = fftPlan(this.size).length;

		const N = this.size;
		const C = this.count;

		// The seeded Gaussian field. Amplitudes are scaled per frame from the sea state,
		// so changing the wind never re-rolls the phases.
		this.noise = new Float32Array(N * N * C * 2);
		const random = gaussian(mulberry32(this.seed));
		for (let i = 0; i < this.noise.length; i++) this.noise[i] = random();
		this.noiseTexture = new THREE.DataTexture(this.noise, N, N * C, THREE.RGFormat, THREE.FloatType);
		this.noiseTexture.minFilter = THREE.NearestFilter;
		this.noiseTexture.magFilter = THREE.NearestFilter;
		this.noiseTexture.needsUpdate = true;

		this.fftTargets = [
			floatTarget(N, N * C, { count: 3 }),
			floatTarget(N, N * C, { count: 3 })
		];

		/** One four-target output per cascade (see CascadeTextures). */
		/** @type {THREE.WebGLRenderTarget[]} */
		this.outputs = [];
		for (let c = 0; c < C; c++) {
			this.outputs.push(
				new THREE.WebGLRenderTarget(N, N, {
					count: 4,
					type: THREE.HalfFloatType,
					format: THREE.RGBAFormat,
					minFilter: THREE.LinearMipmapLinearFilter,
					magFilter: THREE.LinearFilter,
					wrapS: THREE.RepeatWrapping,
					wrapT: THREE.RepeatWrapping,
					generateMipmaps: true,
					anisotropy: 4,
					depthBuffer: false,
					stencilBuffer: false
				})
			);
		}

		const pad4 = (/** @type {number[]} */ a, /** @type {number} */ fill) => [...a, ...Array(4 - a.length).fill(fill)];
		this.evolve = new FullscreenPass(EVOLVE, {
			uNoise: { value: this.noiseTexture },
			uSize: { value: N },
			uPatch: { value: pad4(this.patches, 1) },
			uKMin: { value: pad4(this.bands.map((b) => b[0]), 0) },
			uKMax: { value: pad4(this.bands.map((b) => Math.min(b[1], 1e9)), 0) },
			uWind: { value: new THREE.Vector4() },
			uWindDir: { value: 0 },
			uSpreadTaper: { value: 1.2 },
			uSwell: { value: new THREE.Vector4() },
			uSwellDir: { value: 0 },
			uSwellSpread: { value: 28 },
			uDamp: { value: 0 },
			uTime: { value: 0 },
			uOmega0: { value: OMEGA_0 },
			uSuper: { value: SUPERSAMPLE_RADIUS }
		});
		this.butterfly = butterflyTexture(N);
		this.fft = new FullscreenPass(FFT, {
			uIn0: { value: null },
			uIn1: { value: null },
			uIn2: { value: null },
			uButterfly: { value: this.butterfly },
			uSize: { value: N },
			uStage: { value: 0 },
			uHorizontal: { value: 1 }
		});
		this.assemble = new FullscreenPass(ASSEMBLE, {
			uFft0: { value: null },
			uFft1: { value: null },
			uFft2: { value: null },
			uCascade: { value: 0 },
			uSize: { value: N },
			uChop: { value: 1 }
		});

		/** Per cascade: mean-square slope (both axes), m0 and orbital speed variance of its band. */
		this.bandStats = this.bands.map(() => ({ mss: 0, m0: 0, speedVariance: 0 }));
		/** Mean-square slope of waves finer than the finest cascade resolves. */
		this.tailSlopeVariance = 0;
		this.choppiness = 1;
		/** @type {string} */
		this.lastKey = '';
		/** Unit vector (east, north) the wind sea runs toward. */
		this.windDir = new THREE.Vector2(0, 1);
		/** m/s (east, north) foam drifts with the surface. */
		this.foamDrift = new THREE.Vector2();
		this.windSpeed = 0;
		/** 0–1: no whitecaps in light airs. */
		this.foamEnabled = 0;
		this.foamLifetime = FOAM_LIFETIME;
		/**
		 * Where crests break: the Jacobian of the cascades together, weighted by
		 * JACOBIAN_WEIGHTS, below `bias` turns to foam, completely over 1/`gain`.
		 */
		this.activeFoam = { bias: -1, gain: 1 };
	}

	/**
	 * Set the sea state. Cheap when nothing changed; otherwise recomputes the band
	 * parameters (a few hundred JONSWAP evaluations on the CPU).
	 * @param {ResolvedSea} sea
	 */
	setSea(sea) {
		const key = [
			sea.windSpeed,
			sea.windDirection,
			sea.fetch,
			sea.spread,
			sea.swellHeight,
			sea.swellDirection,
			sea.swellPeriod,
			sea.choppiness
		]
			.map((v) => v.toFixed(4))
			.join(',');
		if (key === this.lastKey) return;
		this.lastKey = key;

		const wind = windSeaParams(sea.windSpeed, sea.fetch);
		const swell = swellParams(sea.swellHeight, sea.swellPeriod);
		const u = this.evolve.uniforms;
		/** @type {THREE.Vector4} */ (u.uWind.value).set(wind.alpha, wind.omegaPeak, wind.gamma, Math.max(0.5, sea.spread));
		u.uWindDir.value = sea.windDirection * DEG;
		/** @type {THREE.Vector4} */ (u.uSwell.value).set(
			swell ? swell.alpha : 0,
			swell ? swell.omegaPeak : 1,
			swell ? swell.gamma : 1,
			swell ? 1 : 0
		);
		u.uSwellDir.value = sea.swellDirection * DEG;
		// Small-wave suppression: fade the last half-octave below the finest Nyquist.
		u.uDamp.value = 0.9 / this.nyquist;
		this.choppiness = sea.choppiness;
		this.windSpeed = sea.windSpeed;

		/** @param {number} w */
		const S = (w) => jonswap(w, wind) + (swell ? jonswap(w, swell) : 0);
		this.bandStats = this.bands.map(([k0, k1]) => ({
			mss: bandMoment(S, k0, Math.min(k1, this.nyquist), 2),
			m0: bandMoment(S, k0, Math.min(k1, this.nyquist), 0),
			// Orbital speed variance at the surface, horizontal plus vertical: 2·∫ω²S = 2g·∫k·S.
			speedVariance: 2 * G * bandMoment(S, k0, Math.min(k1, this.nyquist), 1)
		}));
		// Slopes of waves too short for the finest cascade, down to capillary scales.
		this.tailSlopeVariance = bandMoment(S, this.nyquist * 0.7, 400, 2);

		// Whitecaps: the weighted Jacobian folds below a threshold set in its standard
		// deviations (σ_J ≈ λ·√(Σ w²·mss)), lower as the wind rises. None below ~4 m/s,
		// plenty by force 7.
		const sigmas = breakThreshold(sea.windSpeed);
		this.foamEnabled = smoothstep(3.6, 5.5, sea.windSpeed);
		const weighted = this.bandStats.reduce((s, b, c) => s + JACOBIAN_WEIGHTS[c] ** 2 * b.mss, 0);
		const sigmaJ = Math.max(0.02, sea.choppiness * Math.sqrt(Math.max(weighted, 1e-6)));
		this.activeFoam = {
			bias: this.foamEnabled > 0 ? 1 - sigmas * sigmaJ : -1,
			gain: 1.3 / sigmaJ
		};
		const driftSpeed = 0.035 * sea.windSpeed;
		const dir = sea.windDirection * DEG;
		this.windDir.set(Math.sin(dir), Math.cos(dir));
		this.foamDrift.set(Math.sin(dir) * driftSpeed, Math.cos(dir) * driftSpeed);
	}

	/**
	 * Along-wind diffusion step for foam: a 1-2-1 filter of tap offset d adds d²/2 of
	 * variance, so d = √(4·D·dt) spreads foam with diffusivity D (m²/s) along the wind.
	 * @param {number} dt seconds
	 * @returns {number} tap offset, m
	 */
	foamSpread(dt) {
		return Math.sqrt(4 * FOAM_DIFFUSIVITY * (0.4 + 0.06 * this.windSpeed) * Math.max(dt, 0));
	}

	/**
	 * Advance the sea to `time` and rebuild the cascade textures.
	 * @param {number} time seconds of simulation time
	 */
	update(time) {
		const renderer = this.renderer;
		const previousTarget = renderer.getRenderTarget();

		this.evolve.uniforms.uTime.value = time % LOOP_SECONDS;
		this.evolve.render(renderer, this.fftTargets[0]);

		let src = 0;
		const fu = this.fft.uniforms;
		for (const horizontal of [1, 0]) {
			fu.uHorizontal.value = horizontal;
			for (let stage = 0; stage < this.passCount; stage++) {
				const from = this.fftTargets[src];
				fu.uIn0.value = from.textures[0];
				fu.uIn1.value = from.textures[1];
				fu.uIn2.value = from.textures[2];
				fu.uStage.value = stage;
				this.fft.render(renderer, this.fftTargets[1 - src]);
				src = 1 - src;
			}
		}

		const result = this.fftTargets[src];
		const au = this.assemble.uniforms;
		au.uFft0.value = result.textures[0];
		au.uFft1.value = result.textures[1];
		au.uFft2.value = result.textures[2];
		au.uChop.value = this.choppiness;
		for (let c = 0; c < this.count; c++) {
			au.uCascade.value = c;
			this.assemble.render(renderer, this.outputs[c]);
		}
		renderer.setRenderTarget(previousTarget);
	}

	/**
	 * The textures of a cascade as of the last update.
	 * @param {number} c
	 * @returns {CascadeTextures}
	 */
	textures(c) {
		const t = this.outputs[c].textures;
		return { disp: t[0], slope: t[1], motion: t[2], jac: t[3] };
	}

	/**
	 * Texture-space offset of each cascade for a floating origin at (east, north): the
	 * fractional patch count, computed in double precision so the shaders never see
	 * large coordinates.
	 * @param {number} east
	 * @param {number} north
	 * @param {THREE.Vector2[]} out
	 */
	offsets(east, north, out) {
		for (let c = 0; c < this.count; c++) {
			const L = this.patches[c];
			const fx = east / L - Math.floor(east / L);
			const fy = north / L - Math.floor(north / L);
			out[c].set(fx, fy);
		}
		return out;
	}

	/** Total mean-square slope the simulation resolves, plus the unresolved tail. */
	get meanSquareSlope() {
		return this.bandStats.reduce((s, b) => s + b.mss, 0) + this.tailSlopeVariance;
	}

	dispose() {
		this.noiseTexture.dispose();
		this.butterfly.dispose();
		for (const t of this.fftTargets) t.dispose();
		for (const target of this.outputs) target.dispose();
		this.evolve.dispose();
		this.fft.dispose();
		this.assemble.dispose();
	}
}

