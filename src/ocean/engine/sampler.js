/**
 * The same sea, on the CPU: the strongest long-wave components of the GPU simulation,
 * evaluated at a point. It takes the seeded Gaussian field and the spectrum formulas
 * the GPU uses (same cell integration, same quantised dispersion, same time), keeps
 * the few dozen pairs ±k with the most energy, and sums them — enough for a free
 * camera riding the sea (its heave, pitch and roll), the camera's height above the
 * water, and keeping a chase camera clear of the crests.
 */
import { G, swellParams, windSeaParams } from '../spectrum.js';
import { LOOP_SECONDS, OMEGA_0 } from './waves.js';
import { DEG } from './util.js';

const RADIUS = 18;
const KEEP = 56;

/**
 * @param {number} w
 * @param {{ alpha: number, omegaPeak: number, gamma: number }} band
 */
function jonswap(w, { alpha, omegaPeak, gamma }) {
	if (alpha <= 0 || w <= 0) return 0;
	const q = omegaPeak / w;
	if (q > 3.5) return 0;
	const sigma = w <= omegaPeak ? 0.07 : 0.09;
	const o = w - omegaPeak;
	const r = Math.exp(-(o * o) / (2 * sigma * sigma * omegaPeak * omegaPeak));
	return ((alpha * G * G) / w ** 5) * Math.exp(-1.25 * q ** 4) * Math.pow(gamma, r);
}

/** @param {number} s */
const spreadNorm = (s) => Math.sqrt(s + 0.25 + 1 / (32 * (s + 0.25))) / (2 * Math.sqrt(Math.PI));

export class SeaSampler {
	/**
	 * @param {Float32Array} noise the simulation's Gaussian field (RG per bin, cascades stacked)
	 * @param {number} size FFT size
	 * @param {number} patch cascade 0 patch size, m
	 */
	constructor(noise, size, patch) {
		this.noise = noise;
		this.size = size;
		this.patch = patch;
		/** Component table: k_e, k_n, ω, then h0(k) re/im and h0(−k) re/im. */
		this.table = new Float64Array(KEEP * 7);
		this.count = 0;
		this.key = '';
		this.lastUpdate = -Infinity;
	}

	/**
	 * Rebuild the component table for a sea state (throttled; cheap to call per frame).
	 * @param {import('./waves.js').ResolvedSea} sea
	 * @param {number} now seconds
	 * @param {number} [minInterval] seconds between rebuilds while the sea is changing; the
	 *   manual engine passes 0, so the table always matches the sea being rendered
	 */
	update(sea, now, minInterval = 0.25) {
		const key = [sea.windSpeed, sea.windDirection, sea.fetch, sea.spread, sea.swellHeight, sea.swellDirection, sea.swellPeriod]
			.map((v) => v.toFixed(3))
			.join(',');
		if (key === this.key || (now - this.lastUpdate < minInterval && this.count > 0)) return;
		this.key = key;
		this.lastUpdate = now;
		const wind = windSeaParams(sea.windSpeed, sea.fetch);
		const swell = swellParams(sea.swellHeight, sea.swellPeriod);
		const windDir = sea.windDirection * DEG;
		const swellDir = sea.swellDirection * DEG;
		const sBase = Math.max(0.5, sea.spread);
		const taper = 1.2;
		const swellSpread = 28;
		/** @param {number} ke @param {number} kn */
		const density = (ke, kn) => {
			const kl = Math.hypot(ke, kn);
			if (kl < 1e-6) return 0;
			const w = Math.sqrt(G * kl);
			const theta = Math.atan2(ke, kn);
			let s = sBase;
			if (w > wind.omegaPeak) s = Math.max(Math.min(s, 2), s * Math.pow(w / wind.omegaPeak, -taper));
			let S = jonswap(w, wind) * spreadNorm(s) * Math.pow(Math.abs(Math.cos(0.5 * (theta - windDir))), 2 * s);
			if (swell) S += jonswap(w, swell) * spreadNorm(swellSpread) * Math.pow(Math.abs(Math.cos(0.5 * (theta - swellDir))), 2 * swellSpread);
			return (S * (G / (2 * w))) / kl;
		};
		const N = this.size;
		const dk = (2 * Math.PI) / this.patch;
		/** @param {number} ix @param {number} iy */
		const amplitude = (ix, iy) => {
			let S = 0;
			// The GPU integrates these bins over 4×4 points; 3×3 is close enough here.
			for (let a = 0; a < 3; a++) {
				for (let b = 0; b < 3; b++) S += density((ix + (a + 0.5) / 3 - 0.5) * dk, (iy + (b + 0.5) / 3 - 0.5) * dk);
			}
			S /= 9;
			const t = (((iy + N) % N) * N + ((ix + N) % N)) * 2;
			const amp = 0.5 * Math.sqrt(S) * dk;
			return [this.noise[t] * amp, this.noise[t + 1] * amp];
		};
		/** @type {Array<[number, number, number, number, number, number, number, number]>} */
		const candidates = [];
		for (let iy = 0; iy <= RADIUS; iy++) {
			for (let ix = -RADIUS; ix <= RADIUS; ix++) {
				if (iy === 0 && ix <= 0) continue;
				const [ar, ai] = amplitude(ix, iy);
				const [br, bi] = amplitude(-ix, -iy);
				const energy = ar * ar + ai * ai + br * br + bi * bi;
				const ke = ix * dk;
				const kn = iy * dk;
				let w = Math.sqrt(G * Math.hypot(ke, kn));
				w = Math.max(OMEGA_0, Math.round(w / OMEGA_0) * OMEGA_0);
				candidates.push([energy, ke, kn, w, ar, ai, br, bi]);
			}
		}
		candidates.sort((p, q) => q[0] - p[0]);
		this.count = Math.min(KEEP, candidates.length);
		for (let i = 0; i < this.count; i++) {
			const c = candidates[i];
			for (let j = 0; j < 7; j++) this.table[i * 7 + j] = c[j + 1];
		}
	}

	/**
	 * Height and slopes of the sea at (east, north), absolute metres, at time t.
	 * @param {number} east
	 * @param {number} north
	 * @param {number} time seconds of simulation time
	 */
	sample(east, north, time) {
		const t = time % LOOP_SECONDS;
		let h = 0;
		let se = 0;
		let sn = 0;
		const T = this.table;
		for (let i = 0; i < this.count; i++) {
			const o = i * 7;
			const ke = T[o];
			const kn = T[o + 1];
			const w = T[o + 2];
			// h̃ = h0(k) e^{−iωt} + conj(h0(−k)) e^{iωt}
			const c = Math.cos(w * t);
			const s = Math.sin(w * t);
			const hr = T[o + 3] * c + T[o + 4] * s + T[o + 5] * c + T[o + 6] * s;
			const hi = T[o + 4] * c - T[o + 3] * s - T[o + 6] * c + T[o + 5] * s;
			// 2·Re(h̃ e^{ik·x}) and its gradient
			const ph = ke * east + kn * north;
			const pc = Math.cos(ph);
			const ps = Math.sin(ph);
			const re = hr * pc - hi * ps;
			const im = hr * ps + hi * pc;
			h += 2 * re;
			se += -2 * ke * im;
			sn += -2 * kn * im;
		}
		return { height: h, slopeE: se, slopeN: sn };
	}

	/**
	 * How a small float riding the sea at a point would move, for a free camera's sway.
	 * @param {number} east
	 * @param {number} north
	 * @param {number} time
	 * @param {number} heading compass degrees the float points toward
	 */
	motion(east, north, time, heading) {
		const { height, slopeE, slopeN } = this.sample(east, north, time);
		const h = heading * DEG;
		const along = slopeE * Math.sin(h) + slopeN * Math.cos(h);
		const across = slopeE * Math.cos(h) - slopeN * Math.sin(h);
		return { heave: height, pitch: Math.atan(along) / DEG, roll: Math.atan(across) / DEG };
	}
}
