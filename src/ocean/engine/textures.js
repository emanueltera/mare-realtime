/**
 * Procedural textures generated once at start-up, deterministically: a tiling
 * multi-octave noise for the cloud deck, and a foam texture (bubble cells and the
 * lace between them) that breaks whitecap coverage into something that reads as foam.
 */
import * as THREE from 'three';
import { mulberry32 } from './util.js';

/**
 * Tiling gradient-free value noise with a smooth quintic fade.
 * @param {number} size texture size
 * @param {number} period lattice cells across the tile
 * @param {() => number} random
 */
function valueNoise(size, period, random) {
	const lattice = new Float32Array(period * period);
	for (let i = 0; i < lattice.length; i++) lattice[i] = random();
	const out = new Float32Array(size * size);
	/** @param {number} t */
	const fade = (t) => t * t * t * (t * (t * 6 - 15) + 10);
	for (let y = 0; y < size; y++) {
		const fy = (y / size) * period;
		const y0 = Math.floor(fy);
		const ty = fade(fy - y0);
		const ya = y0 % period;
		const yb = (y0 + 1) % period;
		for (let x = 0; x < size; x++) {
			const fx = (x / size) * period;
			const x0 = Math.floor(fx);
			const tx = fade(fx - x0);
			const xa = x0 % period;
			const xb = (x0 + 1) % period;
			const a = lattice[ya * period + xa];
			const b = lattice[ya * period + xb];
			const c = lattice[yb * period + xa];
			const d = lattice[yb * period + xb];
			out[y * size + x] = a + (b - a) * tx + (c - a) * ty + (a - b - c + d) * tx * ty;
		}
	}
	return out;
}

/**
 * Four channels of tiling fBm at different base frequencies, for clouds.
 * @param {number} [size]
 */
export function createNoiseTexture(size = 256) {
	const random = mulberry32(911);
	const channels = [4, 8, 16, 32].map((base) => {
		const acc = new Float32Array(size * size);
		let amp = 0.5;
		let norm = 0;
		for (let o = 0; o < 4; o++) {
			const layer = valueNoise(size, base << o, random);
			for (let i = 0; i < acc.length; i++) acc[i] += layer[i] * amp;
			norm += amp;
			amp *= 0.5;
		}
		for (let i = 0; i < acc.length; i++) acc[i] /= norm;
		return acc;
	});
	const data = new Uint8Array(size * size * 4);
	for (let i = 0; i < size * size; i++) {
		for (let c = 0; c < 4; c++) data[i * 4 + c] = Math.round(Math.min(1, Math.max(0, channels[c][i])) * 255);
	}
	const texture = new THREE.DataTexture(data, size, size, THREE.RGBAFormat, THREE.UnsignedByteType);
	texture.wrapS = THREE.RepeatWrapping;
	texture.wrapT = THREE.RepeatWrapping;
	texture.minFilter = THREE.LinearMipmapLinearFilter;
	texture.magFilter = THREE.LinearFilter;
	texture.generateMipmaps = true;
	texture.needsUpdate = true;
	return texture;
}

/**
 * Foam detail. R and B are two independent foam laces, histogram-equalised (see
 * `lace` below); G is a soft large-scale break-up; A is value noise with 64 cells
 * across, which foam.js's break gate reads.
 * @param {number} [size]
 */
export function createFoamTexture(size = 256) {
	const random = mulberry32(4242);
	const n = size * size;

	/**
	 * Tiling Worley F1/F2 for about `cells` points, looked up through a periodic domain
	 * warp so the cells come out irregular rather than honeycomb.
	 * @param {number} cells
	 * @param {Float32Array} warpU
	 * @param {Float32Array} warpV
	 * @param {number} warp warp amplitude in cells
	 */
	const worley = (cells, warpU, warpV, warp) => {
		const grid = Math.max(2, Math.round(Math.sqrt(cells)));
		/** @type {Array<[number, number]>} */
		const points = [];
		for (let gy = 0; gy < grid; gy++) {
			for (let gx = 0; gx < grid; gx++) points.push([(gx + random()) / grid, (gy + random()) / grid]);
		}
		const f1 = new Float32Array(n);
		const f2 = new Float32Array(n);
		for (let y = 0; y < size; y++) {
			for (let x = 0; x < size; x++) {
				const i = y * size + x;
				let u = (x + 0.5) / size + ((warpU[i] - 0.5) * warp) / grid;
				let v = (y + 0.5) / size + ((warpV[i] - 0.5) * warp) / grid;
				u -= Math.floor(u);
				v -= Math.floor(v);
				let d1 = 81;
				let d2 = 81;
				const cx = Math.floor(u * grid);
				const cy = Math.floor(v * grid);
				for (let oy = -2; oy <= 2; oy++) {
					const gy = (((cy + oy) % grid) + grid) % grid;
					for (let ox = -2; ox <= 2; ox++) {
						const gx = (((cx + ox) % grid) + grid) % grid;
						const p = points[gy * grid + gx];
						let dx = p[0] - u;
						let dy = p[1] - v;
						dx -= Math.round(dx);
						dy -= Math.round(dy);
						const d = (dx * dx + dy * dy) * grid * grid;
						if (d < d1) {
							d2 = d1;
							d1 = d;
						} else if (d < d2) d2 = d;
					}
				}
				f1[i] = Math.sqrt(d1);
				f2[i] = Math.sqrt(d2);
			}
		}
		return { f1, f2 };
	};

	/** @param {{ f1: Float32Array, f2: Float32Array }} w @param {number} i @param {number} width */
	const rim = (w, i, width) => Math.exp(-((w.f2[i] - w.f1[i]) ** 2) / width);

	/**
	 * Foam lace: a network of bright filaments round dark holes at three scales, like the
	 * foam a whitecap leaves behind. Histogram-equalised to a uniform distribution, so
	 * thresholding it at 1 − c leaves exactly a fraction c of it white: the shader can
	 * turn a foam coverage into lace whose mean is that coverage, near or far.
	 * @param {[number, number, number]} cells coarse, medium, fine cell counts per tile
	 */
	const lace = (cells) => {
		const warpU = valueNoise(size, 8, random);
		const warpV = valueNoise(size, 8, random);
		const coarse = worley(cells[0], warpU, warpV, 1.6);
		const medium = worley(cells[1], warpU, warpV, 2.4);
		const fine = worley(cells[2], warpU, warpV, 3);
		const drift = valueNoise(size, 16, random);
		const raw = new Float32Array(n);
		for (let i = 0; i < n; i++) {
			raw[i] = 0.5 * rim(coarse, i, 0.035) + 0.3 * rim(medium, i, 0.05) + 0.12 * rim(fine, i, 0.08) + 0.08 * drift[i];
		}
		// Equalise through a fine histogram: plenty for an 8-bit texture, and linear time.
		let lo = Infinity;
		let hi = -Infinity;
		for (let i = 0; i < n; i++) {
			lo = Math.min(lo, raw[i]);
			hi = Math.max(hi, raw[i]);
		}
		const bins = 8192;
		const scale = (bins - 1) / Math.max(hi - lo, 1e-9);
		const cdf = new Float32Array(bins);
		for (let i = 0; i < n; i++) cdf[Math.round((raw[i] - lo) * scale)]++;
		let running = 0;
		for (let b = 0; b < bins; b++) {
			const count = cdf[b];
			cdf[b] = (running + count * 0.5) / n;
			running += count;
		}
		const equalised = new Float32Array(n);
		for (let i = 0; i < n; i++) equalised[i] = cdf[Math.round((raw[i] - lo) * scale)];
		return equalised;
	};

	const laceA = lace([36, 144, 900]);
	const laceB = lace([49, 196, 1024]);
	const soft = valueNoise(size, 8, random);
	const speck = valueNoise(size, 64, random);
	const data = new Uint8Array(n * 4);
	for (let i = 0; i < n; i++) {
		data[i * 4] = Math.round(laceA[i] * 255);
		data[i * 4 + 1] = Math.round(soft[i] * 255);
		data[i * 4 + 2] = Math.round(laceB[i] * 255);
		data[i * 4 + 3] = Math.round(speck[i] * 255);
	}
	const texture = new THREE.DataTexture(data, size, size, THREE.RGBAFormat, THREE.UnsignedByteType);
	texture.wrapS = THREE.RepeatWrapping;
	texture.wrapT = THREE.RepeatWrapping;
	texture.minFilter = THREE.LinearMipmapLinearFilter;
	texture.magFilter = THREE.LinearFilter;
	texture.generateMipmaps = true;
	texture.anisotropy = 4;
	texture.needsUpdate = true;
	return texture;
}
