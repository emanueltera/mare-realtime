/**
 * Glare: the soft halo a camera lens throws round anything very bright, the sun's disc,
 * its glitter on the water, a sunlit airframe. A mip chain of the frame's linear
 * radiance (the sea and sky with the aircraft composited over them) is filtered down
 * and back up (after Jimenez 2014: a Karis-weighted 4×4 box to quarter resolution, so
 * single bright pixels don't flicker, 13-tap downsamples and 3×3 tent upsamples), and the post pass
 * adds a small fraction of the result before tone mapping. Physically it is the lens's
 * point spread function; in practice it is what makes the sun read as blinding.
 */
import * as THREE from 'three';
import { FullscreenPass } from './gpu.js';

const LEVELS = 5;

// The first step: the full-resolution frame (sea and sky with the aircraft over it) to
// quarter resolution, as the average of its 4×4 block through four bilinear taps, so
// every pixel counts exactly once. Each tap is weighted by 1/(1 + luma) (Karis), so a
// single blazing glint cannot flicker the glare.
const FIRST = /* glsl */ `
uniform sampler2D uSrc;
uniform sampler2D uAir;
uniform vec2 uTexel;     // source texel size
in vec2 vUv;
out vec4 fragColor;

vec3 fetch(vec2 uv) {
	vec4 a = texture(uAir, uv);
	vec3 c = texture(uSrc, uv).rgb * (1.0 - a.a) + a.rgb;
	// Guard against NaNs and extreme values before they spread through the chain.
	return min(max(c, vec3(0.0)), vec3(6.0e4));
}

void main() {
	vec2 t = uTexel;
	vec3 a = fetch(vUv + t * vec2(-1.0, -1.0));
	vec3 b = fetch(vUv + t * vec2(1.0, -1.0));
	vec3 c = fetch(vUv + t * vec2(-1.0, 1.0));
	vec3 d = fetch(vUv + t * vec2(1.0, 1.0));
	vec4 w = 1.0 / (1.0 + vec4(dot(a, vec3(0.2126, 0.7152, 0.0722)), dot(b, vec3(0.2126, 0.7152, 0.0722)),
		dot(c, vec3(0.2126, 0.7152, 0.0722)), dot(d, vec3(0.2126, 0.7152, 0.0722))));
	fragColor = vec4((a * w.x + b * w.y + c * w.z + d * w.w) / (w.x + w.y + w.z + w.w), 1.0);
}`;

const DOWN = /* glsl */ `
uniform sampler2D uSrc;
uniform vec2 uTexel;     // source texel size
in vec2 vUv;
out vec4 fragColor;

vec3 fetch(vec2 uv) {
	return texture(uSrc, uv).rgb;
}

void main() {
	vec2 t = uTexel;
	vec3 a = fetch(vUv + t * vec2(-2.0, 2.0));
	vec3 b = fetch(vUv + t * vec2(0.0, 2.0));
	vec3 c = fetch(vUv + t * vec2(2.0, 2.0));
	vec3 d = fetch(vUv + t * vec2(-2.0, 0.0));
	vec3 e = fetch(vUv);
	vec3 f = fetch(vUv + t * vec2(2.0, 0.0));
	vec3 g = fetch(vUv + t * vec2(-2.0, -2.0));
	vec3 h = fetch(vUv + t * vec2(0.0, -2.0));
	vec3 i = fetch(vUv + t * vec2(2.0, -2.0));
	vec3 j = fetch(vUv + t * vec2(-1.0, 1.0));
	vec3 k = fetch(vUv + t * vec2(1.0, 1.0));
	vec3 l = fetch(vUv + t * vec2(-1.0, -1.0));
	vec3 m = fetch(vUv + t * vec2(1.0, -1.0));
	vec3 g0 = (j + k + l + m) * 0.25;
	vec3 g1 = (a + b + d + e) * 0.25;
	vec3 g2 = (b + c + e + f) * 0.25;
	vec3 g3 = (d + e + g + h) * 0.25;
	vec3 g4 = (e + f + h + i) * 0.25;
	fragColor = vec4(g0 * 0.5 + (g1 + g2 + g3 + g4) * 0.125, 1.0);
}`;

const UP = /* glsl */ `
uniform sampler2D uLow;    // the coarser level, already accumulated
uniform sampler2D uHigh;   // this level's downsample
uniform vec2 uTexel;       // texel size of uLow
uniform float uSpread;     // how much of the wider glare to carry up
in vec2 vUv;
out vec4 fragColor;
void main() {
	vec2 t = uTexel;
	vec3 s = texture(uLow, vUv + t * vec2(-1.0, 1.0)).rgb
		+ texture(uLow, vUv + t * vec2(0.0, 1.0)).rgb * 2.0
		+ texture(uLow, vUv + t * vec2(1.0, 1.0)).rgb
		+ texture(uLow, vUv + t * vec2(-1.0, 0.0)).rgb * 2.0
		+ texture(uLow, vUv).rgb * 4.0
		+ texture(uLow, vUv + t * vec2(1.0, 0.0)).rgb * 2.0
		+ texture(uLow, vUv + t * vec2(-1.0, -1.0)).rgb
		+ texture(uLow, vUv + t * vec2(0.0, -1.0)).rgb * 2.0
		+ texture(uLow, vUv + t * vec2(1.0, -1.0)).rgb;
	fragColor = vec4(texture(uHigh, vUv).rgb + s * (uSpread / 16.0), 1.0);
}`;

export class Bloom {
	/** @param {THREE.WebGLRenderer} renderer */
	constructor(renderer) {
		this.renderer = renderer;
		/** @returns {THREE.WebGLRenderTarget} */
		const make = () =>
			new THREE.WebGLRenderTarget(1, 1, {
				type: THREE.HalfFloatType,
				format: THREE.RGBAFormat,
				minFilter: THREE.LinearFilter,
				magFilter: THREE.LinearFilter,
				wrapS: THREE.ClampToEdgeWrapping,
				wrapT: THREE.ClampToEdgeWrapping,
				depthBuffer: false,
				generateMipmaps: false
			});
		/** @type {THREE.WebGLRenderTarget[]} */
		this.down = [];
		/** @type {THREE.WebGLRenderTarget[]} */
		this.up = [];
		for (let i = 0; i < LEVELS; i++) {
			this.down.push(make());
			this.up.push(make());
		}
		this.firstPass = new FullscreenPass(FIRST, {
			uSrc: { value: null },
			uAir: { value: null },
			uTexel: { value: new THREE.Vector2() }
		});
		this.downPass = new FullscreenPass(DOWN, {
			uSrc: { value: null },
			uTexel: { value: new THREE.Vector2() }
		});
		this.upPass = new FullscreenPass(UP, {
			uLow: { value: null },
			uHigh: { value: null },
			uTexel: { value: new THREE.Vector2() },
			uSpread: { value: 0.85 }
		});
		this.width = 1;
		this.height = 1;
		/** 1 / the sum of the levels' weights, so `strength` is the fraction of light spread. */
		const spread = /** @type {number} */ (this.upPass.uniforms.uSpread.value);
		let sum = 0;
		for (let i = 0; i < LEVELS; i++) sum += spread ** i;
		this.normalization = 1 / sum;
	}

	/**
	 * @param {number} width the frame's drawing-buffer size
	 * @param {number} height
	 */
	setSize(width, height) {
		this.width = width;
		this.height = height;
		for (let i = 0; i < LEVELS; i++) {
			const w = Math.max(1, Math.round(width / 2 ** (i + 2)));
			const h = Math.max(1, Math.round(height / 2 ** (i + 2)));
			this.down[i].setSize(w, h);
			this.up[i].setSize(w, h);
		}
	}

	/**
	 * @param {THREE.Texture} color the frame's linear radiance
	 * @param {THREE.Texture} air the aircraft layer (premultiplied linear radiance)
	 * @returns {THREE.Texture} the glare, at quarter resolution, in the same units
	 */
	render(color, air) {
		const r = this.renderer;
		const previous = r.getRenderTarget();
		const f = this.firstPass.uniforms;
		f.uSrc.value = color;
		f.uAir.value = air;
		/** @type {THREE.Vector2} */ (f.uTexel.value).set(1 / this.width, 1 / this.height);
		this.firstPass.render(r, this.down[0]);
		const d = this.downPass.uniforms;
		for (let i = 1; i < LEVELS; i++) {
			const src = this.down[i - 1];
			d.uSrc.value = src.texture;
			/** @type {THREE.Vector2} */ (d.uTexel.value).set(1 / src.width, 1 / src.height);
			this.downPass.render(r, this.down[i]);
		}
		const u = this.upPass.uniforms;
		let low = this.down[LEVELS - 1];
		for (let i = LEVELS - 2; i >= 0; i--) {
			u.uLow.value = low.texture;
			u.uHigh.value = this.down[i].texture;
			/** @type {THREE.Vector2} */ (u.uTexel.value).set(1 / low.width, 1 / low.height);
			this.upPass.render(r, this.up[i]);
			low = this.up[i];
		}
		r.setRenderTarget(previous);
		return this.up[0].texture;
	}

	dispose() {
		for (const t of [...this.down, ...this.up]) t.dispose();
		this.firstPass.dispose();
		this.downPass.dispose();
		this.upPass.dispose();
	}
}
