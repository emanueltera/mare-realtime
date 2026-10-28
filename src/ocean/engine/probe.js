/**
 * The sea's height at a point, read off the GPU cascades a frame was drawn from: the
 * ground truth behind FrameTruth's `seaHeightUnderCamera`.
 *
 * The CPU sampler (sampler.js) keeps only a few dozen long components of cascade 0, which
 * is plenty for a free camera's sway but misses the waves that carry the energy of the
 * lighter seas (their peak is shorter than anything it keeps). This pass sums every
 * cascade's displacement texture at full resolution instead, so it is the simulated
 * surface itself.
 *
 * The surface is displaced sideways as well as up (choppiness): the water that lies over
 * a point (east, north) started from somewhere else, x₀ with x₀ + D(x₀) = x. A fixed-point
 * iteration x₀ ← x − D(x₀) finds it (the horizontal displacement varies slowly enough
 * that it converges in a few steps for any sea the engine draws), and the height there
 * is D_up(x₀). One texel per point, read back synchronously, so it is meant for the
 * manual engine's once-a-frame truth, not for the interactive loop.
 */
import * as THREE from 'three';
import { FullscreenPass, floatTarget } from './gpu.js';

/** Points a single probe can answer at once. */
export const PROBE_POINTS = 4;
const ITERATIONS = 8;

const FRAGMENT = /* glsl */ `
uniform vec2 uPoints[${PROBE_POINTS}];   // east, north relative to the floating origin
uniform vec4 uPatch;
uniform vec2 uOffset0;
uniform vec2 uOffset1;
uniform vec2 uOffset2;
uniform vec2 uOffset3;
uniform int uCascades;
uniform sampler2D uDisp0;
uniform sampler2D uDisp1;
uniform sampler2D uDisp2;
uniform sampler2D uDisp3;
out vec4 outColor;

// Displacement east, up, north (the surface shader's xyz) summed over the cascades, at
// full resolution (mip 0).
vec3 displacement(vec2 en) {
	vec3 d = textureLod(uDisp0, en / uPatch.x + uOffset0, 0.0).xyz;
	if (uCascades > 1) d += textureLod(uDisp1, en / uPatch.y + uOffset1, 0.0).xyz;
	if (uCascades > 2) d += textureLod(uDisp2, en / uPatch.z + uOffset2, 0.0).xyz;
	if (uCascades > 3) d += textureLod(uDisp3, en / uPatch.w + uOffset3, 0.0).xyz;
	return d;
}

void main() {
	int i = int(gl_FragCoord.x);
	vec2 target = uPoints[i];
	vec2 p = target;
	vec3 d = vec3(0.0);
	for (int k = 0; k < ${ITERATIONS}; k++) {
		d = displacement(p);
		p = target - d.xz;
	}
	d = displacement(p);
	// height, how far the found point misses the target (m, a convergence check), source east, north
	outColor = vec4(d.y, length(p + d.xz - target), p);
}`;

export class SeaProbe {
	/** @param {THREE.WebGLRenderer} renderer */
	constructor(renderer) {
		this.renderer = renderer;
		this.target = floatTarget(PROBE_POINTS, 1);
		this.pixels = new Float32Array(PROBE_POINTS * 4);
		this.pass = new FullscreenPass(FRAGMENT, {
			uPoints: { value: Array.from({ length: PROBE_POINTS }, () => new THREE.Vector2()) },
			uPatch: { value: new THREE.Vector4(1, 1, 1, 1) },
			uOffset0: { value: new THREE.Vector2() },
			uOffset1: { value: new THREE.Vector2() },
			uOffset2: { value: new THREE.Vector2() },
			uOffset3: { value: new THREE.Vector2() },
			uCascades: { value: 1 },
			uDisp0: { value: null },
			uDisp1: { value: null },
			uDisp2: { value: null },
			uDisp3: { value: null }
		});
	}

	/**
	 * Sea surface height (m above mean sea level) at up to PROBE_POINTS points, from the
	 * cascades as the waves' last update left them.
	 * @param {import('./waves.js').WaveSimulation} waves
	 * @param {{ east: number, north: number }} origin the floating origin the offsets are for
	 * @param {Array<[number, number]>} points absolute east, north metres
	 * @returns {Array<{ height: number, miss: number }>} `miss` is how far (m) the surface
	 *   point found lies from the one asked for; it is millimetres or less when converged
	 */
	sample(waves, origin, points) {
		if (points.length > PROBE_POINTS) throw new Error(`SeaProbe: at most ${PROBE_POINTS} points at once`);
		const u = this.pass.uniforms;
		const count = Math.min(4, waves.count);
		const patches = waves.patches;
		/** @type {THREE.Vector4} */ (u.uPatch.value).set(patches[0], patches[1] ?? 1, patches[2] ?? 1, patches[3] ?? 1);
		const offsets = [u.uOffset0.value, u.uOffset1.value, u.uOffset2.value, u.uOffset3.value];
		waves.offsets(origin.east, origin.north, offsets.slice(0, count));
		u.uCascades.value = count;
		for (let c = 0; c < 4; c++) u[`uDisp${c}`].value = waves.textures(Math.min(c, count - 1)).disp;
		const pts = /** @type {THREE.Vector2[]} */ (u.uPoints.value);
		points.forEach(([e, n], i) => pts[i].set(e - origin.east, n - origin.north));
		const previous = this.renderer.getRenderTarget();
		this.pass.render(this.renderer, this.target);
		this.renderer.readRenderTargetPixels(this.target, 0, 0, PROBE_POINTS, 1, this.pixels);
		this.renderer.setRenderTarget(previous);
		return points.map((_, i) => ({ height: this.pixels[i * 4], miss: this.pixels[i * 4 + 1] }));
	}

	dispose() {
		this.target.dispose();
		this.pass.dispose();
	}
}
