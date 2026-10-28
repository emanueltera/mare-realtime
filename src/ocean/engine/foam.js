/**
 * Foam on the water: the whitecaps where the dominant waves break, and the foam they
 * leave behind, drifting downwind, drawn out into streaks and thinning over a few
 * seconds.
 *
 * The simulation cascades tile, so foam left on the water cannot live in them without
 * tiling too. It lives here, in world space: two square fields that follow the camera's
 * view of the water, a fine one near the camera and a coarse one reaching a few
 * kilometres, each a ping-pong pair of float textures indexed by the undisplaced
 * east/north position. That is the cascades' own Lagrangian coordinate, so the foam rides
 * the orbital motion of the waves exactly as the water does.
 *
 * Every frame each field moves by whole texels as its centre follows the camera (so the
 * foam on it is never resampled out of shape), drifts downwind, diffuses a little along
 * the wind, decays, and takes new foam wherever the crests fold: the Jacobian of the
 * cascades, weighted toward the dominant waves, below a threshold set by the wind. Texels
 * that move into the fine field are seeded from the coarse one, so turning the camera
 * never reveals bare water.
 */
import * as THREE from 'three';
import { FullscreenPass } from './gpu.js';

/** Foam a breaking crest leaves on the water, as a coverage: thinner than the crest's own. */
const RESIDUAL = 0.6;

/**
 * The break gate tiles every this many metres: one tile of the foam texture, whose alpha
 * channel holds 64 noise cells across, so the cells are 12 m. Long enough that no view
 * shows the gate repeat; a short period shows from altitude as a lattice in the foam.
 */
export const GATE_PERIOD = 768;

/**
 * Not every steep crest breaks. A pattern of cells about a dozen metres across, fixed on
 * the water and slowly changing, picks the crests that do: a crest breaks while it runs
 * through one, about a second, and leaves a trail about a cell long. Needs `uFoamTex`
 * and `uGateOffset` (the floating origin plus the slow scroll, wrapped to GATE_PERIOD).
 */
export const BREAK_GATE_GLSL = /* glsl */ `
float breakGate(vec2 en) {
	float n = textureLod(uFoamTex, (en + uGateOffset) / ${GATE_PERIOD.toFixed(1)}, 0.0).a;
	// A soft edge, several metres wide: a cell's border is where its crests break
	// least, so the lace in the surface frays it rather than drawing the cell's outline.
	return smoothstep(0.45, 0.67, n);
}`;

const UPDATE = /* glsl */ `
uniform sampler2D uPrev;
uniform sampler2D uSeed;
uniform float uHasSeed;
uniform vec3 uRect;       // this frame: centre east, north (from the floating origin), extent
uniform vec3 uPrevRect;   // the same, last frame
uniform vec3 uSeedRect;   // the coarser field, this frame
uniform sampler2D uJac0;
uniform sampler2D uJac1;
uniform sampler2D uJac2;
uniform vec3 uPatch;
uniform vec2 uOffset0;
uniform vec2 uOffset1;
uniform vec2 uOffset2;
uniform vec3 uJacW;       // each cascade's weight in the breaking Jacobian
uniform vec3 uLod;        // mip level per cascade for this field's texel size
uniform vec2 uBreak;      // J below x breaks, fully over 1/y
uniform float uInject;
uniform float uKeep;      // exp(−dt/lifetime)
uniform vec2 uDrift;      // metres the surface drifted this frame
uniform vec2 uSpread;     // metres: along-wind diffusion tap offset
uniform int uCascades;
uniform sampler2D uFoamTex;
uniform vec2 uGateOffset;
in vec2 vUv;
out vec4 fragColor;

${BREAK_GATE_GLSL}

bool inside(vec2 uv) {
	return all(greaterThan(uv, vec2(0.0))) && all(lessThan(uv, vec2(1.0)));
}

void main() {
	vec2 en = uRect.xy + (vUv - 0.5) * uRect.z;
	// The water here was upwind of here a frame ago.
	vec2 src = en - uDrift;
	vec2 puv = (src - uPrevRect.xy) / uPrevRect.z + 0.5;
	float prev = 0.0;
	if (inside(puv)) {
		vec2 tap = uSpread / uPrevRect.z;
		prev = 0.5 * textureLod(uPrev, puv, 0.0).r
			+ 0.25 * (textureLod(uPrev, puv + tap, 0.0).r + textureLod(uPrev, puv - tap, 0.0).r);
	} else if (uHasSeed > 0.5) {
		vec2 suv = (src - uSeedRect.xy) / uSeedRect.z + 0.5;
		if (inside(suv)) prev = textureLod(uSeed, suv, 0.0).r;
	}
	vec3 jac = uJacW.x * textureLod(uJac0, en / uPatch.x + uOffset0, uLod.x).xyz;
	if (uCascades > 1) jac += uJacW.y * textureLod(uJac1, en / uPatch.y + uOffset1, uLod.y).xyz;
	if (uCascades > 2) jac += uJacW.z * textureLod(uJac2, en / uPatch.z + uOffset2, uLod.z).xyz;
	float J = (1.0 + jac.x) * (1.0 + jac.y) - jac.z * jac.z;
	float inject = clamp((uBreak.x - J) * uBreak.y, 0.0, 1.0) * breakGate(en) * uInject;
	fragColor = vec4(max(prev * uKeep, inject), 0.0, 0.0, 1.0);
}`;

/**
 * @typedef {object} FoamLevel
 * @property {number} extent metres across
 * @property {number} lookAhead fraction of the extent the centre may sit ahead of the camera
 * @property {[THREE.WebGLRenderTarget, THREE.WebGLRenderTarget]} targets
 * @property {number} current index of the target holding the latest field
 * @property {{ east: number, north: number }} center absolute metres, on the texel grid
 * @property {boolean} valid whether the field holds anything yet
 */

/**
 * @typedef {object} FoamStep
 * @property {number} dt seconds of simulation time
 * @property {number} camEast absolute camera position, m
 * @property {number} camNorth
 * @property {number} camHeight m above the water
 * @property {number} forwardEast horizontal unit vector the camera looks along
 * @property {number} forwardNorth
 * @property {number} forwardDown sine of the camera's look-down angle (negative pitch → positive)
 * @property {number} originEast floating origin, absolute m
 * @property {number} originNorth
 * @property {import('./waves.js').WaveSimulation} waves
 * @property {THREE.Vector2[]} offsets per cascade texture offsets for the floating origin
 * @property {[number, number, number, number]} jacobianWeights
 * @property {THREE.Texture} foamTexture the foam detail texture (its G channel drives the break gate)
 * @property {THREE.Vector2} gateOffset see BREAK_GATE_GLSL
 */

export class FoamField {
	/**
	 * @param {THREE.WebGLRenderer} renderer
	 * @param {{ size?: number, extents?: [number, number] }} [options]
	 */
	constructor(renderer, options = {}) {
		this.renderer = renderer;
		this.size = options.size ?? 1024;
		const [fine, coarse] = options.extents ?? [420, 3600];
		/** @returns {THREE.WebGLRenderTarget} */
		const make = () =>
			new THREE.WebGLRenderTarget(this.size, this.size, {
				type: THREE.HalfFloatType,
				format: THREE.RedFormat,
				minFilter: THREE.LinearMipmapLinearFilter,
				magFilter: THREE.LinearFilter,
				wrapS: THREE.ClampToEdgeWrapping,
				wrapT: THREE.ClampToEdgeWrapping,
				generateMipmaps: true,
				anisotropy: 4,
				depthBuffer: false,
				stencilBuffer: false
			});
		/** @type {FoamLevel[]} fine first */
		this.levels = [
			{ extent: fine, lookAhead: 0.32, targets: [make(), make()], current: 0, center: { east: 0, north: 0 }, valid: false },
			{ extent: coarse, lookAhead: 0.3, targets: [make(), make()], current: 0, center: { east: 0, north: 0 }, valid: false }
		];
		this.pass = new FullscreenPass(UPDATE, {
			uPrev: { value: null },
			uSeed: { value: null },
			uHasSeed: { value: 0 },
			uRect: { value: new THREE.Vector3() },
			uPrevRect: { value: new THREE.Vector3() },
			uSeedRect: { value: new THREE.Vector3() },
			uJac0: { value: null },
			uJac1: { value: null },
			uJac2: { value: null },
			uPatch: { value: new THREE.Vector3(1, 1, 1) },
			uOffset0: { value: new THREE.Vector2() },
			uOffset1: { value: new THREE.Vector2() },
			uOffset2: { value: new THREE.Vector2() },
			uJacW: { value: new THREE.Vector3() },
			uLod: { value: new THREE.Vector3() },
			uBreak: { value: new THREE.Vector2(-1, 1) },
			uInject: { value: 0 },
			uKeep: { value: 1 },
			uDrift: { value: new THREE.Vector2() },
			uSpread: { value: new THREE.Vector2() },
			uCascades: { value: 3 },
			uFoamTex: { value: null },
			uGateOffset: { value: new THREE.Vector2() }
		});
		/** Uniform values the surface reads: per level, centre (from the origin), 1/extent, weight. */
		this.rects = [new THREE.Vector4(), new THREE.Vector4()];
	}

	/** @param {number} i fine 0, coarse 1 */
	texture(i) {
		const level = this.levels[i];
		return level.targets[level.current].texture;
	}

	/** @param {FoamStep} s */
	update(s) {
		const renderer = this.renderer;
		const waves = s.waves;
		const previous = renderer.getRenderTarget();
		const u = this.pass.uniforms;
		const count = Math.min(3, waves.count);
		const tex = [0, 1, 2].map((c) => waves.textures(Math.min(c, waves.count - 1)));
		u.uJac0.value = tex[0].jac;
		u.uJac1.value = tex[1].jac;
		u.uJac2.value = tex[2].jac;
		const patches = waves.patches;
		/** @type {THREE.Vector3} */ (u.uPatch.value).set(patches[0], patches[1] ?? 1, patches[2] ?? 1);
		/** @type {THREE.Vector2} */ (u.uOffset0.value).copy(s.offsets[0]);
		/** @type {THREE.Vector2} */ (u.uOffset1.value).copy(s.offsets[Math.min(1, count - 1)]);
		/** @type {THREE.Vector2} */ (u.uOffset2.value).copy(s.offsets[Math.min(2, count - 1)]);
		const w = s.jacobianWeights;
		/** @type {THREE.Vector3} */ (u.uJacW.value).set(w[0], count > 1 ? w[1] : 0, count > 2 ? w[2] : 0);
		u.uCascades.value = count;
		/** @type {THREE.Vector2} */ (u.uBreak.value).set(waves.activeFoam.bias, waves.activeFoam.gain);
		u.uInject.value = waves.foamEnabled * RESIDUAL;
		u.uFoamTex.value = s.foamTexture;
		/** @type {THREE.Vector2} */ (u.uGateOffset.value).copy(s.gateOffset);
		// Decay, drift and spread are first-order steps, good for short ones only: both
		// drivers keep a step at or under 0.1 s (the interactive loop clamps its frame time,
		// a manual advance sub-steps), and this clamp only guards against a caller that does not.
		const dt = Math.min(s.dt, 0.1);
		u.uKeep.value = Math.exp(-dt / waves.foamLifetime);
		/** @type {THREE.Vector2} */ (u.uDrift.value).copy(waves.foamDrift).multiplyScalar(dt);
		/** @type {THREE.Vector2} */ (u.uSpread.value).copy(waves.windDir).multiplyScalar(waves.foamSpread(dt));

		// Coarse first, so the fine field seeds from this frame's coarse one.
		for (let i = this.levels.length - 1; i >= 0; i--) {
			const level = this.levels[i];
			const texel = level.extent / this.size;
			// Centre the field on the water the camera looks at, ahead of it but never so far
			// that the water right below drops out.
			const hit = s.forwardDown > 0.01 ? s.camHeight / s.forwardDown : Infinity;
			const ahead = Math.min(hit, level.lookAhead * level.extent);
			const east = Math.round((s.camEast + s.forwardEast * ahead) / texel) * texel;
			const north = Math.round((s.camNorth + s.forwardNorth * ahead) / texel) * texel;
			const prevCenter = level.valid ? level.center : { east, north };
			/** @type {THREE.Vector3} */ (u.uRect.value).set(east - s.originEast, north - s.originNorth, level.extent);
			/** @type {THREE.Vector3} */ (u.uPrevRect.value).set(
				level.valid ? prevCenter.east - s.originEast : 1e9,
				level.valid ? prevCenter.north - s.originNorth : 1e9,
				level.extent
			);
			const coarser = this.levels[i + 1];
			u.uHasSeed.value = coarser ? 1 : 0;
			if (coarser) {
				u.uSeed.value = coarser.targets[coarser.current].texture;
				/** @type {THREE.Vector3} */ (u.uSeedRect.value).set(
					coarser.center.east - s.originEast,
					coarser.center.north - s.originNorth,
					coarser.extent
				);
			}
			/** @param {number} c */
			const lod = (c) => Math.max(0, Math.log2(texel / (patches[Math.min(c, waves.count - 1)] / waves.size)));
			/** @type {THREE.Vector3} */ (u.uLod.value).set(lod(0), lod(1), lod(2));
			u.uPrev.value = level.targets[level.current].texture;
			const next = 1 - level.current;
			this.pass.render(renderer, level.targets[next]);
			level.current = next;
			level.center = { east, north };
			level.valid = true;
			this.rects[i].set(east - s.originEast, north - s.originNorth, 1 / level.extent, 1);
		}
		renderer.setRenderTarget(previous);
	}

	dispose() {
		for (const level of this.levels) {
			level.targets[0].dispose();
			level.targets[1].dispose();
		}
		this.pass.dispose();
	}
}
