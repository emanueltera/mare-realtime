/**
 * The sea surface mesh and its material.
 *
 * Geometry is a camera-centred clipmap: a full grid at the finest visible spacing and
 * up to a dozen square rings around it, each twice as coarse as the last, every level
 * snapped to its own coarser grid so vertices never swim over the water. Odd vertices
 * near a level's outer edge morph onto the even ones (CDLOD), so each ring meets the
 * next without cracks or popping. Rings reach past the horizon; the vertex shader
 * bends the sea over the earth's curvature, so the horizon sits where the HUD says.
 *
 * The material samples every cascade, builds the normal and the slope covariance
 * (LEAN-style second moments, so detail that mip-maps away turns into roughness), finds
 * the crests breaking right now and the foam they left (`foam.js`), and writes three
 * targets: the shaded radiance, and the ground-truth channels the deck's label views
 * show (normal + foam, height + vertical velocity + range + water mask).
 */
import * as THREE from 'three';
import { SKY_GLSL, WATER_GLSL } from './sky.js';
import { BREAK_GATE_GLSL } from './foam.js';
import { JACOBIAN_WEIGHTS } from './waves.js';

/** Every foam lace tile divides this many metres (see laceAt), so lace offsets wrap by it. */
export const LACE_PERIOD = 256;

/** Cells from a level's centre to its edge. */
export const CLIP_HALF = 64;
/** Finest vertex spacing, m. */
export const BASE_SPACING = 0.25;
const MAX_LEVELS = 16;

/**
 * Grid indices for the full level or for a ring whose hole is offset by (ox, oz) cells.
 * @param {number} M
 * @param {number | null} ox
 * @param {number} oz
 */
function gridIndices(M, ox, oz) {
	const side = 2 * M + 1;
	/** @type {number[]} */
	const indices = [];
	for (let j = -M; j < M; j++) {
		for (let i = -M; i < M; i++) {
			if (ox !== null) {
				const inHole = i >= -M / 2 + ox && i < M / 2 + ox && j >= -M / 2 + oz && j < M / 2 + oz;
				if (inHole) continue;
			}
			const a = (j + M) * side + (i + M);
			const b = a + 1;
			const c = a + side + 1;
			const d = a + side;
			indices.push(a, b, c, a, c, d);
		}
	}
	return indices;
}

function buildGeometries() {
	const M = CLIP_HALF;
	const side = 2 * M + 1;
	const positions = new Float32Array(side * side * 3);
	for (let j = -M; j <= M; j++) {
		for (let i = -M; i <= M; i++) {
			const v = ((j + M) * side + (i + M)) * 3;
			positions[v] = i;
			positions[v + 1] = j;
		}
	}
	const position = new THREE.BufferAttribute(positions, 3);
	/** @param {number[]} idx */
	const make = (idx) => {
		const g = new THREE.BufferGeometry();
		g.setAttribute('position', position);
		g.setIndex(idx);
		g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e9);
		return g;
	};
	const full = make(gridIndices(M, null, 0));
	/** @type {THREE.BufferGeometry[]} */
	const rings = [];
	for (let oz = -1; oz <= 1; oz++) {
		for (let ox = -1; ox <= 1; ox++) rings.push(make(gridIndices(M, ox, oz)));
	}
	return { full, rings };
}

const VERTEX = /* glsl */ `
uniform vec2 uLevelCenter;
uniform float uLevelSpacing;
uniform vec2 uCamEN;
uniform vec4 uPatch;
uniform vec4 uTexel;
uniform vec2 uOffset0;
uniform vec2 uOffset1;
uniform vec2 uOffset2;
uniform vec2 uOffset3;
uniform sampler2D uDisp0;
uniform sampler2D uDisp1;
uniform sampler2D uDisp2;
uniform sampler2D uDisp3;
out vec2 vEN;
out vec3 vWorld;

const float M = ${CLIP_HALF.toFixed(1)};
const float R_EARTH = 6360000.0;

vec3 disp(sampler2D t, vec2 en, float L, vec2 off, float texel, float spacing) {
	float lod = max(0.0, log2(spacing / texel) - 0.5);
	return textureLod(t, en / L + off, lod).xyz;
}

void main() {
	vec2 g = position.xy;
	vec2 p = uLevelCenter + g * uLevelSpacing;
	vec2 dc = abs(p - uCamEN) / (M * uLevelSpacing);
	float morph = smoothstep(0.62, 0.92, max(dc.x, dc.y));
	p -= fract(g * 0.5) * 2.0 * uLevelSpacing * morph;
	float spacing = uLevelSpacing * (1.0 + morph);
	vec3 d = disp(uDisp0, p, uPatch.x, uOffset0, uTexel.x, spacing);
#if CASCADES > 1
	d += disp(uDisp1, p, uPatch.y, uOffset1, uTexel.y, spacing);
#endif
#if CASCADES > 2
	d += disp(uDisp2, p, uPatch.z, uOffset2, uTexel.z, spacing);
#endif
#if CASCADES > 3
	d += disp(uDisp3, p, uPatch.w, uOffset3, uTexel.w, spacing);
#endif
	vec2 rel = p - uCamEN;
	float drop = dot(rel, rel) / (2.0 * R_EARTH);
	vec3 world = vec3(p.x + d.x, d.y - drop, -(p.y + d.z));
	vEN = p;
	vWorld = world;
	gl_Position = projectionMatrix * viewMatrix * vec4(world, 1.0);
}`;

const FRAGMENT = /* glsl */ `
${SKY_GLSL}
${WATER_GLSL}
uniform vec4 uPatch;
uniform vec2 uOffset0;
uniform vec2 uOffset1;
uniform vec2 uOffset2;
uniform vec2 uOffset3;
uniform sampler2D uDisp0;
uniform sampler2D uDisp1;
uniform sampler2D uDisp2;
uniform sampler2D uDisp3;
uniform sampler2D uSlope0;
uniform sampler2D uSlope1;
uniform sampler2D uSlope2;
uniform sampler2D uSlope3;
uniform sampler2D uJac0;
uniform sampler2D uJac1;
uniform sampler2D uJac2;
uniform sampler2D uJac3;
uniform vec4 uFadeNear;
uniform vec4 uFadeFar;
uniform vec4 uJacWeights;   // each cascade's share in the Jacobian that breaks into whitecaps
uniform vec2 uActiveFoam;   // J below x turns to fresh foam over 1/y
uniform sampler2D uFoamField0;
uniform sampler2D uFoamField1;
uniform vec4 uFoamRect0;    // fine foam field: centre (from the origin), 1/extent, on
uniform vec4 uFoamRect1;    // coarse foam field
uniform vec2 uGateOffset;
${BREAK_GATE_GLSL}
uniform float uTailVariance;
uniform vec2 uLaceCam;     // camera east/north from the origin
uniform vec2 uLaceOffset;  // the camera's lace coordinates, wrapped to LACE_PERIOD
uniform vec2 uWindEN;
in vec2 vEN;
in vec3 vWorld;
layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outData0;
layout(location = 2) out vec4 outData1;

void accumulate(sampler2D tDisp, sampler2D tSlope, sampler2D tJac, vec2 uv, float fade, float jacW,
		inout vec2 slope, inout vec3 sig, inout vec3 jac, inout float height, inout float vel) {
	vec4 S = texture(tSlope, uv);
	vec4 D = texture(tDisp, uv);
	vec4 K = texture(tJac, uv);
	slope += fade * S.xy;
	float f2 = fade * fade;
	sig += vec3(S.z - f2 * S.x * S.x, S.w - f2 * S.y * S.y, K.w - f2 * S.x * S.y);
	jac += fade * jacW * K.xyz;
	height += D.y;
	vel += D.w;
}

// 1 well inside a foam field, falling to 0 over the outer eighth of it.
float fieldWeight(vec2 uv) {
	vec2 d = min(uv, 1.0 - uv);
	return smoothstep(0.0, 0.12, min(d.x, d.y));
}

// The foam whitecaps have left on the water here: the fine field where it reaches, the
// coarse one round it.
float residualFoam(vec2 en) {
	vec2 uv1 = (en - uFoamRect1.xy) * uFoamRect1.z + 0.5;
	vec2 uv0 = (en - uFoamRect0.xy) * uFoamRect0.z + 0.5;
	float f1 = texture(uFoamField1, uv1).r;
	float f0 = texture(uFoamField0, uv0).r;
	float f = f1 * fieldWeight(uv1) * uFoamRect1.w;
	return mix(f, f0, fieldWeight(uv0) * uFoamRect0.w);
}

// A mixture a·U1 + b·U2 of two uniform variables (a + b = 1, a ≥ b) mapped back to
// uniform through its CDF, so two independent laces combine and stay evenly spread.
float uniformMix(float x, float a, float b) {
	if (x < b) return x * x / (2.0 * a * b);
	if (x <= a) return (x - 0.5 * b) / a;
	float y = 1.0 - x;
	return 1.0 - y * y / (2.0 * a * b);
}

// One octave of foam lace at a scale: two independent laces combined, still uniform.
// Tile sizes (16 × 8 m and 6.4 × 3.2 m, times the octave's scale up to 8) all divide
// LACE_PERIOD, so the lace offset can wrap without a seam.
float laceAt(vec2 a, float scale, float seed) {
	vec2 p = a / scale + seed;
	float l1 = texture(uFoamTex, vec2(p.x / 16.0, p.y / 8.0)).r;
	float l2 = texture(uFoamTex, vec2(p.x / 6.4 + 0.31, p.y / 3.2 + 0.77)).b;
	return uniformMix(0.65 * l1 + 0.35 * l2, 0.65, 0.35);
}

// The lace blended between the two octaves either side of a fractional octave, still
// uniform. The stretch squeezes it across the wind (2.5 keeps every tile dividing
// LACE_PERIOD), drawing it out into streaks along the wind.
float laceOctaves(vec2 a, float o0, float f, float stretch, float seed) {
	vec2 p = vec2(a.x, a.y * stretch);
	float lace0 = laceAt(p, exp2(o0), o0 * 0.37 + seed);
	float lace1 = laceAt(p, exp2(o0 + 1.0), (o0 + 1.0) * 0.37 + seed);
	float x = mix(lace0, lace1, f);
	float big = max(f, 1.0 - f);
	return big > 0.99 ? x : uniformMix(x, big, 1.0 - big);
}

// Visible foam for a coverage c (0–1, the fraction of the water foam covers): c
// thresholds a foam lace whose histogram is uniform, so as foam ages and thins it opens
// into lace and streaks drawn out along the wind, covering on average about c. Even a
// fresh whitecap is churned water rather than a sheet of white: the threshold never
// passes about 0.72 of the lace, so dark gaps still run through the densest foam, and
// above half coverage a wind-stretched lace is mixed in so they read as streaks. The
// lace's octave follows the pixel footprint, so its holes are always a few pixels
// across: lacy near the camera, patchy further out, and never finer than the screen
// can show.
float foamDensity = 0.0;

float foamMask(float coverage, vec2 en) {
	float c = clamp(coverage, 0.0, 1.0);
	foamDensity = 0.0;
	if (c < 0.002) return 0.0;
	// The share of the lace filled: c up to about half, then easing off to 0.72.
	float fill = c * (1.0 - 0.28 * smoothstep(0.45, 1.0, c));
	// Wind-aligned frame, the lace drawn out along the wind. It turns about the camera,
	// so a change of wind turns the lace in view rather than sliding it across the water;
	// the rest of the rotated position (and the drift) comes in wrapped, from the CPU.
	vec2 d = en - uLaceCam;
	vec2 w = uWindEN;
	vec2 a = vec2(dot(d, w), dot(d, vec2(-w.y, w.x))) + uLaceOffset;
	float footprint = length(fwidth(en));
	float octave = min(log2(max(footprint / 0.06, 1.0)), 3.0);
	float o0 = floor(octave);
	float f = octave - o0;
	float lace = laceOctaves(a, o0, f, 1.0, 0.0);
	float streaky = 0.4 * smoothstep(0.45, 0.9, c);
	if (streaky > 0.01) {
		float streak = laceOctaves(a, o0, f, 2.5, 0.53);
		lace = uniformMix((1.0 - streaky) * lace + streaky * streak, 1.0 - streaky, streaky);
	}
	// A soft threshold, a few pixels wide, so patch borders fray rather than cut out.
	float edge = clamp(fwidth(lace) * 1.5, 0.04, 0.5);
	float mask = smoothstep(1.0 - fill - edge, 1.0 - fill + edge, lace);
	// How far inside the lace this point is: the core of a patch is dense, bright foam,
	// its fringe a thin grey film.
	foamDensity = smoothstep(0.0, 0.35, lace - (1.0 - fill)) * smoothstep(0.1, 0.8, c);
	// Seen from far enough that even the coarsest lace is a pixel or two, the foam is
	// its coverage: what the lace averages to.
	float far = smoothstep(0.5, 1.5, footprint);
	mask = mix(mask, fill, far);
	foamDensity = mix(foamDensity, smoothstep(0.1, 0.7, c), far);
	// Old foam thins as it spreads, so it fades as well as opening into lace: a
	// whitecap's trail is bright where it was just laid and soon only a faint film.
	return mask * smoothstep(0.04, 0.45, c);
}

void main() {
	vec3 toCam = cameraPosition - vWorld;
	float dist = length(toCam);
	vec3 V = toCam / dist;

	vec2 slope = vec2(0.0);
	vec3 sig = vec3(0.0);
	vec3 jac = vec3(0.0);
	float height = 0.0;
	float vel = 0.0;
	vec4 fade = vec4(1.0) - smoothstep(uFadeNear, uFadeFar, vec4(dist));
	accumulate(uDisp0, uSlope0, uJac0, vEN / uPatch.x + uOffset0, fade.x, uJacWeights.x,
		slope, sig, jac, height, vel);
#if CASCADES > 1
	accumulate(uDisp1, uSlope1, uJac1, vEN / uPatch.y + uOffset1, fade.y, uJacWeights.y,
		slope, sig, jac, height, vel);
#endif
#if CASCADES > 2
	accumulate(uDisp2, uSlope2, uJac2, vEN / uPatch.z + uOffset2, fade.z, uJacWeights.z,
		slope, sig, jac, height, vel);
#endif
#if CASCADES > 3
	accumulate(uDisp3, uSlope3, uJac3, vEN / uPatch.w + uOffset3, fade.w, uJacWeights.w,
		slope, sig, jac, height, vel);
#endif
	sig.xy += vec2(uTailVariance * 0.5);
	sig.xy = max(sig.xy, vec2(4e-5));
	float lim = 0.95 * sqrt(sig.x * sig.y);
	sig.z = clamp(sig.z, -lim, lim);

	vec3 N = normalize(vec3(-slope.x, 1.0, slope.y));
	float shadow = cloudShadow(vWorld);
	float crest = height / max(0.5 * uHs, 0.05);
	// Fresh whitecaps where the crests fold right now (the cascades' Jacobian terms
	// together, weighted toward the dominant waves), over the foam left on the water.
	float J = (1.0 + jac.x) * (1.0 + jac.y) - jac.z * jac.z;
	float breaking = clamp((uActiveFoam.x - J) * uActiveFoam.y, 0.0, 1.0) * breakGate(vEN);
	float coverage = max(residualFoam(vEN), breaking);
	float foamVis = foamMask(coverage, vEN);

	gFoamAlbedo = mix(0.5, 1.0, foamDensity);
	vec3 radiance = shadeWater(vWorld, V, N, sig, crest, foamVis, shadow);
	// The bubble cloud under and around whitecaps lifts the water toward turquoise.
	float bubbles = smoothstep(0.25, 0.9, coverage) * (1.0 - foamVis);
	radiance += uSssColor * (uSunE * shadow * max(uSunDir.y, 0.0) + uSkyE) * bubbles * 0.05;
	vec3 col = applyHaze(radiance, -V, dist, max(vWorld.y, 0.0));

	outColor = vec4(col, 1.0);
	outData0 = vec4(N.x, -N.z, N.y, foamVis);
	outData1 = vec4(height, vel, min(dist, 60000.0), 1.0);
}`;

/**
 * @typedef {object} SurfaceUniforms
 */

export class OceanSurface {
	/**
	 * @param {import('./sky.js').SkyUniforms} skyUniforms
	 * @param {number} cascades
	 */
	constructor(skyUniforms, cascades) {
		this.group = new THREE.Group();
		this.group.name = 'ocean-surface';
		this.geometries = buildGeometries();
		this.uniforms = {
			...skyUniforms,
			uCamEN: { value: new THREE.Vector2() },
			uPatch: { value: new THREE.Vector4(1, 1, 1, 1) },
			uTexel: { value: new THREE.Vector4(1, 1, 1, 1) },
			uOffset0: { value: new THREE.Vector2() },
			uOffset1: { value: new THREE.Vector2() },
			uOffset2: { value: new THREE.Vector2() },
			uOffset3: { value: new THREE.Vector2() },
			uDisp0: { value: /** @type {THREE.Texture | null} */ (null) },
			uDisp1: { value: /** @type {THREE.Texture | null} */ (null) },
			uDisp2: { value: /** @type {THREE.Texture | null} */ (null) },
			uDisp3: { value: /** @type {THREE.Texture | null} */ (null) },
			uSlope0: { value: /** @type {THREE.Texture | null} */ (null) },
			uSlope1: { value: /** @type {THREE.Texture | null} */ (null) },
			uSlope2: { value: /** @type {THREE.Texture | null} */ (null) },
			uSlope3: { value: /** @type {THREE.Texture | null} */ (null) },
			uJac0: { value: /** @type {THREE.Texture | null} */ (null) },
			uJac1: { value: /** @type {THREE.Texture | null} */ (null) },
			uJac2: { value: /** @type {THREE.Texture | null} */ (null) },
			uJac3: { value: /** @type {THREE.Texture | null} */ (null) },
			uFadeNear: { value: new THREE.Vector4(1e9, 1e9, 1e9, 1e9) },
			uFadeFar: { value: new THREE.Vector4(2e9, 2e9, 2e9, 2e9) },
			uJacWeights: { value: new THREE.Vector4(...JACOBIAN_WEIGHTS) },
			uActiveFoam: { value: new THREE.Vector2(-1, 1) },
			uFoamField0: { value: /** @type {THREE.Texture | null} */ (null) },
			uFoamField1: { value: /** @type {THREE.Texture | null} */ (null) },
			uFoamRect0: { value: new THREE.Vector4() },
			uFoamRect1: { value: new THREE.Vector4() },
			uGateOffset: { value: new THREE.Vector2() },
			uTailVariance: { value: 0.004 },
			uLaceCam: { value: new THREE.Vector2() },
			uLaceOffset: { value: new THREE.Vector2() },
			uWindEN: { value: new THREE.Vector2(0, 1) }
		};
		/** @type {Array<{ mesh: THREE.Mesh, material: THREE.ShaderMaterial, center: THREE.Vector2, spacing: { value: number } }>} */
		this.levels = [];
		this.cascades = cascades;
		this.build(cascades);
		this.visibleLevels = 0;
		this.firstLevel = 0;
	}

	/** @param {number} cascades */
	build(cascades) {
		for (const level of this.levels) {
			this.group.remove(level.mesh);
			level.material.dispose();
		}
		this.levels = [];
		this.cascades = cascades;
		for (let i = 0; i < MAX_LEVELS; i++) {
			const center = new THREE.Vector2();
			const spacing = { value: BASE_SPACING };
			const material = new THREE.ShaderMaterial({
				glslVersion: THREE.GLSL3,
				vertexShader: VERTEX,
				fragmentShader: FRAGMENT,
				defines: { CASCADES: cascades },
				uniforms: { ...this.uniforms, uLevelCenter: { value: center }, uLevelSpacing: spacing },
				side: THREE.DoubleSide
			});
			const mesh = new THREE.Mesh(this.geometries.full, material);
			mesh.frustumCulled = false;
			mesh.renderOrder = i;
			mesh.visible = false;
			this.group.add(mesh);
			this.levels.push({ mesh, material, center, spacing });
		}
	}

	/**
	 * Lay the clipmap out around the camera.
	 * @param {number} camEast camera east, absolute metres (double precision)
	 * @param {number} camNorth camera north, absolute metres
	 * @param {number} originEast floating origin east, absolute metres
	 * @param {number} originNorth
	 * @param {number} heightAboveSea camera height above the water, m
	 */
	layout(camEast, camNorth, originEast, originNorth, heightAboveSea) {
		const h = Math.max(1, heightAboveSea);
		// Finest level whose spacing is still a few pixels at the nearest water.
		const first = Math.max(0, Math.min(8, Math.floor(Math.log2((0.006 * h) / BASE_SPACING))));
		// Reach well past the horizon: √(2Rh), with a floor for low cameras.
		const horizon = Math.sqrt(2 * 6.36e6 * (h + 5));
		const reach = Math.max(24000, horizon * 1.35);
		let count = 1;
		while (count < MAX_LEVELS && CLIP_HALF * BASE_SPACING * Math.pow(2, first + count - 1) < reach) count++;
		this.firstLevel = first;
		this.visibleLevels = count;

		/** @type {Array<[number, number]>} */
		const centers = [];
		for (let i = 0; i < MAX_LEVELS; i++) {
			const level = this.levels[i];
			if (i >= count) {
				level.mesh.visible = false;
				continue;
			}
			const s = BASE_SPACING * Math.pow(2, first + i);
			const cx = Math.round(camEast / (2 * s)) * 2 * s;
			const cy = Math.round(camNorth / (2 * s)) * 2 * s;
			centers.push([cx, cy]);
			level.center.set(cx - originEast, cy - originNorth);
			level.spacing.value = s;
			level.mesh.visible = true;
			if (i === 0) {
				level.mesh.geometry = this.geometries.full;
			} else {
				const [px, py] = centers[i - 1];
				const ox = Math.round((px - cx) / s);
				const oz = Math.round((py - cy) / s);
				level.mesh.geometry = this.geometries.rings[(oz + 1) * 3 + (ox + 1)];
			}
		}
	}

	dispose() {
		for (const level of this.levels) level.material.dispose();
		this.geometries.full.dispose();
		for (const g of this.geometries.rings) g.dispose();
	}
}
