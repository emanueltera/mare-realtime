/**
 * The sky: a sky-view LUT rendered from the atmosphere model whenever the light or
 * the camera altitude changes, a cloud deck for overcast scenes, the sun's disc, and
 * the GLSL every surface uses to look the sky up — the dome, the sea's reflections and
 * the far ocean beyond the mesh — so the reflection always matches the sky above it.
 *
 * Also here: the water's optics (Fresnel, Bruneton-style sun glitter with the slope
 * covariance, upwelling body colour, backlit crests, foam), shared by the surface and
 * by the analytic far ocean, and the PMREM environment the aircraft reflects.
 */
import * as THREE from 'three';
import { ATMOSPHERE_GLSL, CLOUD_ALTITUDE, MS_FACTOR, MS_LIFT, MS_MU, R_GROUND, SUN_E0, TWILIGHT_GLOW } from './atmosphere.js';
import { FullscreenPass } from './gpu.js';

export const LUT_WIDTH = 256;
export const LUT_HEIGHT = 144;
/** The cloud texture tiles every this many metres (a multiple of every octave). */
export const CLOUD_PERIOD = 24576;

const LUT_FRAGMENT = /* glsl */ `
${ATMOSPHERE_GLSL}
const float MS_LIFT = ${MS_LIFT.toFixed(1)};
const float MS_MU = ${MS_MU.toFixed(3)};
const float TWILIGHT_GLOW = ${TWILIGHT_GLOW.toFixed(4)};
uniform float uSunElevation;  // radians
uniform float uViewAlt;
uniform float uHazeB;
uniform float uDip;
uniform vec3 uSunE0;
uniform float uMs;
in vec2 vUv;
out vec4 fragColor;

void main() {
	float phi = vUv.x * PI;
	float vv = vUv.y * 2.0 - 1.0;
	float l = sign(vv) * vv * vv * (PI * 0.5);
	float e = l - uDip;
	vec3 dir = vec3(cos(e) * sin(phi), sin(e), cos(e) * cos(phi));
	vec3 sun = vec3(0.0, sin(uSunElevation), cos(uSunElevation));

	float r0 = R_GROUND + max(uViewAlt, 1.0);
	float b = r0 * dir.y;
	float tTop = -b + sqrt(max(0.0, b * b - (r0 * r0 - R_TOP * R_TOP)));
	float discG = b * b - (r0 * r0 - R_GROUND * R_GROUND);
	float tMax = tTop;
	if (discG > 0.0) {
		float tg = -b - sqrt(discG);
		if (tg > 0.0) tMax = min(tMax, tg);
	}
	float nu = dot(dir, sun);
	float pr = phaseRayleigh(nu);
	float pm = phaseMie(nu, 0.78);
	vec3 L = vec3(0.0);
	vec3 T = vec3(1.0);
	const int STEPS = 40;
	float tPrev = 0.0;
	for (int i = 0; i < STEPS; i++) {
		float f = (float(i) + 0.5) / float(STEPS);
		float t = tMax * f * f;
		float f1 = float(i + 1) / float(STEPS);
		float dt = tMax * f1 * f1 - tPrev;
		tPrev += dt;
		vec3 p = vec3(0.0, r0, 0.0) + dir * t;
		float r = length(p);
		float h = max(0.0, r - R_GROUND);
		float mu = dot(p, sun) / r;
		float dR = exp(-h / H_R);
		float dM = exp(-h / H_M);
		float dH = exp(-h / H_HAZE);
		float dO = max(0.0, 1.0 - abs(h - 25000.0) / 15000.0);
		vec3 ts = transmittanceToSpace(h, mu, uHazeB);
		// Multiple scattering, isotropic. While the sun is up at a point, light scattered
		// more than once reaches it mostly from the sky above, lit by a sun that has crossed
		// far less air, so it is driven by the sun as seen from up to MS_LIFT higher and no
		// lower than MS_MU (without this a low sun leaves the sky away from it an olive mix
		// of reddened light and blue). As the sun sets at the point that fades out, so
		// sunset and twilight light stay reddened; in the earth's shadow a faint bluish
		// glow from the lit sky above remains. Weighted toward the Rayleigh coefficient.
		float lift = (0.4 + 0.6 * smoothstep(0.0, 0.2, mu)) * smoothstep(-0.05, 0.02, mu);
		vec3 msSun = transmittanceToSpace(h + MS_LIFT * lift, max(mu, MS_MU * lift), uHazeB) * smoothstep(-0.1, 0.12, mu)
			+ vec3(0.3, 0.45, 1.0) * TWILIGHT_GLOW * smoothstep(-0.35, 0.0, mu) * (1.0 - smoothstep(-0.05, 0.1, mu));
		vec3 sR = RAYLEIGH * dR;
		float sM = MIE_S * dM + 0.95 * uHazeB * dH;
		vec3 ext = sR + vec3(MIE_E * dM + uHazeB * dH) + OZONE * dO;
		vec3 S = ((sR * pr + vec3(sM * pm)) * ts + (sR + vec3(0.3 * sM)) * uMs * msSun / (4.0 * PI)) * uSunE0;
		vec3 segT = exp(-ext * dt);
		L += T * (S - S * segT) / max(ext, vec3(1e-12));
		T *= segT;
	}
	fragColor = vec4(L, dot(T, vec3(1.0 / 3.0)));
}`;

/**
 * Uniforms every sky-aware material shares (the objects themselves are shared, so
 * one update reaches the dome, the sea and the environment map).
 */
export function createSkyUniforms() {
	return {
		uSkyLut: { value: /** @type {THREE.Texture | null} */ (null) },
		uSunDir: { value: new THREE.Vector3(0, 0.5, -0.8).normalize() },
		uDip: { value: 0 },
		uSunE: { value: new THREE.Vector3(1, 1, 1) },
		uSunEClear: { value: new THREE.Vector3(1, 1, 1) },
		uSkyE: { value: new THREE.Vector3(0.1, 0.12, 0.15) },
		uCloudBase: { value: new THREE.Vector3(0.1, 0.1, 0.1) },
		uCloudCover: { value: 0 },
		uHazeB: { value: 1e-5 },
		uCamAlt: { value: 10 },
		uNoiseTex: { value: /** @type {THREE.Texture | null} */ (null) },
		uCloudShift: { value: new THREE.Vector2() },
		uSunDiscRadiance: { value: new THREE.Vector3() },
		uSunVisible: { value: 1 },
		uWaterColor: { value: new THREE.Vector3(0.0011, 0.0068, 0.0165) },
		uSssColor: { value: new THREE.Vector3(0.06, 0.42, 0.36) },
		uFoamTex: { value: /** @type {THREE.Texture | null} */ (null) },
		uFarVariance: { value: new THREE.Vector3(0.01, 0.01, 0) },
		uHs: { value: 1 }
	};
}

/** @typedef {ReturnType<typeof createSkyUniforms>} SkyUniforms */

export const SKY_GLSL = /* glsl */ `
${ATMOSPHERE_GLSL}
uniform sampler2D uSkyLut;
uniform vec3 uSunDir;
uniform float uDip;
uniform vec3 uSunE;
uniform vec3 uSunEClear;
uniform vec3 uSkyE;
uniform vec3 uCloudBase;
uniform float uCloudCover;
uniform float uHazeB;
uniform float uCamAlt;
uniform sampler2D uNoiseTex;
uniform vec2 uCloudShift;
uniform vec3 uSunDiscRadiance;
uniform float uSunVisible;  // fraction of the sky's area where the sun shows through the deck
const float CLOUD_ALT = ${CLOUD_ALTITUDE.toFixed(1)};
const float CLOUD_PERIOD = ${CLOUD_PERIOD.toFixed(1)};

// In-scattered radiance toward the camera along dir (rgb) and the path's mean
// transmittance (a): the full atmosphere above the horizon, the path to the sea below it.
vec4 skyLut(vec3 dir) {
	float e = asin(clamp(dir.y, -1.0, 1.0));
	float l = e + uDip;
	float v = 0.5 + 0.5 * sign(l) * sqrt(min(abs(l), PI * 0.5) / (PI * 0.5));
	vec2 dh = dir.xz;
	vec2 sh = uSunDir.xz;
	float u = 0.0;
	float ld = length(dh);
	float ls = length(sh);
	if (ld > 1e-4 && ls > 1e-4) u = acos(clamp(dot(dh, sh) / (ld * ls), -1.0, 1.0)) / PI;
	return texture(uSkyLut, vec2(u, v));
}

float cloudField(vec2 en, float lod) {
	vec2 uv = (en + uCloudShift) / CLOUD_PERIOD;
	float n = textureLod(uNoiseTex, uv, lod).r * 0.5;
	n += textureLod(uNoiseTex, uv * 3.0 + vec2(0.37, 0.11), lod).g * 0.3;
	n += textureLod(uNoiseTex, uv * 9.0 + vec2(0.71, 0.53), lod + 0.5).b * 0.14;
	n += textureLod(uNoiseTex, uv * 27.0 + vec2(0.13, 0.91), lod + 1.0).a * 0.06;
	return n;
}

float cloudDistance(vec3 origin, vec3 dir) {
	float r0 = R_GROUND + max(origin.y, 0.0);
	float rc = R_GROUND + CLOUD_ALT;
	float mu = dir.y;
	float disc = r0 * r0 * (mu * mu - 1.0) + rc * rc;
	if (disc < 0.0 || origin.y > CLOUD_ALT) return -1.0;
	return -r0 * mu + sqrt(disc);
}

float cloudDensity(vec2 en, float lod) {
	// The deck thickens and opens over tens of kilometres, so the cells never look tiled.
	float broad = textureLod(uNoiseTex, (en + uCloudShift * 0.6) / (CLOUD_PERIOD * 3.0) + vec2(0.57, 0.19), lod + 1.0).r;
	float cover = clamp(uCloudCover + (broad - 0.5) * 0.7 * (1.0 - uCloudCover * uCloudCover), 0.0, 1.0);
	float n = cloudField(en, lod);
	float edge = 1.0 - cover * 0.95 - 0.1;
	return smoothstep(edge, edge + 0.16 + 0.12 * cover, n);
}

// The cloud deck along a ray: rgb radiance arriving at the origin from the deck (with
// the haze in front of it) and its coverage in a.
vec4 cloudLayer(vec3 origin, vec3 dir, float lod) {
	if (uCloudCover < 0.005) return vec4(0.0);
	float t = cloudDistance(origin, dir);
	if (t <= 0.0) return vec4(0.0);
	vec3 p = origin + dir * t;
	float lodT = lod + log2(1.0 + t / 3500.0);
	float d = cloudDensity(vec2(p.x, -p.z), lodT);
	if (d <= 0.0) return vec4(0.0);
	float nu = dot(dir, uSunDir);
	// Thickness from a second, finer read of the field: bases darken where the deck is deep.
	float detail = textureLod(uNoiseTex, (vec2(p.x, -p.z) + uCloudShift) / CLOUD_PERIOD * 11.0 + 0.23, lodT).b;
	float thick = clamp(d * (0.45 + 0.55 * uCloudCover) * (0.75 + 0.5 * detail), 0.0, 1.0);
	// Bases lit diffusely through the deck; thin edges glow toward the sun.
	vec3 col = uCloudBase * mix(1.35, 0.42, thick)
		+ uSunEClear * max(uSunDir.y, 0.0) * (1.0 - thick) * (0.08 + 1.1 * phaseMie(nu, 0.62)) * (1.0 - 0.6 * uCloudCover);
	vec3 T = exp(-pathOpticalDepth(origin.y, CLOUD_ALT, t, uHazeB));
	vec4 sky = skyLut(dir);
	col = col * T + sky.rgb * (1.0 - T);
	float alpha = d * smoothstep(-0.01, 0.05, dir.y) * (0.75 + 0.25 * uCloudCover);
	return vec4(col, alpha);
}

// How much direct sun a full patch of the deck stops: broken cloud lets some through
// its thin edges, a closed deck hides the sun.
float deckOpacity() {
	return mix(0.75, 0.985, smoothstep(0.5, 1.0, uCloudCover));
}

// Fraction of direct sun reaching a point through the deck.
float cloudShadow(vec3 p) {
	if (uCloudCover < 0.005 || uSunDir.y < 0.02) return 1.0;
	float t = (CLOUD_ALT - p.y) / uSunDir.y;
	vec3 q = p + uSunDir * t;
	float d = cloudDensity(vec2(q.x, -q.z), 2.0);
	return 1.0 - d * deckOpacity();
}

// Sky radiance (no sun disc) seen from origin along dir; lod blurs the clouds for
// rough reflections.
vec3 skyRadiance(vec3 origin, vec3 dir, float lod) {
	vec4 lut = skyLut(dir);
	vec4 cl = cloudLayer(origin, dir, lod);
	return mix(lut.rgb, cl.rgb, cl.a);
}

vec3 sunDisc(vec3 dir) {
	float c = dot(dir, uSunDir);
	float cosR = 0.999989;  // 0.27°
	if (c < cosR - 2e-6) return vec3(0.0);
	float r = sqrt(max(0.0, 1.0 - c * c)) / sqrt(1.0 - cosR * cosR);
	float mu = sqrt(max(0.0, 1.0 - r * r));
	vec3 limb = vec3(0.4, 0.5, 0.6) + vec3(0.6, 0.5, 0.4) * pow(vec3(mu), vec3(0.48, 0.66, 0.85));
	return uSunDiscRadiance * limb * smoothstep(1.0, 0.9, r);
}
`;

export const WATER_GLSL = /* glsl */ `
uniform vec3 uWaterColor;
uniform vec3 uSssColor;
uniform sampler2D uFoamTex;
uniform vec3 uFarVariance;
uniform float uHs;

// Albedo of the foam being shaded, 0–1 of dense fresh foam's; set by the surface.
float gFoamAlbedo = 1.0;

float erfcApprox(float x) { return 2.0 * exp(-x * x) / (2.319 * x + sqrt(4.0 + 1.52 * x * x)); }

float smithLambda(float cosTheta, float sigmaSq) {
	float c = clamp(cosTheta, 1e-3, 0.9999);
	float v = c / sqrt(max(1e-7, (1.0 - c * c) * 2.0 * sigmaSq));
	return max(0.0, (exp(-v * v) - v * sqrt(PI) * erfcApprox(v)) / (2.0 * v * sqrt(PI)));
}

float meanFresnel(float cosThetaV, float sigmaV) {
	return pow(1.0 - clamp(cosThetaV, 0.0, 1.0), 5.0 * exp(-2.69 * sigmaV)) / (1.0 + 22.7 * pow(sigmaV, 1.5));
}

// Variance of the slope distribution along a horizontal direction (east, north).
float directionalVariance(vec2 d, vec3 sig) {
	float l2 = dot(d, d);
	if (l2 < 1e-8) return 0.5 * (sig.x + sig.y);
	return (d.x * d.x * sig.x + 2.0 * d.x * d.y * sig.z + d.y * d.y * sig.y) / l2;
}

// Sun glitter, after Bruneton, Neyret & Holzschuch 2010, with the full slope covariance
// sig = (var_e, var_n, cov_en) instead of wind-aligned axes. Multiply by the sun's irradiance.
float sunGlitter(vec3 L, vec3 V, vec3 N, vec3 sig) {
	vec3 Te = normalize(vec3(1.0, 0.0, 0.0) - N * N.x);
	vec3 Tn = cross(N, Te);
	vec3 H = normalize(L + V);
	float zH = max(dot(H, N), 1e-3);
	float zL = dot(L, N);
	float zV = dot(V, N);
	if (zL <= 0.0) return 0.0;
	zV = max(zV, 0.01);
	vec2 zeta = vec2(dot(H, Te), dot(H, Tn)) / zH;
	float det = max(sig.x * sig.y - sig.z * sig.z, 1e-9);
	float q = (zeta.x * zeta.x * sig.y - 2.0 * zeta.x * zeta.y * sig.z + zeta.y * zeta.y * sig.x) / det;
	float p = exp(-0.5 * q) / (2.0 * PI * sqrt(det));
	float sV = directionalVariance(vec2(dot(V, Te), dot(V, Tn)), sig);
	float sL = directionalVariance(vec2(dot(L, Te), dot(L, Tn)), sig);
	float fresnel = 0.02 + 0.98 * pow(1.0 - max(dot(V, H), 0.0), 5.0);
	float zH2 = zH * zH;
	return fresnel * p / ((1.0 + smithLambda(zL, sL) + smithLambda(zV, sV)) * zV * zH2 * zH2 * 4.0);
}

struct WaterSample {
	vec3 radiance;
	float fresnel;
};

// Radiance leaving the sea toward the viewer at P (world), before the haze in between.
// crest: height above mean / (Hs/2), roughly −1…1; foam: visible foam 0–1.
vec3 shadeWater(vec3 P, vec3 V, vec3 N, vec3 sig, float crest, float foam, float shadow) {
	vec3 L = uSunDir;
	float zV = max(dot(V, N), 1e-3);
	float sV = directionalVariance(vec2(dot(V, vec3(1.0, 0.0, 0.0)), dot(V, vec3(0.0, 0.0, -1.0))), sig);
	float F = 0.02 + 0.98 * meanFresnel(zV, sqrt(sV));

	vec3 R = reflect(-V, N);
	R.y = abs(R.y) * 0.9 + 0.012;
	R = normalize(R);
	float lod = clamp(log2(1.0 + sqrt(sig.x + sig.y) * 60.0), 0.0, 6.0);
	vec3 sky = skyRadiance(P, R, lod);

	vec3 sunE = uSunE * shadow;
	float glitter = sunGlitter(L, V, N, sig);
	vec3 Ed = sunE * max(L.y, 0.0) + uSkyE;

	vec3 body = uWaterColor * Ed / PI;
	// Light through the thin tops of crests, strongest looking toward the sun.
	float toward = pow(max(dot(-V, L), 0.0), 4.0);
	float thin = pow(clamp(0.5 - 0.5 * dot(L, N), 0.0, 1.0), 3.0);
	float lift = clamp(crest * 0.7 + 0.25, 0.0, 1.4);
	vec3 sss = uSssColor * (sunE * max(L.y + 0.15, 0.0)) * (lift * toward * thin * 0.42 + lift * 0.011 * pow(zV, 2.0))
		+ uSssColor * uSkyE * lift * 0.006;

	vec3 water = F * sky + sunE * glitter + (1.0 - F) * (body + sss);

	vec3 foamLight = (sunE * clamp(dot(N, L) * 0.75 + 0.25, 0.0, 1.0) * max(L.y, 0.0) * 1.2 + uSkyE) / PI;
	vec3 foamCol = vec3(0.82, 0.88, 0.9) * gFoamAlbedo * foamLight;
	return mix(water, foamCol, foam);
}

// Aerial perspective for a point at distance d along view direction dir (toward the
// point) from the camera: returns (transmittance, in-scatter) packed as rgb·T + add.
vec3 applyHaze(vec3 radiance, vec3 dir, float d, float pointAlt) {
	vec3 T = exp(-pathOpticalDepth(uCamAlt, pointAlt, d, uHazeB));
	vec4 lut = skyLut(dir);
	float tPath = dot(T, vec3(1.0 / 3.0));
	vec3 inscatter = lut.rgb * clamp((1.0 - tPath) / max(1.0 - lut.a, 1e-3), 0.0, 1.0);
	return radiance * T + inscatter;
}
`;

/**
 * Renders the sky-view LUT. `update` reruns it only when the light or the camera
 * altitude has moved enough to notice.
 */
export class SkyLut {
	/** @param {THREE.WebGLRenderer} renderer */
	constructor(renderer) {
		this.renderer = renderer;
		this.target = new THREE.WebGLRenderTarget(LUT_WIDTH, LUT_HEIGHT, {
			type: THREE.HalfFloatType,
			format: THREE.RGBAFormat,
			minFilter: THREE.LinearFilter,
			magFilter: THREE.LinearFilter,
			wrapS: THREE.ClampToEdgeWrapping,
			wrapT: THREE.ClampToEdgeWrapping,
			depthBuffer: false,
			generateMipmaps: false
		});
		this.pass = new FullscreenPass(LUT_FRAGMENT, {
			uSunElevation: { value: 0.3 },
			uViewAlt: { value: 10 },
			uHazeB: { value: 1e-5 },
			uDip: { value: 0 },
			uSunE0: { value: new THREE.Vector3(...SUN_E0) },
			uMs: { value: MS_FACTOR }
		});
		this.key = '';
	}

	/**
	 * @param {number} sunElevation radians
	 * @param {number} altitude camera altitude, m
	 * @param {number} hazeB
	 * @param {number} dip horizon dip, radians
	 */
	update(sunElevation, altitude, hazeB, dip) {
		// Altitude matters on a log scale (the path to the horizon), the rest linearly.
		const key = `${sunElevation.toFixed(4)}|${Math.log(Math.max(altitude, 1)).toFixed(2)}|${hazeB.toExponential(3)}`;
		if (key === this.key) return false;
		this.key = key;
		const u = this.pass.uniforms;
		u.uSunElevation.value = sunElevation;
		u.uViewAlt.value = altitude;
		u.uHazeB.value = hazeB;
		u.uDip.value = dip;
		const previous = this.renderer.getRenderTarget();
		this.pass.render(this.renderer, this.target);
		this.renderer.setRenderTarget(previous);
		return true;
	}

	dispose() {
		this.target.dispose();
		this.pass.dispose();
	}
}

/**
 * Horizon dip below the horizontal for a viewer at `altitude` metres, radians.
 * @param {number} altitude
 */
export function horizonDip(altitude) {
	return Math.acos(R_GROUND / (R_GROUND + Math.max(altitude, 0.01)));
}

const DOME_VERTEX = /* glsl */ `
out vec3 vDir;
void main() {
	vec4 world = modelMatrix * vec4(position, 1.0);
	vDir = world.xyz - cameraPosition;
	gl_Position = projectionMatrix * viewMatrix * world;
	gl_Position.z = gl_Position.w * 0.99999;
}`;

const ENV_FRAGMENT = /* glsl */ `
${SKY_GLSL}
${WATER_GLSL}
in vec3 vDir;
out vec4 fragColor;
void main() {
	vec3 dir = normalize(vDir);
	vec3 origin = vec3(0.0, uCamAlt, 0.0);
	vec3 col;
	if (dir.y > -0.02) {
		col = skyRadiance(origin, dir, 1.5);
	} else {
		// The sea below, as the aircraft's underside would see it: flat and far.
		vec3 V = -dir;
		float d = uCamAlt / max(-dir.y, 0.02);
		vec3 P = origin + dir * d;
		vec3 sea = shadeWater(P, V, vec3(0.0, 1.0, 0.0), uFarVariance, 0.0, 0.0, uSunVisible);
		col = applyHaze(sea, dir, d, 0.0);
	}
	fragColor = vec4(col, 1.0);
}`;

/**
 * The sky (and the sea under it) as a prefiltered environment for the aircraft's
 * reflections. Rendered into a small cube and run through PMREM when the light
 * changes noticeably.
 */
export class EnvironmentMap {
	/**
	 * @param {THREE.WebGLRenderer} renderer
	 * @param {SkyUniforms} skyUniforms
	 */
	constructor(renderer, skyUniforms) {
		this.renderer = renderer;
		this.cube = new THREE.WebGLCubeRenderTarget(64, { type: THREE.HalfFloatType, generateMipmaps: false });
		this.camera = new THREE.CubeCamera(0.1, 10, this.cube);
		this.scene = new THREE.Scene();
		this.material = new THREE.ShaderMaterial({
			glslVersion: THREE.GLSL3,
			vertexShader: DOME_VERTEX,
			fragmentShader: ENV_FRAGMENT,
			uniforms: { ...skyUniforms },
			side: THREE.BackSide,
			depthTest: false,
			depthWrite: false
		});
		this.dome = new THREE.Mesh(new THREE.SphereGeometry(5, 32, 16), this.material);
		this.dome.frustumCulled = false;
		this.scene.add(this.dome);
		this.pmrem = new THREE.PMREMGenerator(renderer);
		/** @type {THREE.WebGLRenderTarget | null} */
		this.target = null;
		this.key = '';
		this.lastTime = -Infinity;
	}

	/** The prefiltered texture, once one has been rendered. */
	get texture() {
		return this.target ? this.target.texture : null;
	}

	/**
	 * Re-render when the light has changed enough; at most every `minInterval` seconds.
	 * @param {string} key a coarse fingerprint of the light
	 * @param {number} now seconds
	 * @param {number} [minInterval]
	 */
	update(key, now, minInterval = 0.35) {
		if (key === this.key) return false;
		if (now - this.lastTime < minInterval && this.target) return false;
		this.key = key;
		this.lastTime = now;
		const previous = this.renderer.getRenderTarget();
		const toneMapping = this.renderer.toneMapping;
		this.renderer.toneMapping = THREE.NoToneMapping;
		this.camera.update(this.renderer, this.scene);
		this.target = this.pmrem.fromCubemap(this.cube.texture, this.target);
		this.renderer.toneMapping = toneMapping;
		this.renderer.setRenderTarget(previous);
		return true;
	}

	dispose() {
		this.cube.dispose();
		this.material.dispose();
		this.dome.geometry.dispose();
		this.pmrem.dispose();
		this.target?.dispose();
	}
}
