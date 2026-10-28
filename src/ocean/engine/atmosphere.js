/**
 * The atmosphere over the simulated sea: Rayleigh + Mie single scattering with ozone
 * absorption and the earth's shadow, a marine haze layer near the water, and a
 * cheap isotropic multiple-scattering term. The GPU renders it into a small
 * sky-view LUT every frame the light changes (see sky.js); this module also mirrors
 * the same physics on the CPU, coarsely, for the numbers the CPU needs: the sun's
 * colour at the camera, the sky's ambient irradiance and the haze colour toward the
 * aircraft. Units are relative: the sun's irradiance above the atmosphere is SUN_E0.
 */
import { DEG, clamp, lerp, smoothstep } from './util.js';

export const R_GROUND = 6360e3;
export const R_TOP = 6460e3;
const RAYLEIGH = [5.802e-6, 13.558e-6, 33.1e-6];
const MIE_S = 3.996e-6;
const MIE_E = 4.44e-6;
const OZONE = [0.65e-6, 1.881e-6, 0.085e-6];
const H_R = 8000;
const H_M = 1200;
const H_HAZE = 900;
/** Sun irradiance above the atmosphere, linear sRGB, relative units. */
export const SUN_E0 = [1.0, 0.98, 0.95];
/** Isotropic multiple scattering, as a fraction of single scattering's driving term. */
export const MS_FACTOR = 0.8;
/** Multiple scattering is lit by the sun as seen this many metres higher… */
export const MS_LIFT = 10000;
/** …and never lower than this zenith cosine (see the sky LUT). */
export const MS_MU = 0.3;
/** Bluish fill in the earth's shadow from the lit sky above, at twilight. */
export const TWILIGHT_GLOW = 0.0125;
/** Cloud deck altitude, m. */
export const CLOUD_ALTITUDE = 1400;

/**
 * Extra aerosol extinction at sea level for a haze setting: clean maritime air at 0
 * (visibility ~150 km), ~35 km at 0.55, ~13 km at 1.
 * @param {number} haze 0–1
 */
export function hazeBeta(haze) {
	const h = clamp(haze, 0, 1.5);
	return 2.2e-6 + h * 4.5e-5 + h * h * 2.6e-4;
}

export const ATMOSPHERE_GLSL = /* glsl */ `
const float PI = 3.141592653589793;
const float R_GROUND = ${R_GROUND.toFixed(1)};
const float R_TOP = ${R_TOP.toFixed(1)};
const vec3 RAYLEIGH = vec3(${RAYLEIGH.map((v) => v.toExponential(4)).join(', ')});
const float MIE_S = ${MIE_S.toExponential(4)};
const float MIE_E = ${MIE_E.toExponential(4)};
const vec3 OZONE = vec3(${OZONE.map((v) => v.toExponential(4)).join(', ')});
const float H_R = ${H_R.toFixed(1)};
const float H_M = ${H_M.toFixed(1)};
const float H_HAZE = ${H_HAZE.toFixed(1)};

float chapman(float x, float cosChi) {
	float c = sqrt(1.5707963 * x);
	if (cosChi >= 0.0) return c / ((c - 1.0) * cosChi + 1.0);
	float sinChi = sqrt(max(0.0, 1.0 - cosChi * cosChi));
	return c / ((c - 1.0) * cosChi - 1.0) + 2.0 * c * exp(min(x - x * sinChi, 60.0)) * sqrt(sinChi);
}

// ∫ exp(−alt/H) ds from altitude h toward zenith cosine mu, to space.
float column(float H, float h, float mu) {
	return H * exp(-h / H) * chapman((R_GROUND + h) / H, mu);
}

// Length of a ray from radius r (zenith cosine mu) inside a sphere of radius rho.
float insideSphere(float r, float mu, float rho) {
	float disc = r * r * (mu * mu - 1.0) + rho * rho;
	if (disc < 0.0) return 0.0;
	float q = sqrt(disc);
	if (r < rho) return -r * mu + q;
	return mu < 0.0 ? 2.0 * q : 0.0;
}

// Ozone along a ray: a uniform shell from 15 to 35 km at 0.75 density, the same
// 15 km column as the tent profile.
float ozoneColumn(float h, float mu) {
	float r = R_GROUND + h;
	return 0.75 * (insideSphere(r, mu, R_GROUND + 35000.0) - insideSphere(r, mu, R_GROUND + 15000.0));
}

// Transmittance from altitude h toward a direction of zenith cosine mu (the sun).
vec3 transmittanceToSpace(float h, float mu, float hazeB) {
	float r = R_GROUND + h;
	float horizon = -sqrt(max(0.0, 1.0 - (R_GROUND / r) * (R_GROUND / r)));
	if (mu < horizon) return vec3(0.0);
	vec3 tau = RAYLEIGH * column(H_R, h, mu) + vec3(MIE_E * column(H_M, h, mu) + hazeB * column(H_HAZE, h, mu))
		+ OZONE * ozoneColumn(h, mu);
	return exp(-tau);
}

float phaseRayleigh(float nu) { return 3.0 / (16.0 * PI) * (1.0 + nu * nu); }

float phaseMie(float nu, float g) {
	float g2 = g * g;
	return 3.0 / (8.0 * PI) * ((1.0 - g2) * (1.0 + nu * nu)) / ((2.0 + g2) * pow(max(1e-4, 1.0 + g2 - 2.0 * g * nu), 1.5));
}

// Extinction per metre at sea level, for a short near-horizontal path.
vec3 seaLevelExtinction(float hazeB) {
	return RAYLEIGH + vec3(MIE_E + hazeB);
}

// Optical depth of a straight path between altitudes y0 and y1 (m) of length d,
// through the exponential layers (flat-earth form, fine below ~30 km of range).
vec3 pathOpticalDepth(float y0, float y1, float d, float hazeB) {
	float dy = y1 - y0;
	vec3 tau = vec3(0.0);
	float k;
	// ∫ exp(−y/H) ds over the path = d·H·(e^{−y0/H} − e^{−y1/H}) / dy, or d·e^{−y/H} when level.
	if (abs(dy) < 1.0) {
		float y = max(0.0, 0.5 * (y0 + y1));
		tau = RAYLEIGH * exp(-y / H_R) * d + vec3(MIE_E * exp(-y / H_M) + hazeB * exp(-y / H_HAZE)) * d;
	} else {
		float a = max(0.0, y0);
		float b = max(0.0, y1);
		k = d / dy;
		tau = RAYLEIGH * (H_R * (exp(-a / H_R) - exp(-b / H_R)) * k)
			+ vec3(MIE_E * H_M * (exp(-a / H_M) - exp(-b / H_M)) * k + hazeB * H_HAZE * (exp(-a / H_HAZE) - exp(-b / H_HAZE)) * k);
	}
	return max(tau, vec3(0.0));
}
`;

/**
 * @param {number} x
 * @param {number} cosChi
 */
function chapman(x, cosChi) {
	const c = Math.sqrt(1.5707963 * x);
	if (cosChi >= 0) return c / ((c - 1) * cosChi + 1);
	const sinChi = Math.sqrt(Math.max(0, 1 - cosChi * cosChi));
	return c / ((c - 1) * cosChi - 1) + 2 * c * Math.exp(Math.min(x - x * sinChi, 60)) * Math.sqrt(sinChi);
}

/** @param {number} H @param {number} h @param {number} mu */
const column = (H, h, mu) => H * Math.exp(-h / H) * chapman((R_GROUND + h) / H, mu);

/** @param {number} r @param {number} mu @param {number} rho */
function insideSphere(r, mu, rho) {
	const disc = r * r * (mu * mu - 1) + rho * rho;
	if (disc < 0) return 0;
	const q = Math.sqrt(disc);
	if (r < rho) return -r * mu + q;
	return mu < 0 ? 2 * q : 0;
}

/** @param {number} h @param {number} mu */
function ozoneColumn(h, mu) {
	const r = R_GROUND + h;
	return 0.75 * (insideSphere(r, mu, R_GROUND + 35000) - insideSphere(r, mu, R_GROUND + 15000));
}

/**
 * Transmittance from altitude h to space along a direction of zenith cosine mu.
 * @param {number} h
 * @param {number} mu
 * @param {number} hb haze extinction at sea level
 * @returns {[number, number, number]}
 */
export function transmittanceToSpace(h, mu, hb) {
	const r = R_GROUND + h;
	const horizon = -Math.sqrt(Math.max(0, 1 - (R_GROUND / r) ** 2));
	if (mu < horizon) return [0, 0, 0];
	const cr = column(H_R, h, mu);
	const cm = MIE_E * column(H_M, h, mu) + hb * column(H_HAZE, h, mu);
	const co = ozoneColumn(h, mu);
	return rgb((i) => Math.exp(-(RAYLEIGH[i] * cr + cm + OZONE[i] * co)));
}

/** @param {number} nu */
const phaseRayleigh = (nu) => (3 / (16 * Math.PI)) * (1 + nu * nu);

/** @param {number} nu @param {number} g */
function phaseMie(nu, g) {
	const g2 = g * g;
	return ((3 / (8 * Math.PI)) * ((1 - g2) * (1 + nu * nu))) / ((2 + g2) * Math.pow(Math.max(1e-4, 1 + g2 - 2 * g * nu), 1.5));
}

/**
 * Sky radiance along a direction from a viewer at altitude h0 — the same single
 * scattering integral as the GPU LUT, with fewer steps.
 * @param {[number, number, number]} dir unit vector, y up
 * @param {[number, number, number]} sun unit vector toward the sun, y up
 * @param {number} h0 viewer altitude, m
 * @param {number} hb haze extinction at sea level
 * @param {number} [steps]
 * @returns {{ radiance: [number, number, number], transmittance: number }}
 */
export function skyRadiance(dir, sun, h0, hb, steps = 18) {
	const r0 = R_GROUND + Math.max(h0, 1);
	// Ray from (0, r0, 0) along dir: where it leaves the top or meets the ground.
	const b = r0 * dir[1];
	const cTop = r0 * r0 - R_TOP * R_TOP;
	const tTop = -b + Math.sqrt(Math.max(0, b * b - cTop));
	const cGround = r0 * r0 - R_GROUND * R_GROUND;
	const discG = b * b - cGround;
	const tGround = discG > 0 && -b - Math.sqrt(discG) > 0 ? -b - Math.sqrt(discG) : Infinity;
	const tMax = Math.min(tTop, tGround);
	const nu = dir[0] * sun[0] + dir[1] * sun[1] + dir[2] * sun[2];
	const pr = phaseRayleigh(nu);
	const pm = phaseMie(nu, 0.78);
	const L = [0, 0, 0];
	const T = [1, 1, 1];
	let tPrev = 0;
	for (let i = 0; i < steps; i++) {
		const f = (i + 0.5) / steps;
		const t = tMax * f * f;
		const dt = tMax * ((i + 1) / steps) ** 2 - tPrev;
		tPrev += dt;
		const px = dir[0] * t;
		const py = r0 + dir[1] * t;
		const pz = dir[2] * t;
		const r = Math.hypot(px, py, pz);
		const h = Math.max(0, r - R_GROUND);
		const mu = (px * sun[0] + py * sun[1] + pz * sun[2]) / r;
		const dR = Math.exp(-h / H_R);
		const dM = Math.exp(-h / H_M);
		const dH = Math.exp(-h / H_HAZE);
		const dO = Math.max(0, 1 - Math.abs(h - 25000) / 15000);
		const ts = transmittanceToSpace(h, mu, hb);
		const lift = (0.4 + 0.6 * smoothstep(0, 0.2, mu)) * smoothstep(-0.05, 0.02, mu);
		const msSun = transmittanceToSpace(h + MS_LIFT * lift, Math.max(mu, MS_MU * lift), hb);
		const msFade = smoothstep(-0.1, 0.12, mu);
		const glow = TWILIGHT_GLOW * smoothstep(-0.35, 0, mu) * (1 - smoothstep(-0.05, 0.1, mu));
		for (let c = 0; c < 3; c++) {
			const sR = RAYLEIGH[c] * dR;
			const sM = MIE_S * dM + 0.95 * hb * dH;
			const ext = sR + MIE_E * dM + hb * dH + OZONE[c] * dO;
			const single = (sR * pr + sM * pm) * ts[c];
			const source = msSun[c] * msFade + [0.3, 0.45, 1.0][c] * glow;
			const multi = ((sR + 0.3 * sM) * MS_FACTOR * source) / (4 * Math.PI);
			const segT = Math.exp(-ext * dt);
			// Energy-conserving segment integral (Hillaire 2020).
			const S = (single + multi) * SUN_E0[c];
			L[c] += (T[c] * (S - S * segT)) / Math.max(ext, 1e-12);
			T[c] *= segT;
		}
	}
	return { radiance: /** @type {[number, number, number]} */ (L), transmittance: (T[0] + T[1] + T[2]) / 3 };
}

/**
 * Everything the CPU needs to know about the light in a scene.
 * @typedef {object} LightState
 * @property {[number, number, number]} sunDir unit vector toward the sun: x east, y up, z north
 * @property {[number, number, number]} sunIrradiance at the sea surface, through the haze and the cloud deck
 * @property {[number, number, number]} sunIrradianceClear without the cloud deck (lights the cloud tops)
 * @property {[number, number, number]} skyIrradiance on a horizontal surface, from the whole sky
 * @property {[number, number, number]} cloudRadiance radiance of the cloud base overhead
 * @property {number} hazeBeta
 * @property {number} cloudCover 0–1
 * @property {number} night 0 daylight – 1 dark
 * @property {number} luminance a scalar for exposure: total horizontal irradiance
 */

/**
 * The CPU side of the light: sun colour, ambient and haze for a lighting preset.
 * @param {Required<import('../scene.js').Lighting>} light
 * @param {number} altitude camera altitude, m
 * @returns {LightState}
 */
export function computeLight(light, altitude) {
	const el = light.sunElevation * DEG;
	const az = light.sunAzimuth * DEG;
	/** @type {[number, number, number]} */
	const sunDir = [Math.cos(el) * Math.sin(az), Math.sin(el), Math.cos(el) * Math.cos(az)];
	const hb = hazeBeta(light.haze);
	const cover = cloudCover(light.overcast);
	const h = Math.max(2, Math.min(altitude, 3000));
	const tSun = transmittanceToSpace(h, sunDir[1], hb);
	// The disc sets over a third of a degree; soften it so the light doesn't snap off.
	const disc = smoothstep(-0.9 * DEG, 0.35 * DEG, el + Math.acos(R_GROUND / (R_GROUND + h)));
	const cloudT = sunVisible(cover);
	const sunClear = rgb((c) => SUN_E0[c] * tSun[c] * disc);
	const sunIrradiance = rgb((c) => sunClear[c] * cloudT);

	// Sky irradiance on a horizontal plane: ∫ L cosθ dω over the upper hemisphere.
	const sky = [0, 0, 0];
	const nEl = 6;
	const nAz = 8;
	for (let i = 0; i < nEl; i++) {
		const e = ((i + 0.5) / nEl) * (Math.PI / 2);
		const ce = Math.cos(e);
		const se = Math.sin(e);
		const dOmega = ((Math.PI / 2) / nEl) * ((2 * Math.PI) / nAz) * ce;
		for (let j = 0; j < nAz; j++) {
			const a = ((j + 0.5) / nAz) * 2 * Math.PI;
			const { radiance } = skyRadiance([ce * Math.sin(a), se, ce * Math.cos(a)], sunDir, h, hb, 12);
			for (let c = 0; c < 3; c++) sky[c] += radiance[c] * se * dOmega;
		}
	}
	// The overcast deck: lit from above by sun and sky, it glows a soft grey below.
	const above = [0, 1, 2].map((c) => (sunClear[c] * Math.max(0.05, sunDir[1]) + sky[c]) / Math.PI);
	const deckT = lerp(0.62, 0.3, cover);
	const cloudRadiance = rgb((c) => above[c] * deckT * 1.05 + 0.002);
	const skyIrradiance = rgb((c) => lerp(sky[c], Math.PI * cloudRadiance[c], cover * 0.92));
	const luminance =
		0.2126 * (sunIrradiance[0] * Math.max(0, sunDir[1]) + skyIrradiance[0]) +
		0.7152 * (sunIrradiance[1] * Math.max(0, sunDir[1]) + skyIrradiance[1]) +
		0.0722 * (sunIrradiance[2] * Math.max(0, sunDir[1]) + skyIrradiance[2]);
	const night = 1 - smoothstep(0.004, 0.06, luminance);
	return {
		sunDir,
		sunIrradiance,
		sunIrradianceClear: sunClear,
		skyIrradiance,
		cloudRadiance,
		hazeBeta: hb,
		cloudCover: cover,
		night,
		luminance
	};
}

/**
 * @param {(channel: number) => number} fn
 * @returns {[number, number, number]}
 */
function rgb(fn) {
	return [fn(0), fn(1), fn(2)];
}

/**
 * Fraction of the sky the cloud deck covers for an overcast setting: nothing below
 * ~0.25, scattered cumulus by 0.5, a closed deck at 1.
 * @param {number} overcast
 */
export function cloudCover(overcast) {
	return smoothstep(0.18, 1.0, overcast);
}

/**
 * How much direct sun a full patch of the cloud deck stops, for a cover (sky.js's
 * deckOpacity): broken cloud lets some through its thin edges, a closed deck hides the sun.
 * @param {number} cover 0–1
 */
export function deckOpacity(cover) {
	return lerp(0.75, 0.985, smoothstep(0.5, 1, cover));
}

/**
 * Fraction of direct sun reaching the sea on average under a deck of this cover.
 * @param {number} cover 0–1
 */
export function sunVisible(cover) {
	return 1 - cover * deckOpacity(cover);
}

/**
 * Horizontal-path transmittance over d metres at altitude y (flat-earth form).
 * @param {number} d
 * @param {number} y
 * @param {number} hb
 */
export function pathTransmittance(d, y, hb) {
	const g = (RAYLEIGH[1] * Math.exp(-y / H_R) + MIE_E * Math.exp(-y / H_M) + hb * Math.exp(-y / H_HAZE)) * d;
	return Math.exp(-g);
}
