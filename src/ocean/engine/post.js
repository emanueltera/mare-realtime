/**
 * The final pass: everything the camera sees, composed into the canvas.
 *
 * Inputs are the main pass's three targets (linear radiance; normal + foam; height,
 * vertical velocity, range and the water mask), the aircraft layer (premultiplied
 * radiance) and the overlay layer (premultiplied, already in display space). It tone
 * maps the radiance, draws the decals that lie on the water — the metric grid and the
 * gimbal camera's footprint, positioned by reconstructing each pixel's point on the sea
 * from its range — and builds the label views from the ground-truth channels with
 * ramps taken from `channels.js`, for the whole frame or right of the split divider.
 * Mode changes crossfade. Lens glare (`bloom.js`) is added before tone mapping. Output is
 * sRGB with a little dither against banding.
 */
import * as THREE from 'three';
import { channels } from '../channels.js';
import { FullscreenPass } from './gpu.js';

/** Render modes in shader order. */
export const MODES = /** @type {const} */ (['shaded', 'height', 'velocity', 'normals', 'foam', 'depth']);

/** @param {import('../scene.js').RenderMode} mode */
export const modeIndex = (mode) => Math.max(0, MODES.indexOf(mode));

/**
 * Ramps for every channel, sampled linearly in sRGB between evenly spaced stops:
 * one 256-texel row per channel, rows in MODES order from 'height'.
 */
function createRampTexture() {
	const width = 256;
	const rows = MODES.length - 1;
	const data = new Uint8Array(width * rows * 4);
	MODES.slice(1).forEach((mode, row) => {
		const ch = channels.find((c) => c.id === mode);
		const stops = (ch?.stops ?? ['#000000', '#ffffff']).map((hex) => {
			const n = parseInt(hex.slice(1), 16);
			return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
		});
		for (let x = 0; x < width; x++) {
			const t = (x / (width - 1)) * (stops.length - 1);
			const i = Math.min(stops.length - 2, Math.floor(t));
			const f = t - i;
			for (let c = 0; c < 3; c++) data[(row * width + x) * 4 + c] = Math.round(stops[i][c] + (stops[i + 1][c] - stops[i][c]) * f);
			data[(row * width + x) * 4 + 3] = 255;
		}
	});
	const texture = new THREE.DataTexture(data, width, rows, THREE.RGBAFormat, THREE.UnsignedByteType);
	texture.minFilter = THREE.LinearFilter;
	texture.magFilter = THREE.LinearFilter;
	texture.needsUpdate = true;
	return texture;
}

/** Value ranges per mode (index 1…5), from channels.js. */
function channelRanges() {
	return MODES.slice(1).map((mode) => {
		const ch = channels.find((c) => c.id === mode);
		return new THREE.Vector2(...(ch?.range ?? [0, 1]));
	});
}

const FRAGMENT = /* glsl */ `
uniform sampler2D uColor;
uniform sampler2D uData0;
uniform sampler2D uData1;
uniform sampler2D uAir;
uniform sampler2D uUi;
uniform sampler2D uBloom;
uniform float uBloomStrength;
uniform float uAirExposure;   // exposure trim on the airframe, so a white one keeps its shading
uniform sampler2D uRamp;
uniform vec2 uRanges[5];
uniform float uExposure;
uniform int uModeA;
uniform int uModeB;
uniform float uModeMix;
uniform int uSplitA;
uniform int uSplitB;
uniform float uSplitModeMix;
uniform float uSplitPos;
uniform float uSplitAmount;
uniform vec2 uResolution;
uniform float uTime;
uniform float uDither;
uniform int uToneMapper;
// Decals on the water.
uniform vec2 uTanHalf;
uniform mat4 uCameraWorld;
uniform sampler2D uAdapted;
uniform float uKey;
uniform vec3 uCameraPos;
uniform float uGridAmount;
uniform vec2 uGridShift;
uniform float uFootAmount;
uniform mat4 uFootViewProj;
uniform float uFootScan;
uniform vec3 uFootOrigin;
uniform vec3 uAccent;
in vec2 vUv;
out vec4 fragColor;

vec3 agx(vec3 v) {
	// Troy Sobotka's AgX, as in three.js (base look), on linear sRGB.
	const mat3 inset = mat3(0.856627153315983, 0.137318972929847, 0.11189821299995, 0.0951212405381588, 0.761241990602591,
		0.0767994186031903, 0.0482516061458583, 0.101439036467562, 0.811302321396782);
	const mat3 outset = mat3(1.1271005818144368, -0.1413297634984383, -0.14132976349843826, -0.11060664309660323,
		1.157823702216272, -0.11060664309660294, -0.016493938717834573, -0.016493938717834257, 1.2519364065950405);
	const float minEv = -12.47393;
	const float maxEv = 4.026069;
	v = inset * v;
	v = clamp(log2(max(v, vec3(1e-10))), minEv, maxEv);
	v = (v - minEv) / (maxEv - minEv);
	vec3 x2 = v * v;
	vec3 x4 = x2 * x2;
	v = 15.5 * x4 * x2 - 40.14 * x4 * v + 31.96 * x4 - 6.868 * x2 * v + 0.4298 * x2 + 0.1191 * v - 0.00232;
	v = outset * v;
	v = pow(max(vec3(0.0), v), vec3(2.2));
	return clamp(v, 0.0, 1.0);
}

vec3 aces(vec3 color) {
	const mat3 ACESInputMat = mat3(0.59719, 0.07600, 0.02840, 0.35458, 0.90834, 0.13383, 0.04823, 0.01566, 0.83777);
	const mat3 ACESOutputMat = mat3(1.60475, -0.10208, -0.00327, -0.53108, 1.10813, -0.07276, -0.07367, -0.00605, 1.07602);
	color = ACESInputMat * color;
	vec3 a = color * (color + 0.0245786) - 0.000090537;
	vec3 b = color * (0.983729 * color + 0.4329510) + 0.238081;
	color = ACESOutputMat * (a / b);
	return clamp(color, 0.0, 1.0);
}

float exposureNow = 1.0;

vec3 tonemap(vec3 linear) {
	vec3 c = max(linear * uExposure * exposureNow, vec3(0.0));
	return uToneMapper == 1 ? agx(c) : aces(c / 0.6);
}

vec3 toSrgb(vec3 c) {
	return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}

vec3 ramp(int row, float t) {
	return texture(uRamp, vec2(clamp(t, 0.0, 1.0) * (255.0 / 256.0) + 0.5 / 256.0, (float(row) + 0.5) / 5.0)).rgb;
}

// A label channel in display space; the sky is a flat dataset mask, the aircraft a
// flat neutral silhouette.
vec3 channelColor(int mode, vec4 d0, vec4 d1, float airA) {
	vec3 c;
	if (d1.w < 0.5) {
		c = vec3(0.016, 0.03, 0.047);
	} else if (mode == 3) {
		vec3 n = normalize(d0.xyz);
		c = 0.5 + 0.5 * n;
	} else {
		int row = mode - 1;
		float v = mode == 1 ? d1.x : mode == 2 ? d1.y : mode == 4 ? d0.w : d1.z;
		vec2 r = uRanges[row];
		c = ramp(row, (v - r.x) / (r.y - r.x));
	}
	return mix(c, vec3(0.6, 0.64, 0.68), airA);
}

float gridLine(float coord, float spacing, float width) {
	float w = fwidth(coord);
	float d = abs(fract(coord / spacing + 0.5) - 0.5) * spacing;
	return 1.0 - smoothstep(width * w * 0.5, width * w * 0.5 + w * 1.2, d);
}

void main() {
	vec2 uv = vUv;
	vec4 col = texture(uColor, uv);
	vec4 d0 = texture(uData0, uv);
	vec4 d1 = texture(uData1, uv);
	vec4 air = texture(uAir, uv);
	vec4 ui = texture(uUi, uv);

	// Auto-exposure from the adapted log-average luminance, only partly compensated so
	// dusk stays dusky and noon stays bright.
	float avgLog = texture(uAdapted, vec2(0.5)).r;
	exposureNow = uKey * exp2(-avgLog * 0.86);
	vec3 glare = texture(uBloom, uv).rgb * uBloomStrength;
	vec3 shaded = tonemap(col.rgb + glare);

	// The pixel's point on the sea, from its range along the view ray.
	vec3 viewRay = vec3((uv * 2.0 - 1.0) * uTanHalf, -1.0);
	vec3 rayDir = normalize(mat3(uCameraWorld) * viewRay);
	vec3 P = uCameraPos + rayDir * d1.z;
	bool water = d1.w > 0.5;

	if (water && uGridAmount > 0.001) {
		vec2 g = vec2(P.x, -P.z) + uGridShift;
		float dist = d1.z;
		vec2 fw = fwidth(g);
		float px = max(fw.x, fw.y);
		// Minor lines vanish before they crowd (under ~9 px apart), major ones under ~6 px.
		float minorFade = 1.0 - smoothstep(20.0 / 16.0, 20.0 / 9.0, px);
		float majorFade = 1.0 - smoothstep(100.0 / 14.0, 100.0 / 6.0, px);
		float major = max(gridLine(g.x, 100.0, 0.9), gridLine(g.y, 100.0, 0.9));
		float minor = max(gridLine(g.x, 20.0, 0.6), gridLine(g.y, 20.0, 0.6));
		float range = max(uCameraPos.y, 50.0);
		float fadeDist = 1.0 - smoothstep(range * 1.6, range * 4.5, dist);
		float a = (major * 0.075 * majorFade + minor * 0.035 * minorFade) * fadeDist * uGridAmount;
		shaded = mix(shaded, vec3(0.6, 0.84, 0.92), a);
	}

	if (water && uFootAmount > 0.001) {
		vec4 clip = uFootViewProj * vec4(P, 1.0);
		if (clip.w > 2.0) {
			vec2 ndc = clip.xy / clip.w;
			float m = max(abs(ndc.x), abs(ndc.y));
			float aa = fwidth(m);
			if (aa < 0.25 && m < 1.6) {
				aa = max(aa, 1e-4);
				float inside = 1.0 - smoothstep(1.0 - aa, 1.0 + aa, m);
				float edge = 1.0 - smoothstep(aa * 0.6, aa * 1.8, abs(m - 1.0));
				float glow = exp(-abs(m - 1.0) / (aa * 6.0)) * 0.22;
				float scanD = abs(ndc.y - uFootScan);
				float scan = inside * (1.0 - smoothstep(0.0, max(fwidth(ndc.y) * 1.4, 1e-4), scanD)) * 0.7
					+ inside * exp(-max(0.0, uFootScan - ndc.y) * 9.0) * step(ndc.y, uFootScan) * 0.12;
				float fill = inside * 0.07;
				float a = clamp(fill + edge * 0.85 + glow + scan, 0.0, 1.0) * uFootAmount;
				shaded = mix(shaded, uAccent, a);
			}
		}
	}

	shaded = shaded * (1.0 - ui.a) + ui.rgb;
	vec3 airShaded = air.a > 0.0 ? tonemap(air.rgb / air.a * uAirExposure + glare) : vec3(0.0);
	shaded = mix(shaded, airShaded, air.a);

	vec3 shadedSrgb = toSrgb(shaded);
	vec3 left = shadedSrgb;
	if (uModeA != 0 || uModeB != 0) {
		vec3 a = uModeA == 0 ? shadedSrgb : channelColor(uModeA, d0, d1, air.a);
		vec3 b = uModeB == 0 ? shadedSrgb : channelColor(uModeB, d0, d1, air.a);
		left = mix(a, b, uModeMix);
	}
	vec3 outCol = left;
	if (uSplitAmount > 0.001) {
		vec3 a = channelColor(max(uSplitA, 1), d0, d1, air.a);
		vec3 b = channelColor(max(uSplitB, 1), d0, d1, air.a);
		vec3 right = mix(a, b, uSplitModeMix);
		float px = uv.x * uResolution.x;
		float divider = uSplitPos * uResolution.x;
		float side = smoothstep(divider - 0.5, divider + 0.5, px);
		outCol = mix(left, mix(left, right, uSplitAmount), side);
		float line = 1.0 - smoothstep(0.6, 1.6, abs(px - divider));
		float shadow = (1.0 - smoothstep(1.0, 7.0, abs(px - divider))) * 0.35;
		outCol = mix(outCol, outCol * 0.4, shadow * uSplitAmount);
		outCol = mix(outCol, vec3(0.93, 0.97, 1.0), line * uSplitAmount);
	}
	// Triangular dither, ±1 LSB (off for label channels written out as data).
	vec2 seed = gl_FragCoord.xy + fract(uTime) * 61.0;
	float n1 = fract(sin(dot(seed, vec2(12.9898, 78.233))) * 43758.5453);
	float n2 = fract(sin(dot(seed, vec2(39.3468, 11.135))) * 24634.6345);
	outCol += (n1 + n2 - 1.0) / 255.0 * uDither;
	fragColor = vec4(outCol, 1.0);
}`;

const LUMINANCE = /* glsl */ `
uniform sampler2D uColor;
in vec2 vUv;
out vec4 fragColor;
void main() {
	vec3 c = texture(uColor, vUv).rgb;
	float L = dot(c, vec3(0.2126, 0.7152, 0.0722));
	vec2 d = (vUv - 0.5) * vec2(1.6, 1.0);
	float w = 1.0 / (1.0 + 3.0 * dot(d, d));
	fragColor = vec4(log2(clamp(L, 1e-5, 3.0)) * w, w, 0.0, 1.0);
}`;

const ADAPT = /* glsl */ `
uniform sampler2D uLum;
uniform sampler2D uPrev;
uniform float uLevel;
uniform float uK;
out vec4 fragColor;
void main() {
	vec2 m = textureLod(uLum, vec2(0.5), uLevel).rg;
	float avg = m.x / max(m.y, 1e-4);
	float prev = texelFetch(uPrev, ivec2(0), 0).r;
	fragColor = vec4(mix(prev, avg, uK), 0.0, 0.0, 1.0);
}`;

/**
 * Auto-exposure: the frame's centre-weighted log-average luminance, adapted over
 * about half a second like a camera's meter.
 */
export class Exposure {
	/** @param {THREE.WebGLRenderer} renderer */
	constructor(renderer) {
		this.renderer = renderer;
		this.lum = new THREE.WebGLRenderTarget(128, 72, {
			type: THREE.HalfFloatType,
			format: THREE.RGBAFormat,
			minFilter: THREE.LinearMipmapLinearFilter,
			magFilter: THREE.LinearFilter,
			generateMipmaps: true,
			depthBuffer: false
		});
		/** @returns {THREE.WebGLRenderTarget} */
		const tiny = () =>
			new THREE.WebGLRenderTarget(1, 1, {
				type: THREE.FloatType,
				format: THREE.RGBAFormat,
				minFilter: THREE.NearestFilter,
				magFilter: THREE.NearestFilter,
				depthBuffer: false
			});
		this.adapted = [tiny(), tiny()];
		this.current = 0;
		this.lumPass = new FullscreenPass(LUMINANCE, { uColor: { value: null } });
		this.adaptPass = new FullscreenPass(ADAPT, {
			uLum: { value: this.lum.texture },
			uPrev: { value: null },
			uLevel: { value: 7 },
			uK: { value: 1 }
		});
		this.fresh = true;
	}

	/**
	 * @param {THREE.Texture} color the frame's linear radiance
	 * @param {number} dt seconds
	 * @param {number} [tau] adaptation time constant
	 */
	update(color, dt, tau = 0.45) {
		const r = this.renderer;
		this.lumPass.uniforms.uColor.value = color;
		this.lumPass.render(r, this.lum);
		const next = 1 - this.current;
		this.adaptPass.uniforms.uPrev.value = this.adapted[this.current].texture;
		this.adaptPass.uniforms.uK.value = this.fresh ? 1 : 1 - Math.exp(-dt / tau);
		this.adaptPass.render(r, this.adapted[next]);
		this.current = next;
		this.fresh = false;
		return this.adapted[this.current].texture;
	}

	dispose() {
		this.lum.dispose();
		for (const t of this.adapted) t.dispose();
		this.lumPass.dispose();
		this.adaptPass.dispose();
	}
}

export class PostPass {
	constructor() {
		this.ramp = createRampTexture();
		this.pass = new FullscreenPass(FRAGMENT, {
			uColor: { value: null },
			uData0: { value: null },
			uData1: { value: null },
			uAir: { value: null },
			uUi: { value: null },
			uBloom: { value: null },
			uBloomStrength: { value: 0 },
			uAirExposure: { value: 1 },
			uRamp: { value: this.ramp },
			uRanges: { value: channelRanges() },
			uExposure: { value: 1 },
			uModeA: { value: 0 },
			uModeB: { value: 0 },
			uModeMix: { value: 1 },
			uSplitA: { value: 1 },
			uSplitB: { value: 1 },
			uSplitModeMix: { value: 1 },
			uSplitPos: { value: 0.5 },
			uSplitAmount: { value: 0 },
			uResolution: { value: new THREE.Vector2(1, 1) },
			uTime: { value: 0 },
			uDither: { value: 1 },
			uToneMapper: { value: 0 },
			uTanHalf: { value: new THREE.Vector2(1, 1) },
			uCameraWorld: { value: new THREE.Matrix4() },
			uAdapted: { value: null },
			uKey: { value: 0.27 },
			uCameraPos: { value: new THREE.Vector3() },
			uGridAmount: { value: 0 },
			uGridShift: { value: new THREE.Vector2() },
			uFootAmount: { value: 0 },
			uFootViewProj: { value: new THREE.Matrix4() },
			uFootScan: { value: 0 },
			uFootOrigin: { value: new THREE.Vector3() },
			uAccent: { value: new THREE.Vector3(0.31, 0.847, 0.902) }
		});
		this.uniforms = this.pass.uniforms;
	}

	/** @param {THREE.WebGLRenderer} renderer */
	render(renderer) {
		this.pass.render(renderer, null);
	}

	dispose() {
		this.ramp.dispose();
		this.pass.dispose();
	}
}
