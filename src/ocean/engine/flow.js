/**
 * The `flow` overlay: the simulated surface velocity made visible on the water.
 *
 * Particles live in the sea's Lagrangian frame — a texture of undisplaced positions
 * on the surface — so drawing each at its displaced position makes it ride the waves'
 * orbital motion exactly. Each is drawn as a short comet streak along the orbital
 * velocity the simulation computes (the same field the velocity channel shows), in
 * an additive cyan that brightens where water is rising. They spawn mostly inside the
 * aircraft camera's footprint, when one is drawn, so the field reads strongest on the
 * patch of sea the model is reading.
 */
import * as THREE from 'three';
import { FullscreenPass, floatTarget } from './gpu.js';

const STATE_W = 192;
const STATE_H = 128;
/** Share of the second cascade's motion drawn with the first's. */
export const FINE_SHARE = 0.35;

const UPDATE = /* glsl */ `
uniform sampler2D uState;
uniform float uDt;
uniform float uTime;
uniform vec2 uDrift;
uniform vec2 uQuad0;
uniform vec2 uQuad1;
uniform vec2 uQuad2;
uniform vec2 uQuad3;
uniform vec2 uCenter;
uniform float uRadius;
uniform float uQuadShare;
uniform vec2 uShift;
uniform float uLife;
uniform float uReset;
out vec4 fragColor;

float hash(vec2 p) {
	vec3 p3 = fract(vec3(p.xyx) * 0.1031);
	p3 += dot(p3, p3.yzx + 33.33);
	return fract((p3.x + p3.y) * p3.z);
}

void main() {
	vec2 id = gl_FragCoord.xy;
	vec4 s = texelFetch(uState, ivec2(id), 0);
	s.xy -= uShift;
	s.z += uDt;
	if (uReset > 0.5 || s.z >= s.w || s.w <= 0.0) {
		float seed = uTime * 17.13 + s.w * 3.7;
		float r1 = hash(id + seed);
		float r2 = hash(id * 1.37 + seed + 11.0);
		float r3 = hash(id * 0.71 + seed + 23.0);
		float r4 = hash(id * 2.13 + seed + 37.0);
		vec2 pos;
		if (r1 < uQuadShare) {
			vec2 a = mix(uQuad0, uQuad1, r2);
			vec2 b = mix(uQuad3, uQuad2, r2);
			pos = mix(a, b, r3);
		} else {
			float ang = r2 * 6.2831853;
			float rad = uRadius * sqrt(r3);
			pos = uCenter + vec2(cos(ang), sin(ang)) * rad;
		}
		float life = uLife * (0.55 + 0.9 * r4);
		// On a reset, start part-way through a life so the field doesn't pulse.
		float age = uReset > 0.5 ? life * hash(id + 5.0) : 0.0;
		s = vec4(pos, age, life);
	} else {
		s.xy += uDrift * uDt;
	}
	fragColor = s;
}`;

const VERTEX = /* glsl */ `
in vec2 corner;
uniform sampler2D uState;
uniform sampler2D uDisp0;
uniform sampler2D uDisp1;
uniform sampler2D uMotion0;
uniform sampler2D uMotion1;
uniform vec4 uPatch;
uniform vec2 uOffset0;
uniform vec2 uOffset1;
uniform vec2 uResolution;
uniform float uStreak;
uniform float uOpacity;
uniform vec2 uFade;
uniform mat4 uFootVP;
uniform float uHasFoot;
uniform float uFine;        // share of the second cascade's motion in the drawn field
uniform float uSpeedScale;  // 1 / (about two standard deviations of the drawn speed)
out float vAlong;
out float vAcross;
out float vAlpha;
out float vRise;

void main() {
	int id = gl_InstanceID;
	vec4 s = texelFetch(uState, ivec2(id % ${STATE_W}, id / ${STATE_W}), 0);
	vec2 p = s.xy;
	vec2 uv0 = p / uPatch.x + uOffset0;
	vec2 uv1 = p / uPatch.y + uOffset1;
	vec4 d0 = textureLod(uDisp0, uv0, 0.0);
	vec4 d1 = textureLod(uDisp1, uv1, 0.0);
	vec4 m0 = textureLod(uMotion0, uv0, 0.0);
	vec4 m1 = textureLod(uMotion1, uv1, 0.0);
	// Where the particle is: the full displaced surface. How it is drawn moving: the
	// dominant waves' orbital velocity, with a little of the shorter waves', so the field
	// reads coherently across a wavelength: water surging forward under the crests and
	// back in the troughs, rising ahead of them and falling behind.
	vec3 disp = d0.xyz + d1.xyz;
	float w = d0.w + uFine * d1.w;
	vec2 u = m0.xy + uFine * m1.xy;
	vec3 head = vec3(p.x + disp.x, disp.y, -(p.y + disp.z));
	vec3 vel = vec3(u.x, w, -u.y);
	vec3 tail = head - vel * uStreak;
	vec4 vh = modelViewMatrix * vec4(head, 1.0);
	vec4 vt = modelViewMatrix * vec4(tail, 1.0);
	float range = length(vh.xyz);
	float lift = 0.25 + range * 0.004;
	vh.xyz += normalize(-vh.xyz) * lift;
	vt.xyz += normalize(-vt.xyz) * lift;
	vec4 a = projectionMatrix * vt;
	vec4 b = projectionMatrix * vh;
	if (a.w < 0.05 || b.w < 0.05) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
	vec2 sa = (a.xy / a.w * 0.5 + 0.5) * uResolution;
	vec2 sb = (b.xy / b.w * 0.5 + 0.5) * uResolution;
	vec2 d = sb - sa;
	float len = length(d);
	vec2 dir = len > 1e-3 ? d / len : vec2(1.0, 0.0);
	vec2 nrm = vec2(-dir.y, dir.x);
	float width = clamp(1.9 - range * 0.003, 0.9, 1.9);
	float minLen = 2.0;
	vec2 sTail = len < minLen ? sb - dir * minLen : sa;
	vec2 sp = mix(sTail, sb, corner.x) + dir * (corner.x * 2.0 - 1.0) * width + nrm * corner.y * width;
	vec4 c = mix(a, b, corner.x);
	gl_Position = vec4((sp / uResolution * 2.0 - 1.0) * c.w, c.z, c.w);
	vAlong = corner.x;
	vAcross = corner.y;
	float lifeFade = sin(3.14159 * clamp(s.z / max(s.w, 1e-3), 0.0, 1.0));
	float speed = smoothstep(0.08, 1.0, length(vel) * uSpeedScale);
	// Strongest inside the aircraft camera's footprint, a faint field outside it.
	float inside = 1.0;
	if (uHasFoot > 0.5) {
		vec4 fc = uFootVP * vec4(head, 1.0);
		float m = fc.w > 0.0 ? max(abs(fc.x), abs(fc.y)) / fc.w : 9.0;
		inside = mix(0.16, 1.0, 1.0 - smoothstep(0.92, 1.25, m));
	}
	vAlpha = uOpacity * lifeFade * (0.12 + 0.88 * speed) * inside * (1.0 - smoothstep(uFade.x, uFade.y, range));
	vRise = clamp(w * uSpeedScale * 2.0, -1.0, 1.0);
}`;

const FRAGMENT = /* glsl */ `
in float vAlong;
in float vAcross;
in float vAlpha;
in float vRise;
uniform vec3 uColor;
out vec4 fragColor;
void main() {
	float across = 1.0 - smoothstep(0.35, 1.0, abs(vAcross));
	float along = smoothstep(0.0, 0.85, vAlong) * (1.0 - smoothstep(0.92, 1.0, vAlong) * 0.4);
	float a = across * along * vAlpha;
	vec3 c = uColor * (0.8 + 0.5 * vRise) + uColor * along * along * across * 0.6;
	fragColor = vec4(c * a, 0.0);
}`;

export class FlowField {
	/** @param {THREE.WebGLRenderer} renderer */
	constructor(renderer) {
		this.renderer = renderer;
		this.targets = [
			floatTarget(STATE_W, STATE_H),
			floatTarget(STATE_W, STATE_H)
		];
		this.current = 0;
		this.update = new FullscreenPass(UPDATE, {
			uState: { value: null },
			uDt: { value: 0 },
			uTime: { value: 0 },
			uDrift: { value: new THREE.Vector2() },
			uQuad0: { value: new THREE.Vector2() },
			uQuad1: { value: new THREE.Vector2() },
			uQuad2: { value: new THREE.Vector2() },
			uQuad3: { value: new THREE.Vector2() },
			uCenter: { value: new THREE.Vector2() },
			uRadius: { value: 100 },
			uQuadShare: { value: 0 },
			uShift: { value: new THREE.Vector2() },
			uLife: { value: 3.5 },
			uReset: { value: 1 }
		});
		const geometry = new THREE.InstancedBufferGeometry();
		geometry.setAttribute('corner', new THREE.Float32BufferAttribute([0, -1, 1, -1, 1, 1, 0, 1], 2));
		geometry.setIndex([0, 1, 2, 0, 2, 3]);
		geometry.instanceCount = STATE_W * STATE_H;
		this.material = new THREE.ShaderMaterial({
			glslVersion: THREE.GLSL3,
			vertexShader: VERTEX,
			fragmentShader: FRAGMENT,
			uniforms: {
				uState: { value: null },
				uDisp0: { value: null },
				uDisp1: { value: null },
				uMotion0: { value: null },
				uMotion1: { value: null },
				uPatch: { value: new THREE.Vector4(1, 1, 1, 1) },
				uOffset0: { value: new THREE.Vector2() },
				uOffset1: { value: new THREE.Vector2() },
				uResolution: { value: new THREE.Vector2(1, 1) },
				uStreak: { value: 1.6 },
				uOpacity: { value: 0 },
				uFade: { value: new THREE.Vector2(250, 900) },
				uFootVP: { value: new THREE.Matrix4() },
				uHasFoot: { value: 0 },
				uFine: { value: FINE_SHARE },
				uSpeedScale: { value: 1 },
				uColor: { value: new THREE.Vector3(0.08, 0.62, 0.74) }
			},
			transparent: true,
			depthTest: true,
			depthWrite: false,
			blending: THREE.CustomBlending,
			blendSrc: THREE.OneFactor,
			blendDst: THREE.OneFactor,
			blendSrcAlpha: THREE.ZeroFactor,
			blendDstAlpha: THREE.OneFactor
		});
		this.mesh = new THREE.Mesh(geometry, this.material);
		this.mesh.frustumCulled = false;
		this.mesh.visible = false;
		this.needsReset = true;
	}

	/**
	 * Step the particles.
	 * @param {object} p
	 * @param {number} p.dt
	 * @param {number} p.time
	 * @param {THREE.Vector2} p.drift m/s east/north
	 * @param {THREE.Vector2[] | null} p.quad footprint corners (east/north relative to the origin), or null
	 * @param {THREE.Vector2} p.center spawn centre when there is no footprint
	 * @param {number} p.radius
	 * @param {THREE.Vector2} p.shift floating-origin shift since the last step
	 */
	step({ dt, time, drift, quad, center, radius, shift }) {
		const u = this.update.uniforms;
		u.uState.value = this.targets[this.current].texture;
		u.uDt.value = dt;
		u.uTime.value = time % 1000;
		/** @type {THREE.Vector2} */ (u.uDrift.value).copy(drift);
		if (quad) {
			/** @type {THREE.Vector2} */ (u.uQuad0.value).copy(quad[0]);
			/** @type {THREE.Vector2} */ (u.uQuad1.value).copy(quad[1]);
			/** @type {THREE.Vector2} */ (u.uQuad2.value).copy(quad[2]);
			/** @type {THREE.Vector2} */ (u.uQuad3.value).copy(quad[3]);
			u.uQuadShare.value = 0.86;
		} else {
			u.uQuadShare.value = 0;
		}
		/** @type {THREE.Vector2} */ (u.uCenter.value).copy(center);
		u.uRadius.value = radius;
		/** @type {THREE.Vector2} */ (u.uShift.value).copy(shift);
		u.uReset.value = this.needsReset ? 1 : 0;
		this.needsReset = false;
		const next = 1 - this.current;
		const previous = this.renderer.getRenderTarget();
		this.update.render(this.renderer, this.targets[next]);
		this.renderer.setRenderTarget(previous);
		this.current = next;
		this.material.uniforms.uState.value = this.targets[this.current].texture;
	}

	/** @param {number} opacity */
	setOpacity(opacity) {
		this.material.uniforms.uOpacity.value = opacity;
		this.mesh.visible = opacity > 0.002;
	}

	dispose() {
		for (const t of this.targets) t.dispose();
		this.update.dispose();
		this.mesh.geometry.dispose();
		this.material.dispose();
	}
}
