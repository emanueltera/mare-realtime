/**
 * Screen-space ribbons for the overlay layer: polylines of constant pixel width with
 * round caps and joins, optional dashes measured in metres (so they stay put on the
 * water as the camera moves), a distance fade, and premultiplied output in display
 * space. They depth-test against the sea, which the overlay layer shares a depth
 * buffer with, so a track behind a crest or under the aircraft is hidden properly.
 *
 * Each segment is its own instance, so a polyline's joins would be covered twice (a
 * bead at every sample of a translucent track); a segment leaves out what its
 * predecessor already covers, so a translucent line has one even alpha along its
 * length. Long recorded tracks are drawn once and left alone: a segment beyond the
 * fade range is culled in the vertex shader, and a polyline can carry a `lod` range
 * past which it fades out (drop lines thin out with distance that way, with no
 * per-frame rebuild).
 */
import * as THREE from 'three';

/** iTime of an untimed line. */
const NO_TIME = -1e30;
const COMMON = /* glsl */ `
#define NO_TIME -1e29
`;

const VERTEX = /* glsl */ `${COMMON}
in vec2 corner;            // x: 0 at start, 1 at end; y: side, −1 or 1
in vec3 iStart;
in vec3 iEnd;
in vec3 iPrev;             // the previous segment's start, when iStyle.w = 1
in vec2 iDist;             // metres along the line at start and end
in vec4 iColor;            // linear display rgb, alpha
in vec2 iFade;             // alpha multiplier at start and end
in vec2 iTime;             // recording seconds at start and end; NO_TIME when untimed
in vec4 iStyle;            // width px, dash period m (0 = solid), lod range m (0 = none), has prev
uniform vec2 uResolution;
uniform float uOpacity;
uniform float uLift;       // metres toward the camera, against z-fighting
uniform float uLiftRange;  // and this much more per metre of range, so far crests never cut a line
uniform vec2 uFade;        // range at which lines start and finish fading, m
uniform vec4 uWindow;      // timed lines: now, seconds shown before it, seconds after it, alpha
out vec2 vLocal;           // x: px along from start, y: px across
out float vLen;            // segment length, px
out float vHalf;           // half width, px
out float vDist;
out vec4 vColor;
out float vDash;
out float vRange;
out float vLod;
out float vTime;
flat out vec3 vPrev;       // previous segment's start in the local frame (xy), and whether there is one

vec2 screenOf(vec4 c) {
	return (c.xy / c.w * 0.5 + 0.5) * uResolution;
}

vec4 clipOf(vec3 p) {
	vec4 view = modelViewMatrix * vec4(p, 1.0);
	// Along the line of sight: only the depth changes, never where it lands on screen.
	float range = length(view.xyz);
	view.xyz -= view.xyz / max(range, 1e-3) * min(uLift + uLiftRange * range, 0.9 * range);
	return projectionMatrix * view;
}

void main() {
	vec4 a = clipOf(iStart);
	vec4 b = clipOf(iEnd);
	float near = 1e-3;
	// Clip against the camera plane so a segment reaching behind the camera stays sane.
	if (a.w < near && b.w < near) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
	// Wholly beyond the fade, or beyond its own level of detail, or outside the time
	// window of a recorded track: nothing to draw.
	float closest = max(near, min(a.w, b.w));
	bool timed = iTime.x > NO_TIME;
	bool outside = timed && (max(iTime.x, iTime.y) < uWindow.x - uWindow.y || min(iTime.x, iTime.y) > uWindow.x + uWindow.z);
	if (closest > uFade.y || (iStyle.z > 0.0 && closest > iStyle.z) || outside) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
	float ta = 0.0;
	float tb = 1.0;
	if (a.w < near) { ta = (near - a.w) / (b.w - a.w); a = mix(a, b, ta); }
	else if (b.w < near) { tb = (near - a.w) / (b.w - a.w); b = mix(a, b, tb); }
	vec2 sa = screenOf(a);
	vec2 sb = screenOf(b);
	vec2 d = sb - sa;
	float len = length(d);
	vec2 dir = len > 1e-4 ? d / len : vec2(1.0, 0.0);
	vec2 nrm = vec2(-dir.y, dir.x);
	float half_ = 0.5 * iStyle.x + 1.0;
	vec2 s = mix(sa, sb, corner.x) + dir * (corner.x * 2.0 - 1.0) * half_ + nrm * corner.y * half_;
	vec4 c = mix(a, b, corner.x);
	gl_Position = vec4((s / uResolution * 2.0 - 1.0) * c.w, c.z, c.w);
	vLocal = vec2(corner.x * len + (corner.x * 2.0 - 1.0) * half_, corner.y * half_);
	vLen = len;
	vHalf = 0.5 * iStyle.x;
	vDist = mix(mix(iDist.x, iDist.y, ta), mix(iDist.x, iDist.y, tb), corner.x);
	float fade = mix(mix(iFade.x, iFade.y, ta), mix(iFade.x, iFade.y, tb), corner.x);
	vColor = iColor * vec4(1.0, 1.0, 1.0, uOpacity * fade);
	vDash = iStyle.y;
	vRange = c.w;
	vLod = iStyle.z;
	vTime = timed ? mix(mix(iTime.x, iTime.y, ta), mix(iTime.x, iTime.y, tb), corner.x) : NO_TIME;
	if (timed) vColor.a *= uWindow.w;
	vPrev = vec3(0.0);
	if (iStyle.w > 0.5 && ta == 0.0) {
		vec4 p = clipOf(iPrev);
		if (p.w > near) {
			vec2 sp = screenOf(p) - sa;
			vPrev = vec3(dot(sp, dir), dot(sp, nrm), 1.0);
		}
	}
}`;

const FRAGMENT = /* glsl */ `${COMMON}
in vec2 vLocal;
in float vLen;
in float vHalf;
in float vDist;
in vec4 vColor;
in float vDash;
in float vRange;
in float vLod;
in float vTime;
flat in vec3 vPrev;
uniform vec4 uWindow;
uniform vec2 uFade;  // range at which lines start and finish fading, m
uniform vec2 uResolution;
uniform float uClear; // timed lines: seconds just behind the window's now over which they fade in
out vec4 fragColor;

float coverage(float d) {
	return 1.0 - smoothstep(vHalf - 0.5, vHalf + 0.6, d);
}

void main() {
	// Distance to the segment (a capsule), in pixels.
	float x = clamp(vLocal.x, 0.0, vLen);
	float d = length(vec2(vLocal.x - x, vLocal.y));
	float a = coverage(d);
	if (vDash > 0.0) {
		float f = fract(vDist / vDash);
		float w = fwidth(vDist / vDash);
		a *= smoothstep(0.0, w + 1e-4, f) * (1.0 - smoothstep(0.6 - w, 0.6, f));
	}
	// The line's own alpha here: colour, distance, level of detail and time window.
	float alpha = vColor.a * (1.0 - smoothstep(uFade.x, uFade.y, vRange));
	if (vLod > 0.0) alpha *= 1.0 - smoothstep(0.7 * vLod, vLod, vRange);
	if (vTime > NO_TIME) {
		// Fade over the oldest quarter of the window behind the clock, and the furthest ahead.
		float behind = uWindow.x - vTime;
		float ahead = vTime - uWindow.x;
		float past = max(uWindow.y, 1e-3);
		float future = max(uWindow.z, 1e-3);
		// The older half of the window behind the clock fades out, so what was flown long
		// ago (a pass that reaches across the copy) is faint rather than cut off.
		alpha *= 1.0 - smoothstep(0.5 * past, past, behind);
		alpha *= 1.0 - smoothstep(0.75 * future, future, ahead);
		// Leave the last moments before the clock clear, so an aircraft flying the line has
		// a silhouette against the sea rather than ending a ribbon of its own colour.
		if (uClear > 0.0) alpha *= smoothstep(0.3 * uClear, uClear, behind);
	}
	if (vPrev.z > 0.5) {
		// The previous segment (from vPrev to this one's start), drawn just before, already
		// covers c of this pixel at this alpha: add only what brings the union to the
		// line's alpha, so a translucent line has no bead at its joins (over: a + b(1 − a)).
		vec2 q = vLocal;
		vec2 e = -vPrev.xy;
		float h = clamp(dot(q - vPrev.xy, e) / max(dot(e, e), 1e-6), 0.0, 1.0);
		float c = coverage(length(q - vPrev.xy - e * h));
		a = clamp((a - c) / max(1.0 - alpha * c, 1e-3), 0.0, 1.0);
	}
	a *= alpha;
	if (a < 0.003) discard;
	fragColor = vec4(vColor.rgb * a, a);
}`;

/**
 * @typedef {object} Polyline
 * @property {THREE.Vector3[]} points local coordinates (three world axes, relative to the set's anchor)
 * @property {[number, number, number]} color linear display rgb
 * @property {number} alpha
 * @property {number} width px
 * @property {number} [dash] dash period, m; 0 or absent for solid
 * @property {number} [alphaEnd] alpha at the end of the line, faded along its length from
 *   `alpha` at the start; default `alpha`
 * @property {number} [lod] range, m, beyond which the line fades out; 0 or absent for none
 * @property {number[]} [times] recording seconds, one per point: the line shows only within
 *   the set's time window (`setWindow`)
 */

export class LineSet {
	/**
	 * @param {{ lift?: number, liftRange?: number, fade?: [number, number], capacity?: number }} [options]
	 *   lift: metres toward the camera (depth only), plus liftRange per metre of range
	 */
	constructor(options = {}) {
		this.capacity = options.capacity ?? 512;
		const geometry = new THREE.InstancedBufferGeometry();
		geometry.setAttribute('corner', new THREE.Float32BufferAttribute([0, -1, 1, -1, 1, 1, 0, 1], 2));
		geometry.setIndex([0, 1, 2, 0, 2, 3]);
		this.attrs = {
			iStart: new THREE.InstancedBufferAttribute(new Float32Array(this.capacity * 3), 3),
			iEnd: new THREE.InstancedBufferAttribute(new Float32Array(this.capacity * 3), 3),
			iPrev: new THREE.InstancedBufferAttribute(new Float32Array(this.capacity * 3), 3),
			iDist: new THREE.InstancedBufferAttribute(new Float32Array(this.capacity * 2), 2),
			iColor: new THREE.InstancedBufferAttribute(new Float32Array(this.capacity * 4), 4),
			iFade: new THREE.InstancedBufferAttribute(new Float32Array(this.capacity * 2), 2),
			iTime: new THREE.InstancedBufferAttribute(new Float32Array(this.capacity * 2), 2),
			iStyle: new THREE.InstancedBufferAttribute(new Float32Array(this.capacity * 4), 4)
		};
		for (const [name, attr] of Object.entries(this.attrs)) {
			attr.setUsage(THREE.DynamicDrawUsage);
			geometry.setAttribute(name, attr);
		}
		geometry.instanceCount = 0;
		this.geometry = geometry;
		this.material = new THREE.ShaderMaterial({
			glslVersion: THREE.GLSL3,
			vertexShader: VERTEX,
			fragmentShader: FRAGMENT,
			uniforms: {
				uResolution: { value: new THREE.Vector2(1, 1) },
				uOpacity: { value: 1 },
				uLift: { value: options.lift ?? 0 },
				uLiftRange: { value: options.liftRange ?? 0 },
				uFade: { value: new THREE.Vector2(...(options.fade ?? [1e7, 2e7])) },
				uWindow: { value: new THREE.Vector4(0, 1e9, 1e9, 1) },
				uClear: { value: 0 }
			},
			transparent: true,
			depthWrite: false,
			depthTest: true,
			blending: THREE.CustomBlending,
			blendSrc: THREE.OneFactor,
			blendDst: THREE.OneMinusSrcAlphaFactor,
			blendSrcAlpha: THREE.OneFactor,
			blendDstAlpha: THREE.OneMinusSrcAlphaFactor
		});
		this.mesh = new THREE.Mesh(geometry, this.material);
		this.mesh.frustumCulled = false;
		this.count = 0;
	}

	/**
	 * Instances a set of lines needs: one per segment.
	 * @param {Polyline[]} lines
	 */
	static segments(lines) {
		let n = 0;
		for (const line of lines) n += Math.max(1, line.points.length - 1);
		return n;
	}

	/**
	 * Replace the lines.
	 * @param {Polyline[]} lines
	 */
	set(lines) {
		let n = 0;
		const { iStart, iEnd, iPrev, iDist, iColor, iFade, iTime, iStyle } = this.attrs;
		for (const line of lines) {
			const pts = line.points;
			const segments = Math.max(1, pts.length - 1);
			let total = 0;
			const fading = line.alphaEnd !== undefined && line.alpha > 0;
			if (fading) for (let i = 0; i < segments; i++) total += pts[i].distanceTo(pts[Math.min(i + 1, pts.length - 1)]);
			const endFade = fading ? /** @type {number} */ (line.alphaEnd) / line.alpha : 1;
			// A closed loop (a ring) joins its last segment to its first, too.
			const closed = pts.length > 3 && pts[0].distanceToSquared(pts[pts.length - 1]) < 1e-12;
			let dist = 0;
			for (let i = 0; i < segments && n < this.capacity; i++) {
				const a = pts[i];
				const b = pts[Math.min(i + 1, pts.length - 1)];
				const len = a.distanceTo(b);
				const prev = i > 0 ? pts[i - 1] : closed ? pts[pts.length - 2] : null;
				iStart.setXYZ(n, a.x, a.y, a.z);
				iEnd.setXYZ(n, b.x, b.y, b.z);
				if (prev) iPrev.setXYZ(n, prev.x, prev.y, prev.z);
				iDist.setXY(n, dist, dist + len);
				iColor.setXYZW(n, line.color[0], line.color[1], line.color[2], line.alpha);
				if (fading && total > 0) iFade.setXY(n, 1 + (endFade - 1) * (dist / total), 1 + (endFade - 1) * ((dist + len) / total));
				else iFade.setXY(n, 1, 1);
				iStyle.setXYZW(n, line.width, line.dash ?? 0, line.lod ?? 0, prev ? 1 : 0);
				const times = line.times;
				if (times) iTime.setXY(n, times[i], times[Math.min(i + 1, pts.length - 1)]);
				else iTime.setXY(n, NO_TIME, NO_TIME);
				dist += len;
				n++;
			}
		}
		this.count = n;
		this.geometry.instanceCount = n;
		for (const attr of Object.values(this.attrs)) {
			// Only what changed goes to the GPU.
			attr.clearUpdateRanges();
			attr.addUpdateRange(0, Math.max(1, n) * attr.itemSize);
			attr.needsUpdate = true;
		}
	}

	/** @param {number} width @param {number} height */
	setResolution(width, height) {
		/** @type {THREE.Vector2} */ (this.material.uniforms.uResolution.value).set(width, height);
	}

	/**
	 * Show timed lines only from `past` seconds before `now` to `future` after it, at `alpha`.
	 * @param {number} now
	 * @param {number} past
	 * @param {number} future
	 * @param {number} alpha
	 */
	setWindow(now, past, future, alpha) {
		/** @type {THREE.Vector4} */ (this.material.uniforms.uWindow.value).set(now, past, future, alpha);
	}

	/**
	 * Timed lines only: leave the last `seconds` before the window's now clear (fading in
	 * behind it), so an aircraft flying the line stands out against the sea.
	 * @param {number} seconds 0 for none
	 */
	setClear(seconds) {
		this.material.uniforms.uClear.value = seconds;
	}

	/** @param {number} opacity */
	setOpacity(opacity) {
		this.material.uniforms.uOpacity.value = opacity;
		this.mesh.visible = opacity > 0.002 && this.count > 0;
	}

	/**
	 * The range, m, over which lines fade out and are then culled. A set built for a chase
	 * view wants a short one, so other passes far off do not clutter the frame; one drawn
	 * for a whole route seen from tens of kilometres up needs the lines to reach.
	 * @param {number} near
	 * @param {number} far
	 */
	setFade(near, far) {
		/** @type {THREE.Vector2} */ (this.material.uniforms.uFade.value).set(near, far);
	}

	dispose() {
		this.geometry.dispose();
		this.material.dispose();
	}
}
