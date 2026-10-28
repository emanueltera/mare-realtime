/**
 * Procedural geometry for the UAV: NACA 4-digit airfoils, lofted lifting surfaces
 * (with hinged control-surface cut-outs), a superelliptic fuselage loft, lathe parts
 * and a few small helpers. Everything is generated in the model frame (metres, nose
 * toward −Z, right wing +X, up +Y) with position, normal, uv and a per-vertex grey
 * `color` used as a shade multiplier on the shared materials.
 */
import * as THREE from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';

/**
 * @typedef {[number, number, number, number]} LoopPoint chord x, thickness y (both in
 *   chords), texture u, shade
 * @typedef {LoopPoint[]} Segment a run of points with smooth normals; segments meet at
 *   creases
 * @typedef {object} Station one cross-section of a lofted surface
 * @property {THREE.Vector3} o leading-edge point
 * @property {number} c chord, m
 * @property {THREE.Vector3} d unit chord direction, leading edge → trailing edge
 * @property {THREE.Vector3} u unit thickness direction (the airfoil's upper side)
 * @property {number} v texture v along the span
 * @property {number} [shade] vertex-colour multiplier for this station; default 1
 */

/**
 * NACA 4-digit airfoil with a closed trailing edge.
 * @param {number} m max camber (fraction of chord)
 * @param {number} p position of max camber (fraction of chord)
 * @param {number} t thickness (fraction of chord)
 */
export function naca4(m, p, t) {
	/** @param {number} x */
	const thickness = (x) =>
		5 * t * (0.2969 * Math.sqrt(x) - 0.126 * x - 0.3516 * x * x + 0.2843 * x ** 3 - 0.1036 * x ** 4);
	/** @param {number} x @returns {[number, number]} camber and its slope */
	const camber = (x) => {
		if (m === 0) return [0, 0];
		if (x < p) return [(m / (p * p)) * (2 * p * x - x * x), ((2 * m) / (p * p)) * (p - x)];
		return [(m / (1 - p) ** 2) * (1 - 2 * p + 2 * p * x - x * x), ((2 * m) / (1 - p) ** 2) * (p - x)];
	};
	return {
		thickness,
		camber,
		/**
		 * @param {number} x
		 * @param {1 | -1} side 1 upper, −1 lower
		 * @returns {[number, number]}
		 */
		point(x, side) {
			const yt = thickness(x);
			const [yc, dyc] = camber(x);
			const th = Math.atan(dyc);
			return [x - side * yt * Math.sin(th), yc + side * yt * Math.cos(th)];
		}
	};
}

/** @typedef {ReturnType<typeof naca4>} Airfoil */

/**
 * Cosine-spaced chord stations between x0 and x1 (inclusive), clustered at the leading
 * and trailing edges.
 * @param {number} x0
 * @param {number} x1
 * @param {number} n points per full chord
 */
function chordStations(x0, x1, n) {
	const b0 = Math.acos(1 - 2 * x0);
	const b1 = Math.acos(1 - 2 * x1);
	const count = Math.max(2, Math.round((n * Math.abs(b1 - b0)) / Math.PI) + 1);
	return Array.from({ length: count }, (_, i) => (1 - Math.cos(b0 + ((b1 - b0) * i) / (count - 1))) / 2);
}

/**
 * Shade along the chord: a darker leading-edge erosion strip.
 * @param {number} x
 * @param {number} strip shade of the strip (1 for none)
 */
function chordShade(x, strip) {
	const k = Math.min(1, Math.max(0, (x - 0.025) / 0.04));
	return strip + (1 - strip) * k * k * (3 - 2 * k);
}

/**
 * The whole airfoil as one smooth segment: upper surface trailing edge → leading
 * edge, then lower surface back to the trailing edge (a crease at the TE).
 * @param {Airfoil} af
 * @param {{ n?: number, strip?: number }} [opts]
 * @returns {Segment[]}
 */
export function fullLoop(af, { n = 28, strip = 1 } = {}) {
	const xs = chordStations(0, 1, n);
	/** @type {Segment} */
	const seg = [];
	for (let i = xs.length - 1; i >= 0; i--) {
		const [x, y] = af.point(xs[i], 1);
		seg.push([x, y, 0.5 * (1 - xs[i]), chordShade(xs[i], strip)]);
	}
	for (let i = 1; i < xs.length; i++) {
		const [x, y] = af.point(xs[i], -1);
		seg.push([x, y, 0.5 + 0.5 * xs[i], chordShade(xs[i], strip)]);
	}
	return [seg];
}

/**
 * The fixed part of a surface ahead of a control surface: chord 0 → `cut`, closed by a
 * flat face at the cut.
 * @param {Airfoil} af
 * @param {number} cut chord fraction
 * @param {{ n?: number, strip?: number }} [opts]
 * @returns {Segment[]}
 */
export function mainLoop(af, cut, { n = 28, strip = 1 } = {}) {
	const xs = chordStations(0, cut, n);
	/** @type {Segment} */
	const skin = [];
	for (let i = xs.length - 1; i >= 0; i--) {
		const [x, y] = af.point(xs[i], 1);
		skin.push([x, y, 0.5 * (1 - xs[i]), chordShade(xs[i], strip)]);
	}
	for (let i = 1; i < xs.length; i++) {
		const [x, y] = af.point(xs[i], -1);
		skin.push([x, y, 0.5 + 0.5 * xs[i], chordShade(xs[i], strip)]);
	}
	const top = af.point(cut, 1);
	const bottom = af.point(cut, -1);
	const shade = 0.45;
	/** @type {Segment} */
	const face = [
		[bottom[0], bottom[1], 0.5 + 0.5 * cut, shade],
		[(top[0] + bottom[0]) / 2, (top[1] + bottom[1]) / 2, 0.5, shade],
		[top[0], top[1], 0.5 * (1 - cut), shade]
	];
	return [skin, face];
}

/**
 * A control surface aft of a hinge: upper skin TE → hinge, a rounded nose about the
 * hinge line, lower skin back to the TE. Its nose sits inside the fixed part, so it
 * can rotate about the hinge without opening a hole.
 * @param {Airfoil} af
 * @param {number} hinge chord fraction
 * @param {{ n?: number }} [opts]
 * @returns {Segment[]}
 */
export function controlLoop(af, hinge, { n = 28 } = {}) {
	const xs = chordStations(hinge, 1, n);
	const [yc] = af.camber(hinge);
	const r = af.thickness(hinge);
	/** @type {Segment} */
	const seg = [];
	for (let i = xs.length - 1; i >= 0; i--) {
		const [x, y] = af.point(xs[i], 1);
		seg.push([x, y, 0.5 * (1 - xs[i]), i === 0 ? 0.6 : 0.97]);
	}
	const arc = 7;
	for (let k = 1; k < arc; k++) {
		const phi = Math.PI / 2 + (Math.PI * k) / arc;
		seg.push([hinge + r * Math.cos(phi), yc + r * Math.sin(phi), 0.5, 0.4]);
	}
	for (let i = 0; i < xs.length; i++) {
		const [x, y] = af.point(xs[i], -1);
		seg.push([x, y, 0.5 + 0.5 * xs[i], i === 0 ? 0.6 : 0.97]);
	}
	return [seg];
}

/**
 * Hinge point of a station: on the camber line at chord fraction `x`.
 * @param {Station} st
 * @param {Airfoil} af
 * @param {number} x
 */
export function hingePoint(st, af, x) {
	const [yc] = af.camber(x);
	return st.o.clone().addScaledVector(st.d, x * st.c).addScaledVector(st.u, yc * st.c);
}

/**
 * Loft airfoil segments through stations. Faces point outward when each station's
 * `u` equals d × (span direction) — the convention every caller follows.
 * @param {Station[]} stations
 * @param {Segment[]} segments
 * @param {{ capStart?: boolean, capEnd?: boolean }} [opts]
 */
export function loft(stations, segments, { capStart = true, capEnd = true } = {}) {
	/** @type {number[]} */
	const pos = [];
	/** @type {number[]} */
	const uv = [];
	/** @type {number[]} */
	const col = [];
	/** @type {number[]} */
	const idx = [];
	const p = new THREE.Vector3();
	for (const seg of segments) {
		const base = pos.length / 3;
		const n = seg.length;
		for (const st of stations) {
			for (const [x, y, u, shade] of seg) {
				p.copy(st.o).addScaledVector(st.d, x * st.c).addScaledVector(st.u, y * st.c);
				pos.push(p.x, p.y, p.z);
				uv.push(u, st.v);
				const k = shade * (st.shade ?? 1);
				col.push(k, k, k);
			}
		}
		for (let j = 0; j < stations.length - 1; j++) {
			for (let i = 0; i < n - 1; i++) {
				const a = base + j * n + i;
				const b = a + 1;
				const c = a + n;
				const d = c + 1;
				idx.push(a, c, b, b, c, d);
			}
		}
	}
	const geometry = new THREE.BufferGeometry();
	geometry.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
	geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
	geometry.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
	geometry.setIndex(idx);
	geometry.computeVertexNormals();
	const parts = [geometry];
	const span = stations[stations.length - 1].o.clone().sub(stations[0].o).normalize();
	if (capStart) parts.push(cap(stations[0], segments, span.clone().negate()));
	if (capEnd) parts.push(cap(stations[stations.length - 1], segments, span));
	return parts.length === 1 ? geometry : merge(parts);
}

/**
 * A flat cap closing a section.
 * @param {Station} st
 * @param {Segment[]} segments
 * @param {THREE.Vector3} outward
 */
function cap(st, segments, outward) {
	/** @type {THREE.Vector2[]} */
	const contour = [];
	/** @type {LoopPoint[]} */
	const points = [];
	for (const seg of segments) {
		for (const pt of seg) {
			const last = points[points.length - 1];
			if (last && Math.hypot(last[0] - pt[0], last[1] - pt[1]) < 1e-7) continue;
			points.push(pt);
			contour.push(new THREE.Vector2(pt[0], pt[1]));
		}
	}
	if (points.length > 2 && contour[0].distanceTo(contour[contour.length - 1]) < 1e-7) {
		points.pop();
		contour.pop();
	}
	const tris = THREE.ShapeUtils.triangulateShape(contour, []);
	const normal = new THREE.Vector3().crossVectors(st.d, st.u).normalize();
	const flip = normal.dot(outward) < 0;
	if (flip) normal.negate();
	/** @type {number[]} */
	const pos = [];
	/** @type {number[]} */
	const nor = [];
	/** @type {number[]} */
	const uv = [];
	/** @type {number[]} */
	const col = [];
	const p = new THREE.Vector3();
	for (const [x, y, u, shade] of points) {
		p.copy(st.o).addScaledVector(st.d, x * st.c).addScaledVector(st.u, y * st.c);
		pos.push(p.x, p.y, p.z);
		nor.push(normal.x, normal.y, normal.z);
		uv.push(u, st.v);
		const k = shade * 0.92 * (st.shade ?? 1);
		col.push(k, k, k);
	}
	/** @type {number[]} */
	const idx = [];
	// A triangle counter-clockwise in (x, y) = (d, u) faces d × u; orient each to `normal`.
	for (const [a, b, c] of tris) {
		const pa = contour[a];
		const pb = contour[b];
		const pc = contour[c];
		const ccw = (pb.x - pa.x) * (pc.y - pa.y) - (pb.y - pa.y) * (pc.x - pa.x) > 0;
		if (ccw !== flip) idx.push(a, b, c);
		else idx.push(a, c, b);
	}
	const geometry = new THREE.BufferGeometry();
	geometry.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
	geometry.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
	geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
	geometry.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
	geometry.setIndex(idx);
	return geometry;
}

/**
 * Monotone cubic (Fritsch–Carlson) interpolation through a table, flat beyond its ends.
 * @param {number[]} xs increasing
 * @param {number[]} ys
 */
export function pchip(xs, ys) {
	const n = xs.length;
	const h = xs.slice(1).map((x, i) => x - xs[i]);
	const delta = h.map((hi, i) => (ys[i + 1] - ys[i]) / hi);
	const m = new Array(n).fill(0);
	m[0] = delta[0];
	m[n - 1] = delta[n - 2];
	for (let i = 1; i < n - 1; i++) {
		if (delta[i - 1] * delta[i] <= 0) m[i] = 0;
		else {
			const w1 = 2 * h[i] + h[i - 1];
			const w2 = h[i] + 2 * h[i - 1];
			m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]);
		}
	}
	/** @param {number} x */
	return (x) => {
		if (x <= xs[0]) return ys[0];
		if (x >= xs[n - 1]) return ys[n - 1];
		let i = 0;
		while (i < n - 2 && x > xs[i + 1]) i++;
		const t = (x - xs[i]) / h[i];
		const t2 = t * t;
		const t3 = t2 * t;
		return (
			(2 * t3 - 3 * t2 + 1) * ys[i] +
			(t3 - 2 * t2 + t) * h[i] * m[i] +
			(-2 * t3 + 3 * t2) * ys[i + 1] +
			(t3 - t2) * h[i] * m[i + 1]
		);
	};
}

/**
 * @typedef {object} BodyProfile a lofted body of superelliptic sections along z
 * @property {number} zNose
 * @property {number} zTail
 * @property {number} noseLength length of the elliptical nose cap
 * @property {(z: number) => number} halfWidth
 * @property {(z: number) => number} centre centreline height
 * @property {(z: number) => number} top half-height above the centreline
 * @property {(z: number) => number} bottom half-depth below the centreline
 * @property {number} [exponent] superellipse exponent; 2 is an ellipse; default 2.4
 * @property {number} [noseSharpness] 0.5 gives an elliptical (blunt) nose cap, 1 a cone;
 *   default 0.5
 */

/**
 * Loft a body from a nose cap to a closed tail, with UVs (u round, seam underneath;
 * v along the length) for panel-line maps.
 * @param {BodyProfile} profile
 * @param {{ around?: number, along?: number, nose?: number }} [opts]
 */
export function bodyLoft(profile, { around = 64, along = 64, nose = 16 } = {}) {
	const { zNose, zTail, noseLength } = profile;
	const ex = 2 / (profile.exponent ?? 2.4);
	/** @type {Array<[number, number]>} z and nose factor */
	const rings = [];
	for (let k = 0; k <= nose; k++) {
		const phi = (k / nose) * (Math.PI / 2);
		rings.push([zNose + noseLength * (1 - Math.cos(phi)), Math.sin(phi) ** (2 * (profile.noseSharpness ?? 0.5))]);
	}
	for (let k = 1; k <= along; k++) rings.push([zNose + noseLength + ((zTail - zNose - noseLength) * k) / along, 1]);
	rings.push([zTail, 0]); // close the tail
	/** @param {number} s */
	const spow = (s) => Math.sign(s) * Math.abs(s) ** ex;
	const cols = around + 1;
	/** @type {number[]} */
	const pos = [];
	/** @type {number[]} */
	const uv = [];
	/** @type {number[]} */
	const col = [];
	for (const [z, e] of rings) {
		const w = profile.halfWidth(z) * e;
		const c = profile.centre(z);
		const a = profile.top(z) * e;
		const b = profile.bottom(z) * e;
		for (let j = 0; j < cols; j++) {
			const th = (2 * Math.PI * j) / around; // 0 underneath, π/2 right side, π on top
			const sx = Math.sin(th);
			const sy = -Math.cos(th);
			pos.push(w * spow(sx), c + (sy > 0 ? a : b) * spow(sy), z);
			uv.push(j / around, (z - zNose) / (zTail - zNose));
			col.push(1, 1, 1);
		}
	}
	/** @type {number[]} */
	const idx = [];
	for (let i = 0; i < rings.length - 1; i++) {
		for (let j = 0; j < around; j++) {
			const a = i * cols + j;
			const b = a + 1;
			const c = a + cols;
			const d = c + 1;
			idx.push(a, b, c, b, d, c);
		}
	}
	const geometry = new THREE.BufferGeometry();
	geometry.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
	geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
	geometry.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
	geometry.setIndex(idx);
	geometry.computeVertexNormals();
	const normal = /** @type {THREE.BufferAttribute} */ (geometry.getAttribute('normal'));
	const v = new THREE.Vector3();
	for (let i = 0; i < rings.length; i++) {
		// Weld the seam's normals, and point the nose tip straight ahead.
		const first = i * cols;
		const last = first + around;
		if (i === 0) {
			for (let j = 0; j < cols; j++) normal.setXYZ(first + j, 0, 0, -1);
			continue;
		}
		if (i === rings.length - 1) {
			for (let j = 0; j < cols; j++) normal.setXYZ(first + j, 0, 0, 1);
			continue;
		}
		v.set(normal.getX(first) + normal.getX(last), normal.getY(first) + normal.getY(last), normal.getZ(first) + normal.getZ(last)).normalize();
		normal.setXYZ(first, v.x, v.y, v.z);
		normal.setXYZ(last, v.x, v.y, v.z);
	}
	return geometry;
}

/**
 * A body of revolution about the z axis from a [radius, z] profile.
 * @param {Array<[number, number]>} profile
 * @param {number} [segments]
 */
export function latheZ(profile, segments = 32) {
	const geometry = new THREE.LatheGeometry(
		profile.map(([r, z]) => new THREE.Vector2(Math.max(0, r), z)),
		segments
	);
	// Lathe revolves about +Y; turn its axis onto +Z.
	geometry.rotateX(Math.PI / 2);
	return prepare(geometry);
}

/**
 * Give a geometry the attribute set every part shares (position, normal, uv, color)
 * so parts can be merged per material.
 * @param {THREE.BufferGeometry} geometry
 * @param {number} [shade] vertex colour
 */
export function prepare(geometry, shade = 1) {
	const count = geometry.getAttribute('position').count;
	if (!geometry.getAttribute('normal')) geometry.computeVertexNormals();
	if (!geometry.getAttribute('uv')) geometry.setAttribute('uv', new THREE.Float32BufferAttribute(new Float32Array(count * 2), 2));
	geometry.setAttribute('color', new THREE.Float32BufferAttribute(new Float32Array(count * 3).fill(shade), 3));
	for (const name of Object.keys(geometry.attributes)) {
		if (!['position', 'normal', 'uv', 'color'].includes(name)) geometry.deleteAttribute(name);
	}
	return geometry;
}

/**
 * Merge geometries that share a material (indexed or not), disposing the inputs.
 * @param {THREE.BufferGeometry[]} parts
 */
export function merge(parts) {
	const indexed = parts.every((g) => g.index !== null);
	const ready = parts.map((g) => (indexed || g.index === null ? g : g.toNonIndexed()));
	const merged = mergeGeometries(ready, false);
	if (!merged) throw new Error('aircraft: could not merge geometry');
	for (const g of parts) g.dispose();
	for (const g of ready) g.dispose();
	return merged;
}

/**
 * Mirror a geometry across the x = 0 plane (left side from right), keeping faces
 * outward.
 * @param {THREE.BufferGeometry} geometry
 */
export function mirrorX(geometry) {
	const out = geometry.clone();
	out.scale(-1, 1, 1);
	const index = out.index;
	if (index) {
		for (let i = 0; i < index.count; i += 3) {
			const b = index.getX(i + 1);
			index.setX(i + 1, index.getX(i + 2));
			index.setX(i + 2, b);
		}
		index.needsUpdate = true;
	} else {
		for (const attribute of Object.values(out.attributes)) {
			const a = /** @type {THREE.BufferAttribute} */ (attribute);
			const size = a.itemSize;
			for (let i = 0; i < a.count; i += 3) {
				for (let k = 0; k < size; k++) {
					const t = a.array[(i + 1) * size + k];
					a.array[(i + 1) * size + k] = a.array[(i + 2) * size + k];
					a.array[(i + 2) * size + k] = t;
				}
			}
			a.needsUpdate = true;
		}
	}
	return out;
}
