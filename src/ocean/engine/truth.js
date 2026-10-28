/**
 * A rendered camera as exported ground truth: its pose in the east/north/up frame and
 * its pinhole intrinsics, in the OpenCV camera convention (x right, y down, z along the
 * optical axis) that docs/ARCHITECTURE.md fixes for every exported pose.
 *
 * Everything is read off the three.js camera that drew the frame — its world matrix and
 * its projection matrix — never re-derived from the scene spec, so whatever the rigs did
 * (stabilisation, aim offsets, a shifted projection) is in it.
 *
 * Frames: three.js world is x east, y up, z south (−north), relative to the engine's
 * floating origin; the three.js camera looks down its −z with +y up. So with
 * M = [[1, 0, 0], [0, 0, −1], [0, 1, 0]] taking three.js world vectors to ENU and
 * F = diag(1, −1, −1) taking OpenCV camera axes to three.js camera axes,
 * R_wc = M · R_three · F, and t_wc = M · p_three + origin.
 *
 * Projection: for three.js's perspective matrix P, a camera-frame point (X, Y, Z) in
 * OpenCV axes lands at NDC x = P₀₀·X/Z − P₀₂ and y = −P₁₁·Y/Z − P₁₂. With pixel (0, 0) at
 * the top-left corner of the top-left pixel, u = (x + 1)·W/2 and v = (1 − y)·H/2, so
 *   fx = P₀₀·W/2, cx = (1 − P₀₂)·W/2, fy = P₁₁·H/2, cy = (1 + P₁₂)·H/2.
 * A centred projection with aspect W/H gives fx = fy = (H/2)/tan(fovY/2) and the
 * principal point at the centre; a view offset (setViewOffset) moves it.
 */
import * as THREE from 'three';

/**
 * @typedef {object} CameraTruth
 * @property {[number, number, number]} position ENU metres (t_wc)
 * @property {number[]} rotation 3×3 row-major R_wc: OpenCV camera axes → ENU
 * @property {[number, number, number, number]} quaternion the same rotation, [x, y, z, w], w ≥ 0
 * @property {number} fovY degrees, as the intrinsics give it (2·atan(H / 2fy))
 * @property {number} width drawing-buffer pixels
 * @property {number} height
 * @property {number[]} K 3×3 row-major intrinsics [fx, 0, cx, 0, fy, cy, 0, 0, 1]
 */

const _m = new THREE.Matrix4();
const _q = new THREE.Quaternion();

/**
 * The truth of a perspective camera, after its world and projection matrices are current
 * (as they are straight after it rendered).
 * @param {THREE.PerspectiveCamera} camera
 * @param {{ east: number, north: number }} origin the floating origin its world sits on
 * @param {number} width drawing-buffer pixels
 * @param {number} height
 * @returns {CameraTruth}
 */
export function cameraTruth(camera, origin, width, height) {
	const w = camera.matrixWorld.elements; // column-major
	// The camera's axes in three.js world (columns of its world matrix; unit scale).
	const right = [w[0], w[1], w[2]];
	const up = [w[4], w[5], w[6]];
	const back = [w[8], w[9], w[10]];
	/** three.js world vector → ENU. @param {number[]} v */
	const enu = (v) => [v[0], -v[2], v[1]];
	// R_wc's columns: where the OpenCV x (right), y (down) and z (forward) axes point.
	const cx = enu(right);
	const cy = enu(up).map((c) => -c);
	const cz = enu(back).map((c) => -c);
	const rotation = [cx[0], cy[0], cz[0], cx[1], cy[1], cz[1], cx[2], cy[2], cz[2]];
	_m.set(cx[0], cy[0], cz[0], 0, cx[1], cy[1], cz[1], 0, cx[2], cy[2], cz[2], 0, 0, 0, 0, 1);
	_q.setFromRotationMatrix(_m);
	const sign = _q.w < 0 ? -1 : 1;

	const p = camera.projectionMatrix.elements; // column-major: P₀₀ = p[0], P₁₁ = p[5], P₀₂ = p[8], P₁₂ = p[9]
	const fx = (p[0] * width) / 2;
	const fy = (p[5] * height) / 2;
	const ppx = ((1 - p[8]) * width) / 2;
	const ppy = ((1 + p[9]) * height) / 2;

	return {
		position: [w[12] + origin.east, -w[14] + origin.north, w[13]],
		rotation,
		quaternion: [_q.x * sign, _q.y * sign, _q.z * sign, _q.w * sign],
		fovY: (2 * Math.atan(height / (2 * fy)) * 180) / Math.PI,
		width,
		height,
		K: [fx, 0, ppx, 0, fy, ppy, 0, 0, 1]
	};
}

/**
 * Project an ENU point with a camera truth (the convention above), for checks.
 * @param {CameraTruth} truth
 * @param {[number, number, number]} point ENU metres
 * @returns {[number, number, number]} u, v (pixels) and the depth Z along the optical axis
 */
export function projectTruth(truth, point) {
	const R = truth.rotation;
	const d = [point[0] - truth.position[0], point[1] - truth.position[1], point[2] - truth.position[2]];
	// p_c = R_wcᵀ (P − t_wc)
	const X = R[0] * d[0] + R[3] * d[1] + R[6] * d[2];
	const Y = R[1] * d[0] + R[4] * d[1] + R[7] * d[2];
	const Z = R[2] * d[0] + R[5] * d[1] + R[8] * d[2];
	const K = truth.K;
	return [(K[0] * X) / Z + K[2], (K[4] * Y) / Z + K[5], Z];
}
