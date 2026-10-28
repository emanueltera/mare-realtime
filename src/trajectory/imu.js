/**
 * What an ideal strapdown IMU on the airframe reads, from the integrated flight state:
 * no noise, no bias, no lever arm (it sits at the centre of gravity).
 *
 * Body axes are FRD (x forward along the nose, y out the right wing, z down through the
 * belly). The attitude is the yaw–pitch–roll (Z-Y-X) sequence from NED: heading ψ
 * (compass, the NED yaw), pitch θ nose-up, bank φ right-wing-down, so a body vector maps
 * to NED by R_nb = Rz(ψ)·Ry(θ)·Rx(φ).
 *
 * Specific force is what the accelerometers feel, acceleration minus gravity:
 *
 *   f_body = R_nbᵀ · (a_ned − g_ned),   g_ned = (0, 0, +g)
 *
 * so level, unaccelerated flight reads f = (g·sin θ, 0, −g·cos θ): (0, 0, −9.81) with
 * the nose a couple of degrees up for its angle of attack showing as a small forward
 * component. Body rates are the Euler-angle kinematics:
 *
 *   p = φ̇ − ψ̇ sin θ
 *   q = θ̇ cos φ + ψ̇ sin φ cos θ
 *   r = −θ̇ sin φ + ψ̇ cos φ cos θ
 */

export const G = 9.81;

/**
 * Rotate an NED vector into FRD body axes.
 * @param {number} n
 * @param {number} e
 * @param {number} d
 * @param {number} heading radians, compass (NED yaw)
 * @param {number} pitch radians, nose up
 * @param {number} bank radians, right wing down
 * @param {{ x: number, y: number, z: number }} out
 */
export function nedToBody(n, e, d, heading, pitch, bank, out) {
	const cy = Math.cos(heading);
	const sy = Math.sin(heading);
	const cp = Math.cos(pitch);
	const sp = Math.sin(pitch);
	const cr = Math.cos(bank);
	const sr = Math.sin(bank);
	// Undo the yaw, then the pitch, then the roll: Rx(φ)ᵀ · Ry(θ)ᵀ · Rz(ψ)ᵀ.
	const x1 = cy * n + sy * e;
	const y1 = -sy * n + cy * e;
	const x2 = cp * x1 - sp * d;
	const z2 = sp * x1 + cp * d;
	out.x = x2;
	out.y = cr * y1 + sr * z2;
	out.z = -sr * y1 + cr * z2;
	return out;
}

/**
 * Specific force in body axes from the ENU acceleration and the attitude.
 * @param {number} aE acceleration, m/s² east
 * @param {number} aN m/s² north
 * @param {number} aU m/s² up
 * @param {number} heading radians
 * @param {number} pitch radians
 * @param {number} bank radians
 * @param {{ x: number, y: number, z: number }} out m/s², FRD
 */
export function specificForce(aE, aN, aU, heading, pitch, bank, out) {
	// ENU → NED, and take gravity (down, +g in NED) away.
	return nedToBody(aN, aE, -aU - G, heading, pitch, bank, out);
}

/**
 * Body rates from the attitude and its rates of change.
 * @param {number} headingRate ψ̇, rad/s
 * @param {number} pitchRate θ̇, rad/s
 * @param {number} bankRate φ̇, rad/s
 * @param {number} pitch θ, radians
 * @param {number} bank φ, radians
 * @param {{ x: number, y: number, z: number }} out p, q, r in x, y, z; rad/s
 */
export function bodyRates(headingRate, pitchRate, bankRate, pitch, bank, out) {
	const cp = Math.cos(pitch);
	const sp = Math.sin(pitch);
	const cr = Math.cos(bank);
	const sr = Math.sin(bank);
	out.x = bankRate - headingRate * sp;
	out.y = pitchRate * cr + headingRate * sr * cp;
	out.z = -pitchRate * sr + headingRate * cr * cp;
	return out;
}
