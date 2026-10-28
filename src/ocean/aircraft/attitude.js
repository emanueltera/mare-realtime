/**
 * Aircraft state helpers shared by the flight model, the 3D model and the engine:
 * an empty state, interpolation between two states, and the attitude quaternion.
 *
 * Attitude conventions: heading in compass degrees (0 north, 90 east), pitch in
 * degrees nose-up positive, bank in degrees right-wing-down positive. The world
 * frame is three.js's: x east, y up, z south. The model's nose points toward −Z,
 * its right wing toward +X.
 */
import * as THREE from 'three';

const DEG = Math.PI / 180;

/** @returns {import('../aircraft.js').AircraftState} */
export function emptyState() {
	return { x: 0, y: 0, altitude: 0, heading: 0, pitch: 0, bank: 0, speed: 0, turnRate: 0, time: 0, visible: 1 };
}

/**
 * Interpolate two states (headings by the shortest arc) into `out`.
 * @param {import('../aircraft.js').AircraftState} a
 * @param {import('../aircraft.js').AircraftState} b
 * @param {number} t 0 → a, 1 → b
 * @param {import('../aircraft.js').AircraftState} [out]
 */
export function blendStates(a, b, t, out = emptyState()) {
	/** @param {number} p @param {number} q */
	const mix = (p, q) => p + (q - p) * t;
	// Shortest signed arc from a to b, robust to headings outside [0, 360).
	const dh = ((((b.heading - a.heading) % 360) + 540) % 360) - 180;
	const heading = a.heading + dh * t;
	out.x = mix(a.x, b.x);
	out.y = mix(a.y, b.y);
	out.altitude = mix(a.altitude, b.altitude);
	out.heading = ((heading % 360) + 360) % 360;
	out.pitch = mix(a.pitch, b.pitch);
	out.bank = mix(a.bank, b.bank);
	out.speed = mix(a.speed, b.speed);
	out.turnRate = mix(a.turnRate, b.turnRate);
	out.time = b.time;
	out.visible = mix(a.visible, b.visible);
	return out;
}

const euler = new THREE.Euler();

/**
 * Quaternion for an attitude, in the world frame where x is east, y up and z south.
 * Heading, then pitch, then bank (intrinsic yaw–pitch–roll), so the model's −Z nose
 * points along the compass heading, raised by `pitch`, and its +X right wing dips
 * for positive `bank`.
 * @param {number} heading compass degrees
 * @param {number} pitch degrees, nose up positive
 * @param {number} bank degrees, right wing down positive
 * @param {THREE.Quaternion} [out]
 */
export function attitudeQuaternion(heading, pitch, bank, out = new THREE.Quaternion()) {
	euler.set(pitch * DEG, -heading * DEG, -bank * DEG, 'YXZ');
	return out.setFromEuler(euler);
}
