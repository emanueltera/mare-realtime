/**
 * The trajectory's IMU (src/trajectory/imu.js through createTrajectory): specific force
 * and body rates in FRD, checked against the cases with known answers and against
 * finite differences of the trajectory's own velocity and attitude.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createTrajectory, nedToBody } from '../src/trajectory/index.js';

const G = 9.81;
const DEG = Math.PI / 180;

test('straight, level, unaccelerated flight reads f = (g·sin θ, 0, −g·cos θ) and no rotation', () => {
	for (const wind of /** @type {const} */ (['none', { speed: 10, direction: 90 }])) {
		// A cross wind first drifts the aircraft off its line; give the guidance time to
		// crab it back on and settle.
		const traj = createTrajectory({ duration: 120, altitude: 100, turbulence: 0, wind, pattern: { type: 'straight' } }, {});
		for (const s of traj.sample(50).filter((s) => s.t > 80)) {
			const theta = s.pitch * DEG;
			assert.ok(Math.abs(s.fx - G * Math.sin(theta)) < 1e-6, `fx ${s.fx}`);
			assert.ok(Math.abs(s.fy) < 1e-6, `fy ${s.fy}`);
			assert.ok(Math.abs(s.fz + G * Math.cos(theta)) < 1e-6, `fz ${s.fz}`);
			// The nose sits ~2° up for its angle of attack: close to (0, 0, −9.81).
			assert.ok(Math.hypot(s.fx, s.fy, s.fz + G) < 0.4);
			assert.ok(Math.abs(s.p) < 1e-6 && Math.abs(s.q) < 1e-6 && Math.abs(s.r) < 1e-6);
		}
	}
});

test('steady coordinated turn: fz = −g/cos φ, fy = 0, and ψ̇ resolved in body axes', () => {
	for (const direction of /** @type {const} */ ([1, -1])) {
		const traj = createTrajectory(
			{ duration: 120, altitude: 100, turbulence: 0, wind: 'none', pattern: { type: 'orbit', radius: 150, direction } },
			{}
		);
		for (const s of traj.sample(20).filter((s) => s.t > 60)) {
			const phi = s.bank * DEG;
			const theta = s.pitch * DEG;
			const psiDot = s.turnRate * DEG;
			assert.ok(Math.abs(s.bank) > 20, `bank ${s.bank}`);
			assert.ok(Math.abs(s.fz + G / Math.cos(phi)) < 0.01 * G, `fz ${s.fz} vs ${-G / Math.cos(phi)}`);
			assert.ok(Math.abs(s.fy) < 0.02, `fy ${s.fy}`);
			assert.ok(Math.abs(s.p + psiDot * Math.sin(theta)) < 1e-3, `p ${s.p}`);
			assert.ok(Math.abs(s.q - psiDot * Math.sin(phi) * Math.cos(theta)) < 1e-3, `q ${s.q}`);
			assert.ok(Math.abs(s.r - psiDot * Math.cos(phi) * Math.cos(theta)) < 1e-3, `r ${s.r}`);
			// A right turn yaws and pitches the body right / nose-up: r and q share the
			// turn's sign, q always positive (the nose follows the turn into the sky).
			assert.ok(Math.sign(s.r) === direction && s.q > 0);
		}
	}
});

/**
 * Body → NED rotation matrix for heading ψ, pitch θ, bank φ (Z-Y-X), row-major.
 * @param {number} psi @param {number} theta @param {number} phi
 */
function dcm(psi, theta, phi) {
	const [cy, sy, cp, sp, cr, sr] = [Math.cos(psi), Math.sin(psi), Math.cos(theta), Math.sin(theta), Math.cos(phi), Math.sin(phi)];
	return [
		[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
		[sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
		[-sp, cp * sr, cp * cr]
	];
}

test('finite differences of the flown trajectory match the IMU across a turbulent random mission', () => {
	const traj = createTrajectory(
		{
			seed: 11,
			duration: 400,
			altitude: 120,
			pattern: { type: 'random' },
			altitudeProfile: { type: 'sine', amplitude: 25, period: 70 },
			airspeedVariation: 3,
			turbulence: 0.6
		},
		{ sea: { windSpeed: 12, windDirection: 240 } }
	);
	const rate = traj.rate;
	const h = 1 / rate;
	const s = traj.sample(rate);
	const forceErrors = [];
	const rateErrors = [];
	const body = { x: 0, y: 0, z: 0 };
	for (let i = 1; i < s.length - 1; i++) {
		const a = s[i - 1];
		const b = s[i];
		const c = s[i + 1];
		// Acceleration from the velocity the trajectory reports, rotated into body axes
		// with gravity taken away: what the accelerometers should read.
		const aE = (c.vE - a.vE) / (2 * h);
		const aN = (c.vN - a.vN) / (2 * h);
		const aU = (c.vU - a.vU) / (2 * h);
		nedToBody(aN, aE, -aU - G, b.heading * DEG, b.pitch * DEG, b.bank * DEG, body);
		forceErrors.push(Math.hypot(body.x - b.fx, body.y - b.fy, body.z - b.fz));
		// Body rates from the attitude: the skew part of Rᵀ·Ṙ.
		const R = dcm(b.heading * DEG, b.pitch * DEG, b.bank * DEG);
		const R0 = dcm(a.heading * DEG, a.pitch * DEG, a.bank * DEG);
		const R1 = dcm(c.heading * DEG, c.pitch * DEG, c.bank * DEG);
		/** (Rᵀ · dR/dt)[i][j] @param {number} r @param {number} k */
		const W = (r, k) => {
			let v = 0;
			for (let m = 0; m < 3; m++) v += (R[m][r] * (R1[m][k] - R0[m][k])) / (2 * h);
			return v;
		};
		const p = (W(2, 1) - W(1, 2)) / 2;
		const q = (W(0, 2) - W(2, 0)) / 2;
		const r = (W(1, 0) - W(0, 1)) / 2;
		rateErrors.push(Math.hypot(p - b.p, q - b.q, r - b.r));
	}
	const sorted = (/** @type {number[]} */ e) => [...e].sort((x, y) => x - y);
	const f = sorted(forceErrors);
	const w = sorted(rateErrors);
	const pct = (/** @type {number[]} */ e, /** @type {number} */ q) => e[Math.floor(q * (e.length - 1))];
	// The autopilot's commands are held over each step, so the bank and flight-path rates
	// step slightly from one step to the next (and the rate limits switch in and out); a
	// central difference straddles those steps and averages them. Apart from that the
	// two agree to the integration error.
	assert.ok(pct(f, 0.99) < 0.01, `specific force: 99th percentile error ${pct(f, 0.99)} m/s²`);
	assert.ok(pct(f, 1) < 0.2, `specific force: largest error ${pct(f, 1)} m/s²`);
	assert.ok(pct(w, 0.5) < 1e-3, `body rates: median error ${pct(w, 0.5)} rad/s`);
	assert.ok(pct(w, 0.99) < 5e-3, `body rates: 99th percentile error ${pct(w, 0.99)} rad/s`);
	assert.ok(pct(w, 1) < 0.05, `body rates: largest error ${pct(w, 1)} rad/s`);
	// And the mission did exercise the IMU: turns, climbs and gusts.
	assert.ok(Math.max(...s.map((x) => Math.abs(x.bank))) > 25);
	assert.ok(Math.max(...s.map((x) => Math.abs(x.fz + G))) > 1);
});

test('the IMU sees gusts: turbulence moves f and the rates, still air does not', () => {
	/** @param {number} turbulence */
	const spread = (turbulence) => {
		const traj = createTrajectory({ duration: 60, altitude: 100, turbulence, wind: 'none', pattern: { type: 'straight' } }, {});
		const s = traj.sample(50).filter((x) => x.t > 5);
		const sd = (/** @type {number[]} */ v) => {
			const m = v.reduce((a, b) => a + b, 0) / v.length;
			return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length);
		};
		return { fz: sd(s.map((x) => x.fz)), p: sd(s.map((x) => x.p)) };
	};
	const still = spread(0);
	const light = spread(0.15);
	const strong = spread(1);
	assert.ok(still.fz < 1e-9 && still.p < 1e-9);
	assert.ok(light.fz > 0.01 && light.p > 0.001);
	assert.ok(strong.fz > 2 * light.fz && strong.p > 2 * light.p);
	// Strong turbulence is felt but stays an airframe's, not a crash's: under ~1 g RMS.
	assert.ok(strong.fz < 9.81, `fz RMS ${strong.fz}`);
});
