/**
 * The trajectory generator (src/trajectory): determinism, interpolation, the flight
 * model's physics (wind triangle, coordinated turns, pattern geometry) and the writers.
 * The IMU has its own file, trajectory-imu.test.js.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
	autopilot,
	createTrajectory,
	enuToGeodetic,
	gimbalDefaults,
	imuColumns,
	imuToCsv,
	parseCsv,
	resolveTrajectorySpec,
	sampleColumns,
	samplesToCsv,
	samplesToJson,
	toAircraftState,
	trajectoryDefaults,
	windAloft
} from '../src/trajectory/index.js';
import { createAltitudeProfile } from '../src/trajectory/profiles.js';

const G = 9.81;
const DEG = Math.PI / 180;
const sea = { windSpeed: 9, windDirection: 70 };

/** Shortest signed difference b − a between two compass angles, degrees. @param {number} a @param {number} b */
const angleDiff = (a, b) => ((((b - a) % 360) + 540) % 360) - 180;

test('defaults are filled in and the input is left alone', () => {
	const input = { duration: 10, altitude: 100, pattern: /** @type {const} */ ({ type: 'orbit' }) };
	const spec = resolveTrajectorySpec(input);
	assert.equal(spec.seed, trajectoryDefaults.seed);
	assert.equal(spec.airspeed, 24);
	assert.equal(spec.turbulence, 0.15);
	assert.equal(spec.wind, 'sea');
	assert.deepEqual(spec.start, { x: 0, y: 0, heading: 0 });
	assert.deepEqual(spec.gimbal, { ...gimbalDefaults });
	assert.deepEqual(spec.pattern, { type: 'orbit', radius: 180, direction: 1, center: [180, 0] });
	assert.deepEqual(input.pattern, { type: 'orbit' });
	// JSON round trip: the resolved spec is plain data.
	assert.deepEqual(JSON.parse(JSON.stringify(spec)), spec);
	assert.throws(() => createTrajectory({ duration: 10, altitude: 100, pattern: { type: 'straight' } }), /sea/);
});

test('the same spec gives bit-identical samples; another seed does not', () => {
	const spec = {
		seed: 7,
		duration: 120,
		altitude: 90,
		pattern: /** @type {const} */ ({ type: 'random' }),
		altitudeProfile: /** @type {const} */ ({ type: 'steps', amplitude: 20, interval: 30 }),
		airspeedVariation: 2,
		turbulence: 0.6
	};
	const a = createTrajectory(spec, { sea }).sample(25);
	const b = createTrajectory(structuredClone(spec), { sea }).sample(25);
	assert.deepEqual(a, b);
	const c = createTrajectory({ ...spec, seed: 8 }, { sea }).sample(25);
	assert.notDeepEqual(a, c);
});

test('stateAt is continuous between internal steps and wraps the heading at 0/360', () => {
	// An anticlockwise orbit from heading 10° crosses north every lap.
	const traj = createTrajectory(
		{ duration: 120, altitude: 120, start: { heading: 10 }, pattern: { type: 'orbit', direction: -1 }, turbulence: 0.5 },
		{ sea }
	);
	const dt = 1 / 1000; // five lookups per internal step
	let prev = traj.stateAt(0);
	let wraps = 0;
	for (let i = 1; i <= 120 * 1000; i++) {
		const s = traj.stateAt(i * dt);
		const move = Math.hypot(s.x - prev.x, s.y - prev.y, s.altitude - prev.altitude);
		assert.ok(move <= Math.max(s.groundSpeed, prev.groundSpeed) * 1.2 * dt + 1e-6, `jump of ${move} m at t=${s.t}`);
		assert.ok(s.heading >= 0 && s.heading < 360);
		assert.ok(s.course >= 0 && s.course < 360);
		const dh = angleDiff(prev.heading, s.heading);
		assert.ok(Math.abs(dh) <= Math.abs(s.turnRate) * dt * 1.5 + 0.01, `heading jump ${dh}° at t=${s.t}`);
		assert.ok(Math.abs(s.bank - prev.bank) < 0.1);
		assert.ok(Math.abs(s.fz - prev.fz) < 0.5);
		if (Math.abs(s.heading - prev.heading) > 180) wraps++;
		prev = s;
	}
	assert.ok(wraps >= 2, `expected the heading to wrap, saw ${wraps}`);
	// At the ends: clamped, and exactly the nodes there.
	assert.deepEqual(traj.stateAt(-5), traj.stateAt(0));
	assert.deepEqual(traj.stateAt(1e9), traj.stateAt(120));
	assert.equal(traj.sample(10).length, 1201);
	assert.equal(traj.sample(10)[1200].t, 120);
});

test('wind triangle: ground velocity − wind is the airspeed along the heading and flight path', () => {
	const traj = createTrajectory(
		{
			duration: 200,
			altitude: 150,
			pattern: { type: 'random' },
			altitudeProfile: { type: 'sine', amplitude: 15, period: 50 },
			turbulence: 0.4,
			wind: { speed: 11, direction: 120 }
		},
		{}
	);
	let crab = 0;
	for (const s of traj.sample(7)) {
		const aE = s.vE - s.windE;
		const aN = s.vN - s.windN;
		const aU = s.vU - s.windU;
		assert.ok(Math.abs(Math.hypot(aE, aN, aU) - s.airspeed) < 0.01, `|v_air| ${Math.hypot(aE, aN, aU)} vs ${s.airspeed}`);
		assert.ok(Math.abs(angleDiff(s.heading, Math.atan2(aE, aN) / DEG)) < 0.05, `air track vs heading at t=${s.t}`);
		// Flight-path angle = pitch − angle of attack (α = 2°·n·(24/V)²).
		const alpha = (2 * (24 / s.airspeed) ** 2) / Math.cos(s.bank * DEG);
		const gamma = Math.atan2(aU, Math.hypot(aE, aN)) / DEG;
		assert.ok(Math.abs(s.pitch - alpha - gamma) < 0.05, `pitch ${s.pitch} vs γ + α ${gamma + alpha}`);
		crab = Math.max(crab, Math.abs(angleDiff(s.heading, s.course)));
	}
	// An 11 m/s cross wind on a 24 m/s airframe crabs it by up to ~27°.
	assert.ok(crab > 20 && crab < 35, `crab ${crab}`);
});

test('a flight starts trimmed: on the start course, crabbed into the wind, at the leg\'s bank', () => {
	// The sea wind runs toward 70°: every one of these courses has a cross wind.
	for (const heading of [0, 160, 300]) {
		// Straight: the ground track is the start heading from t = 0, with no correction turn.
		const straight = createTrajectory({ duration: 20, altitude: 120, start: { heading }, pattern: { type: 'straight' }, turbulence: 0 }, { sea });
		const s0 = straight.stateAt(0);
		assert.ok(Math.abs(angleDiff(heading, s0.course)) < 0.01, `course ${s0.course} vs ${heading}`);
		assert.ok(Math.abs(angleDiff(s0.heading, s0.course)) > 5, `no crab at heading ${heading}`);
		for (const s of straight.sample(20)) {
			assert.ok(Math.abs(s.bank) < 0.05 && Math.abs(s.p) < 1e-3, `t=${s.t}: bank ${s.bank}, p ${s.p}`);
		}
		// Orbit: already banked into the circle, so the roll rate at t = 0 is small.
		const orbit = createTrajectory({ duration: 10, altitude: 120, start: { heading }, pattern: { type: 'orbit' }, turbulence: 0 }, { sea });
		const o0 = orbit.stateAt(0);
		assert.ok(o0.bank > 3 && Math.abs(o0.p) < 0.05, `orbit start bank ${o0.bank}, p ${o0.p}`);
	}
});

test('coordinated turn: turnRate = g·tan(bank)/V in level flight', () => {
	const traj = createTrajectory({ duration: 150, altitude: 120, pattern: { type: 'figure8' }, turbulence: 0 }, { sea });
	for (const s of traj.sample(200)) {
		const expected = (G * Math.tan(s.bank * DEG)) / s.airspeed / DEG;
		assert.ok(Math.abs(s.turnRate - expected) < 0.02, `t=${s.t}: ${s.turnRate} vs ${expected}`);
	}
});

test('climbing and descending turns stay coordinated: no side force without gusts', () => {
	// The lift that bends the path up also turns it when banked: ψ̇ = (V·γ̇ + g·cos γ)·tan φ / (V·cos γ).
	// With ψ̇ = g·tan φ / V instead, a climbing orbit read up to 1 m/s² of side force.
	for (const altitudeProfile of /** @type {const} */ ([
		{ type: 'sine', amplitude: 20, period: 30 },
		{ type: 'steps', amplitude: 30, interval: 20 }
	])) {
		for (const pattern of /** @type {const} */ ([{ type: 'orbit' }, { type: 'figure8' }, { type: 'lawnmower' }])) {
			const traj = createTrajectory(
				{ duration: 120, altitude: 120, pattern, altitudeProfile, turbulence: 0, wind: { speed: 6, direction: 30 } },
				{}
			);
			const fy = Math.max(...traj.sample(50).map((s) => Math.abs(s.fy)));
			assert.ok(fy < 0.08, `${pattern.type} ${altitudeProfile.type}: max |fy| ${fy.toFixed(3)} m/s²`);
		}
	}
});

test('pitch rate and load factor ramp at an altitude-command corner, never step', () => {
	// The steps profile switches its climb feed-forward between ±2 m/s and 0, and a
	// waypoint with a new altitude saturates the climb: a first-order flight path made q
	// and fz jump by 0.13 rad/s and 3 m/s² in one 5 ms step there.
	/** @type {import('../src/trajectory/index.js').TrajectorySpec[]} */
	const specs = [
		{ duration: 120, altitude: 120, pattern: { type: 'straight' }, altitudeProfile: { type: 'steps', amplitude: 30, interval: 20 } },
		{
			duration: 120,
			altitude: 120,
			pattern: { type: 'waypoints', points: [[0, 800, 160], [600, 800, 90], [600, 0, 140]] }
		}
	];
	for (const spec of specs) {
		const traj = createTrajectory({ ...spec, turbulence: 0, wind: 'none' }, {});
		const s = traj.sample(traj.rate);
		let dq = 0;
		let dfz = 0;
		for (let i = 1; i < s.length; i++) {
			dq = Math.max(dq, Math.abs(s[i].q - s[i - 1].q));
			dfz = Math.max(dfz, Math.abs(s[i].fz - s[i - 1].fz));
		}
		// Pitch acceleration ≤ 20°/s² is 0.0017 rad/s per step; leave room for the bank's own.
		assert.ok(dq < 0.01, `${spec.pattern.type}: q jumps ${dq.toFixed(4)} rad/s in one step`);
		assert.ok(dfz < 0.3, `${spec.pattern.type}: fz jumps ${dfz.toFixed(3)} m/s² in one step`);
	}
});

test('geometry that cannot be flown is refused', () => {
	for (const pattern of [
		{ type: 'orbit', radius: 0 },
		{ type: 'orbit', radius: -50 },
		{ type: 'orbit', direction: 0 },
		{ type: 'racetrack', radius: 0 },
		{ type: 'racetrack', length: -1 },
		{ type: 'figure8', radius: 0 },
		{ type: 'lawnmower', spacing: 0 },
		{ type: 'lawnmower', legs: 0 },
		{ type: 'random', legLength: [0, 100] },
		{ type: 'waypoints', points: [[0, 1], [NaN, 2]] }
	]) {
		assert.throws(() => createTrajectory({ duration: 60, altitude: 100, wind: 'none', pattern: /** @type {any} */ (pattern) }), RangeError, JSON.stringify(pattern));
	}
});

test('a wind the aircraft cannot outfly is flagged', () => {
	const fine = createTrajectory({ duration: 10, altitude: 100, airspeed: 24, wind: { speed: 15, direction: 90 }, pattern: { type: 'straight' } });
	assert.deepEqual(fine.warnings, []);
	const strong = createTrajectory({ duration: 10, altitude: 100, airspeed: 20, wind: { speed: 21, direction: 90 }, pattern: { type: 'straight' } });
	assert.equal(strong.warnings.length, 1);
	assert.match(strong.warnings[0], /at or above the airspeed/);
});

test('orbit: holds its radius round its centre once settled', () => {
	for (const [wind, tol] of /** @type {const} */ ([
		['none', 1],
		[{ speed: 6, direction: 30 }, 6]
	])) {
		for (const direction of /** @type {const} */ ([1, -1])) {
			const traj = createTrajectory(
				{ duration: 240, altitude: 120, start: { x: 50, y: -20, heading: 45 }, wind, turbulence: 0, pattern: { type: 'orbit', direction } },
				{}
			);
			const [cr, ca] = /** @type {[number, number]} */ (/** @type {any} */ (traj.spec.pattern).center);
			// The centre is [right, ahead] of the start: 180 m to the side the orbit turns.
			const h = 45 * DEG;
			const cx = 50 + cr * Math.cos(h) + ca * Math.sin(h);
			const cy = -20 - cr * Math.sin(h) + ca * Math.cos(h);
			const settled = traj.sample(5).filter((s) => s.t > 60);
			for (const s of settled) {
				const r = Math.hypot(s.x - cx, s.y - cy);
				assert.ok(Math.abs(r - 180) < tol, `wind ${JSON.stringify(wind)} dir ${direction}: r=${r.toFixed(2)} at t=${s.t}`);
				// Turning the right way round: positive turn rate is clockwise.
				assert.ok(Math.sign(s.turnRate) === direction);
			}
			// The circle that fits the flown track best is centred on the asked-for centre.
			const fit = fitCircle(settled);
			assert.ok(Math.hypot(fit.x - cx, fit.y - cy) < tol, `centre off by ${Math.hypot(fit.x - cx, fit.y - cy)}`);
			assert.ok(Math.abs(fit.r - 180) < tol, `fitted radius ${fit.r}`);
		}
	}
});

/**
 * Least-squares circle through points (Kåsa's fit: x² + y² + D·x + E·y + F = 0, linear).
 * @param {Array<{ x: number, y: number }>} pts
 */
function fitCircle(pts) {
	// Normal equations for [D, E, F].
	const A = [
		[0, 0, 0],
		[0, 0, 0],
		[0, 0, 0]
	];
	const b = [0, 0, 0];
	for (const { x, y } of pts) {
		const row = [x, y, 1];
		const z = -(x * x + y * y);
		for (let i = 0; i < 3; i++) {
			for (let j = 0; j < 3; j++) A[i][j] += row[i] * row[j];
			b[i] += row[i] * z;
		}
	}
	// Cramer's rule.
	/** @param {number[][]} m */
	const det = (m) =>
		m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
	const d = det(A);
	/** @param {number} c */
	const solve = (c) => det(A.map((row, i) => row.map((v, j) => (j === c ? b[i] : v)))) / d;
	const D = solve(0);
	const E = solve(1);
	const F = solve(2);
	return { x: -D / 2, y: -E / 2, r: Math.sqrt((D * D + E * E) / 4 - F) };
}

test('lawnmower: parallel legs at the spacing, alternating direction', () => {
	for (const spacing of [120, 250]) {
		// An even number of legs, so the way home (from the last leg's near end back round
		// to the first) stays clear of the middle band where the legs are measured.
		const legs = 4;
		const traj = createTrajectory(
			{ duration: 600, altitude: 100, turbulence: 0, wind: 'none', pattern: { type: 'lawnmower', legs, spacing, legLength: 800 } },
			{}
		);
		const seen = new Set();
		for (const s of traj.sample(2)) {
			// The middle of the legs, flown along them (not the turns or the way home).
			if (s.y < 350 || s.y > 650) continue;
			if (Math.abs(angleDiff(s.course, 0)) > 8 && Math.abs(angleDiff(s.course, 180)) > 8) continue;
			const k = Math.round(s.x / spacing);
			assert.ok(k >= 0 && k < legs, `leg ${k} at x=${s.x}`);
			assert.ok(Math.abs(s.x - k * spacing) < 3, `spacing ${spacing}: x=${s.x.toFixed(2)} off leg ${k}`);
			// Even legs run ahead (north), odd ones back.
			assert.ok(Math.abs(angleDiff(s.course, k % 2 === 0 ? 0 : 180)) < 4, `leg ${k} course ${s.course}`);
			seen.add(k);
		}
		assert.equal(seen.size, legs);
	}
});

test('racetrack, figure8 and waypoints follow their geometry', () => {
	// Racetrack: the two straights 2·radius apart, to the right of the start. Rolling out
	// of each half circle the aircraft swings a few metres wide and settles back.
	const rt = createTrajectory({ duration: 400, altitude: 100, turbulence: 0, wind: 'none', pattern: { type: 'racetrack' } }, {});
	const legs = rt.sample(2).filter((s) => s.t > 120 && s.y > 150 && s.y < 450);
	const off = legs.map((s) => Math.min(Math.abs(s.x), Math.abs(s.x - 300)));
	assert.ok(Math.max(...off) < 6, `racetrack: ${Math.max(...off)} m off the straights`);
	assert.ok(off.reduce((a, b) => a + b, 0) / off.length < 2);
	// Figure 8: on one of the two circles of radius 150 either side of the start.
	const f8 = createTrajectory({ duration: 400, altitude: 100, turbulence: 0, wind: 'none', pattern: { type: 'figure8' } }, {});
	// The bank reverses where the circles meet, through the roll-rate limit, so the
	// aircraft swings wide of the crossing for a moment; elsewhere it is on the circle.
	let sum = 0;
	let count = 0;
	for (const s of f8.sample(2).filter((s) => s.t > 60)) {
		const r = Math.min(Math.abs(Math.hypot(s.x - 150, s.y) - 150), Math.abs(Math.hypot(s.x + 150, s.y) - 150));
		assert.ok(r < 15, `figure8 off both circles by ${r} at t=${s.t}`);
		sum += r;
		count++;
	}
	assert.ok(sum / count < 3, `figure8 mean distance from the circles ${sum / count}`);
	// Waypoints with altitudes, looping: passes near every point, at its altitude.
	const points = /** @type {Array<[number, number, number]>} */ ([
		[0, 600, 120],
		[600, 600, 150],
		[600, 0, 100]
	]);
	const wp = createTrajectory({ duration: 500, altitude: 100, turbulence: 0, wind: 'none', pattern: { type: 'waypoints', points } }, {});
	const samples = wp.sample(5);
	for (const [px, py] of points) {
		const nearest = Math.min(...samples.map((s) => Math.hypot(s.x - px, s.y - py)));
		assert.ok(nearest < 90, `waypoint ${px},${py}: closest ${nearest}`);
	}
	const top = Math.max(...samples.map((s) => s.altitude));
	assert.ok(top > 148 && top < 152, `waypoint altitude ${top}`);
});

test('random pattern: differs per seed, the same for the same seed', () => {
	/** @param {number} seed */
	const fly = (seed) => createTrajectory({ seed, duration: 300, altitude: 100, turbulence: 0, wind: 'none', pattern: { type: 'random' } }, {});
	const a = fly(1).stateAt(250);
	const b = fly(1).stateAt(250);
	const c = fly(2).stateAt(250);
	assert.deepEqual(a, b);
	assert.ok(Math.hypot(a.x - c.x, a.y - c.y) > 100);
	// The mission does not depend on how long it is flown.
	const longer = createTrajectory({ seed: 1, duration: 400, altitude: 100, turbulence: 0, wind: 'none', pattern: { type: 'random' } }, {});
	assert.deepEqual(longer.stateAt(250), a);
});

test('every pattern flies 300 s in calm, moderate and strong wind', () => {
	/** @type {import('../src/trajectory/index.js').PatternSpec[]} */
	const patterns = [
		{ type: 'straight' },
		{ type: 'orbit' },
		{ type: 'racetrack' },
		{ type: 'figure8' },
		{ type: 'lawnmower' },
		{ type: 'waypoints', points: [[300, 500], [-400, 900], [-200, -300]] },
		{ type: 'random' }
	];
	const winds = [0, 8, 14];
	/** @type {import('../src/trajectory/index.js').AltitudeProfile[]} */
	const profiles = [
		{ type: 'constant' },
		{ type: 'sine', amplitude: 20, period: 90 },
		{ type: 'steps', amplitude: 30, interval: 45 }
	];
	for (const pattern of patterns) {
		for (const [i, windSpeed] of winds.entries()) {
			const profile = profiles[(i + patterns.indexOf(pattern)) % profiles.length];
			const spec = { seed: 3, duration: 300, altitude: 110, pattern, altitudeProfile: profile, start: { heading: 37 } };
			const traj = createTrajectory(spec, { sea: { windSpeed, windDirection: 200 } });
			const command = createAltitudeProfile(profile, 110, 3, 300);
			const cmd = { altitude: 0, rate: 0 };
			const label = `${pattern.type} / U10 ${windSpeed} / ${profile.type}`;
			for (const s of traj.sample(10)) {
				for (const c of sampleColumns) assert.ok(Number.isFinite(/** @type {any} */ (s)[c]), `${label}: ${c} at t=${s.t}`);
				// The command is ±35°; a gust can carry the bank a little past it.
				assert.ok(Math.abs(s.bank) <= autopilot.maxBank + 5, `${label}: bank ${s.bank} at t=${s.t}`);
				if (s.t > 30) {
					command(s.t, cmd);
					assert.ok(Math.abs(s.altitude - cmd.altitude) < 5, `${label}: altitude ${s.altitude} vs ${cmd.altitude} at t=${s.t}`);
				}
			}
		}
	}
});

test('windAloft: the sea wind at 10 m, rising gently with height along the log profile', () => {
	const w10 = windAloft({ windSpeed: 12, windDirection: 90 }, 10);
	assert.ok(Math.abs(w10.speed - 12) < 1e-9);
	assert.ok(Math.abs(w10.east - 12) < 1e-9 && Math.abs(w10.north) < 1e-9);
	const w120 = windAloft({ windSpeed: 12, windDirection: 90 }, 120);
	assert.ok(w120.speed > 13 && w120.speed < 16, `U(120) ${w120.speed}`);
	assert.ok(w120.z0 > 1e-5 && w120.z0 < 1e-2, `z0 ${w120.z0}`);
	// u* ≈ 0.4 m/s at 12 m/s (a drag coefficient of about 1.2e-3).
	const cd = (w120.frictionVelocity / 12) ** 2;
	assert.ok(cd > 0.8e-3 && cd < 2e-3, `Cd ${cd}`);
	assert.equal(windAloft({ windSpeed: 0, windDirection: 0 }, 100).speed, 0);
});

test('CSV and JSON writers', () => {
	const traj = createTrajectory({ duration: 5, altitude: 80, pattern: { type: 'orbit' } }, { sea });
	const samples = traj.sample(20);
	const csv = samplesToCsv(samples);
	assert.ok(csv.endsWith('\n') && !csv.includes('\r'));
	assert.ok(!/[eE][+-]?\d/.test(csv.split('\n').slice(1).join('\n')), 'no exponents');
	const { columns, rows } = parseCsv(csv);
	assert.deepEqual(columns, [...sampleColumns]);
	assert.equal(rows.length, samples.length);
	for (const [i, row] of rows.entries()) {
		assert.equal(row.length, columns.length);
		for (const [j, c] of columns.entries()) assert.ok(Math.abs(row[j] - /** @type {any} */ (samples[i])[c]) < 1e-6);
	}
	const imu = parseCsv(imuToCsv(samples));
	assert.deepEqual(imu.columns, ['t', 'fx', 'fy', 'fz', 'p', 'q', 'r']);
	assert.deepEqual(imu.columns, [...imuColumns]);
	assert.equal(imu.rows[3].length, 7);
	const two = parseCsv(samplesToCsv(samples, ['t', 'x']));
	assert.deepEqual(two.columns, ['t', 'x']);
	const json = JSON.parse(samplesToJson(samples));
	assert.equal(json.length, samples.length);
	assert.deepEqual(json[4], samples[4]);
	const state = toAircraftState(samples[4]);
	assert.deepEqual(Object.keys(state).sort(), ['altitude', 'bank', 'heading', 'pitch', 'speed', 'time', 'turnRate', 'visible', 'x', 'y']);
	assert.equal(state.speed, samples[4].airspeed);
	assert.equal(state.time, samples[4].t);
	assert.equal(samples[4].gimbalPitch, -35);
});

test('enuToGeodetic: the tangent-plane point, to well under a millimetre', () => {
	// Forward: geodetic (on the ellipsoid) → ECEF → ENU at the origin; the inverse of
	// that is what enuToGeodetic does. A plane point sits above the ellipsoid, so go
	// forward from the lat/lon found with the height that puts it back on the plane.
	const a = 6378137;
	const e2 = 6.69437999014e-3;
	/** @param {number} lat @param {number} lon @param {number} h */
	const ecef = (lat, lon, h) => {
		const [p, l] = [lat * DEG, lon * DEG];
		const n = a / Math.sqrt(1 - e2 * Math.sin(p) ** 2);
		return [(n + h) * Math.cos(p) * Math.cos(l), (n + h) * Math.cos(p) * Math.sin(l), (n * (1 - e2) + h) * Math.sin(p)];
	};
	for (const origin of [{ lat: 10, lon: -30 }, { lat: 45, lon: 5.3 }, { lat: 60, lon: 170 }, { lat: -52, lon: -70 }]) {
		const toGeo = enuToGeodetic(origin);
		const o = ecef(origin.lat, origin.lon, 0);
		const [sp, cp, sl, cl] = [Math.sin(origin.lat * DEG), Math.cos(origin.lat * DEG), Math.sin(origin.lon * DEG), Math.cos(origin.lon * DEG)];
		for (const [e, n] of [[1000, 1000], [3000, 3000], [5000, -5000], [-4000, 2500]]) {
			const [lat, lon] = toGeo(e, n);
			// The plane point's ECEF, and the ellipsoid normal through (lat, lon) must pass through it.
			const P = [o[0] - sl * e - sp * cl * n, o[1] + cl * e - sp * sl * n, o[2] + cp * n];
			let h = 0;
			for (let i = 0; i < 3; i++) {
				const q = ecef(lat, lon, h);
				const up = [Math.cos(lat * DEG) * Math.cos(lon * DEG), Math.cos(lat * DEG) * Math.sin(lon * DEG), Math.sin(lat * DEG)];
				h += (P[0] - q[0]) * up[0] + (P[1] - q[1]) * up[1] + (P[2] - q[2]) * up[2];
			}
			const q = ecef(lat, lon, h);
			const miss = Math.hypot(P[0] - q[0], P[1] - q[1], P[2] - q[2]);
			assert.ok(miss < 1e-3, `origin ${JSON.stringify(origin)}, (${e}, ${n}): off by ${miss} m`);
			// About d²/2R above the ellipsoid.
			assert.ok(Math.abs(h - (e * e + n * n) / (2 * 6371000)) < 0.05 * h + 0.01, `height ${h}`);
		}
		// The origin itself.
		const [lat0, lon0] = toGeo(0, 0);
		assert.ok(Math.abs(lat0 - origin.lat) < 1e-10 && Math.abs(lon0 - origin.lon) < 1e-10);
	}
});
