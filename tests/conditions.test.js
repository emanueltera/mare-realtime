/**
 * The conditions module: presets, run resolution, job expansion and the randomiser.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import {
	expandJob,
	lightPresets,
	patternPresets,
	randomRun,
	renderDefaults,
	resolveRun,
	runSummary,
	seaPresets,
	seaSeedFor,
	variedDefaults,
	variedRun,
	variedWaters,
	WIND_MARGIN
} from '../src/conditions/index.js';
import { seaDefaults, lightDefaults, lookDefaults } from '../src/ocean/scene.js';
import { CLOUD_PERIOD } from '../src/ocean/engine/sky.js';
import { seaStats } from '../src/ocean/spectrum.js';
import { createTrajectory, windAloft } from '../src/trajectory/index.js';

const SEA_FIELDS = ['windSpeed', 'windDirection', ...Object.keys(seaDefaults)].sort();
const LIGHT_FIELDS = ['sunElevation', 'sunAzimuth', ...Object.keys(lightDefaults)].sort();
const jobsDir = fileURLToPath(new URL('../jobs/', import.meta.url));

test('presets have the documented names and exactly the scene fields', () => {
	for (const name of ['glassy', 'calm', 'light', 'moderate', 'fresh', 'strong', 'gale', 'moderate-swell', 'cross-swell']) {
		assert.ok(seaPresets[name], `sea preset ${name}`);
	}
	for (const name of ['midday', 'morning', 'low-sun', 'overcast', 'hazy', 'dusk']) assert.ok(lightPresets[name], name);
	for (const name of ['transit', 'loiter', 'survey', 'racetrack', 'figure8', 'random']) assert.ok(patternPresets[name], name);

	for (const [name, p] of Object.entries(seaPresets)) {
		assert.deepEqual(Object.keys(p.sea).sort(), SEA_FIELDS, name);
		for (const v of Object.values(p.sea)) assert.ok(Number.isFinite(v), name);
		assert.ok(p.label && p.description, name);
		// Every preset resolves to itself.
		assert.deepEqual(resolveRun({ sea: name }).sea, p.sea);
	}
	for (const [name, p] of Object.entries(lightPresets)) {
		assert.deepEqual(Object.keys(p.light).sort(), LIGHT_FIELDS, name);
		assert.deepEqual(resolveRun({ light: name }).light, p.light);
		// The randomiser's rules: no closed deck with the sun near the horizon, thick air under one.
		if (p.light.sunElevation < 6) assert.ok(p.light.overcast <= 0.4, name);
		if (p.light.overcast > 0.7) assert.ok(p.light.haze >= 0.3, name);
	}
	for (const [name, p] of Object.entries(patternPresets)) {
		assert.deepEqual(resolveRun({ trajectory: { pattern: name } }).trajectory.pattern, p.pattern);
	}
	assert.ok(Object.isFrozen(seaPresets.moderate.sea));
});

test('sea presets rise from glassy to gale, and their descriptions quote the simulated sea', () => {
	const order = ['glassy', 'calm', 'light', 'moderate', 'fresh', 'strong', 'gale'];
	const heights = order.map((n) => seaStats(seaPresets[n].sea).significantHeight);
	for (let i = 1; i < heights.length; i++) assert.ok(heights[i] > heights[i - 1], `${order[i]} above ${order[i - 1]}`);
	for (const p of Object.values(seaPresets)) {
		const stats = seaStats(p.sea);
		assert.match(p.description, new RegExp(`Hs ${stats.significantHeight.toFixed(stats.significantHeight < 0.5 ? 2 : 1).replace('.', '\\.')} m`));
		assert.match(p.description, new RegExp(`Tp ${stats.peakPeriod.toFixed(1).replace('.', '\\.')} s`));
	}
	const cross = seaPresets['cross-swell'].sea;
	const off = Math.abs(((cross.swellDirection - cross.windDirection + 540) % 360) - 180);
	assert.ok(off >= 70 && off <= 90, 'cross swell crosses the wind sea');
});

test('resolveRun fills every default and returns a fresh object', () => {
	const partial = { sea: 'calm', trajectory: { duration: 12, gimbal: { pitch: -50 } }, render: { fps: 5 } };
	const before = structuredClone(partial);
	const run = resolveRun(partial);
	assert.deepEqual(partial, before, 'input untouched');
	assert.deepEqual(run.render, { ...renderDefaults, fps: 5, modes: ['shaded'] });
	assert.equal(run.imuRate, 200);
	assert.deepEqual(run.trajectory.gimbal, { pitch: -50, yaw: 0, fov: 48, stabilized: 0.55 });
	assert.deepEqual(run.trajectory.start, { x: 0, y: 0, heading: 0 });
	assert.equal(run.trajectory.seed, 1);
	assert.equal(run.trajectory.airspeed, 24);
	assert.equal(run.trajectory.turbulence, 0.15);
	assert.equal(run.trajectory.wind, 'sea');
	assert.deepEqual(run.trajectory.altitudeProfile, { type: 'constant' });
	assert.equal(run.id, 'calm-morning-transit-s1');
	assert.ok(!('origin' in run));

	// Fresh: editing the result touches neither the presets nor the defaults.
	run.sea.windSpeed = 30;
	run.render.modes.push('height');
	assert.equal(seaPresets.calm.sea.windSpeed, 3);
	assert.deepEqual(renderDefaults.modes, ['shaded']);

	// A resolved run resolves to itself (run.json can be fed straight back in).
	assert.deepEqual(resolveRun(JSON.parse(JSON.stringify(resolveRun(partial)))), resolveRun(partial));
	// An object sea is filled from the scene defaults.
	assert.deepEqual(resolveRun({ sea: { windSpeed: 12, windDirection: 270 } }).sea, { ...seaDefaults, windSpeed: 12, windDirection: 270 });
	assert.deepEqual(resolveRun({ origin: { lat: 43.1, lon: 5.9 } }).origin, { lat: 43.1, lon: 5.9 });
	// The sea runs in real time unless a run freezes (0) or rescales it.
	assert.equal(run.render.seaRate, 1);
	assert.deepEqual(run.render.look, lookDefaults);
	assert.deepEqual(resolveRun({ render: { look: { foam: 0.5 } } }).render.look, { ...lookDefaults, foam: 0.5 });
	assert.equal(resolveRun({ render: { seaRate: 0 } }).render.seaRate, 0);
	assert.equal(expandJob({ name: 'x', sweep: { 'render.seaRate': [0, 1] } })[0].render.seaRate, 0);
});

test('resolveRun rejects bad input, naming the field', () => {
	/** @type {Array<[object, RegExp]>} */
	const cases = [
		[{ sea: 'choppy' }, /sea preset 'choppy'/],
		[{ light: 'noon' }, /light preset 'noon'/],
		[{ trajectory: { pattern: 'spiral' } }, /pattern preset 'spiral'/],
		[{ trajectory: { pattern: { type: 'spiral' } } }, /trajectory\.pattern\.type/],
		[{ trajectory: { pattern: { type: 'orbit', radius: -5 } } }, /trajectory\.pattern\.radius/],
		[{ trajectory: { pattern: { type: 'straight', radius: 5 } } }, /trajectory\.pattern\.radius/],
		[{ trajectory: { duration: 0 } }, /trajectory\.duration must be positive/],
		[{ trajectory: { duration: -3 } }, /trajectory\.duration/],
		[{ trajectory: { altitude: 'high' } }, /trajectory\.altitude must be a finite number/],
		[{ trajectory: { turbulence: 2 } }, /trajectory\.turbulence/],
		[{ trajectory: { gimbal: { tilt: 3 } } }, /trajectory\.gimbal\.tilt/],
		[{ trajectory: { wind: 'gusty' } }, /trajectory\.wind/],
		[{ render: { modes: ['shaded', 'thermal'] } }, /render mode 'thermal'/],
		[{ render: { modes: [] } }, /render\.modes/],
		[{ render: { fsp: 10 } }, /render\.fsp/],
		[{ render: { width: 640.5 } }, /render\.width must be an integer/],
		[{ render: { quality: 'ultra' } }, /render\.quality/],
		[{ render: { view: 'side' } }, /render\.view/],
		[{ render: { seaRate: -0.5 } }, /render\.seaRate must be at least 0/],
		[{ render: { look: { foam: -1 } } }, /render\.look\.foam must be at least 0/],
		[{ render: { look: { waterColor: [0, 0] } } }, /render\.look\.waterColor must be a list of 3 numbers/],
		[{ render: { look: { tint: 1 } } }, /tint/],
		[{ render: { seaRate: '0' } }, /render\.seaRate must be a finite number/],
		[{ sea: { windSpeed: 5 } }, /sea\.windDirection/],
		[{ sea: { windSpeed: 5, windDirection: 0, choppiness: 3 } }, /sea\.choppiness/],
		[{ light: { sunElevation: 10, sunAzimuth: 0, overcast: 1.5 } }, /light\.overcast/],
		[{ imuRate: 0 }, /imuRate/],
		[{ origin: { lat: 95, lon: 0 } }, /origin\.lat/],
		[{ id: '../escape' }, /id must be/],
		[{ colour: 'blue' }, /run\.colour/]
	];
	for (const [partial, message] of cases) {
		assert.throws(() => resolveRun(partial), message, JSON.stringify(partial));
	}
});

test('expandJob: a grid is the product of its axes, with stable, unique, safe ids', () => {
	const job = JSON.parse(readFileSync(jobsDir + 'conditions-grid.json', 'utf8'));
	const runs = expandJob(job);
	assert.equal(runs.length, 4 * 3 * 3 * 2);
	assert.deepEqual(expandJob(job), runs, 'same job, same runs');
	const ids = runs.map((r) => r.id);
	assert.equal(new Set(ids).size, ids.length, 'unique');
	for (const id of ids) assert.match(id, /^conditions-grid-\d{3}-[a-z0-9-]+$/);
	assert.equal(ids[0], 'conditions-grid-000-calm-midday-transit-s1');
	// The last axis varies fastest.
	assert.equal(runs[1].trajectory.seed, 2);
	assert.equal(runs[2].trajectory.pattern.type, 'orbit');
	// Every combination appears exactly once.
	const combos = new Set(runs.map(runSummary));
	assert.equal(combos.size, runs.length);
	// The base is shared.
	for (const r of runs) assert.equal(r.trajectory.duration, 30);
});

test('expandJob: seed moves the trajectory and the sea together', () => {
	const runs = expandJob({ name: 'seeds', sweep: { seed: [0, 3, 41] } });
	assert.deepEqual(
		runs.map((r) => [r.trajectory.seed, r.render.seaSeed]),
		[0, 3, 41].map((s) => [s, seaSeedFor(s)])
	);
	assert.equal(runs[0].render.seaSeed, renderDefaults.seaSeed);
	assert.deepEqual(runs.map((r) => r.id), ['seeds-000-s0', 'seeds-001-s3', 'seeds-002-s41']);
});

test('expandJob: dotted paths reach into presets and objects', () => {
	const runs = expandJob({
		name: 'Wind & altitude!',
		base: { sea: 'moderate-swell', trajectory: { pattern: 'loiter' } },
		sweep: { 'sea.windSpeed': [4, 12.5], 'trajectory.altitude': [60, 300], 'trajectory.pattern.radius': [400] }
	});
	assert.equal(runs.length, 4);
	assert.equal(runs[0].id, 'wind-altitude-000-windspeed4-altitude60');
	assert.equal(runs[3].id, 'wind-altitude-003-windspeed12p5-altitude300');
	// The rest of the preset survives the override.
	assert.equal(runs[3].sea.swellHeight, seaPresets['moderate-swell'].sea.swellHeight);
	assert.equal(runs[3].sea.windSpeed, 12.5);
	assert.deepEqual(runs[0].trajectory.pattern, { ...patternPresets.loiter.pattern, radius: 400 });
	// A sweep over the sea with no base sea starts from the default preset.
	assert.equal(expandJob({ name: 'x', sweep: { 'sea.choppiness': [0.4] } })[0].sea.windSpeed, resolveRun({}).sea.windSpeed);
	// An object value in a preset slot is named when it is a preset.
	const named = expandJob({ name: 'o', sweep: { sea: [seaPresets.gale.sea, { windSpeed: 2, windDirection: 0 }] } });
	assert.equal(named[0].id, 'o-000-gale');
	assert.match(named[1].id, /^o-001-sea-[0-9a-f]{6}$/);
});

test('expandJob rejects bad jobs clearly', () => {
	assert.throws(() => expandJob({ name: 'x', sweep: { sea: [] } }), /sweep\.sea must be a non-empty list/);
	assert.throws(() => expandJob({ name: 'x', sweep: { sea: ['rough'] } }), /run x-000: .*sea preset 'rough'/);
	assert.throws(() => expandJob({ name: 'x', sweep: { 'render.fsp': [5] } }), /render\.fsp/);
	assert.throws(() => expandJob({ name: 'x', sweep: { '__proto__.polluted': [1] } }), /dotted path/);
	assert.throws(() => expandJob({ name: 'x', sweep: { seed: [1.5] } }), /seed values must be integers/);
	assert.throws(() => expandJob({ name: 'x', random: { count: 0, seed: 1 } }), /random\.count/);
	assert.throws(() => expandJob({ name: 'x', varied: { count: 0, seed: 1 } }), /varied\.count/);
	assert.throws(() => expandJob({ name: 'x', varied: /** @type {any} */ ({ count: 1 }) }), /varied\.seed/);
	assert.throws(() => expandJob({ name: 'x', varied: /** @type {any} */ ({ count: 1, seed: 1, fov: [60, 90] }) }), /unknown field varied\.fov/);
	assert.throws(() => expandJob({ name: 'x', varied: { count: 1, seed: 1, altitude: [200, 30] } }), /varied\.altitude must be \[low, high\]/);
	assert.throws(() => expandJob({ name: 'x', varied: { count: 1, seed: 1, pitch: [-120, -60] } }), /varied\.pitch/);
	// Pitch is negative down: a range written as angles below the horizon would film the sky.
	assert.throws(() => expandJob({ name: 'x', varied: { count: 1, seed: 1, pitch: [60, 90] } }), /varied\.pitch must be \[low, high\] with -90 ≤ low ≤ high ≤ 0/);
	assert.throws(() => expandJob({ name: 'x', varied: { count: 1, seed: 1, pitch: [-90, 1] } }), /varied\.pitch/);
	assert.throws(() => expandJob({ name: 'x', varied: /** @type {any} */ ({ count: 1, seed: 1, altitude: [30, 100, 200] }) }), /varied\.altitude/);
	assert.throws(() => expandJob({ name: 'x', varied: { count: 1, seed: 1, airspeed: [15, Infinity] } }), /varied\.airspeed/);
	assert.throws(() => expandJob({ name: 'x', varied: { count: 1, seed: 1, altitude: [NaN, 50] } }), /varied\.altitude/);
	assert.throws(() => expandJob({ name: 'x', varied: { count: 100001, seed: 1 } }), /more than 100000 runs/);
	assert.throws(() => expandJob({ name: 'x', random: { count: 99999, seed: 1 }, varied: { count: 2, seed: 1 } }), /more than 100000 runs/);
	// The block's airspeeds answer to the sea's wind, not to one given in numbers.
	assert.throws(() => expandJob({ name: 'x', base: { trajectory: { wind: { speed: 40, direction: 0 } } }, varied: { count: 1, seed: 1 } }), /varied: base\.trajectory\.wind must be 'sea' or 'none'/);
	assert.equal(expandJob({ name: 'x', base: { trajectory: { wind: 'none' } }, varied: { count: 2, seed: 1 } }).length, 2);
	// An airspeed range hardly any wind lets through: the job is refused, not left short.
	assert.throws(() => expandJob({ name: 'x', varied: { count: 200, seed: 1, airspeed: [1, 2] } }), /varied: no flight for run \d+/);
	assert.throws(() => expandJob(/** @type {any} */ ({ sweep: {} })), /needs a name/);
	assert.throws(() => expandJob(/** @type {any} */ ({ name: 'x', sweeps: {} })), /unknown field 'sweeps'/);
	assert.equal(/** @type {any} */ ({}).polluted, undefined);
});

test('expandJob: random runs are deterministic and follow the grid', () => {
	const job = { name: 'mix', base: { trajectory: { duration: 20 } }, sweep: { light: ['midday', 'dusk'] }, random: { count: 5, seed: 7 } };
	const runs = expandJob(job);
	assert.equal(runs.length, 7);
	assert.deepEqual(expandJob(structuredClone(job)), runs);
	assert.equal(new Set(runs.map((r) => r.id)).size, 7);
	for (const r of runs) assert.equal(r.trajectory.duration, 20);
	for (const r of runs.slice(2)) assert.match(r.id, /^mix-00[2-6]-wind\d+-sunm?\d+-[a-z0-9]+$/);
	// A different seed draws different runs.
	const other = expandJob({ ...job, random: { count: 5, seed: 8 } });
	assert.notDeepEqual(other.slice(2).map((r) => r.sea), runs.slice(2).map((r) => r.sea));
});

test('randomRun is deterministic per seed and stays within its ranges', () => {
	assert.deepEqual(randomRun(42), randomRun(42));
	assert.notDeepEqual(randomRun(42).sea, randomRun(43).sea);
	const patterns = new Set();
	let swells = 0;
	const N = 400;
	for (let seed = 0; seed < N; seed++) {
		const r = randomRun(seed, { trajectory: { duration: 15 }, render: { fps: 4 } });
		const { sea, light, trajectory } = r;
		assert.equal(trajectory.seed, seed);
		assert.equal(r.render.seaSeed, seaSeedFor(seed));
		assert.equal(trajectory.duration, 15);
		assert.equal(r.render.fps, 4);
		assert.ok(sea.windSpeed >= 2 && sea.windSpeed <= 20);
		assert.ok(sea.windDirection >= 0 && sea.windDirection < 360);
		assert.ok(sea.swellDirection >= 0 && sea.swellDirection < 360);
		if (sea.swellHeight > 0) swells++;
		assert.ok(sea.swellHeight === 0 || (sea.swellHeight >= 0.1 && sea.swellHeight <= 2.5));
		assert.ok(light.sunElevation >= -4 && light.sunElevation <= 65);
		if (light.sunElevation < 6) assert.ok(light.overcast <= 0.4);
		if (light.sunElevation < 0) assert.ok(light.haze <= 0.55);
		if (light.overcast > 0.7) assert.ok(light.haze >= 0.3);
		assert.ok(trajectory.altitude >= 30 && trajectory.altitude <= 400);
		// 18–32 m/s, raised where needed to outfly the wind aloft by WIND_MARGIN.
		const aloft = windAloft(sea, trajectory.altitude).speed;
		assert.ok(trajectory.airspeed >= 18 && (trajectory.airspeed <= 32 || trajectory.airspeed <= WIND_MARGIN * aloft + 0.01), `airspeed ${trajectory.airspeed}`);
		assert.ok(trajectory.airspeed >= WIND_MARGIN * aloft - 0.01, `seed ${seed}: airspeed ${trajectory.airspeed} in a ${aloft.toFixed(1)} m/s wind`);
		assert.ok(trajectory.gimbal.pitch <= -15 && trajectory.gimbal.pitch >= -80);
		assert.ok(trajectory.gimbal.fov >= 35 && trajectory.gimbal.fov <= 75);
		assert.ok(trajectory.turbulence >= 0 && trajectory.turbulence <= 0.8);
		patterns.add(runSummary(r).split(' · ')[2]);
	}
	assert.deepEqual([...patterns].sort(), Object.keys(patternPresets).sort(), 'every pattern preset is drawn');
	assert.ok(swells > N * 0.6 && swells < N * 0.9, `about three in four have a swell (${swells}/${N})`);
	assert.throws(() => randomRun(1.5), /seed must be an integer/);
	// The base's id, sea and light fields the randomiser leaves alone come through.
	const r = randomRun(3, { id: 'mine', sea: { windSpeed: 1, windDirection: 0, spread: 13 }, light: { sunElevation: 0, sunAzimuth: 0, exposure: 1.4 } });
	assert.equal(r.id, 'mine');
	assert.equal(r.sea.spread, 13);
	assert.equal(r.light.exposure, 1.4);
});

test('runSummary is short and names presets', () => {
	assert.equal(runSummary(resolveRun({ sea: 'moderate', light: 'low-sun', trajectory: { pattern: 'survey' } })), 'moderate · low-sun · survey · s1');
	const custom = resolveRun({ sea: { windSpeed: 9.25, windDirection: 10, swellHeight: 1 }, light: { sunElevation: 20.4, sunAzimuth: 0, overcast: 0.9 }, trajectory: { pattern: { type: 'figure8', radius: 90 }, seed: 4 } });
	assert.equal(runSummary(custom), 'wind 9.3 m/s + swell 1 m · sun 20° overcast · figure8 · s4');
	for (let seed = 0; seed < 50; seed++) assert.ok(runSummary(randomRun(seed * 9973 + 123456789)).length <= 80);
});

test('the example job files expand', () => {
	const expected = { 'quick.json': 1, 'sea-state-sweep.json': 14, 'conditions-grid.json': 72, 'random-100.json': 100, 'varied-12.json': 12 };
	const files = readdirSync(jobsDir).filter((f) => f.endsWith('.json'));
	for (const [file, count] of Object.entries(expected)) {
		assert.ok(files.includes(file), file);
		const runs = expandJob(JSON.parse(readFileSync(jobsDir + file, 'utf8')));
		assert.equal(runs.length, count, file);
		assert.equal(new Set(runs.map((r) => r.id)).size, count, file);
	}
	const [quick] = expandJob(JSON.parse(readFileSync(jobsDir + 'quick.json', 'utf8')));
	assert.deepEqual([quick.trajectory.duration, quick.render.width, quick.render.height, quick.render.fps], [10, 640, 360, 5]);
});

test('every shipped job flies in a wind its aircraft can outfly', () => {
	for (const file of readdirSync(jobsDir).filter((f) => f.endsWith('.json'))) {
		for (const run of expandJob(JSON.parse(readFileSync(jobsDir + file, 'utf8')))) {
			const traj = createTrajectory({ ...run.trajectory, duration: 1 }, { sea: run.sea });
			assert.deepEqual(traj.warnings, [], `${run.id}`);
		}
	}
});

test('expandJob: listed runs lie over the base, keep their order, and take their ids as slugs', () => {
	/** @type {import('../src/conditions/job.js').Job} */
	const job = {
		name: 'listed',
		base: { trajectory: { duration: 8, gimbal: { fov: 92 } }, render: { fps: 10 } },
		runs: [
			{ id: 'first', trajectory: { altitude: 50, gimbal: { pitch: -70 } }, sea: 'calm' },
			{ trajectory: { altitude: 300 }, render: { seaRate: 0 } }
		]
	};
	const runs = expandJob(job);
	assert.deepEqual(runs.map((r) => r.id), ['listed-000-first', 'listed-001']);
	assert.equal(runs[0].trajectory.altitude, 50);
	assert.equal(runs[0].trajectory.gimbal.pitch, -70);
	assert.equal(runs[0].trajectory.gimbal.fov, 92, 'nested objects merge with the base');
	assert.equal(runs[0].trajectory.duration, 8);
	assert.deepEqual(runs[0].sea, resolveRun({ sea: 'calm' }).sea);
	assert.equal(runs[1].render.seaRate, 0);
	assert.equal(runs[1].render.fps, 10);
	assert.deepEqual(expandJob(structuredClone(job)), runs, 'same job, same runs');
	assert.equal(/** @type {any} */ (job.base).trajectory.gimbal.pitch, undefined, 'the base is not changed');
	// A listed run is resolved like any other: a randomRun written out comes back as itself.
	const drawn = randomRun(1234, { trajectory: { duration: 8 } });
	const [again] = expandJob({ name: 'r', runs: [{ ...drawn, id: 'x' }] });
	assert.deepEqual({ ...again, id: drawn.id }, drawn);
	assert.throws(() => expandJob({ name: 'x', runs: [] }), /runs must be a non-empty list/);
	assert.throws(() => expandJob({ name: 'x', runs: [{ render: { fsp: 5 } }] }), /render\.fsp/);
	assert.throws(() => expandJob({ name: 'x', runs: [/** @type {any} */ (3)] }), /runs\[0\] must be an object/);
});

test('expandJob: varied runs are deterministic, lie over the base and come between the random block and the listed runs', () => {
	/** @type {import('../src/conditions/job.js').Job} */
	const job = {
		name: 'mix',
		base: { trajectory: { duration: 12, gimbal: { yaw: 0, fov: 61, stabilized: 0 } }, render: { width: 640, height: 360 } },
		sweep: { light: ['midday', 'dusk'] },
		random: { count: 2, seed: 7 },
		varied: { count: 5, seed: 7 },
		runs: [{ id: 'last' }]
	};
	const runs = expandJob(job);
	assert.equal(runs.length, 10);
	assert.deepEqual(expandJob(structuredClone(job)), runs);
	assert.equal(new Set(runs.map((r) => r.id)).size, 10);
	assert.equal(runs[9].id, 'mix-009-last');
	const varied = runs.slice(4, 9);
	for (const r of varied) {
		assert.match(r.id, /^mix-00[4-8]-w\d+-sun-?\d+-a\d+-v\d+-[a-z]+-[a-z0-9]+$/);
		// What the block does not draw is the base's.
		assert.deepEqual([r.trajectory.duration, r.trajectory.gimbal.yaw, r.trajectory.gimbal.fov, r.trajectory.gimbal.stabilized], [12, 0, 61, 0]);
		assert.deepEqual([r.render.width, r.render.height, r.render.seaRate], [640, 360, 1]);
		assert.equal(r.render.seaSeed, seaSeedFor(r.trajectory.seed));
	}
	assert.equal(new Set(varied.map((r) => r.trajectory.seed)).size, 5);
	// A different seed draws different runs; the block's own ranges are kept.
	const other = expandJob({ name: 'mix', varied: { count: 5, seed: 8 } });
	assert.notDeepEqual(other.map((r) => r.sea), varied.map((r) => r.sea));
	for (const r of expandJob({ name: 'low', varied: { count: 40, seed: 3, altitude: [40, 60], airspeed: [50, 60], pitch: [-90, -85] } })) {
		assert.ok(r.trajectory.altitude >= 40 && r.trajectory.altitude <= 60);
		assert.ok(r.trajectory.airspeed >= 50 && r.trajectory.airspeed <= 60);
		assert.ok(r.trajectory.gimbal.pitch >= -90 && r.trajectory.gimbal.pitch <= -85);
	}
	// Equal bounds give exactly that value.
	for (const r of expandJob({ name: 'one', varied: { count: 20, seed: 3, altitude: [100, 100], airspeed: [60, 60], pitch: [-90, -90] } })) {
		assert.deepEqual([r.trajectory.altitude, r.trajectory.airspeed, r.trajectory.gimbal.pitch], [100, 60, -90]);
	}
	// Whatever the block draws wins over the base: a base never changes what a seed draws.
	const plain = expandJob({ name: 'b', varied: { count: 3, seed: 7 } });
	const loud = expandJob({
		name: 'b',
		base: {
			sea: 'gale',
			light: 'dusk',
			trajectory: { seed: 5, altitude: 1000, airspeed: 5, turbulence: 0.9, pattern: 'loiter', start: { heading: 1, x: 250 }, gimbal: { pitch: -10 } },
			render: { seaSeed: 9, look: { foam: 0 } }
		},
		varied: { count: 3, seed: 7 }
	});
	loud.forEach((r, i) => {
		assert.deepEqual([r.id, r.sea, r.light, r.render.look, r.render.seaSeed], [plain[i].id, plain[i].sea, plain[i].light, plain[i].render.look, plain[i].render.seaSeed]);
		assert.deepEqual({ ...r.trajectory, start: { ...r.trajectory.start, x: 0 } }, { ...plain[i].trajectory, start: { ...plain[i].trajectory.start, x: 0 } });
		assert.equal(r.trajectory.start.x, 250, 'what is not drawn is still the base\'s');
	});
});

test('expandJob: a varied block still draws the runs it always drew', () => {
	// Sets rendered from this draw have been trained on: a seed must keep giving its run.
	const [first, , third] = expandJob({ name: 'pinned', varied: { count: 3, seed: 4919 } });
	assert.equal(first.id, 'pinned-000-w13-sun-2-a288-v50-grey-random');
	assert.deepEqual(first.sea, {
		...seaDefaults,
		windSpeed: 13.085,
		windDirection: 235.798,
		fetch: 194.512,
		spread: 5.495,
		swellHeight: 3.343,
		swellPeriod: 14.352,
		swellDirection: 217.559,
		choppiness: 0.568
	});
	assert.deepEqual(first.light, { ...lightDefaults, sunElevation: -2.318, sunAzimuth: 204.742, overcast: 0.195, haze: 0.775, exposure: 0.602 });
	assert.deepEqual(
		[first.trajectory.seed, first.trajectory.altitude, first.trajectory.airspeed, first.trajectory.turbulence, first.trajectory.start.heading, first.trajectory.gimbal.pitch],
		[282719779, 287.782, 50.16, 0.16, 314.843, -65.765]
	);
	assert.deepEqual(first.render.look, { waterColor: [0.00239, 0.0052, 0.00508], sssColor: [0.15284, 0.19357, 0.20308], foam: 0.321, cloudOffset: [9216.949, 7431.567] });
	assert.equal(first.render.seaSeed, 302960390);
	assert.equal(third.id, 'pinned-002-w1-sun51-a308-v31-sediment-survey');

	// Many runs at once, by a digest of everything expandJob gives back (FNV-1a over its JSON):
	// the default ranges, another altitude and airspeed range, and a range slow enough that
	// seeds are refused and the next candidate is taken.
	/** @param {import('../src/conditions/job.js').Job} job */
	const digest = (job) => {
		const text = JSON.stringify(expandJob(job));
		let h = 0x811c9dc5;
		for (let i = 0; i < text.length; i++) h = Math.imul(h ^ text.charCodeAt(i), 0x01000193);
		return (h >>> 0).toString(16).padStart(8, '0');
	};
	assert.equal(digest({ name: 'pinned', varied: { count: 500, seed: 4919 } }), '0a7ff29a');
	assert.equal(digest({ name: 'pinned', varied: { count: 500, seed: 4101, altitude: [30, 200], airspeed: [15, 60] } }), '1bddf3d9');
	const slow = { name: 'z', varied: { count: 50, seed: 11, airspeed: /** @type {[number, number]} */ ([15, 25]) } };
	assert.equal(digest(slow), '01a25372');
	assert.deepEqual(expandJob(slow).slice(0, 4).map((r) => r.trajectory.seed), [356300964, 165193558, 713914631, 246432354]);

	// Which seed a run takes: never 0 (block seed 0 would give it to run 0), never one the
	// block has used (block seed 323 would give run 2644 the seed of run 1232).
	assert.equal(expandJob({ name: 'z', varied: { count: 1, seed: 0 } })[0].trajectory.seed, 301794027);
	const long = expandJob({ name: 'z', varied: { count: 2645, seed: 323 } });
	assert.equal(new Set(long.map((r) => r.trajectory.seed)).size, 2645);
	assert.deepEqual([long[1232].trajectory.seed, long[2644].trajectory.seed], [931771976, 411740409]);
});

test('variedRun stays within its ranges and reaches every water, pattern and wind band', () => {
	assert.deepEqual(variedRun(42), variedRun(42));
	assert.throws(() => variedRun(1.5), /seed must be an integer/);
	assert.throws(() => variedRun(5, { altitude: [-1, -1] }), /variedRun: altitude must be \[low, high\]/);
	assert.throws(() => variedRun(5, { airspeed: [75, 15] }), /variedRun: airspeed must be \[low, high\]/);
	assert.throws(() => variedRun(5, { pitch: [60, 90] }), /variedRun: pitch must be \[low, high\]/);
	assert.equal(variedRun(5, { airspeed: [1, 2] }), null, 'no airspeed in the range outflies this wind');
	const waters = new Set();
	const patterns = new Set();
	const bands = { glassy: 0, usual: 0, storm: 0 };
	let swells = 0;
	let drawn = 0;
	let refused = 0;
	const N = 1500;
	for (let seed = 1; seed <= N; seed++) {
		const run = variedRun(seed);
		if (!run) {
			refused++;
			continue;
		}
		drawn++;
		const { water, ...partial } = run;
		const r = resolveRun(partial);
		const { sea, light, trajectory } = r;
		assert.equal(trajectory.seed, seed);
		assert.equal(r.render.seaSeed, seaSeedFor(seed));
		assert.ok(sea.windSpeed >= 0.7 && sea.windSpeed <= 28);
		bands[sea.windSpeed < 2 ? 'glassy' : sea.windSpeed <= 20 ? 'usual' : 'storm']++;
		assert.ok(sea.fetch >= 5 && sea.fetch <= 800);
		assert.ok(sea.spread >= 2 && sea.spread <= 24);
		assert.ok(sea.choppiness >= 0.5 && sea.choppiness <= 1.6);
		assert.ok(sea.swellDirection >= 0 && sea.swellDirection <= 360);
		if (sea.swellHeight > 0) swells++;
		assert.ok(sea.swellHeight === 0 || (sea.swellHeight >= 0.1 && sea.swellHeight <= 4 && sea.swellPeriod >= 6 && sea.swellPeriod <= 18));
		assert.ok(light.sunElevation >= -8 && light.sunElevation <= 88);
		assert.ok(light.overcast >= 0 && light.overcast <= 1 && light.haze >= 0 && light.haze <= 1);
		assert.ok(light.exposure >= 0.5 && light.exposure <= 2);
		// The water: one of the bodies, each channel within its gain and its ±15 %.
		const body = variedWaters[water];
		for (const key of /** @type {const} */ (['waterColor', 'sssColor'])) {
			r.render.look[key].forEach((x, c) => assert.ok(x >= body[key][c] * 0.6 * 0.85 - 1e-5 && x <= body[key][c] * 1.6 * 1.15 + 1e-5, `${water} ${key}`));
		}
		assert.ok(r.render.look.foam >= 0.3 && r.render.look.foam <= 1.5);
		// The cloud map never starts within a minute's flight and drift of its wrap.
		for (const x of r.render.look.cloudOffset) assert.ok(x >= 6000 && x <= CLOUD_PERIOD - 6000, `cloud offset ${x} of a ${CLOUD_PERIOD} m map`);
		assert.ok(trajectory.altitude >= variedDefaults.altitude[0] && trajectory.altitude <= variedDefaults.altitude[1]);
		assert.ok(trajectory.airspeed >= variedDefaults.airspeed[0] && trajectory.airspeed <= variedDefaults.airspeed[1]);
		const aloft = windAloft(sea, trajectory.altitude).speed;
		assert.ok(trajectory.airspeed >= WIND_MARGIN * aloft - 0.01, `seed ${seed}: airspeed ${trajectory.airspeed} in a ${aloft.toFixed(1)} m/s wind`);
		assert.deepEqual(createTrajectory({ ...trajectory, duration: 1 }, { sea }).warnings, [], `seed ${seed}`);
		assert.ok(trajectory.gimbal.pitch >= -90 && trajectory.gimbal.pitch <= -60);
		assert.ok(trajectory.turbulence >= 0 && trajectory.turbulence <= 0.8);
		waters.add(water);
		patterns.add(trajectory.pattern.type);
	}
	assert.deepEqual([...waters].sort(), Object.keys(variedWaters).sort(), 'every water is drawn');
	assert.equal(patterns.size, Object.keys(patternPresets).length, 'every pattern preset is drawn');
	assert.deepEqual(variedWaters.ocean, { waterColor: lookDefaults.waterColor, sssColor: lookDefaults.sssColor });
	// Seven in ten in the usual band, the rest split between glassy and storm; three in four with a swell.
	assert.ok(bands.usual > drawn * 0.64 && bands.usual < drawn * 0.78, `usual ${bands.usual}/${drawn}`);
	assert.ok(bands.glassy > drawn * 0.1 && bands.glassy < drawn * 0.2, `glassy ${bands.glassy}/${drawn}`);
	assert.ok(bands.storm > drawn * 0.08 && bands.storm < drawn * 0.2, `storm ${bands.storm}/${drawn}`);
	assert.ok(swells > drawn * 0.7 && swells < drawn * 0.8, `swell ${swells}/${drawn}`);
	// With the default ranges no seed is refused: 1.3 × the strongest wind at the greatest
	// height is still well under the top airspeed.
	assert.equal(refused, 0);
	assert.ok(WIND_MARGIN * windAloft({ windSpeed: 28, windDirection: 0 }, variedDefaults.altitude[1]).speed < variedDefaults.airspeed[1]);
});
