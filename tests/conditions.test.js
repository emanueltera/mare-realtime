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
	WIND_MARGIN
} from '../src/conditions/index.js';
import { seaDefaults, lightDefaults } from '../src/ocean/scene.js';
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
	const expected = { 'quick.json': 1, 'sea-state-sweep.json': 14, 'conditions-grid.json': 72, 'random-100.json': 100 };
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
