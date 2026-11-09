#!/usr/bin/env node
/**
 * E2's render jobs: domain-randomised runs with E2's own distributions, written out one by
 * one as `runs` jobs the generator takes (docs/ARCHITECTURE.md, "Runs and jobs").
 *
 *   node experiments/e2_learned_velocity/make_jobs.mjs [--train 5000] [--val 150] [--iid 300]
 *     [--sea 150] [--alt 150] [--rate 100] [--pilot 16] [--out experiments/e2_learned_velocity/jobs]
 *
 * Each run starts as `randomRun(seed, base)` (sea, sky, turbulence, gimbal stabilisation,
 * start heading and pattern as it draws them; wind aloft 'sea', so the aircraft crabs), and
 * then E2 redraws, from a stream of its own seeded by the run seed:
 *   - gimbal pitch uniform in [-90, -60], fov 92 (the camera of E1), yaw 0;
 *   - altitude log-uniform in 30–200 m (test-alt: 220–400 m);
 *   - airspeed uniform in [max(15, WIND_MARGIN × wind aloft), 60] m/s.
 * The held-out sea regime (wind 16–20 m/s, or a swell of period 14–16 s) is rejected for
 * every set but test-sea, which keeps only it: a run seed whose sea falls on the wrong side
 * is skipped for the next candidate (seed = mixSeed(setSeed, 64·i + j) mod 1e9, j = 0, 1 …),
 * so the expansion stays a pure function of the arguments.
 *
 * Run seeds (hence sea seeds, render.seaSeed = 20240611 + seed) are unique across all sets
 * and never 0, the seed of every E1 run (E1's jobs set no seed). test-rate draws 100
 * conditions like test-iid and renders each at render.seaRate 0, 0.5, 1 and 2 with the same
 * seeds (paired). seeds.csv lists every run's set, index, seed and sea seed.
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { expandJob, randomRun, WIND_MARGIN } from '../../src/conditions/index.js';
import { mixSeed, sampler } from '../../src/conditions/random.js';
import { windAloft } from '../../src/trajectory/index.js';

const HERE = fileURLToPath(new URL('.', import.meta.url));

/** The base every E2 run shares. */
export const BASE = Object.freeze({
	trajectory: { duration: 8, wind: 'sea', gimbal: { yaw: 0, fov: 92 } },
	render: { width: 720, height: 1280, fps: 10, warmup: 8, quality: 'high', modes: ['shaded'] }
});

/** Per-set stream seeds (any integers, distinct). */
export const SET_SEEDS = Object.freeze({
	pilot: 2909,
	train: 2101,
	val: 2202,
	'test-iid': 2303,
	'test-sea': 2404,
	'test-alt': 2505,
	'test-rate': 2606
});

export const SEA_RATES = [0, 0.5, 1, 2];
const ALT_IID = [30, 200];
const ALT_HIGH = [220, 400];
const PITCH = [-90, -60];
const AIRSPEED = [15, 60];

/**
 * The held-out sea regime: wind 16–20 m/s, or a swell (present) of period 14–16 s.
 * @param {{ windSpeed: number, swellHeight: number, swellPeriod: number }} sea
 */
export const heldOutSea = (sea) => sea.windSpeed >= 16 || (sea.swellHeight > 0 && sea.swellPeriod >= 14);

const r3 = (/** @type {number} */ x) => Math.round(x * 1000) / 1000;

/**
 * One E2 run for a candidate seed, or null if the seed's sea is on the wrong side of the
 * held-out regime (or no airspeed in range outflies the wind).
 * @param {number} seed
 * @param {{ heldOut: boolean, altitude: number[] }} want
 */
function e2Run(seed, want) {
	const run = randomRun(seed, structuredClone(BASE));
	if (heldOutSea(run.sea) !== want.heldOut) return null;
	const { uniform, logUniform } = sampler(mixSeed(seed, 0xe2));
	const pitch = uniform(PITCH[0], PITCH[1]);
	const altitude = logUniform(want.altitude[0], want.altitude[1]);
	const u = uniform(0, 1);
	const aloft = windAloft({ windSpeed: run.sea.windSpeed, windDirection: 0 }, r3(altitude)).speed;
	const lo = Math.max(AIRSPEED[0], WIND_MARGIN * aloft);
	if (lo > AIRSPEED[1]) return null;
	const t = run.trajectory;
	return {
		sea: run.sea,
		light: run.light,
		trajectory: {
			seed: t.seed,
			pattern: t.pattern,
			altitude: r3(altitude),
			airspeed: r3(lo + (AIRSPEED[1] - lo) * u),
			turbulence: t.turbulence,
			start: { heading: t.start.heading },
			gimbal: { pitch: r3(pitch), stabilized: t.gimbal.stabilized }
		},
		render: { seaSeed: run.render.seaSeed }
	};
}

/**
 * @param {string} set
 * @param {number} count
 * @param {Set<number>} used run seeds already taken (updated)
 */
function drawSet(set, count, used) {
	const want = { heldOut: set === 'test-sea', altitude: set === 'test-alt' ? ALT_HIGH : ALT_IID };
	const out = [];
	for (let i = 0; i < count; i++) {
		let run = null;
		for (let j = 0; j < 64 && !run; j++) {
			const seed = mixSeed(SET_SEEDS[/** @type {keyof typeof SET_SEEDS} */ (set)], 64 * i + j) % 1e9;
			if (seed === 0 || used.has(seed)) continue;
			run = e2Run(seed, want);
		}
		if (!run) throw new Error(`make_jobs: no acceptable seed for ${set} run ${i} in 64 candidates`);
		used.add(run.trajectory.seed);
		out.push(run);
	}
	return out;
}

/** @param {any} run */
function slug(run) {
	const p = run.trajectory.pattern;
	const name = { straight: 'transit', orbit: 'loiter', lawnmower: 'survey', racetrack: 'racetrack', figure8: 'figure8', random: 'random' }[/** @type {string} */ (p.type)] ?? p.type;
	return `w${Math.round(run.sea.windSpeed)}-sun${Math.round(run.light.sunElevation)}-a${Math.round(run.trajectory.altitude)}-v${Math.round(run.trajectory.airspeed)}-${name}`;
}

/**
 * Every set's job.
 * @param {Record<string, number>} counts
 */
export function makeJobs(counts) {
	const used = new Set();
	/** @type {Record<string, any>} */
	const jobs = {};
	/** @type {string[]} */
	const seeds = ['set,index,seed,seaSeed,seaRate'];
	for (const set of ['pilot', 'train', 'val', 'test-iid', 'test-sea', 'test-alt', 'test-rate']) {
		const n = counts[set] ?? 0;
		if (!n) continue;
		const drawn = drawSet(set, n, used);
		/** @type {any[]} */
		let runs;
		if (set === 'test-rate') {
			runs = drawn.flatMap((run, i) =>
				SEA_RATES.map((rate) => ({ ...structuredClone(run), id: `c${String(i).padStart(3, '0')}-rate${String(rate).replace('.', 'p')}`, render: { ...run.render, seaRate: rate } }))
			);
		} else {
			runs = drawn.map((run) => ({ ...run, id: slug(run) }));
		}
		runs.forEach((r, i) => seeds.push(`${set},${i},${r.trajectory.seed},${r.render.seaSeed},${r.render.seaRate ?? 1}`));
		jobs[set] = {
			name: `e2-${set}`,
			description: `E2 ${set}: ${runs.length} runs, 8 s at 10 fps, 720x1280 portrait, vfov 92, gimbal pitch -90..-60, ` +
				(set === 'test-alt' ? 'altitude 220-400 m' : 'altitude 30-200 m') + ' (log-uniform), airspeed 15-60 m/s, wind aloft from the sea; ' +
				(set === 'test-sea' ? 'only the held-out sea regime (wind >= 16 m/s or swell period >= 14 s)' : 'the held-out sea regime (wind >= 16 m/s or swell period >= 14 s) excluded') +
				(set === 'test-rate' ? `; 100 conditions x render.seaRate ${SEA_RATES.join(', ')}` : '') + '. Written by make_jobs.mjs.',
			base: structuredClone(BASE),
			runs
		};
	}
	return { jobs, seeds: seeds.join('\n') + '\n' };
}

/**
 * Checks the jobs expand, stay in range and use disjoint seeds; throws on the first failure.
 * @param {Record<string, any>} jobs
 */
export function checkJobs(jobs) {
	/** @type {Map<number, string>} */
	const owner = new Map();
	for (const [set, job] of Object.entries(jobs)) {
		const runs = expandJob(job);
		const ids = new Set(runs.map((r) => r.id));
		if (ids.size !== runs.length) throw new Error(`${set}: duplicate run ids`);
		for (const r of runs) {
			const t = r.trajectory;
			const alt = set === 'test-alt' ? ALT_HIGH : ALT_IID;
			const ok =
				t.duration === 8 && r.render.fps === 10 && r.render.width === 720 && r.render.height === 1280 && t.gimbal.fov === 92 &&
				t.gimbal.pitch >= PITCH[0] && t.gimbal.pitch <= PITCH[1] && t.altitude >= alt[0] && t.altitude <= alt[1] &&
				t.airspeed >= AIRSPEED[0] && t.airspeed <= AIRSPEED[1] && t.wind === 'sea' && r.render.seaSeed === 20240611 + t.seed &&
				heldOutSea(r.sea) === (set === 'test-sea') && t.seed !== 0;
			if (!ok) throw new Error(`${set}: run ${r.id} is out of E2's ranges`);
			const prev = owner.get(r.render.seaSeed);
			if (prev !== undefined && !(prev === set && set === 'test-rate')) throw new Error(`sea seed ${r.render.seaSeed} in ${prev} and ${set}`);
			owner.set(r.render.seaSeed, set);
		}
	}
	return owner.size;
}

if (resolve(process.argv[1] ?? '') === fileURLToPath(import.meta.url)) {
	const { values } = parseArgs({
		options: {
			train: { type: 'string', default: '5000' },
			val: { type: 'string', default: '150' },
			iid: { type: 'string', default: '300' },
			sea: { type: 'string', default: '150' },
			alt: { type: 'string', default: '150' },
			rate: { type: 'string', default: '100' },
			pilot: { type: 'string', default: '16' },
			out: { type: 'string', default: join(HERE, 'jobs') }
		}
	});
	const counts = {
		pilot: Number(values.pilot),
		train: Number(values.train),
		val: Number(values.val),
		'test-iid': Number(values.iid),
		'test-sea': Number(values.sea),
		'test-alt': Number(values.alt),
		'test-rate': Number(values.rate)
	};
	const { jobs, seeds } = makeJobs(counts);
	const nSeeds = checkJobs(jobs);
	mkdirSync(values.out, { recursive: true });
	for (const [set, job] of Object.entries(jobs)) {
		writeFileSync(join(values.out, `${set}.json`), JSON.stringify(job, null, '\t') + '\n');
		console.log(`${set}: ${job.runs.length} runs`);
	}
	writeFileSync(join(values.out, 'seeds.csv'), seeds);
	console.log(`checked: ranges, held-out sea regime, ${nSeeds} distinct sea seeds, none shared between sets, none = E1's (seed 0)`);
}
