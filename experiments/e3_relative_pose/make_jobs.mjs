#!/usr/bin/env node
/**
 * E3's render jobs: E2's domain-randomised flights with the camera hard-mounted on the
 * airframe (gimbal.stabilized 0), so the camera turns with the aircraft and the pose
 * between frames has a real rotation as well as a translation.
 *
 *   node experiments/e3_relative_pose/make_jobs.mjs [--train 3000] [--val 100] [--long 120]
 *     [--rate 30] [--pilot 4] [--out experiments/e3_relative_pose/jobs]
 *
 * Each run starts as `randomRun(seed, base)` (sea, sky, turbulence, start heading and
 * pattern as it draws them; wind aloft 'sea', so the aircraft crabs), then E3 redraws, from
 * a stream of its own seeded by the run seed:
 *   - camera mount pitch uniform in [-90, -60] (from the nose), fov 92, yaw 0, stabilized 0;
 *   - altitude log-uniform in 30–200 m;
 *   - airspeed uniform in [max(15, WIND_MARGIN × wind aloft), 60] m/s.
 * Run lengths: train, val and pilot 12 s; test-long 60 s (for drift over a minute);
 * test-rate 20 s, 30 conditions × render.seaRate 0, 1 and 2 with the same seeds (paired).
 * Run seeds are unique across sets and never 0 (E1's).
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { expandJob, randomRun, WIND_MARGIN } from '../../src/conditions/index.js';
import { mixSeed, sampler } from '../../src/conditions/random.js';
import { windAloft } from '../../src/trajectory/index.js';

const HERE = fileURLToPath(new URL('.', import.meta.url));

/** @param {number} duration */
const base = (duration) => ({
	trajectory: { duration, wind: 'sea', gimbal: { yaw: 0, fov: 92, stabilized: 0 } },
	render: { width: 720, height: 1280, fps: 10, warmup: 8, quality: 'high', modes: ['shaded'] }
});

export const SET_SEEDS = Object.freeze({ pilot: 3909, train: 3101, val: 3202, 'test-long': 3303, 'test-rate': 3606 });
export const DURATION = Object.freeze({ pilot: 12, train: 12, val: 12, 'test-long': 60, 'test-rate': 20 });
export const SEA_RATES = [0, 1, 2];
const ALT = [30, 200];
const PITCH = [-90, -60];
const AIRSPEED = [15, 60];

const r3 = (/** @type {number} */ x) => Math.round(x * 1000) / 1000;

/** One E3 run for a candidate seed, or null if no airspeed in range outflies the wind. */
function e3Run(/** @type {number} */ seed, /** @type {number} */ duration) {
	const run = randomRun(seed, base(duration));
	const { uniform, logUniform } = sampler(mixSeed(seed, 0xe3));
	const pitch = uniform(PITCH[0], PITCH[1]);
	const altitude = logUniform(ALT[0], ALT[1]);
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
			gimbal: { pitch: r3(pitch) }
		},
		render: { seaSeed: run.render.seaSeed }
	};
}

/** @param {any} run */
function slug(run) {
	const p = run.trajectory.pattern;
	const name = { straight: 'transit', orbit: 'loiter', lawnmower: 'survey', racetrack: 'racetrack', figure8: 'figure8', random: 'random' }[/** @type {string} */ (p.type)] ?? p.type;
	return `w${Math.round(run.sea.windSpeed)}-sun${Math.round(run.light.sunElevation)}-a${Math.round(run.trajectory.altitude)}-v${Math.round(run.trajectory.airspeed)}-${name}`;
}

/** @param {Record<string, number>} counts */
export function makeJobs(counts) {
	const used = new Set();
	/** @type {Record<string, any>} */
	const jobs = {};
	const seeds = ['set,index,seed,seaSeed,seaRate'];
	for (const set of /** @type {(keyof typeof SET_SEEDS)[]} */ (['pilot', 'train', 'val', 'test-long', 'test-rate'])) {
		const n = counts[set] ?? 0;
		if (!n) continue;
		const drawn = [];
		for (let i = 0; i < n; i++) {
			let run = null;
			for (let j = 0; j < 64 && !run; j++) {
				const seed = mixSeed(SET_SEEDS[set], 64 * i + j) % 1e9;
				if (seed === 0 || used.has(seed)) continue;
				run = e3Run(seed, DURATION[set]);
			}
			if (!run) throw new Error(`make_jobs: no acceptable seed for ${set} run ${i}`);
			used.add(run.trajectory.seed);
			drawn.push(run);
		}
		const runs = set === 'test-rate'
			? drawn.flatMap((run, i) => SEA_RATES.map((rate) => ({ ...structuredClone(run), id: `c${String(i).padStart(3, '0')}-rate${rate}`, render: { ...run.render, seaRate: rate } })))
			: drawn.map((run) => ({ ...run, id: slug(run) }));
		runs.forEach((r, i) => seeds.push(`${set},${i},${r.trajectory.seed},${r.render.seaSeed},${r.render.seaRate ?? 1}`));
		jobs[set] = {
			name: `e3-${set}`,
			description: `E3 ${set}: ${runs.length} runs, ${DURATION[set]} s at 10 fps, 720x1280 portrait, vfov 92, camera hard-mounted at pitch -90..-60, altitude 30-200 m (log-uniform), airspeed 15-60 m/s, wind aloft from the sea` +
				(set === 'test-rate' ? `; ${n} conditions x render.seaRate ${SEA_RATES.join(', ')}` : '') + '. Written by make_jobs.mjs.',
			base: base(DURATION[set]),
			runs
		};
	}
	return { jobs, seeds: seeds.join('\n') + '\n' };
}

/** Checks the jobs expand, stay in range and use disjoint seeds. */
export function checkJobs(/** @type {Record<string, any>} */ jobs) {
	const owner = new Map();
	for (const [set, job] of Object.entries(jobs)) {
		const runs = expandJob(job);
		if (new Set(runs.map((r) => r.id)).size !== runs.length) throw new Error(`${set}: duplicate run ids`);
		for (const r of runs) {
			const t = r.trajectory;
			const ok = t.duration === DURATION[/** @type {keyof typeof DURATION} */ (set)] && r.render.fps === 10 && r.render.width === 720 && r.render.height === 1280 &&
				t.gimbal.fov === 92 && t.gimbal.stabilized === 0 && t.gimbal.pitch >= PITCH[0] && t.gimbal.pitch <= PITCH[1] &&
				t.altitude >= ALT[0] && t.altitude <= ALT[1] && t.airspeed >= AIRSPEED[0] && t.airspeed <= AIRSPEED[1] && t.wind === 'sea' && t.seed !== 0;
			if (!ok) throw new Error(`${set}: run ${r.id} is out of E3's ranges`);
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
			train: { type: 'string', default: '3000' },
			val: { type: 'string', default: '100' },
			long: { type: 'string', default: '120' },
			rate: { type: 'string', default: '30' },
			pilot: { type: 'string', default: '4' },
			out: { type: 'string', default: join(HERE, 'jobs') }
		}
	});
	const counts = { pilot: +values.pilot, train: +values.train, val: +values.val, 'test-long': +values.long, 'test-rate': +values.rate };
	const { jobs, seeds } = makeJobs(counts);
	const n = checkJobs(jobs);
	mkdirSync(values.out, { recursive: true });
	for (const [set, job] of Object.entries(jobs)) {
		writeFileSync(join(values.out, `${set}.json`), JSON.stringify(job, null, '\t') + '\n');
		console.log(`${set}: ${job.runs.length} runs`);
	}
	writeFileSync(join(values.out, 'seeds.csv'), seeds);
	console.log(`checked: ranges, ${n} distinct sea seeds, none shared between sets`);
}
