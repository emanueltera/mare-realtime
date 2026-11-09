#!/usr/bin/env node
/**
 * E4's render jobs: E3's flights (hard-mounted camera, the same altitudes, airspeeds, mount
 * pitches and patterns) over a far wider range of water and sky than `randomRun` draws.
 *
 *   node experiments/e4_varied_scenes/make_jobs.mjs [--train 6000] [--val 100] [--varied 120]
 *     [--pilot 12] [--cam-train 3000] [--cam-val 100] [--cam-test 120] [--cam-pilot 12]
 *     [--out experiments/e4_varied_scenes/jobs]
 *
 * What changes from E3, all from a stream of E4's own seeded by the run seed:
 *   - wind: 70 % E3's uniform 2–20 m/s, 15 % glassy (log-uniform 0.7–2), 15 % storm (uniform
 *     20–28), so E3's whole range stays well covered; fetch 5–800 km (50–800); swell up to 4 m
 *     at 6–18 s from any direction (2.5 m, 8–16 s, within 90° of the wind); directional spread
 *     2–24 (fixed 8); choppiness 0.5–1.6 (0.6–1.4);
 *   - sun −8° to 88° (−4° to 65°), overcast and haze independently 0–1, without randomRun's
 *     rules that keep skies readable (the low-texture blanks are wanted here);
 *   - exposure bias 0.5–2 (fixed 1) and the cloud map moved per run (fixed), never so near a
 *     wrap of its period that a flight crosses it, where the deck's broad cover would jump;
 *   - the water: one of six bodies, its brightness ×0.6–1.6 and its channels ±15 %;
 *   - foam coverage ×0.3–1.5 (fixed 1).
 * The sets `val-varied` and `test-varied` draw from the same ranges as `train`. E3's own
 * test sets stay the comparison on E3's range (experiments/e3_relative_pose/jobs).
 *
 * The `cam-*` sets (`cam-pilot`, `cam-train`, `val-cam`, `test-cam`) draw fresh scenes, with
 * seeds of their own, from the same widened water and sky, and see them through the recorded
 * flights' camera in its rectified mode instead of E3's: 1280 × 720 landscape at a focal
 * length of 610.7 px (61.04° across the short side, 92.7° across the long), the flight
 * direction up the frame, i.e. along the short side, as in those recordings. They also reach
 * the recorded flights' heights and speeds: altitude 30–400 m (30–200), airspeed 15–75 m/s
 * (15–60). The camera stays hard-mounted; the recordings had the camera's own stabilisation
 * on, which is not simulated. With 92.7° across the track, banked turns tip a corner of the
 * frame above the horizon more often than in portrait (about 10 % of test-cam's frames show
 * some sky, 4 % of test-varied's); straight flight never does.
 *
 * No water-fixed features are added (slicks, plumes, debris): nothing here drifts with a
 * current, so such a feature would hand the network a fixed ground reference that real
 * water does not offer.
 */
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { expandJob, patternPresets, WIND_MARGIN } from '../../src/conditions/index.js';
import { mixSeed, sampler } from '../../src/conditions/random.js';
import { windAloft } from '../../src/trajectory/index.js';

const HERE = fileURLToPath(new URL('.', import.meta.url));

/** @typedef {{ width: number, height: number, fov: number }} Cam  render size, vertical field of view in degrees */
/** @typedef {{ alt: number[], airspeed: number[] }} Ranges */
/** @typedef {{ seed: number, duration: number, cam: Cam, range: Ranges }} SetSpec */

/** E3's camera: 720 × 1280 portrait, 92° across the long side, flight direction up the frame. */
export const PORTRAIT = Object.freeze({ width: 720, height: 1280, fov: 92 });
/** The recorded flights' camera (rectified mode), as the team calibrated it: focal length in px at 1280 × 720. */
export const REAL_FOCAL = 610.7;
/** That camera: 1280 × 720 landscape, flight direction up the frame (across the short side). */
export const LANDSCAPE = Object.freeze({ width: 1280, height: 720, fov: (2 * Math.atan(360 / REAL_FOCAL) * 180) / Math.PI });

const E3_RANGE = Object.freeze({ alt: [30, 200], airspeed: [15, 60] });
const REAL_RANGE = Object.freeze({ alt: [30, 400], airspeed: [15, 75] });

/** Every set, in the order they are drawn: a later set never takes an earlier one's seeds. */
export const SETS = /** @type {Readonly<Record<string, SetSpec>>} */ (Object.freeze({
	pilot: { seed: 4909, duration: 12, cam: PORTRAIT, range: E3_RANGE },
	train: { seed: 4101, duration: 12, cam: PORTRAIT, range: E3_RANGE },
	'val-varied': { seed: 4202, duration: 12, cam: PORTRAIT, range: E3_RANGE },
	'test-varied': { seed: 4303, duration: 60, cam: PORTRAIT, range: E3_RANGE },
	'cam-pilot': { seed: 4919, duration: 12, cam: LANDSCAPE, range: REAL_RANGE },
	'cam-train': { seed: 4111, duration: 12, cam: LANDSCAPE, range: REAL_RANGE },
	'val-cam': { seed: 4212, duration: 12, cam: LANDSCAPE, range: REAL_RANGE },
	'test-cam': { seed: 4313, duration: 60, cam: LANDSCAPE, range: REAL_RANGE }
}));

/** @param {number} duration @param {Cam} cam */
const base = (duration, cam) => ({
	trajectory: { duration, wind: 'sea', gimbal: { yaw: 0, fov: cam.fov, stabilized: 0 } },
	render: { width: cam.width, height: cam.height, fps: 10, warmup: 8, quality: 'high', modes: ['shaded'] }
});

const PITCH = [-90, -60];
/** How many metres the cloud map repeats over (sky.js's CLOUD_PERIOD). */
const CLOUD_MAP = 24576;
/** A 60 s flight and its cloud drift stay within this many metres of where they start. */
const CLOUD_MARGIN = 6000;

/** The water bodies E4 draws from: linear RGB, simulator values, not measurements of a real sea. */
export const WATERS = Object.freeze({
	ocean: { waterColor: [0.0011, 0.0068, 0.0165], sssColor: [0.045, 0.32, 0.56] },
	indigo: { waterColor: [0.0006, 0.0028, 0.0085], sssColor: [0.03, 0.16, 0.42] },
	turquoise: { waterColor: [0.002, 0.02, 0.025], sssColor: [0.05, 0.55, 0.55] },
	coastal: { waterColor: [0.003, 0.014, 0.008], sssColor: [0.15, 0.5, 0.25] },
	grey: { waterColor: [0.004, 0.008, 0.007], sssColor: [0.2, 0.3, 0.28] },
	sediment: { waterColor: [0.02, 0.02, 0.01], sssColor: [0.5, 0.45, 0.2] }
});

const r3 = (/** @type {number} */ x) => Math.round(x * 1000) / 1000;
const r5 = (/** @type {number} */ x) => Math.round(x * 100000) / 100000;

/** One E4 run for a candidate seed, or null if no airspeed in range outflies the wind. */
function e4Run(/** @type {number} */ seed, /** @type {Ranges} */ range) {
	const ALT = range.alt;
	const AIRSPEED = range.airspeed;
	const { random, uniform, logUniform, pick } = sampler(mixSeed(seed, 0xe4));
	const band = random();
	const windSpeed = band < 0.7 ? uniform(2, 20) : band < 0.85 ? logUniform(0.7, 2) : uniform(20, 28);
	const windDirection = uniform(0, 360);
	const hasSwell = random() >= 0.25;
	const swellHeight = uniform(0.1, 4);
	const swellOffset = uniform(-180, 180);
	const sea = {
		windSpeed: r3(windSpeed),
		windDirection: r3(windDirection),
		fetch: r3(logUniform(5, 800)),
		spread: r3(logUniform(2, 24)),
		swellHeight: hasSwell ? r3(swellHeight) : 0,
		swellPeriod: r3(uniform(6, 18)),
		swellDirection: r3((((windDirection + swellOffset) % 360) + 360) % 360),
		choppiness: r3(uniform(0.5, 1.6))
	};
	const light = {
		sunElevation: r3(uniform(-8, 88)),
		sunAzimuth: r3(uniform(0, 360)),
		overcast: r3(random()),
		haze: r3(random()),
		exposure: r3(logUniform(0.5, 2))
	};
	const waterName = pick(Object.keys(WATERS));
	const water = WATERS[/** @type {keyof typeof WATERS} */ (waterName)];
	const gain = logUniform(0.6, 1.6);
	const channels = (/** @type {number[]} */ c) => c.map((x) => r5(x * gain * uniform(0.85, 1.15)));
	const look = {
		waterColor: channels(water.waterColor),
		sssColor: channels(water.sssColor),
		foam: r3(logUniform(0.3, 1.5)),
		cloudOffset: [r3(uniform(CLOUD_MARGIN, CLOUD_MAP - CLOUD_MARGIN)), r3(uniform(CLOUD_MARGIN, CLOUD_MAP - CLOUD_MARGIN))]
	};
	const pitch = uniform(PITCH[0], PITCH[1]);
	const altitude = logUniform(ALT[0], ALT[1]);
	const u = uniform(0, 1);
	const turbulence = uniform(0, 0.8);
	const heading = uniform(0, 360);
	const pattern = pick(Object.keys(patternPresets));
	const aloft = windAloft({ windSpeed: sea.windSpeed, windDirection: 0 }, r3(altitude)).speed;
	const lo = Math.max(AIRSPEED[0], WIND_MARGIN * aloft);
	if (lo > AIRSPEED[1]) return null;
	return {
		sea,
		light,
		trajectory: {
			seed,
			pattern,
			altitude: r3(altitude),
			airspeed: r3(lo + (AIRSPEED[1] - lo) * u),
			turbulence: r3(turbulence),
			start: { heading: r3(heading) },
			gimbal: { pitch: r3(pitch) }
		},
		render: { seaSeed: 20240611 + seed, look },
		water: waterName
	};
}

/** @param {any} run */
function slug(run) {
	const p = run.trajectory.pattern;
	const name = typeof p === 'string' ? p : p.type;
	return `w${Math.round(run.sea.windSpeed)}-sun${Math.round(run.light.sunElevation)}-a${Math.round(run.trajectory.altitude)}-v${Math.round(run.trajectory.airspeed)}-${run.water}-${name}`;
}

/** @param {Record<string, number>} counts */
export function makeJobs(counts) {
	const used = new Set();
	/** @type {Record<string, any>} */
	const jobs = {};
	const seeds = ['set,index,seed,seaSeed,water'];
	for (const [set, spec] of Object.entries(SETS)) {
		const n = counts[set] ?? 0;
		if (!n) continue;
		const runs = [];
		for (let i = 0; i < n; i++) {
			let run = null;
			for (let j = 0; j < 64 && !run; j++) {
				const seed = mixSeed(spec.seed, 64 * i + j) % 1e9;
				if (seed === 0 || used.has(seed)) continue;
				run = e4Run(seed, spec.range);
			}
			if (!run) throw new Error(`make_jobs: no acceptable seed for ${set} run ${i}`);
			used.add(run.trajectory.seed);
			const { water, ...rest } = run;
			runs.push({ ...rest, id: slug(run) });
			seeds.push(`${set},${i},${run.trajectory.seed},${run.render.seaSeed},${water}`);
		}
		const what = spec.cam === PORTRAIT
			? "720x1280 portrait, E3's flights over widened water and sky"
			: `1280x720 landscape at focal ${REAL_FOCAL} px (the recorded flights' camera), altitude ${spec.range.alt.join('-')} m, airspeed ${spec.range.airspeed.join('-')} m/s, over E4's widened water and sky`;
		jobs[set] = {
			name: `e4-${set}`,
			description: `E4 ${set}: ${runs.length} runs, ${spec.duration} s at 10 fps, ${what}. Written by make_jobs.mjs.`,
			base: base(spec.duration, spec.cam),
			runs
		};
	}
	return { jobs, seeds: seeds.join('\n') + '\n' };
}

/** Checks the jobs expand, stay in range and share no seed with each other or with E3. */
export function checkJobs(/** @type {Record<string, any>} */ jobs) {
	const e3 = new Set(readFileSync(join(HERE, '../e3_relative_pose/jobs/seeds.csv'), 'utf8').trim().split('\n').slice(1).flatMap((l) => l.split(',').slice(2, 4).map(Number)));
	const seen = new Set();
	for (const [set, job] of Object.entries(jobs)) {
		const { duration, cam, range } = SETS[set];
		const runs = expandJob(job);
		if (new Set(runs.map((r) => r.id)).size !== runs.length) throw new Error(`${set}: duplicate run ids`);
		for (const r of runs) {
			const t = r.trajectory;
			const ok = t.duration === duration && t.gimbal.fov === cam.fov && r.render.width === cam.width && r.render.height === cam.height &&
				t.gimbal.yaw === 0 && t.gimbal.stabilized === 0 && t.gimbal.pitch >= PITCH[0] && t.gimbal.pitch <= PITCH[1] &&
				t.altitude >= range.alt[0] && t.altitude <= range.alt[1] && t.airspeed >= range.airspeed[0] && t.airspeed <= range.airspeed[1] &&
				t.wind === 'sea' && t.seed !== 0 && r.render.seaRate === 1 && r.render.fps === 10;
			if (!ok) throw new Error(`${set}: run ${r.id} is out of E4's ranges`);
			if (seen.has(r.render.seaSeed) || e3.has(r.render.seaSeed) || e3.has(t.seed)) throw new Error(`seed ${t.seed} / sea seed ${r.render.seaSeed} reused (${set})`);
			seen.add(r.render.seaSeed);
		}
	}
	return seen.size;
}

if (resolve(process.argv[1] ?? '') === fileURLToPath(import.meta.url)) {
	const { values } = parseArgs({
		options: {
			train: { type: 'string', default: '6000' },
			val: { type: 'string', default: '100' },
			varied: { type: 'string', default: '120' },
			pilot: { type: 'string', default: '12' },
			'cam-train': { type: 'string', default: '3000' },
			'cam-val': { type: 'string', default: '100' },
			'cam-test': { type: 'string', default: '120' },
			'cam-pilot': { type: 'string', default: '12' },
			out: { type: 'string', default: join(HERE, 'jobs') }
		}
	});
	const counts = {
		pilot: +values.pilot, train: +values.train, 'val-varied': +values.val, 'test-varied': +values.varied,
		'cam-pilot': +values['cam-pilot'], 'cam-train': +values['cam-train'], 'val-cam': +values['cam-val'], 'test-cam': +values['cam-test']
	};
	const { jobs, seeds } = makeJobs(counts);
	const n = checkJobs(jobs);
	mkdirSync(values.out, { recursive: true });
	for (const [set, job] of Object.entries(jobs)) {
		writeFileSync(join(values.out, `${set}.json`), JSON.stringify(job, null, '\t') + '\n');
		console.log(`${set}: ${job.runs.length} runs`);
	}
	writeFileSync(join(values.out, 'seeds.csv'), seeds);
	console.log(`checked: ranges, ${n} distinct sea seeds, none shared between sets or with E3`);
}
