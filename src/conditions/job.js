/**
 * Jobs: a base run and what varies across it (docs/ARCHITECTURE.md, "Runs and jobs").
 *
 * A `sweep` is a grid over dotted paths into the RunSpec; a `random` block draws runs
 * with domain-randomised conditions instead. Either way the expansion is a pure
 * function of the job file — no clock, no `Math.random` — so the same file always
 * gives the same runs with the same ids, and a generator that stopped half-way can be
 * restarted and skip the runs already on disk.
 */
import { lightPresets, patternPresets, seaPresets } from './presets.js';
import { mixSeed, sampler, shortHash } from './random.js';
import { windAloft } from '../trajectory/index.js';
import {
	DEFAULT_LIGHT,
	DEFAULT_PATTERN,
	DEFAULT_SEA,
	isObject,
	presetName,
	resolveLight,
	resolveRun,
	resolveSea,
	seaSeedFor,
	slugify
} from './run.js';

/** @typedef {import('./run.js').RunSpec} RunSpec */
/** @typedef {import('./run.js').PartialRun} PartialRun */

/**
 * @typedef {object} Job   JSON-serialisable
 * @property {string} name          names the output directory and prefixes every run id
 * @property {string} [description]
 * @property {PartialRun} [base]    what every run shares; preset names allowed
 * @property {Record<string, unknown[]>} [sweep]  dotted path → the values it takes; every
 *   combination is a run. `seed` sets `trajectory.seed` and `render.seaSeed` together.
 * @property {{ count: number, seed: number }} [random]  `count` domain-randomised runs on
 *   `base`. With a sweep as well, the random runs follow the grid's.
 */

/** No job expands past this many runs: a mistyped sweep should fail, not fill a disk. */
const MAX_RUNS = 100000;

/** A random run's airspeed is at least this many times the mean wind at its cruise altitude. */
export const WIND_MARGIN = 1.3;

/**
 * The paths that also accept preset names, with the presets, the field each preset
 * holds its value in, and the preset used when the base has none.
 * @type {Record<string, { presets: Record<string, Record<string, any>>, field: 'sea' | 'light' | 'pattern', fallback: string }>}
 */
const PRESET_PATHS = {
	sea: { presets: seaPresets, field: 'sea', fallback: DEFAULT_SEA },
	light: { presets: lightPresets, field: 'light', fallback: DEFAULT_LIGHT },
	'trajectory.pattern': { presets: patternPresets, field: 'pattern', fallback: DEFAULT_PATTERN }
};

/**
 * @param {string} message
 * @returns {never}
 */
function fail(message) {
	throw new Error(`expandJob: ${message}`);
}

/**
 * Set `value` at a dotted path inside `target`, creating objects on the way. Where the
 * path passes through a preset name (`sea.windSpeed` on a base whose sea is
 * 'moderate'), the preset is copied in first, so the sweep varies one field of it.
 * @param {Record<string, any>} target
 * @param {string} path
 * @param {unknown} value
 */
function setPath(target, path, value) {
	const keys = path.split('.');
	for (const key of keys) {
		if (!/^[A-Za-z0-9_]+$/.test(key) || key === '__proto__' || key === 'constructor' || key === 'prototype') {
			fail(`sweep key '${path}' is not a dotted path of field names`);
		}
	}
	let node = target;
	for (let i = 0; i < keys.length - 1; i++) {
		const key = keys[i];
		const prefix = keys.slice(0, i + 1).join('.');
		const named = PRESET_PATHS[prefix];
		let next = node[key];
		if (named && (next === undefined || typeof next === 'string')) {
			const name = next ?? named.fallback;
			if (!Object.hasOwn(named.presets, name)) fail(`unknown ${named.field} preset '${name}' under sweep key '${path}'`);
			next = structuredClone(named.presets[name][named.field]);
		} else if (next === undefined) {
			next = {};
		} else if (!isObject(next)) {
			fail(`sweep key '${path}' goes through ${prefix}, which is not an object`);
		}
		node[key] = next;
		node = next;
	}
	node[keys[keys.length - 1]] = structuredClone(value);
}

/**
 * A short, file-name-safe token for one swept value, for the run id: preset names as
 * they are, numbers after their field name (`altitude120`, `s2` for a seed), anything
 * else by its type and a digest.
 * @param {string} path
 * @param {unknown} value
 * @returns {string}
 */
function token(path, value) {
	const field = path.split('.').pop() ?? path;
	if (path === 'seed') return `s${value}`;
	if (typeof value === 'string') return value;
	if (typeof value === 'number') return `${field}${value}`;
	if (typeof value === 'boolean') return `${field}-${value ? 'on' : 'off'}`;
	const named = PRESET_PATHS[path];
	if (named) {
		const name = presetName(named.presets, named.field, value);
		if (name) return name;
	}
	if (isObject(value) && typeof value.type === 'string') return `${value.type}-${shortHash(value)}`;
	return `${field}-${shortHash(value)}`;
}

/**
 * Expand a job into resolved runs with stable ids `<job>-<index>-<what varies>`, the
 * index zero-padded to the width of the largest. The grid varies its last key fastest,
 * like nested loops written in the order of the sweep's keys.
 * @param {Job} job
 * @returns {RunSpec[]}
 */
export function expandJob(job) {
	if (!isObject(job)) fail('a job must be an object');
	for (const key of Object.keys(job)) {
		if (!['name', 'description', 'base', 'sweep', 'random'].includes(key)) {
			fail(`unknown field '${key}' (expected name, description, base, sweep, random)`);
		}
	}
	if (typeof job.name !== 'string' || !/[a-z0-9]/i.test(job.name)) fail('a job needs a name');
	const name = slugify(job.name, 48);
	const base = structuredClone(job.base ?? {});
	if (!isObject(base)) fail('base must be an object');
	delete base.id;

	/** @type {Array<{ partial: PartialRun, slug: string }>} */
	const runs = [];

	if (job.sweep !== undefined) {
		if (!isObject(job.sweep)) fail('sweep must be an object of path → list of values');
		const axes = Object.entries(job.sweep);
		for (const [path, values] of axes) {
			if (!Array.isArray(values) || values.length === 0) fail(`sweep.${path} must be a non-empty list`);
			if (path === 'seed') {
				for (const v of values) if (!Number.isSafeInteger(v)) fail(`sweep.seed values must be integers (got ${JSON.stringify(v)})`);
			}
		}
		const total = axes.reduce((n, [, values]) => n * values.length, 1);
		if (total > MAX_RUNS) fail(`the sweep has ${total} combinations, more than ${MAX_RUNS}`);
		for (let i = 0; i < total; i++) {
			const partial = structuredClone(base);
			const tokens = [];
			// Mixed-radix digits of i, last axis fastest.
			let rest = i;
			const picks = new Array(axes.length);
			for (let a = axes.length - 1; a >= 0; a--) {
				const values = axes[a][1];
				picks[a] = values[rest % values.length];
				rest = Math.floor(rest / values.length);
			}
			for (let a = 0; a < axes.length; a++) {
				const [path, value] = [axes[a][0], picks[a]];
				if (path === 'seed') {
					setPath(partial, 'trajectory.seed', value);
					setPath(partial, 'render.seaSeed', seaSeedFor(/** @type {number} */ (value)));
				} else {
					setPath(partial, path, value);
				}
				// Only axes that actually vary say anything about the run.
				if (axes[a][1].length > 1) tokens.push(token(path, value));
			}
			runs.push({ partial, slug: tokens.join('-') });
		}
	}

	if (job.random !== undefined) {
		const random = job.random;
		if (!isObject(random)) fail('random must be { count, seed }');
		for (const key of Object.keys(random)) if (key !== 'count' && key !== 'seed') fail(`unknown field random.${key} (expected count, seed)`);
		if (!Number.isSafeInteger(random.count) || random.count < 1) fail(`random.count must be a positive integer (got ${JSON.stringify(random.count)})`);
		if (!Number.isSafeInteger(random.seed)) fail(`random.seed must be an integer (got ${JSON.stringify(random.seed)})`);
		if (runs.length + random.count > MAX_RUNS) fail(`the job has more than ${MAX_RUNS} runs`);
		for (let i = 0; i < random.count; i++) {
			// Nine-digit run seeds: well spread, and small enough that the sea's seed
			// (seaSeedFor) stays inside 32 bits.
			const run = randomRun(mixSeed(random.seed, i) % 1e9, base);
			const pattern = presetName(patternPresets, 'pattern', run.trajectory.pattern) ?? run.trajectory.pattern.type;
			const slug = `wind${Math.round(run.sea.windSpeed)}-sun${Math.round(run.light.sunElevation)}-${pattern}`;
			runs.push({ partial: run, slug });
		}
	}

	if (runs.length === 0) runs.push({ partial: base, slug: '' });

	const width = Math.max(3, String(runs.length - 1).length);
	return runs.map(({ partial, slug }, i) => {
		const index = String(i).padStart(width, '0');
		const id = slugify(slug ? `${name}-${index}-${slug}` : `${name}-${index}`, 120);
		try {
			return resolveRun({ ...partial, id });
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			throw new Error(`expandJob: run ${id}: ${message}`);
		}
	});
}

/**
 * One domain-randomised run:
 * any wind sea from a light breeze to a near gale, a swell three times in four, the
 * sun anywhere from just below the horizon to high, any cloud and haze within the
 * rules that keep a frame worth looking at; and a flight at any altitude from 30 to
 * 400 m (log-uniform, so low passes are as common per octave as high ones), an airspeed
 * of 18–32 m/s raised to at least WIND_MARGIN × the wind aloft, any gimbal angle from
 * shallow to steep, any heading, and one of the pattern presets.
 *
 * `base` supplies everything not randomised (duration, render settings, the sea's
 * spreading, the light's exposure, the wind model). Every draw is made, in a fixed
 * order, whether or not a rule then overrides it, so a change to one rule leaves the
 * other values of every run where they were.
 * @param {number} seed integer; `trajectory.seed` is set to it, and `render.seaSeed` to seaSeedFor(seed)
 * @param {PartialRun} [base]
 * @returns {RunSpec}
 */
export function randomRun(seed, base = {}) {
	if (!Number.isSafeInteger(seed)) throw new Error(`randomRun: seed must be an integer (got ${JSON.stringify(seed)})`);
	if (!isObject(base)) throw new Error('randomRun: base must be an object');
	const { random, uniform, logUniform, pick } = sampler(mixSeed(seed, 0x5eed));

	// The sea: wind 2–20 m/s toward anywhere, a fetch from a coastal 50 km to open
	// ocean, a swell up to 90° off the wind sea (a cross swell at the extreme).
	const windSpeed = uniform(2, 20);
	const windDirection = uniform(0, 360);
	const fetch = logUniform(50, 800);
	const hasSwell = random() >= 0.25;
	const swellHeight = uniform(0.1, 2.5);
	const swellPeriod = uniform(8, 16);
	const swellOffset = uniform(-90, 90);
	const choppiness = uniform(0.6, 1.4);

	// The sky.
	let sunElevation = uniform(-4, 65);
	const sunAzimuth = uniform(0, 360);
	const clearish = random() < 0.45;
	const overcastLow = uniform(0, 0.25);
	const overcastHigh = uniform(0.25, 1);
	let overcast = clearish ? overcastLow : overcastHigh;
	let haze = uniform(0, 0.8);
	// Low sun under a closed deck is just dark grey: keep dusk skies open.
	if (sunElevation < 6) overcast = Math.min(overcast, 0.4);
	if (sunElevation < 0) haze = Math.min(haze, 0.55);
	// A closed deck comes with thick air.
	if (overcast > 0.7) haze = Math.max(haze, 0.3);
	// Calm, overcast and low sun together reads as a blank: lift the sun a little.
	if (windSpeed < 4 && overcast > 0.6) sunElevation = Math.max(sunElevation, 15);

	// The flight.
	const altitude = logUniform(30, 400);
	let airspeed = uniform(18, 32);
	const gimbalPitch = uniform(-15, -80);
	const fov = uniform(35, 75);
	const turbulence = uniform(0, 0.8);
	const stabilized = uniform(0.3, 1);
	const heading = uniform(0, 360);
	const pattern = pick(Object.keys(patternPresets));

	const b = structuredClone(base);
	const baseSea = resolveSea(b.sea ?? DEFAULT_SEA);
	const baseLight = resolveLight(b.light ?? DEFAULT_LIGHT);
	const trajectory = isObject(b.trajectory) ? b.trajectory : {};
	const render = isObject(b.render) ? b.render : {};
	/** @param {number} deg */
	const wrap = (deg) => ((deg % 360) + 360) % 360;
	/** @param {number} x */
	const r3 = (x) => Math.round(x * 1000) / 1000;

	// An aircraft has to outfly the wind to fly a pattern: at a wind near its airspeed it
	// drifts backwards, crabs past any sensible angle and its guidance hunts. So the
	// airspeed is at least WIND_MARGIN times the mean wind at cruise (the draw above
	// stands whenever it already is).
	const wind = trajectory.wind ?? 'sea';
	const aloft =
		wind === 'none' ? 0 : isObject(wind) ? Math.abs(Number(wind.speed) || 0) : windAloft({ windSpeed: r3(windSpeed), windDirection: 0 }, r3(altitude)).speed;
	airspeed = Math.max(airspeed, WIND_MARGIN * aloft);

	// Values are rounded to a millimetre / thousandth of a degree: far below anything
	// the render can show, and run.json stays readable.
	return resolveRun({
		...b,
		id: typeof b.id === 'string' ? b.id : `random-${seed}`,
		sea: {
			...baseSea,
			windSpeed: r3(windSpeed),
			windDirection: r3(windDirection),
			fetch: r3(fetch),
			swellHeight: hasSwell ? r3(swellHeight) : 0,
			swellPeriod: r3(swellPeriod),
			swellDirection: r3(wrap(windDirection + swellOffset)),
			choppiness: r3(choppiness)
		},
		light: { ...baseLight, sunElevation: r3(sunElevation), sunAzimuth: r3(sunAzimuth), overcast: r3(overcast), haze: r3(haze) },
		trajectory: {
			...trajectory,
			seed,
			pattern,
			altitude: r3(altitude),
			airspeed: r3(airspeed),
			turbulence: r3(turbulence),
			start: { ...(isObject(trajectory.start) ? trajectory.start : {}), heading: r3(heading) },
			gimbal: { ...(isObject(trajectory.gimbal) ? trajectory.gimbal : {}), pitch: r3(gimbalPitch), fov: r3(fov), stabilized: r3(stabilized) }
		},
		render: { ...render, seaSeed: seaSeedFor(seed) }
	});
}
