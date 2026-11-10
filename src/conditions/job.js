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
 * @property {VariedBlock} [varied]  `count` runs over varied scenes (variedRun), each laid
 *   over `base` like a listed run. They follow the sweep's and the random block's.
 * @property {PartialRun[]} [runs]  runs listed one by one, each laid over `base` (objects
 *   merged key by key, anything else replaced); an entry's `id` becomes its run id's slug.
 *   For runs drawn by a script with its own distributions. They come last.
 */

/**
 * @typedef {object} VariedBlock
 * @property {number} count
 * @property {number} seed
 * @property {[number, number]} [altitude]  metres, log-uniform; default 30–400
 * @property {[number, number]} [airspeed]  m/s; default 15–75
 * @property {[number, number]} [pitch]     gimbal pitch in degrees, negative down and never above 0; default −90 to −60
 */

/** No job expands past this many runs: a mistyped sweep should fail, not fill a disk. */
const MAX_RUNS = 100000;

/** A random run's airspeed is at least this many times the mean wind at its cruise altitude. */
export const WIND_MARGIN = 1.3;

/** What a varied block draws its flights from, unless it says otherwise. */
export const variedDefaults = Object.freeze({
	altitude: /** @type {[number, number]} */ ([30, 400]),
	airspeed: /** @type {[number, number]} */ ([15, 75]),
	pitch: /** @type {[number, number]} */ ([-90, -60])
});

/**
 * The water bodies a varied run draws from: what `render.look` takes as `waterColor` and
 * `sssColor`, linear RGB. Simulator values chosen to look different from each other, not
 * measurements of a real sea. `ocean` is scene.js's lookDefaults.
 */
export const variedWaters = Object.freeze({
	ocean: { waterColor: [0.0011, 0.0068, 0.0165], sssColor: [0.045, 0.32, 0.56] },
	indigo: { waterColor: [0.0006, 0.0028, 0.0085], sssColor: [0.03, 0.16, 0.42] },
	turquoise: { waterColor: [0.002, 0.02, 0.025], sssColor: [0.05, 0.55, 0.55] },
	coastal: { waterColor: [0.003, 0.014, 0.008], sssColor: [0.15, 0.5, 0.25] },
	grey: { waterColor: [0.004, 0.008, 0.007], sssColor: [0.2, 0.3, 0.28] },
	sediment: { waterColor: [0.02, 0.02, 0.01], sssColor: [0.5, 0.45, 0.2] }
});

/** How many metres the cloud map repeats over: sky.js's CLOUD_PERIOD, which cannot be imported here (it needs three.js). */
const CLOUD_MAP = 24576;
/** A varied run's cloud map starts at least this far from where it wraps, so a minute's flight and cloud drift never cross the wrap. */
const CLOUD_MARGIN = 6000;
/** How many candidate seeds a varied run may try before the job is refused. */
const VARIED_TRIES = 64;

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
 * `top` laid over `under`: objects merged key by key, anything else (numbers, strings,
 * arrays, preset names) taken from `top`. Neither is changed.
 * @param {Record<string, any>} under
 * @param {Record<string, any>} top
 * @returns {Record<string, any>}
 */
function overlay(under, top) {
	const out = structuredClone(under);
	for (const [key, value] of Object.entries(top)) {
		if (key === '__proto__' || key === 'constructor' || key === 'prototype') fail(`runs: field '${key}' is not allowed`);
		out[key] = isObject(value) && isObject(out[key]) ? overlay(out[key], value) : structuredClone(value);
	}
	return out;
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
 * A `[low, high]` pair of a varied block, checked.
 * @param {unknown} value
 * @param {string} what
 * @param {number} min
 * @param {number} max
 * @param {string} [who] the function the error is reported under
 * @returns {[number, number]}
 */
function range(value, what, min, max, who = 'expandJob') {
	const ok = Array.isArray(value) && value.length === 2 && value.every((x) => typeof x === 'number' && Number.isFinite(x) && x >= min && x <= max) && value[0] <= value[1];
	if (!ok) throw new Error(`${who}: ${what} must be [low, high] with ${min} ≤ low ≤ high${max === Infinity ? '' : ` ≤ ${max}`} (got ${JSON.stringify(value)})`);
	return /** @type {[number, number]} */ (value);
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
		if (!['name', 'description', 'base', 'sweep', 'random', 'varied', 'runs'].includes(key)) {
			fail(`unknown field '${key}' (expected name, description, base, sweep, random, varied, runs)`);
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

	if (job.varied !== undefined) {
		const varied = job.varied;
		if (!isObject(varied)) fail('varied must be { count, seed } and, optionally, altitude, airspeed and pitch ranges');
		for (const key of Object.keys(varied)) {
			if (!['count', 'seed', 'altitude', 'airspeed', 'pitch'].includes(key)) fail(`unknown field varied.${key} (expected count, seed, altitude, airspeed, pitch)`);
		}
		if (!Number.isSafeInteger(varied.count) || varied.count < 1) fail(`varied.count must be a positive integer (got ${JSON.stringify(varied.count)})`);
		if (!Number.isSafeInteger(varied.seed)) fail(`varied.seed must be an integer (got ${JSON.stringify(varied.seed)})`);
		if (runs.length + varied.count > MAX_RUNS) fail(`the job has more than ${MAX_RUNS} runs`);
		const ranges = {
			altitude: range(varied.altitude ?? variedDefaults.altitude, 'varied.altitude', 1e-3, Infinity),
			airspeed: range(varied.airspeed ?? variedDefaults.airspeed, 'varied.airspeed', 1e-3, Infinity),
			pitch: range(varied.pitch ?? variedDefaults.pitch, 'varied.pitch', -90, 0)
		};
		// The airspeeds are drawn against the wind of each run's own sea (variedRun), so a
		// wind given in numbers would leave them unchecked.
		if (isObject(base.trajectory) && isObject(base.trajectory.wind)) {
			fail("varied: base.trajectory.wind must be 'sea' or 'none', since each airspeed is drawn against its own sea's wind");
		}
		// No two runs of the block share a seed, so none share a flight or a sea.
		const used = new Set();
		for (let i = 0; i < varied.count; i++) {
			let run = null;
			for (let j = 0; j < VARIED_TRIES && !run; j++) {
				// Nine-digit run seeds, as in the random block. Seed 0 is left out.
				const seed = mixSeed(varied.seed, VARIED_TRIES * i + j) % 1e9;
				if (seed === 0 || used.has(seed)) continue;
				run = variedRun(seed, ranges);
			}
			if (!run) fail(`varied: no flight for run ${i} in ${VARIED_TRIES} tries (airspeed ${ranges.airspeed.join('–')} m/s rarely outflies the wind at ${ranges.altitude.join('–')} m)`);
			used.add(run.trajectory.seed);
			const { water, ...drawn } = run;
			const pattern = /** @type {string} */ (drawn.trajectory.pattern);
			const slug = `w${Math.round(drawn.sea.windSpeed)}-sun${Math.round(drawn.light.sunElevation)}-a${Math.round(drawn.trajectory.altitude)}-v${Math.round(drawn.trajectory.airspeed)}-${water}-${pattern}`;
			runs.push({ partial: overlay(base, drawn), slug });
		}
	}

	if (job.runs !== undefined) {
		if (!Array.isArray(job.runs) || job.runs.length === 0) fail('runs must be a non-empty list of runs');
		if (runs.length + job.runs.length > MAX_RUNS) fail(`the job has more than ${MAX_RUNS} runs`);
		job.runs.forEach((entry, i) => {
			if (!isObject(entry)) fail(`runs[${i}] must be an object`);
			if (entry.id !== undefined && typeof entry.id !== 'string') fail(`runs[${i}].id must be a string`);
			const partial = overlay(base, entry);
			delete partial.id;
			runs.push({ partial, slug: entry.id ?? '' });
		});
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

/**
 * One run over a varied scene: far more kinds of water and sky than `randomRun` draws,
 * for a camera pointed steeply down. Sets drawn this way have been rendered and trained
 * on, so the draw stays as it is: a seed gives the run it always gave.
 *
 * - the sea: wind 2–20 m/s seven times in ten, otherwise glassy (0.7–2, log-uniform) or a
 *   storm (20–28), half each; fetch 5–800 km; a swell three times in four, up to 4 m at
 *   6–18 s from any direction; directional spread 2–24; choppiness 0.5–1.6;
 * - the sky: sun from 8° below the horizon to 88° up, overcast and haze each anywhere from
 *   0 to 1 and independent, without `randomRun`'s rules that keep a frame worth looking at
 *   (the blank, low-texture frames are wanted here); exposure ×0.5–2;
 * - the water (`render.look`): one of `variedWaters`, its brightness ×0.6–1.6 and each
 *   channel ±15 %; foam coverage ×0.3–1.5; the cloud map moved, never so near its wrap
 *   that a minute's flight crosses it;
 * - the flight: altitude log-uniform and airspeed uniform in the given ranges, the
 *   airspeed never under WIND_MARGIN × this sea's wind aloft (the wind the flight meets
 *   under `trajectory.wind: 'sea'`); gimbal pitch in its range; turbulence 0–0.8; any
 *   heading; one of the pattern presets.
 *
 * Nothing fixed to the water is drawn (slicks, debris): nothing in the simulator drifts
 * with a current, so such a thing would be a ground reference real water does not offer.
 *
 * What comes back is laid over a job's base like a listed run, so the base supplies what
 * is not drawn: the duration, the frame size and the camera's field of view, yaw and
 * stabilisation.
 * @param {number} seed integer; `trajectory.seed` is set to it, and `render.seaSeed` to seaSeedFor(seed)
 * @param {{ altitude?: [number, number], airspeed?: [number, number], pitch?: [number, number] }} [ranges]
 * @returns {(PartialRun & { water: keyof typeof variedWaters, sea: Record<string, number>, light: Record<string, number>, trajectory: Record<string, any> }) | null}
 *   `water` names the body drawn and is not a run field: take it off before resolving.
 *   null when even the top of the airspeed range cannot outfly this run's wind.
 */
export function variedRun(seed, ranges = {}) {
	if (!Number.isSafeInteger(seed)) throw new Error(`variedRun: seed must be an integer (got ${JSON.stringify(seed)})`);
	if (!isObject(ranges)) throw new Error('variedRun: ranges must be an object');
	const ALT = range(ranges.altitude ?? variedDefaults.altitude, 'altitude', 1e-3, Infinity, 'variedRun');
	const AIRSPEED = range(ranges.airspeed ?? variedDefaults.airspeed, 'airspeed', 1e-3, Infinity, 'variedRun');
	const PITCH = range(ranges.pitch ?? variedDefaults.pitch, 'pitch', -90, 0, 'variedRun');
	const { random, uniform, logUniform, pick } = sampler(mixSeed(seed, 0xe4));
	/** @param {number} x */
	const r3 = (x) => Math.round(x * 1000) / 1000;
	/** @param {number} x */
	const r5 = (x) => Math.round(x * 100000) / 100000;

	// Every draw below is made in this order whatever its value: do not reorder them.
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
	const water = pick(/** @type {Array<keyof typeof variedWaters>} */ (Object.keys(variedWaters)));
	const gain = logUniform(0.6, 1.6);
	/** @param {number[]} c */
	const channels = (c) => c.map((x) => r5(x * gain * uniform(0.85, 1.15)));
	const look = {
		waterColor: channels(variedWaters[water].waterColor),
		sssColor: channels(variedWaters[water].sssColor),
		foam: r3(logUniform(0.3, 1.5)),
		cloudOffset: [r3(uniform(CLOUD_MARGIN, CLOUD_MAP - CLOUD_MARGIN)), r3(uniform(CLOUD_MARGIN, CLOUD_MAP - CLOUD_MARGIN))]
	};
	const pitch = uniform(PITCH[0], PITCH[1]);
	const altitude = logUniform(ALT[0], ALT[1]);
	const u = uniform(0, 1);
	const turbulence = uniform(0, 0.8);
	const heading = uniform(0, 360);
	const pattern = pick(Object.keys(patternPresets));

	// The airspeed is drawn between the range's top and whichever is higher, its bottom or
	// what outflies the wind aloft (see randomRun). A wind nothing in the range outflies
	// gives no run, and the caller moves to another seed.
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
		render: { seaSeed: seaSeedFor(seed), look },
		water
	};
}
