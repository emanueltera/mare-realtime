/**
 * A run is one flight over one ocean, rendered (docs/ARCHITECTURE.md, "Runs and
 * jobs"). This module turns whatever a job file or the app hands over — preset names,
 * half-filled objects — into a complete RunSpec: every default written out, every
 * field checked. The result is what `run.json` records, so a run can be repeated from
 * its own file without knowing what the defaults were when it was made.
 *
 * Validation is strict about names as well as values: a typo in a job file
 * (`render.fsp`) is an error naming the field, not a setting silently ignored for a
 * hundred runs.
 *
 * Conventions as everywhere: positions are metres east / north of the origin,
 * altitude metres above mean sea level, directions compass degrees (0 north, 90 east,
 * the way seas and wind travel toward).
 */
import { seaDefaults, lightDefaults } from '../ocean/scene.js';
import { seaPresets, lightPresets, patternPresets } from './presets.js';
import { trajectoryDefaults as flightDefaults } from '../trajectory/index.js';

/** @typedef {import('../ocean/scene.js').SeaState} SeaState */
/** @typedef {import('../ocean/scene.js').Lighting} Lighting */
/** @typedef {import('../ocean/scene.js').RenderMode} RenderMode */

/*
 * The trajectory types, as docs/ARCHITECTURE.md gives them, restated in their resolved
 * form (every field present). src/trajectory owns their meaning and their defaults
 * (imported below); this module fills the defaults in and checks the shapes, so a new
 * pattern parameter or gimbal key must be added to KEYS / PATTERN_KEYS here too.
 */

/**
 * @typedef {
 *   | { type: 'straight' }
 *   | { type: 'orbit', radius?: number, direction?: 1 | -1, center?: [number, number] }
 *   | { type: 'racetrack', length?: number, radius?: number, direction?: 1 | -1 }
 *   | { type: 'figure8', radius?: number }
 *   | { type: 'lawnmower', legLength?: number, spacing?: number, legs?: number }
 *   | { type: 'waypoints', points: Array<[number, number] | [number, number, number]>, loop?: boolean }
 *   | { type: 'random', legLength?: [number, number], turn?: [number, number] }
 * } PatternSpec
 */

/**
 * @typedef {
 *   | { type: 'constant' }
 *   | { type: 'sine', amplitude: number, period: number }
 *   | { type: 'steps', amplitude: number, interval: number }
 * } AltitudeProfile
 */

/** @typedef {'sea' | 'none' | { speed: number, direction: number }} WindSpec */

/**
 * @typedef {object} GimbalSpec
 * @property {number} pitch       degrees, negative looks down
 * @property {number} yaw         degrees from the nose, positive right
 * @property {number} fov         vertical field of view, degrees
 * @property {number} stabilized  0 hard-mounted – 1 fully stabilised
 */

/**
 * A TrajectorySpec with every default filled in.
 * @typedef {object} TrajectorySpec
 * @property {number} seed
 * @property {number} duration          s
 * @property {PatternSpec} pattern
 * @property {number} altitude          m above mean sea level
 * @property {AltitudeProfile} altitudeProfile
 * @property {number} airspeed          m/s
 * @property {number} airspeedVariation m/s
 * @property {{ x: number, y: number, heading: number }} start
 * @property {WindSpec} wind
 * @property {number} turbulence        0–1
 * @property {GimbalSpec} gimbal
 */

/**
 * @typedef {object} RenderSpec
 * @property {number} width     pixels
 * @property {number} height    pixels
 * @property {number} fps       frames per trajectory second
 * @property {'high' | 'low'} quality   a fixed preset of the engine's quality.js
 * @property {RenderMode[]} modes
 * @property {'onboard' | 'chase'} view
 * @property {number} warmup    s the sea and foam run before frame 0
 * @property {number} seaTime   the sea's clock at the start of warm-up, s
 * @property {number} seaSeed   the sea's Gaussian field
 * @property {boolean} frames   false writes the flight only (trajectory.csv, imu.csv; no frames, no frames.csv)
 */

/**
 * A resolved run: every field present.
 * @typedef {object} RunSpec
 * @property {string} id
 * @property {Required<SeaState>} sea
 * @property {Required<Lighting>} light
 * @property {TrajectorySpec} trajectory
 * @property {RenderSpec} render
 * @property {number} imuRate   Hz
 * @property {{ lat: number, lon: number }} [origin]
 */

/**
 * What resolveRun accepts: any part of a RunSpec, with `sea`, `light` and
 * `trajectory.pattern` also accepted as preset names. Loosely typed, since it is
 * usually parsed JSON; resolveRun checks it field by field.
 * @typedef {Record<string, any>} PartialRun
 */

/** The sea a run gets when it names none. */
export const DEFAULT_SEA = 'moderate';
/** The sky a run gets when it names none. */
export const DEFAULT_LIGHT = 'morning';
/** The pattern a run flies when it names none. */
export const DEFAULT_PATTERN = 'transit';

/**
 * The seed of the sea's Gaussian field when a run gives none; also the base the job
 * `seed` key offsets from (see seaSeedFor).
 */
export const DEFAULT_SEA_SEED = 20240611;

/** @type {Readonly<RenderSpec>} */
export const renderDefaults = Object.freeze({
	width: 1280,
	height: 720,
	fps: 10,
	quality: /** @type {'high'} */ ('high'),
	modes: /** @type {RenderMode[]} */ (Object.freeze(['shaded'])),
	view: /** @type {'onboard'} */ ('onboard'),
	warmup: 8,
	seaTime: 0,
	seaSeed: DEFAULT_SEA_SEED,
	frames: true
});

/**
 * TrajectorySpec defaults: the trajectory module's own (one source, so the two cannot
 * drift apart), plus `duration` and `altitude`, which it requires. A run that leaves
 * them out gets a minute at 120 m, so even an empty RunSpec resolves to something that
 * can be flown.
 * @type {Readonly<Omit<TrajectorySpec, 'pattern'>>}
 */
export const trajectoryDefaults = Object.freeze({
	...flightDefaults,
	duration: 60,
	altitude: 120
});

export const DEFAULT_IMU_RATE = 200;

/** @type {RenderMode[]} */
export const renderModes = ['shaded', 'height', 'velocity', 'normals', 'foam', 'depth'];
const QUALITIES = ['high', 'low'];
const VIEWS = ['onboard', 'chase'];

/**
 * The sea's seed for a run seed. A job's `seed` key moves the trajectory's random
 * choices and the sea's wave field together, so seed 2 is a different flight over a
 * different (but statistically identical) sea; seed 0 is the default sea.
 * @param {number} seed
 */
export const seaSeedFor = (seed) => DEFAULT_SEA_SEED + seed;

/** Keys each object may carry; anything else is a typo. */
const KEYS = {
	run: ['id', 'sea', 'light', 'trajectory', 'render', 'imuRate', 'origin'],
	sea: Object.keys({ windSpeed: 0, windDirection: 0, ...seaDefaults }),
	light: Object.keys({ sunElevation: 0, sunAzimuth: 0, ...lightDefaults }),
	trajectory: ['pattern', ...Object.keys(trajectoryDefaults)],
	start: ['x', 'y', 'heading'],
	gimbal: ['pitch', 'yaw', 'fov', 'stabilized'],
	render: Object.keys(renderDefaults),
	origin: ['lat', 'lon']
};

/** Pattern parameters by type. */
const PATTERN_KEYS = {
	straight: [],
	orbit: ['radius', 'direction', 'center'],
	racetrack: ['length', 'radius', 'direction'],
	figure8: ['radius'],
	lawnmower: ['legLength', 'spacing', 'legs'],
	waypoints: ['points', 'loop'],
	random: ['legLength', 'turn']
};

const PROFILE_KEYS = { constant: [], sine: ['amplitude', 'period'], steps: ['amplitude', 'interval'] };

/**
 * @param {string} message
 * @returns {never}
 */
function fail(message) {
	throw new Error(`resolveRun: ${message}`);
}

/** @param {unknown} v @returns {v is Record<string, any>} */
export const isObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);

/** @param {unknown} v */
const show = (v) => (typeof v === 'string' ? `'${v}'` : JSON.stringify(v));

/**
 * @param {Record<string, any>} obj
 * @param {readonly string[]} allowed
 * @param {string} where
 */
function checkKeys(obj, allowed, where) {
	for (const key of Object.keys(obj)) {
		if (!allowed.includes(key)) fail(`unknown field ${where}.${key} (expected one of ${allowed.join(', ')})`);
	}
}

/**
 * @param {unknown} value
 * @param {string} where
 * @param {{ min?: number, max?: number, positive?: boolean, integer?: boolean }} [range]
 * @returns {number}
 */
function num(value, where, range = {}) {
	if (typeof value !== 'number' || !Number.isFinite(value)) fail(`${where} must be a finite number (got ${show(value)})`);
	if (range.integer && !Number.isInteger(value)) fail(`${where} must be an integer (got ${value})`);
	if (range.positive && !(value > 0)) fail(`${where} must be positive (got ${value})`);
	if (range.min !== undefined && value < range.min) fail(`${where} must be at least ${range.min} (got ${value})`);
	if (range.max !== undefined && value > range.max) fail(`${where} must be at most ${range.max} (got ${value})`);
	return value;
}

/**
 * @param {unknown} value
 * @param {string} where
 * @returns {Record<string, any>}
 */
function obj(value, where) {
	if (!isObject(value)) fail(`${where} must be an object (got ${show(value)})`);
	return value;
}

/**
 * Look up a preset by name, or fail listing the names there are.
 * @template T
 * @param {Record<string, T>} presets
 * @param {string} name
 * @param {string} where
 * @returns {T}
 */
function preset(presets, name, where) {
	if (!Object.hasOwn(presets, name)) fail(`unknown ${where} preset ${show(name)} (expected one of ${Object.keys(presets).join(', ')})`);
	return presets[name];
}

/**
 * A preset name or partial object → a complete, checked SeaState. An object is filled
 * from the scene defaults, not from a preset: `{ windSpeed: 12, windDirection: 0 }` is
 * exactly that wind sea, with no swell.
 * @param {unknown} input
 * @returns {Required<SeaState>}
 */
export function resolveSea(input) {
	const src = typeof input === 'string' ? preset(seaPresets, input, 'sea').sea : obj(input, 'sea');
	checkKeys(src, KEYS.sea, 'sea');
	/** @type {Record<string, any>} */
	const sea = { ...seaDefaults, ...src };
	return {
		windSpeed: num(sea.windSpeed, 'sea.windSpeed', { min: 0, max: 40 }),
		windDirection: num(sea.windDirection, 'sea.windDirection'),
		fetch: num(sea.fetch, 'sea.fetch', { positive: true }),
		spread: num(sea.spread, 'sea.spread', { positive: true }),
		swellHeight: num(sea.swellHeight, 'sea.swellHeight', { min: 0, max: 15 }),
		swellDirection: num(sea.swellDirection, 'sea.swellDirection'),
		swellPeriod: num(sea.swellPeriod, 'sea.swellPeriod', { positive: true }),
		choppiness: num(sea.choppiness, 'sea.choppiness', { min: 0, max: 1.6 })
	};
}

/**
 * @param {unknown} input
 * @returns {Required<Lighting>}
 */
export function resolveLight(input) {
	const src = typeof input === 'string' ? preset(lightPresets, input, 'light').light : obj(input, 'light');
	checkKeys(src, KEYS.light, 'light');
	/** @type {Record<string, any>} */
	const light = { ...lightDefaults, ...src };
	return {
		sunElevation: num(light.sunElevation, 'light.sunElevation', { min: -90, max: 90 }),
		sunAzimuth: num(light.sunAzimuth, 'light.sunAzimuth'),
		overcast: num(light.overcast, 'light.overcast', { min: 0, max: 1 }),
		haze: num(light.haze, 'light.haze', { min: 0, max: 1 }),
		exposure: num(light.exposure, 'light.exposure', { positive: true })
	};
}

/**
 * A preset name or pattern object → a checked copy. Pattern parameters left out stay
 * out: their defaults belong to the trajectory module.
 * @param {unknown} input
 * @returns {PatternSpec}
 */
export function resolvePattern(input) {
	const where = 'trajectory.pattern';
	const src = typeof input === 'string' ? preset(patternPresets, input, 'pattern').pattern : obj(input, where);
	const type = src.type;
	if (typeof type !== 'string' || !Object.hasOwn(PATTERN_KEYS, type)) {
		fail(`${where}.type must be one of ${Object.keys(PATTERN_KEYS).join(', ')} (got ${show(type)})`);
	}
	checkKeys(src, ['type', ...PATTERN_KEYS[/** @type {keyof PATTERN_KEYS} */ (type)]], where);
	const p = structuredClone(src);
	/** @param {string} key @param {Parameters<typeof num>[2]} range */
	const opt = (key, range) => p[key] === undefined || num(p[key], `${where}.${key}`, range);
	/** @param {string} key @param {number} lo @param {number} hi */
	const interval = (key, lo, hi) => {
		if (p[key] === undefined) return;
		const v = p[key];
		if (!Array.isArray(v) || v.length !== 2) fail(`${where}.${key} must be [min, max] (got ${show(v)})`);
		num(v[0], `${where}.${key}[0]`, { min: lo, max: hi });
		num(v[1], `${where}.${key}[1]`, { min: v[0], max: hi });
	};
	const direction = () => {
		if (p.direction !== undefined && p.direction !== 1 && p.direction !== -1) {
			fail(`${where}.direction must be 1 (clockwise) or -1 (got ${show(p.direction)})`);
		}
	};
	switch (type) {
		case 'orbit':
			opt('radius', { positive: true });
			direction();
			if (p.center !== undefined) {
				if (!Array.isArray(p.center) || p.center.length !== 2) fail(`${where}.center must be [right, ahead] of the start (got ${show(p.center)})`);
				p.center.forEach((/** @type {unknown} */ c, /** @type {number} */ i) => num(c, `${where}.center[${i}]`));
			}
			break;
		case 'racetrack':
			opt('length', { positive: true });
			opt('radius', { positive: true });
			direction();
			break;
		case 'figure8':
			opt('radius', { positive: true });
			break;
		case 'lawnmower':
			opt('legLength', { positive: true });
			opt('spacing', { positive: true });
			opt('legs', { positive: true, integer: true });
			break;
		case 'waypoints':
			if (!Array.isArray(p.points) || p.points.length < 2) fail(`${where}.points must list at least two waypoints`);
			p.points.forEach((/** @type {unknown} */ pt, /** @type {number} */ i) => {
				if (!Array.isArray(pt) || (pt.length !== 2 && pt.length !== 3)) {
					fail(`${where}.points[${i}] must be [right, ahead] or [right, ahead, altitude] (got ${show(pt)})`);
				}
				pt.forEach((c, j) => num(c, `${where}.points[${i}][${j}]`));
			});
			if (p.loop !== undefined && typeof p.loop !== 'boolean') fail(`${where}.loop must be true or false`);
			break;
		case 'random':
			interval('legLength', 1e-9, Infinity);
			// Signed, as the trajectory flies them: positive turns right, negative left.
			interval('turn', -180, 180);
			break;
	}
	return /** @type {PatternSpec} */ (p);
}

/**
 * @param {unknown} input
 * @returns {WindSpec}
 */
function resolveWind(input) {
	if (input === 'sea' || input === 'none') return input;
	const w = obj(input, 'trajectory.wind');
	checkKeys(w, ['speed', 'direction'], 'trajectory.wind');
	return {
		speed: num(w.speed, 'trajectory.wind.speed', { min: 0 }),
		direction: num(w.direction, 'trajectory.wind.direction')
	};
}

/**
 * @param {unknown} input
 * @returns {AltitudeProfile}
 */
function resolveProfile(input) {
	const where = 'trajectory.altitudeProfile';
	const p = obj(input, where);
	if (typeof p.type !== 'string' || !Object.hasOwn(PROFILE_KEYS, p.type)) {
		fail(`${where}.type must be one of ${Object.keys(PROFILE_KEYS).join(', ')} (got ${show(p.type)})`);
	}
	const type = /** @type {keyof PROFILE_KEYS} */ (p.type);
	checkKeys(p, ['type', ...PROFILE_KEYS[type]], where);
	if (type === 'sine') {
		return { type, amplitude: num(p.amplitude, `${where}.amplitude`, { min: 0 }), period: num(p.period, `${where}.period`, { positive: true }) };
	}
	if (type === 'steps') {
		return { type, amplitude: num(p.amplitude, `${where}.amplitude`, { min: 0 }), interval: num(p.interval, `${where}.interval`, { positive: true }) };
	}
	return { type: 'constant' };
}

/**
 * @param {unknown} input
 * @returns {TrajectorySpec}
 */
export function resolveTrajectory(input) {
	/** @type {Record<string, any>} */
	const t = { ...trajectoryDefaults, ...obj(input ?? {}, 'trajectory') };
	checkKeys(t, KEYS.trajectory, 'trajectory');
	const start = { ...trajectoryDefaults.start, ...obj(t.start, 'trajectory.start') };
	checkKeys(start, KEYS.start, 'trajectory.start');
	const gimbal = { ...trajectoryDefaults.gimbal, ...obj(t.gimbal, 'trajectory.gimbal') };
	checkKeys(gimbal, KEYS.gimbal, 'trajectory.gimbal');
	const trajectory = {
		seed: num(t.seed, 'trajectory.seed', { integer: true }),
		duration: num(t.duration, 'trajectory.duration', { positive: true }),
		pattern: resolvePattern(t.pattern ?? DEFAULT_PATTERN),
		altitude: num(t.altitude, 'trajectory.altitude', { positive: true }),
		altitudeProfile: resolveProfile(t.altitudeProfile),
		airspeed: num(t.airspeed, 'trajectory.airspeed', { positive: true }),
		airspeedVariation: num(t.airspeedVariation, 'trajectory.airspeedVariation', { min: 0 }),
		start: {
			x: num(start.x, 'trajectory.start.x'),
			y: num(start.y, 'trajectory.start.y'),
			heading: num(start.heading, 'trajectory.start.heading')
		},
		wind: resolveWind(t.wind),
		turbulence: num(t.turbulence, 'trajectory.turbulence', { min: 0, max: 1 }),
		gimbal: {
			pitch: num(gimbal.pitch, 'trajectory.gimbal.pitch', { min: -90, max: 90 }),
			yaw: num(gimbal.yaw, 'trajectory.gimbal.yaw'),
			fov: num(gimbal.fov, 'trajectory.gimbal.fov', { positive: true, max: 170 }),
			stabilized: num(gimbal.stabilized, 'trajectory.gimbal.stabilized', { min: 0, max: 1 })
		}
	};
	return trajectory;
}

/**
 * @param {unknown} input
 * @returns {RenderSpec}
 */
export function resolveRender(input) {
	const r = { ...renderDefaults, ...obj(input ?? {}, 'render') };
	checkKeys(r, KEYS.render, 'render');
	if (!QUALITIES.includes(r.quality)) fail(`render.quality must be one of ${QUALITIES.join(', ')} (got ${show(r.quality)})`);
	if (!VIEWS.includes(r.view)) fail(`render.view must be one of ${VIEWS.join(', ')} (got ${show(r.view)})`);
	if (!Array.isArray(r.modes) || r.modes.length === 0) fail(`render.modes must be a non-empty list of ${renderModes.join(', ')}`);
	for (const mode of r.modes) {
		if (!renderModes.includes(mode)) fail(`unknown render mode ${show(mode)} in render.modes (expected any of ${renderModes.join(', ')})`);
	}
	if (new Set(r.modes).size !== r.modes.length) fail(`render.modes lists a mode twice (${r.modes.join(', ')})`);
	if (typeof r.frames !== 'boolean') fail(`render.frames must be true or false (got ${show(r.frames)})`);
	return {
		width: num(r.width, 'render.width', { positive: true, integer: true, max: 8192 }),
		height: num(r.height, 'render.height', { positive: true, integer: true, max: 8192 }),
		fps: num(r.fps, 'render.fps', { positive: true, max: 240 }),
		quality: r.quality,
		modes: [...r.modes],
		view: r.view,
		warmup: num(r.warmup, 'render.warmup', { min: 0 }),
		seaTime: num(r.seaTime, 'render.seaTime'),
		seaSeed: num(r.seaSeed, 'render.seaSeed', { integer: true }),
		frames: r.frames
	};
}

/** A run id: a file and directory name on any filesystem. */
const SAFE_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;

/**
 * A RunSpec with presets looked up and every default filled in. Never mutates its
 * argument; the result shares nothing with it or with the presets.
 * @param {PartialRun} [partial]
 * @returns {RunSpec}
 */
export function resolveRun(partial = {}) {
	const src = obj(partial, 'run');
	checkKeys(src, KEYS.run, 'run');
	/** @type {RunSpec} */
	const run = {
		id: '',
		sea: resolveSea(src.sea ?? DEFAULT_SEA),
		light: resolveLight(src.light ?? DEFAULT_LIGHT),
		trajectory: resolveTrajectory(src.trajectory),
		render: resolveRender(src.render),
		imuRate: num(src.imuRate ?? DEFAULT_IMU_RATE, 'imuRate', { positive: true, max: 10000 })
	};
	if (src.origin !== undefined) {
		const o = obj(src.origin, 'origin');
		checkKeys(o, KEYS.origin, 'origin');
		run.origin = { lat: num(o.lat, 'origin.lat', { min: -90, max: 90 }), lon: num(o.lon, 'origin.lon', { min: -180, max: 180 }) };
	}
	if (src.id === undefined) {
		run.id = slugify(runSummary(run));
	} else {
		if (typeof src.id !== 'string' || !SAFE_ID.test(src.id)) {
			fail(`id must be 1–128 letters, digits, '.', '_' or '-', starting with a letter or digit (got ${show(src.id)})`);
		}
		run.id = src.id;
	}
	return run;
}

/**
 * Structural equality for JSON values, ignoring key order.
 * @param {unknown} a
 * @param {unknown} b
 * @returns {boolean}
 */
export function deepEqual(a, b) {
	if (a === b) return true;
	if (Array.isArray(a)) return Array.isArray(b) && a.length === b.length && a.every((v, i) => deepEqual(v, b[i]));
	if (isObject(a) && isObject(b)) {
		const keys = Object.keys(a);
		return keys.length === Object.keys(b).length && keys.every((k) => Object.hasOwn(b, k) && deepEqual(a[k], b[k]));
	}
	return false;
}

/**
 * The name of the preset a resolved value came from, if it is exactly one.
 * @param {Record<string, Record<string, any>>} presets
 * @param {'sea' | 'light' | 'pattern'} field
 * @param {unknown} value
 * @returns {string | null}
 */
export function presetName(presets, field, value) {
	for (const [name, p] of Object.entries(presets)) if (deepEqual(p[field], value)) return name;
	return null;
}

/** @param {number} x */
const round1 = (x) => String(Number(x.toFixed(1)));

/**
 * A short human label, e.g. "moderate · low-sun · survey · s1". Presets are named;
 * a sea or sky that is not exactly a preset is summarised by its main numbers.
 * @param {RunSpec} run a resolved run
 * @returns {string}
 */
export function runSummary(run) {
	const { sea, light, trajectory } = run;
	let seaName = presetName(seaPresets, 'sea', sea);
	if (!seaName) {
		seaName = `wind ${round1(sea.windSpeed)} m/s`;
		if (sea.swellHeight > 0.01) seaName += ` + swell ${round1(sea.swellHeight)} m`;
	}
	let lightName = presetName(lightPresets, 'light', light);
	if (!lightName) {
		lightName = `sun ${Math.round(light.sunElevation)}°`;
		if (light.overcast >= 0.6) lightName += ' overcast';
	}
	const patternName = presetName(patternPresets, 'pattern', trajectory.pattern) ?? trajectory.pattern.type;
	return `${seaName} · ${lightName} · ${patternName} · s${trajectory.seed}`;
}

/**
 * Lower-case letters, digits and single hyphens: safe as a file name everywhere.
 * Decimal points become 'p' and minus signs 'm', so 2.5 and 25, −3 and 3 stay distinct.
 * @param {string} text
 * @param {number} [max] characters
 * @returns {string}
 */
export function slugify(text, max = 64) {
	const slug = text
		.toLowerCase()
		.replace(/(\d)\.(\d)/g, '$1p$2')
		.replace(/(^|[^a-z0-9])-(\d)/g, '$1m$2')
		.replace(/[^a-z0-9]+/g, '-')
		.replace(/^-+|-+$/g, '');
	return slug.slice(0, max).replace(/-+$/, '') || 'run';
}
