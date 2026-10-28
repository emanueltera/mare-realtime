/**
 * The app's run, as data: the RunSpec being designed (docs/ARCHITECTURE.md, "Runs and
 * jobs"), how the view is set up, and how both travel in the URL's hash so a link
 * reproduces what is on screen. Pure and framework-free; App.svelte wraps it in runes.
 *
 * The run in the app is always a resolved RunSpec (every field present) without its
 * id, so every slider has a value to show. Only what differs from `resolveRun({})` goes
 * into the hash, and a sea, sky or pattern that is a preset goes in by name: the link
 * stays short, and resolveRun fills the rest back in on the way out.
 */
import { lightPresets, patternPresets, renderModes, resolveRun, seaPresets } from '../conditions/index.js';
import { deepEqual } from '../conditions/run.js';
import { lightDefaults, seaDefaults } from '../ocean/scene.js';

/** @typedef {import('../conditions/index.js').RunSpec} RunSpec */
/** @typedef {import('../conditions/index.js').PatternSpec} PatternSpec */
/** @typedef {import('../ocean/scene.js').RenderMode} RenderMode */
/** @typedef {import('../ocean/scene.js').OverlayName} OverlayName */

/**
 * The run being designed: a resolved RunSpec, with `id` empty while it is the automatic
 * one (resolveRun's slug of the run's summary).
 * @typedef {Omit<RunSpec, 'id'> & { id: string }} AppRun
 */

/**
 * How the preview is looked at. None of it is part of the run: the run's own camera
 * is `render.view`, chosen in the export section.
 * @typedef {object} ViewState
 * @property {'chase' | 'onboard' | 'overhead'} camera
 * @property {OverlayName[]} overlays
 * @property {RenderMode} mode
 * @property {boolean} track draw the trajectory on the sea
 * @property {boolean} footprint draw the gimbal camera's frustum and footprint
 */

/** @type {ViewState} */
export const viewDefaults = Object.freeze({
	camera: /** @type {const} */ ('chase'),
	overlays: /** @type {OverlayName[]} */ ([]),
	mode: /** @type {RenderMode} */ ('shaded'),
	track: true,
	footprint: false
});

/**
 * What the app opens with when the URL names nothing: a survey over the moderate sea,
 * long enough to fly the whole lawnmower. Its 4 legs of 900 m at 24 m/s take about a
 * minute each with the turns between, which are Omega turns here: in the moderate sea's
 * wind aloft (about 8.5 m/s) the turn radius is more than half the 200 m spacing. The
 * fourth leg ends about 305 s in.
 * @returns {AppRun}
 */
export function defaultRun() {
	return { ...resolveRun({ trajectory: { pattern: 'survey', duration: 310 } }), id: '' };
}

/**
 * The partial run resolveRun takes back: the app's run with its id left out when it is
 * the automatic one. `run` must be plain data (a component passes `$state.snapshot`).
 * @param {AppRun} run
 * @returns {import('../conditions/index.js').PartialRun}
 */
export function toPartial(run) {
	const { id, ...rest } = structuredClone(run);
	return id ? { id, ...rest } : rest;
}

/**
 * An imported or decoded run, checked and filled in by resolveRun. A run whose id was
 * the automatic one (it matches the slug of its own summary) stays automatic, so editing
 * it renames it.
 * @param {import('../conditions/index.js').PartialRun} partial
 * @returns {AppRun}
 */
export function fromPartial(partial) {
	const resolved = resolveRun(partial);
	const auto = resolveRun({ ...partial, id: undefined }).id;
	return { ...resolved, id: resolved.id === auto ? '' : resolved.id };
}

/**
 * The name of the preset `value` is, if any.
 * @template {{ [k: string]: any }} P
 * @param {Record<string, P>} presets
 * @param {keyof P} key
 * @param {unknown} value
 * @returns {string | null}
 */
export function presetOf(presets, key, value) {
	for (const [name, p] of Object.entries(presets)) if (deepEqual(p[key], value)) return name;
	return null;
}

export const seaPresetOf = (/** @type {unknown} */ sea) => presetOf(seaPresets, 'sea', sea);
export const lightPresetOf = (/** @type {unknown} */ light) => presetOf(lightPresets, 'light', light);
export const patternPresetOf = (/** @type {unknown} */ pattern) => presetOf(patternPresets, 'pattern', pattern);

/** @param {unknown} v @returns {v is Record<string, any>} */
const isObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);

/**
 * What of `value` differs from `base`, key by key; undefined when nothing does. Keys in
 * `whole` are compared and kept as a unit (a pattern or profile is not merged with a
 * default of another type by resolveRun, so half of one would be wrong).
 * @param {unknown} value
 * @param {unknown} base
 * @param {readonly string[]} [whole]
 * @returns {any}
 */
function difference(value, base, whole = []) {
	if (deepEqual(value, base)) return undefined;
	if (!isObject(value) || !isObject(base)) return value;
	/** @type {Record<string, any>} */
	const out = {};
	for (const [k, v] of Object.entries(value)) {
		const d = whole.includes(k) ? (deepEqual(v, base[k]) ? undefined : v) : difference(v, base[k], whole);
		if (d !== undefined) out[k] = d;
	}
	return Object.keys(out).length ? out : undefined;
}

/**
 * The run as the shortest partial run that resolves back to it.
 * @param {AppRun} run
 * @returns {import('../conditions/index.js').PartialRun}
 */
export function compactRun(run) {
	const base = resolveRun({});
	/** @type {Record<string, any>} */
	const out = {};
	if (run.id) out.id = run.id;
	// An object sea or sky is filled from the scene defaults (not from a preset), and the
	// defaults have no wind or sun, so those are always written.
	out.sea = seaPresetOf(run.sea) ?? difference(run.sea, seaDefaults);
	out.light = lightPresetOf(run.light) ?? difference(run.light, lightDefaults);
	const { pattern, ...trajectory } = run.trajectory;
	const { pattern: _, ...baseTrajectory } = base.trajectory;
	const t = difference(trajectory, baseTrajectory, ['altitudeProfile', 'wind']) ?? {};
	if (!deepEqual(pattern, base.trajectory.pattern)) t.pattern = patternPresetOf(pattern) ?? pattern;
	if (Object.keys(t).length) out.trajectory = t;
	const render = difference(run.render, base.render, ['modes']);
	if (render) out.render = render;
	if (run.imuRate !== base.imuRate) out.imuRate = run.imuRate;
	if (run.origin) out.origin = run.origin;
	return out;
}

/** Short keys for the view in the hash. */
const VIEW_KEYS = /** @type {const} */ ({ camera: 'c', overlays: 'o', mode: 'm', track: 't', footprint: 'f' });

/**
 * The run and the view as a URL hash: compact JSON, base64url (no padding).
 * @param {AppRun} run
 * @param {ViewState} view
 * @returns {string} without the leading '#'
 */
export function encodeHash(run, view) {
	/** @type {Record<string, any>} */
	const v = {};
	for (const [k, short] of Object.entries(VIEW_KEYS)) {
		const key = /** @type {keyof ViewState} */ (k);
		if (!deepEqual(view[key], viewDefaults[key])) v[short] = view[key];
	}
	/** @type {Record<string, any>} */
	const doc = { r: compactRun(run) };
	if (Object.keys(v).length) doc.v = v;
	const bytes = new TextEncoder().encode(JSON.stringify(doc));
	let binary = '';
	for (const b of bytes) binary += String.fromCharCode(b);
	return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

/**
 * The run and view a hash describes. Throws (resolveRun's message) on a hash that does
 * not decode to a valid run.
 * @param {string} hash with or without the leading '#'
 * @returns {{ run: AppRun, view: ViewState } | null} null for an empty hash
 */
export function decodeHash(hash) {
	const text = hash.replace(/^#/, '');
	if (!text) return null;
	const binary = atob(text.replace(/-/g, '+').replace(/_/g, '/'));
	const bytes = Uint8Array.from(binary, (c) => c.charCodeAt(0));
	const doc = JSON.parse(new TextDecoder().decode(bytes));
	if (!isObject(doc)) throw new Error('the link does not hold a run');
	const run = fromPartial(doc.r ?? {});
	/** @type {ViewState} */
	const view = structuredClone({ ...viewDefaults });
	const v = isObject(doc.v) ? doc.v : {};
	if (['chase', 'onboard', 'overhead'].includes(v.c)) view.camera = v.c;
	if (Array.isArray(v.o)) view.overlays = v.o.filter((/** @type {unknown} */ o) => o === 'flow' || o === 'grid' || o === 'horizon');
	if (renderModes.includes(v.m)) view.mode = v.m;
	if (typeof v.t === 'boolean') view.track = v.t;
	if (typeof v.f === 'boolean') view.footprint = v.f;
	return { run, view };
}

/**
 * Signed difference a − b of two compass angles, degrees in (−180, 180].
 * @param {number} a
 * @param {number} b
 */
export function angleDiff(a, b) {
	const d = (((a - b) % 360) + 360) % 360;
	return d > 180 ? d - 360 : d;
}

/** A compass bearing in [0, 360). @param {number} deg */
export const compass = (deg) => ((deg % 360) + 360) % 360;

/**
 * The command line that renders an exported run.json.
 * @param {string} [file]
 */
export const cliCommand = (file = 'run.json') => `node scripts/generate.mjs ${file} --out out`;
