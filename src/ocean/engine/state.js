/**
 * Scene parameters as numbers the engine can blend: SceneSpec fields with defaults
 * filled, interpolation for transitions (headings by the shortest arc), and a
 * frame-rate independent approach for live overrides such as the wind slider.
 */
import { aircraftDefaults, lightDefaults, seaDefaults } from '../scene.js';
import { angleDelta, lerp, lerpAngle, wrap360 } from './util.js';

/** @typedef {Required<import('../scene.js').SeaState>} ResolvedSea */
/** @typedef {Required<import('../scene.js').Lighting>} ResolvedLight */

/**
 * @typedef {object} ResolvedAircraft
 * @property {number} altitude
 * @property {number} speed
 * @property {number} heading
 * @property {number} x
 * @property {number} y
 * @property {'straight' | 'orbit' | 'weave' | 'track'} path
 * @property {number} orbitRadius
 * @property {1 | -1} orbitDirection
 * @property {string | undefined} track
 * @property {number} turbulence
 * @property {{ pitch: number, yaw: number, fov: number }} gimbal
 * @property {boolean} footprint
 * @property {number} scale
 */

/**
 * @param {import('../scene.js').SeaState} sea
 * @param {Partial<import('../scene.js').SeaState>} [override]
 * @returns {ResolvedSea}
 */
export function resolveSea(sea, override = {}) {
	const s = { ...seaDefaults, ...sea, ...override };
	return {
		windSpeed: s.windSpeed,
		windDirection: wrap360(s.windDirection),
		fetch: s.fetch,
		spread: s.spread,
		swellHeight: s.swellHeight,
		swellDirection: wrap360(s.swellDirection),
		swellPeriod: s.swellPeriod,
		choppiness: s.choppiness
	};
}

/**
 * @param {import('../scene.js').Lighting} light
 * @param {Partial<import('../scene.js').Lighting>} [override]
 * @returns {ResolvedLight}
 */
export function resolveLight(light, override = {}) {
	const l = { ...lightDefaults, ...light, ...override };
	return {
		sunElevation: l.sunElevation,
		sunAzimuth: wrap360(l.sunAzimuth),
		overcast: l.overcast,
		haze: l.haze,
		exposure: l.exposure
	};
}

/**
 * @param {import('../scene.js').AircraftSpec} spec
 * @returns {ResolvedAircraft}
 */
export function resolveAircraft(spec) {
	const s = { ...aircraftDefaults, ...spec };
	return {
		altitude: s.altitude,
		speed: s.speed,
		heading: s.heading,
		x: s.x,
		y: s.y,
		path: s.path,
		orbitRadius: s.orbitRadius,
		orbitDirection: s.orbitDirection,
		track: s.track,
		turbulence: s.turbulence,
		gimbal: { pitch: s.gimbal.pitch, yaw: s.gimbal.yaw ?? 0, fov: s.gimbal.fov ?? 48 },
		footprint: s.footprint,
		scale: s.scale
	};
}

/** @param {ResolvedSea} s */
export const copySea = (s) => ({ ...s });
/** @param {ResolvedLight} l */
export const copyLight = (l) => ({ ...l });

/**
 * @param {ResolvedSea} a
 * @param {ResolvedSea} b
 * @param {number} t
 * @returns {ResolvedSea}
 */
export function blendSea(a, b, t) {
	return {
		windSpeed: lerp(a.windSpeed, b.windSpeed, t),
		windDirection: lerpAngle(a.windDirection, b.windDirection, t),
		fetch: Math.exp(lerp(Math.log(a.fetch), Math.log(b.fetch), t)),
		spread: lerp(a.spread, b.spread, t),
		swellHeight: lerp(a.swellHeight, b.swellHeight, t),
		swellDirection: lerpAngle(a.swellDirection, b.swellDirection, t),
		swellPeriod: lerp(a.swellPeriod, b.swellPeriod, t),
		choppiness: lerp(a.choppiness, b.choppiness, t)
	};
}

/**
 * @param {ResolvedLight} a
 * @param {ResolvedLight} b
 * @param {number} t
 * @returns {ResolvedLight}
 */
export function blendLight(a, b, t) {
	return {
		sunElevation: lerp(a.sunElevation, b.sunElevation, t),
		sunAzimuth: lerpAngle(a.sunAzimuth, b.sunAzimuth, t),
		overcast: lerp(a.overcast, b.overcast, t),
		haze: lerp(a.haze, b.haze, t),
		exposure: lerp(a.exposure, b.exposure, t)
	};
}

/**
 * Move `current` toward `target` by fraction k (angles the short way round).
 * @template {Record<string, number>} T
 * @param {T} current
 * @param {T} target
 * @param {number} k 0–1
 * @param {string[]} angles keys that are compass angles
 */
export function approach(current, target, k, angles) {
	for (const key of Object.keys(target)) {
		const c = /** @type {number} */ (current[key]);
		const t = /** @type {number} */ (target[key]);
		/** @type {Record<string, number>} */ (current)[key] = angles.includes(key) ? wrap360(c + angleDelta(c, t) * k) : c + (t - c) * k;
	}
	return current;
}

export const SEA_ANGLES = ['windDirection', 'swellDirection'];
export const LIGHT_ANGLES = ['sunAzimuth'];
