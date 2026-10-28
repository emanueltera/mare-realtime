/**
 * Named conditions a run can ask for by name: seas, skies and flight patterns.
 *
 * The seas are wind seas on the simulator's own spectrum (`../ocean/spectrum.js`):
 * fetch-limited JONSWAP that stops growing at full development, with a separate swell
 * band for the swell variants. Each wind speed sits in the middle of its Beaufort
 * force, so a preset's name, its force and its sea agree. The numbers in each
 * description (Hs, Tp, whether the sea is fully developed) are computed from that
 * spectrum when this module loads, never typed in, so they cannot drift from what is
 * rendered: they describe the simulated sea, not a measured one.
 *
 * Conventions as everywhere: directions are compass degrees, 0 north, 90 east. Seas,
 * swells and wind are the way they travel *toward* (a sea with `windDirection: 60`
 * runs toward the east-north-east); `sunAzimuth` is the bearing toward the sun, so
 * 180 puts it in the south.
 * Pattern lengths are metres, relative to the trajectory's start and its heading.
 */
import { seaDefaults, lightDefaults } from '../ocean/scene.js';
import { G, beaufortNames, seaStats } from '../ocean/spectrum.js';

/** @typedef {import('../ocean/scene.js').SeaState} SeaState */
/** @typedef {import('../ocean/scene.js').Lighting} Lighting */
/** @typedef {import('./run.js').PatternSpec} PatternSpec */

/**
 * @typedef {object} Preset
 * @property {string} label short name for a menu
 * @property {string} description one or two sentences: what it is, and (seas) its simulated statistics
 */

/** @typedef {Preset & { sea: Required<SeaState> }} SeaPreset */
/** @typedef {Preset & { light: Required<Lighting> }} LightPreset */
/** @typedef {Preset & { pattern: PatternSpec }} PatternPreset */

/** gF/U² past which the simulator's wind sea stops growing (as in spectrum.js). */
const FULLY_DEVELOPED_FETCH = 1.7e4;

/**
 * Every preset sea travels the same way, toward the east-north-east, so a sweep over
 * presets varies the sea state and nothing else; the trajectory's start heading sets
 * the angle the aircraft meets it at.
 */
const SEA_DIRECTION = 60;

/**
 * A complete SeaState, every field present, so a resolved run can be matched back to
 * the preset it came from.
 * @param {SeaState} sea
 * @returns {Required<SeaState>}
 */
function seaOf(sea) {
	return {
		windSpeed: sea.windSpeed,
		windDirection: sea.windDirection,
		fetch: sea.fetch ?? seaDefaults.fetch,
		spread: sea.spread ?? seaDefaults.spread,
		swellHeight: sea.swellHeight ?? seaDefaults.swellHeight,
		swellDirection: sea.swellDirection ?? seaDefaults.swellDirection,
		swellPeriod: sea.swellPeriod ?? seaDefaults.swellPeriod,
		choppiness: sea.choppiness ?? seaDefaults.choppiness
	};
}

/**
 * @param {Lighting} light
 * @returns {Required<Lighting>}
 */
function lightOf(light) {
	return {
		sunElevation: light.sunElevation,
		sunAzimuth: light.sunAzimuth,
		overcast: light.overcast ?? lightDefaults.overcast,
		haze: light.haze ?? lightDefaults.haze,
		exposure: light.exposure ?? lightDefaults.exposure
	};
}

/** An input value as written: no trailing zeros. @param {number} x @param {number} [digits] */
const fmt = (x, digits = 1) => String(Number(x.toFixed(digits)));

/** A computed height: centimetres while it is small, decimetres after. @param {number} h */
const metres = (h) => h.toFixed(h < 0.5 ? 2 : 1);

/** Signed shortest angle from a to b, degrees in (−180, 180]. @param {number} a @param {number} b */
function angleBetween(a, b) {
	const d = ((((b - a) % 360) + 360) % 360);
	return d > 180 ? d - 360 : d;
}

/**
 * The computed half of a sea preset's description: force, fetch, development and the
 * simulated Hs / Tp from `seaStats`.
 * @param {Required<SeaState>} sea
 * @returns {string}
 */
export function describeSea(sea) {
	const stats = seaStats(sea);
	const U = Math.max(sea.windSpeed, 0.5);
	const developed = (G * sea.fetch * 1000) / (U * U) >= FULLY_DEVELOPED_FETCH;
	const force = `Beaufort ${stats.beaufort} (${beaufortNames[stats.beaufort].toLowerCase()})`;
	let text = `${force}: U10 ${fmt(sea.windSpeed)} m/s over ${fmt(sea.fetch, 0)} km of fetch, ${developed ? 'fully developed' : 'fetch-limited'}`;
	if (sea.swellHeight > 0.01) {
		const off = Math.abs(angleBetween(sea.windDirection, sea.swellDirection));
		text += `, with a ${fmt(sea.swellHeight)} m, ${fmt(sea.swellPeriod, 0)} s swell ${off < 20 ? 'running with it' : `${fmt(off, 0)}° across it`}`;
		text += `. Simulated Hs ${metres(stats.significantHeight)} m (wind sea ${metres(stats.windSeaHeight)} m), Tp ${stats.peakPeriod.toFixed(1)} s (${stats.peakBand === 'swell' ? 'the swell' : 'the wind sea'}).`;
	} else {
		text += `. Simulated Hs ${metres(stats.significantHeight)} m, Tp ${stats.peakPeriod.toFixed(1)} s.`;
	}
	return text;
}

/**
 * @param {Required<Lighting>} light
 * @returns {string}
 */
function describeLight(light) {
	const sun =
		light.sunElevation < 0
			? `sun ${fmt(-light.sunElevation, 0)}° below the horizon`
			: `sun ${fmt(light.sunElevation, 0)}° up`;
	return `${sun}, toward ${fmt(light.sunAzimuth, 0)}°; overcast ${fmt(light.overcast, 2)}, haze ${fmt(light.haze, 2)}.`;
}

/**
 * @param {string} label
 * @param {string} lead hand-written: what this sea looks like
 * @param {SeaState} sea
 * @returns {SeaPreset}
 */
function seaPreset(label, lead, sea) {
	const full = seaOf(sea);
	return { label, description: `${lead} ${describeSea(full)}`, sea: full };
}

/**
 * @param {string} label
 * @param {string} lead
 * @param {Lighting} light
 * @returns {LightPreset}
 */
function lightPreset(label, lead, light) {
	const full = lightOf(light);
	const words = describeLight(full);
	return { label, description: `${lead} ${words[0].toUpperCase()}${words.slice(1)}`, light: full };
}

/**
 * Wind seas from glassy to gale, then swell variants built on the moderate and calm
 * seas. Spreading and choppiness rise with the wind: a young light-air sea is ripples
 * on a rolling surface, a gale is short-crested and sharp. The gale and strong seas
 * blow over a limited fetch, so they are still growing; the rest are fully developed.
 * @type {Record<string, SeaPreset>}
 */
export const seaPresets = deepFreeze({
	glassy: seaPreset('Glassy', 'Near-flat water under light air: ripples and little else.', {
		windSpeed: 1,
		windDirection: SEA_DIRECTION,
		fetch: 400,
		spread: 5,
		choppiness: 0.5
	}),
	calm: seaPreset('Calm', 'Small wavelets, crests glassy and not breaking.', {
		windSpeed: 3,
		windDirection: SEA_DIRECTION,
		fetch: 400,
		spread: 6,
		choppiness: 0.7
	}),
	light: seaPreset('Light', 'Large wavelets, crests beginning to break; scattered whitecaps.', {
		windSpeed: 5,
		windDirection: SEA_DIRECTION,
		fetch: 400,
		spread: 7,
		choppiness: 0.85
	}),
	moderate: seaPreset('Moderate', 'Small waves growing longer; fairly frequent whitecaps.', {
		windSpeed: 7,
		windDirection: SEA_DIRECTION,
		fetch: 400,
		spread: 8,
		choppiness: 1
	}),
	fresh: seaPreset('Fresh', 'Moderate waves of a more pronounced long form; many whitecaps.', {
		windSpeed: 9.5,
		windDirection: SEA_DIRECTION,
		fetch: 400,
		spread: 8,
		choppiness: 1.1
	}),
	strong: seaPreset('Strong', 'Large waves forming, white foam crests more extensive everywhere.', {
		windSpeed: 12,
		windDirection: SEA_DIRECTION,
		fetch: 150,
		spread: 9,
		choppiness: 1.2
	}),
	gale: seaPreset('Gale', 'High, long waves; crests break into spindrift, foam in streaks along the wind.', {
		windSpeed: 18,
		windDirection: SEA_DIRECTION,
		fetch: 300,
		spread: 10,
		choppiness: 1.3
	}),
	'moderate-swell': seaPreset('Moderate + swell', 'The moderate sea riding a long swell from the same quarter.', {
		windSpeed: 7,
		windDirection: SEA_DIRECTION,
		fetch: 400,
		spread: 8,
		choppiness: 1,
		swellHeight: 1.5,
		swellPeriod: 13,
		swellDirection: SEA_DIRECTION + 15
	}),
	'cross-swell': seaPreset('Cross swell', 'The moderate sea with a swell crossing it nearly at right angles: a confused, criss-crossed surface.', {
		windSpeed: 7,
		windDirection: SEA_DIRECTION,
		fetch: 400,
		spread: 8,
		choppiness: 1,
		swellHeight: 1.8,
		swellPeriod: 11,
		swellDirection: SEA_DIRECTION + 80
	}),
	'calm-swell': seaPreset('Calm + swell', 'Light wind over a long, low swell: smooth rolling hills with little texture on them.', {
		windSpeed: 3,
		windDirection: SEA_DIRECTION,
		fetch: 400,
		spread: 6,
		choppiness: 0.7,
		swellHeight: 1.2,
		swellPeriod: 14,
		swellDirection: SEA_DIRECTION - 30
	})
});

/**
 * Skies. They follow the rules randomRun keeps (job.js): a sun
 * near or under the horizon has an open sky (overcast ≤ 0.4) and clearer air (haze ≤
 * 0.55 below it), since a closed deck at dusk is a black frame; a closed deck comes
 * with thick air (haze ≥ 0.3). Azimuths are for a northern-hemisphere day: the
 * morning sun in the east-south-east, midday due south, evening in the west.
 * @type {Record<string, LightPreset>}
 */
export const lightPresets = deepFreeze({
	midday: lightPreset('Midday', 'High sun in a clear sky: a hard sun glint, strong contrast.', {
		sunElevation: 60,
		sunAzimuth: 180,
		overcast: 0.05,
		haze: 0.15
	}),
	morning: lightPreset('Morning', 'Mid-height sun with a few clouds.', {
		sunElevation: 22,
		sunAzimuth: 110,
		overcast: 0.15,
		haze: 0.3
	}),
	'low-sun': lightPreset('Low sun', 'Late-afternoon sun a few degrees up: long glitter path, dark wave faces.', {
		sunElevation: 7,
		sunAzimuth: 250,
		overcast: 0.1,
		haze: 0.35
	}),
	overcast: lightPreset('Overcast', 'A closed cloud deck: flat, shadowless light and a grey sea.', {
		sunElevation: 35,
		sunAzimuth: 160,
		overcast: 0.85,
		haze: 0.45
	}),
	hazy: lightPreset('Hazy', 'Thick marine haze washing out the horizon and the distance.', {
		sunElevation: 30,
		sunAzimuth: 200,
		overcast: 0.25,
		haze: 0.75
	}),
	dusk: lightPreset('Dusk', 'Blue hour just after sunset: dim, low-contrast, the sky brighter than the sea.', {
		sunElevation: -3,
		sunAzimuth: 290,
		overcast: 0.2,
		haze: 0.35
	})
});

/**
 * Flight patterns. Sized for a small fixed-wing UAV at about 24 m/s, whose turn radius
 * at its bank limit is of the order of 100 m: survey legs are spaced two turn diameters
 * apart, so the aircraft can reverse onto the next leg without overshooting it.
 * @type {Record<string, PatternPreset>}
 */
export const patternPresets = deepFreeze({
	transit: {
		label: 'Transit',
		description: 'Straight and level along the start heading.',
		pattern: { type: 'straight' }
	},
	loiter: {
		label: 'Loiter',
		description: 'A clockwise orbit of 250 m radius, the sun and the sea passing all the way round.',
		pattern: { type: 'orbit', radius: 250, direction: 1 }
	},
	survey: {
		label: 'Survey',
		description: 'A lawnmower: four 900 m legs 200 m apart.',
		pattern: { type: 'lawnmower', legLength: 900, spacing: 200, legs: 4 }
	},
	racetrack: {
		label: 'Racetrack',
		description: 'Two 900 m straights joined by 200 m-radius turns, clockwise.',
		pattern: { type: 'racetrack', length: 900, radius: 200, direction: 1 }
	},
	figure8: {
		label: 'Figure eight',
		description: 'Two 250 m-radius loops turning opposite ways: both senses of turn in one flight.',
		pattern: { type: 'figure8', radius: 250 }
	},
	random: {
		label: 'Random',
		description: 'Seeded random legs of 300–900 m joined by turns of up to 150° either way.',
		pattern: { type: 'random', legLength: [300, 900], turn: [-150, 150] }
	}
});

/**
 * Freeze an object and everything in it, so the shared presets cannot be edited
 * through a run that forgot to copy them.
 * @template T
 * @param {T} value
 * @returns {T}
 */
export function deepFreeze(value) {
	if (value && typeof value === 'object') {
		for (const v of Object.values(value)) deepFreeze(v);
		Object.freeze(value);
	}
	return value;
}
