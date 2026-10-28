/**
 * The wave physics the simulator, the presets and the readouts share: one set of
 * formulas, so the sea-state numbers (conditions presets, meta.json, the app's panel)
 * and the ocean on screen all describe the same sea.
 *
 * The wind sea is a fetch-limited JONSWAP spectrum (Hasselmann et al., 1973). Past
 * full development — a nondimensional fetch gF/U² of about 1.7e4, where JONSWAP's
 * peak frequency meets the Pierson–Moskowitz limit of 0.855 g/U — the fetch is
 * clamped, since a sea cannot keep growing once the wind has stopped feeding it.
 * The peak enhancement relaxes with fetch rather than staying at the textbook 3.3,
 * γ = 7·(gF/U²)^−0.143 (Mitsuyasu et al., 1980): young seas are sharply peaked and
 * a developed one approaches Pierson–Moskowitz's γ = 1. A constant 3.3 overstates a
 * developed sea's height by a third.
 * A swell, when there is one, is a second JONSWAP-shaped band with a sharp peak
 * (γ = 7), scaled so its significant height is exactly the one asked for.
 *
 * Deep water throughout: ω² = g·k.
 */

export const G = 9.81;

/** gF/U² at which a fetch-limited JONSWAP sea is fully developed. */
const FULLY_DEVELOPED_FETCH = 1.7e4;

const SWELL_GAMMA = 7;

/**
 * @typedef {object} BandParams
 * @property {number} alpha Phillips constant
 * @property {number} omegaPeak peak angular frequency, rad/s
 * @property {number} gamma peak enhancement
 */

/**
 * JONSWAP parameters for a wind sea.
 * @param {number} windSpeed U10, m/s
 * @param {number} fetchKm km
 * @returns {BandParams}
 */
export function windSeaParams(windSpeed, fetchKm) {
	const U = Math.max(windSpeed, 0.5);
	const fetch = Math.max(fetchKm, 0.1) * 1000;
	const fetchNd = Math.min((G * fetch) / (U * U), FULLY_DEVELOPED_FETCH);
	return {
		alpha: 0.076 * Math.pow(fetchNd, -0.22),
		omegaPeak: 22 * (G / U) * Math.pow(fetchNd, -1 / 3),
		gamma: Math.max(1, 7 * Math.pow(fetchNd, -0.143))
	};
}

/**
 * JONSWAP frequency spectrum S(ω), m²·s/rad.
 * @param {number} omega rad/s
 * @param {BandParams} band
 */
export function jonswap(omega, { alpha, omegaPeak, gamma }) {
	if (omega <= 0) return 0;
	const sigma = omega <= omegaPeak ? 0.07 : 0.09;
	const offset = omega - omegaPeak;
	const r = Math.exp(-(offset * offset) / (2 * sigma * sigma * omegaPeak * omegaPeak));
	return (
		((alpha * G * G) / Math.pow(omega, 5)) *
		Math.exp(-1.25 * Math.pow(omegaPeak / omega, 4)) *
		Math.pow(gamma, r)
	);
}

/**
 * Zeroth moment ∫S(ω)dω of one band, by trapezoid over where nearly all its energy is.
 * @param {BandParams} band
 */
function zerothMoment(band) {
	const lo = 0.35 * band.omegaPeak;
	const hi = 8 * band.omegaPeak;
	const steps = 600;
	const dw = (hi - lo) / steps;
	let sum = 0;
	for (let i = 0; i <= steps; i++) {
		const w = lo + i * dw;
		sum += jonswap(w, band) * (i === 0 || i === steps ? 0.5 : 1);
	}
	return sum * dw;
}

/**
 * The swell band, scaled so 4·√m0 equals `height`. Null when there is no swell.
 * @param {number} height significant height, m
 * @param {number} period peak period, s
 * @returns {BandParams | null}
 */
export function swellParams(height, period) {
	if (!(height > 0.01)) return null;
	const unit = { alpha: 1, omegaPeak: (2 * Math.PI) / Math.max(period, 2), gamma: SWELL_GAMMA };
	const target = (height / 4) ** 2;
	return { ...unit, alpha: target / zerothMoment(unit) };
}

/**
 * Normalisation for the cos^2s((θ − θ̄)/2) spreading function, so it integrates to 1
 * over a full turn: Γ(s+1)² · 2^(2s−1) / (π · Γ(2s+1)).
 * @param {number} s spreading exponent
 */
export function spreadingNorm(s) {
	return Math.exp(2 * logGamma(s + 1) + (2 * s - 1) * Math.LN2 - Math.log(Math.PI) - logGamma(2 * s + 1));
}

/**
 * Directional spreading D(θ), per radian.
 * @param {number} theta wave travel direction, radians (any convention, as long as `mean` matches)
 * @param {number} mean mean direction, radians
 * @param {number} s spreading exponent
 */
export function spreading(theta, mean, s) {
	return spreadingNorm(s) * Math.pow(Math.abs(Math.cos((theta - mean) / 2)), 2 * s);
}

/**
 * Lanczos approximation to ln Γ(x), x > 0.
 * @param {number} x
 * @returns {number}
 */
function logGamma(x) {
	const c = [
		676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
		12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
	];
	if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - logGamma(1 - x);
	const z = x - 1;
	let a = 0.99999999999980993;
	const t = z + 7.5;
	for (let i = 0; i < c.length; i++) a += c[i] / (z + i + 1);
	return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(a);
}

/** Deep-water angular frequency for a wavenumber. @param {number} k rad/m */
export const dispersion = (k) => Math.sqrt(G * k);

/** Deep-water wavelength for a period. @param {number} period s */
export const wavelengthOf = (period) => (G * period * period) / (2 * Math.PI);

/**
 * Deep-water phase speed for a period, c = gT/2π: how fast the crests themselves
 * travel across the sea — which a camera over open water sees as motion of the
 * "ground". @param {number} period s
 */
export const phaseSpeedOf = (period) => (G * period) / (2 * Math.PI);

/**
 * Beaufort force for a 10 m wind speed, from U = 0.836·B^(3/2).
 * @param {number} windSpeed m/s
 */
export function beaufort(windSpeed) {
	return Math.max(0, Math.min(12, Math.round(Math.pow(Math.max(windSpeed, 0) / 0.836, 2 / 3))));
}

/** @type {string[]} */
export const beaufortNames = [
	'Calm',
	'Light air',
	'Light breeze',
	'Gentle breeze',
	'Moderate breeze',
	'Fresh breeze',
	'Strong breeze',
	'Near gale',
	'Gale',
	'Strong gale',
	'Storm',
	'Violent storm',
	'Hurricane force'
];

/**
 * @typedef {object} SeaSpectrum
 * @property {BandParams} wind
 * @property {BandParams | null} swell
 */

/**
 * Both bands for a sea state, with defaults filled.
 * @param {import('./scene.js').SeaState} sea
 * @returns {SeaSpectrum}
 */
export function seaSpectrum(sea) {
	return {
		wind: windSeaParams(sea.windSpeed, sea.fetch ?? 400),
		swell: swellParams(sea.swellHeight ?? 0, sea.swellPeriod ?? 11)
	};
}

/**
 * Total S(ω) of a sea, wind sea plus swell.
 * @param {number} omega rad/s
 * @param {SeaSpectrum} spectrum
 */
export function seaDensity(omega, spectrum) {
	return jonswap(omega, spectrum.wind) + (spectrum.swell ? jonswap(omega, spectrum.swell) : 0);
}

/**
 * Peak angular frequency of the whole sea, wind sea plus swell: the ω at which the
 * total S(ω) is largest. When a swell dominates this is the swell's peak, so Tp,
 * λp and the crest speed describe the waves that are biggest on screen, as Hs does.
 * The two bands are sharply peaked, so the maximum lies near one of their peaks; the
 * stronger one is refined with a short scan, since the other band's flank can nudge it.
 * @param {SeaSpectrum} spectrum
 * @returns {{ omega: number, band: 'wind' | 'swell' }}
 */
export function peakOmega(spectrum) {
	/** @type {Array<{ omega: number, band: 'wind' | 'swell' }>} */
	const candidates = [{ omega: spectrum.wind.omegaPeak, band: 'wind' }];
	if (spectrum.swell) candidates.push({ omega: spectrum.swell.omegaPeak, band: 'swell' });
	let best = candidates[0];
	let bestS = seaDensity(best.omega, spectrum);
	for (const c of candidates.slice(1)) {
		const s = seaDensity(c.omega, spectrum);
		if (s > bestS) [best, bestS] = [c, s];
	}
	let omega = best.omega;
	const steps = 40;
	for (let i = 0; i <= steps; i++) {
		const w = best.omega * (0.85 + (0.3 * i) / steps);
		const s = seaDensity(w, spectrum);
		if (s > bestS) [omega, bestS] = [w, s];
	}
	return { omega, band: best.band };
}

/**
 * @typedef {object} SeaStats
 * @property {number} significantHeight Hs = 4·√m0 of wind sea and swell together, m
 * @property {number} windSeaHeight Hs of the wind sea alone, m
 * @property {number} peakPeriod Tp of the whole sea (the period at the peak of S(ω), wind sea or swell), s
 * @property {number} peakWavelength deep-water wavelength at Tp, m
 * @property {'wind' | 'swell'} peakBand which band holds the peak
 * @property {number} beaufort
 */

/**
 * The simulated sea's summary statistics (significant height, peak period and wavelength,
 * Beaufort force): what the app's conditions panel, the README's preset table and each
 * run's meta.json report, from the same spectrum the engine draws.
 * @param {import('./scene.js').SeaState} sea
 * @returns {SeaStats}
 */
export function seaStats(sea) {
	const { wind, swell } = seaSpectrum(sea);
	const m0Wind = zerothMoment(wind);
	const m0Swell = swell ? zerothMoment(swell) : 0;
	const peak = peakOmega({ wind, swell });
	const peakPeriod = (2 * Math.PI) / peak.omega;
	return {
		significantHeight: 4 * Math.sqrt(m0Wind + m0Swell),
		windSeaHeight: 4 * Math.sqrt(m0Wind),
		peakPeriod,
		peakWavelength: wavelengthOf(peakPeriod),
		peakBand: peak.band,
		beaufort: beaufort(sea.windSpeed)
	};
}
