/**
 * The mean wind the aircraft flies in. Directions are compass degrees the air moves
 * toward (the sea's convention, not the meteorological "from"): a wind with
 * `direction: 90` blows east. Vectors are east / north, m/s.
 *
 * `wind: 'sea'` takes the sea state's U10 (the wind that raised the waves, at 10 m)
 * up to the aircraft's altitude with the neutral logarithmic profile of the surface
 * layer over open water:
 *
 *   U(z) = U10 · ln(z / z0) / ln(10 / z0)
 *
 * with the roughness length z0 from Charnock's relation for a wind sea,
 *
 *   z0 = α · u*² / g + 0.11 · ν / u*,      u* = κ · U10 / ln(10 / z0)
 *
 * (α = 0.011, the open-ocean Charnock constant; ν the kinematic viscosity of air;
 * κ = 0.4, von Kármán), solved by fixed-point iteration. The second term keeps z0
 * sensible in light air, where the sea is aerodynamically smooth. The result is a
 * gentle increase with height — about +20 % from 10 m to 120 m in a fresh breeze — and
 * the direction does not turn with height (no Ekman spiral: the profile is the surface
 * layer's, which is what a low-flying UAV is in). Stability is not modelled: the
 * profile is the neutral one at every hour and over every sea.
 */

const G = 9.81;
const DEG = Math.PI / 180;
/** Charnock's constant for open-ocean wind seas. */
const CHARNOCK = 0.011;
/** von Kármán's constant. */
const KARMAN = 0.4;
/** Kinematic viscosity of air, m²/s. */
const NU = 1.5e-5;
/** The profile is not taken below this height, m: it diverges at z0. */
const MIN_HEIGHT = 1;

/**
 * Roughness length and friction velocity of the sea surface for a 10 m wind.
 * @param {number} u10 m/s
 * @returns {{ z0: number, frictionVelocity: number }}
 */
export function seaRoughness(u10) {
	const u = Math.max(0, u10);
	if (u === 0) return { z0: 1e-4, frictionVelocity: 0 };
	let z0 = 2e-4;
	let ustar = (KARMAN * u) / Math.log(10 / z0);
	// Converges to well below a part in a million within a dozen iterations for any
	// realistic wind; a fixed count keeps it deterministic.
	for (let i = 0; i < 30; i++) {
		z0 = (CHARNOCK * ustar * ustar) / G + (0.11 * NU) / ustar;
		ustar = (KARMAN * u) / Math.log(10 / z0);
	}
	return { z0, frictionVelocity: ustar };
}

/**
 * @typedef {object} WindAloft
 * @property {number} speed m/s at the altitude asked for
 * @property {number} direction compass degrees the air moves toward
 * @property {number} east m/s
 * @property {number} north m/s
 * @property {number} z0 the sea's roughness length, m
 * @property {number} frictionVelocity u*, m/s
 */

/**
 * The sea state's wind carried up to `altitude` (m above mean sea level) by the
 * neutral log profile over open water.
 * @param {{ windSpeed: number, windDirection: number }} sea a SeaState (only its wind is read)
 * @param {number} altitude m
 * @returns {WindAloft}
 */
export function windAloft(sea, altitude) {
	const u10 = Math.max(0, sea.windSpeed);
	const { z0, frictionVelocity } = seaRoughness(u10);
	const z = Math.max(MIN_HEIGHT, altitude);
	const speed = u10 === 0 ? 0 : (u10 * Math.log(z / z0)) / Math.log(10 / z0);
	const direction = (((sea.windDirection ?? 0) % 360) + 360) % 360;
	return {
		speed,
		direction,
		east: speed * Math.sin(direction * DEG),
		north: speed * Math.cos(direction * DEG),
		z0,
		frictionVelocity
	};
}

/**
 * The mean wind as a function of altitude, with its vertical gradient (the aircraft
 * sees the wind change as it climbs through the profile, and that change is part of
 * its acceleration).
 * @typedef {(altitude: number, out: MeanWind) => MeanWind} WindField
 * @typedef {{ east: number, north: number, dEast: number, dNorth: number }} MeanWind
 *   dEast, dNorth: ∂/∂z, 1/s
 */

/**
 * @param {import('./index.js').WindSpec} spec
 * @param {{ windSpeed: number, windDirection: number } | undefined} sea
 * @returns {WindField}
 */
export function createWindField(spec, sea) {
	if (spec === 'none') {
		return (_, out) => {
			out.east = out.north = out.dEast = out.dNorth = 0;
			return out;
		};
	}
	if (spec === 'sea') {
		if (!sea) throw new TypeError("trajectory wind: 'sea' needs the sea state: createTrajectory(spec, { sea })");
		const u10 = Math.max(0, sea.windSpeed);
		const { z0 } = seaRoughness(u10);
		const dir = (sea.windDirection ?? 0) * DEG;
		const se = Math.sin(dir);
		const sn = Math.cos(dir);
		const norm = u10 === 0 ? 0 : u10 / Math.log(10 / z0);
		return (altitude, out) => {
			const z = Math.max(MIN_HEIGHT, altitude);
			const speed = norm * Math.log(z / z0);
			// dU/dz = U10 / (z · ln(10 / z0)); zero where the profile is held constant.
			const shear = altitude > MIN_HEIGHT ? norm / z : 0;
			out.east = speed * se;
			out.north = speed * sn;
			out.dEast = shear * se;
			out.dNorth = shear * sn;
			return out;
		};
	}
	const speed = Math.max(0, spec.speed);
	const dir = spec.direction * DEG;
	const east = speed * Math.sin(dir);
	const north = speed * Math.cos(dir);
	return (_, out) => {
		out.east = east;
		out.north = north;
		out.dEast = out.dNorth = 0;
		return out;
	};
}
