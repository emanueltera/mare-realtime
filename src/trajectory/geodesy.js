/**
 * Latitude and longitude for the run's east / north metres, when a run has an `origin`
 * (scripts/generate.mjs writes them beside the positions). Pure, node and browser.
 */

/** WGS-84. */
export const WGS84_A = 6378137;
export const WGS84_E2 = 6.69437999014e-3;

/**
 * East / north metres about `origin` as latitude / longitude, degrees. The run's world is
 * the local tangent plane at the origin (east / north / up, WGS-84): a point (east,
 * north) on it goes to earth-centred coordinates and back to the ellipsoid along its
 * normal (iterated to well under a millimetre). This keeps the meridians' convergence,
 * which a linear scaling by the radii of curvature at the origin would miss by metres
 * a few kilometres out at mid latitudes. The sea is flat, so a point d from the origin
 * sits about d² / 2R above the ellipsoid (2 m at 5 km); only its latitude and longitude
 * are written.
 * @param {{ lat: number, lon: number }} origin degrees
 * @returns {(east: number, north: number) => [number, number]} latitude, longitude in degrees
 */
export function enuToGeodetic(origin) {
	const DEG = Math.PI / 180;
	const phi0 = origin.lat * DEG;
	const lam0 = origin.lon * DEG;
	const [sp, cp, sl, cl] = [Math.sin(phi0), Math.cos(phi0), Math.sin(lam0), Math.cos(lam0)];
	const n0 = WGS84_A / Math.sqrt(1 - WGS84_E2 * sp * sp);
	// The origin, earth-centred (on the ellipsoid).
	const X0 = n0 * cp * cl;
	const Y0 = n0 * cp * sl;
	const Z0 = n0 * (1 - WGS84_E2) * sp;
	/** @param {number} east @param {number} north @returns {[number, number]} */
	return (east, north) => {
		// ENU → ECEF: east is (−sin λ, cos λ, 0), north (−sin φ cos λ, −sin φ sin λ, cos φ).
		const X = X0 - sl * east - sp * cl * north;
		const Y = Y0 + cl * east - sp * sl * north;
		const Z = Z0 + cp * north;
		const p = Math.hypot(X, Y);
		let phi = Math.atan2(Z, p * (1 - WGS84_E2));
		for (let i = 0; i < 6; i++) {
			const s = Math.sin(phi);
			const n = WGS84_A / Math.sqrt(1 - WGS84_E2 * s * s);
			const h = Math.abs(Math.cos(phi)) > 1e-9 ? p / Math.cos(phi) - n : Math.abs(Z) / Math.abs(s) - n * (1 - WGS84_E2);
			phi = Math.atan2(Z, p * (1 - (WGS84_E2 * n) / (n + h)));
		}
		return [phi / DEG, Math.atan2(Y, X) / DEG];
	};
}
