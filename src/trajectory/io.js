/**
 * Writers for trajectory samples: CSV (a header row, then plain decimal numbers, '\n'
 * line ends, no quoting) and JSON. Units are the TrajectorySample's: seconds, metres,
 * m/s, compass degrees, m/s² and rad/s (see index.js).
 */

/** Every TrajectorySample field, in the order the CSV writes them. */
export const sampleColumns = Object.freeze([
	't',
	'x',
	'y',
	'altitude',
	'vE',
	'vN',
	'vU',
	'heading',
	'course',
	'pitch',
	'bank',
	'turnRate',
	'airspeed',
	'groundSpeed',
	'windE',
	'windN',
	'windU',
	'fx',
	'fy',
	'fz',
	'p',
	'q',
	'r',
	'gimbalPitch',
	'gimbalYaw',
	'gimbalFov'
]);

/** The IMU file's columns: time, specific force (FRD, m/s²) and body rates (FRD, rad/s). */
export const imuColumns = Object.freeze(['t', 'fx', 'fy', 'fz', 'p', 'q', 'r']);

/**
 * A number as plain decimal text: no exponent, at most `digits` decimals, trailing zeros
 * trimmed (micrometres and microradians are well below anything the simulation resolves).
 * @param {number} v
 * @param {number} [digits]
 */
export function formatNumber(v, digits = 6) {
	if (!Number.isFinite(v)) return Number.isNaN(v) ? 'NaN' : v > 0 ? 'Infinity' : '-Infinity';
	if (Number.isInteger(v) && Math.abs(v) < 1e21) return String(v);
	let s = v.toFixed(digits);
	if (s.includes('.')) s = s.replace(/0+$/, '').replace(/\.$/, '');
	return s === '-0' ? '0' : s;
}

/**
 * @param {ArrayLike<Record<string, any>>} samples
 * @param {readonly string[]} [columns] default every sample field
 * @param {{ digits?: number }} [options]
 * @returns {string} CSV text, ending in '\n'
 */
export function samplesToCsv(samples, columns = sampleColumns, { digits = 6 } = {}) {
	const lines = [columns.join(',')];
	for (let i = 0; i < samples.length; i++) {
		const s = samples[i];
		lines.push(columns.map((c) => formatNumber(s[c], digits)).join(','));
	}
	return lines.join('\n') + '\n';
}

/**
 * The IMU file: t, fx, fy, fz, p, q, r.
 * @param {ArrayLike<Record<string, any>>} samples
 * @param {{ digits?: number }} [options]
 */
export function imuToCsv(samples, options) {
	return samplesToCsv(samples, imuColumns, options);
}

/**
 * Samples as a JSON array of objects with the given fields.
 * @param {ArrayLike<Record<string, any>>} samples
 * @param {readonly string[]} [columns]
 * @returns {string}
 */
export function samplesToJson(samples, columns = sampleColumns) {
	const rows = [];
	for (let i = 0; i < samples.length; i++) {
		/** @type {Record<string, number>} */
		const row = {};
		for (const c of columns) row[c] = samples[i][c];
		rows.push(row);
	}
	return JSON.stringify(rows);
}

/**
 * Read a CSV written by `samplesToCsv` back: its header and its rows as numbers.
 * @param {string} text
 * @returns {{ columns: string[], rows: number[][] }}
 */
export function parseCsv(text) {
	const lines = text.split('\n').filter((l) => l.length > 0);
	const columns = lines.length ? lines[0].split(',') : [];
	const rows = lines.slice(1).map((l) => l.split(',').map(Number));
	return { columns, rows };
}
