/**
 * Files in and out of the browser: what the export section downloads (run.json and
 * trajectory.csv, the same files the batch generator writes for a run) and the run.json
 * it imports.
 */
import { resolveRun } from '../conditions/index.js';
import { samplesToCsv } from '../trajectory/index.js';

/**
 * Hand the browser a file to save.
 * @param {string} name
 * @param {string} text
 * @param {string} type MIME type
 */
export function download(name, text, type) {
	const url = URL.createObjectURL(new Blob([text], { type }));
	const a = document.createElement('a');
	a.href = url;
	a.download = name;
	a.style.display = 'none';
	document.body.append(a);
	a.click();
	a.remove();
	// Revoked a moment later: some browsers start the download after click() returns.
	setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/**
 * run.json as the batch generator writes it: the RunSpec fully resolved, tab-indented.
 * @param {import('../conditions/index.js').PartialRun} partial
 */
export function runJson(partial) {
	return `${JSON.stringify(resolveRun(partial), null, '\t')}\n`;
}

/**
 * trajectory.csv: every TrajectorySample field at `rate` Hz from t = 0 to the end.
 * @param {import('../trajectory/index.js').Trajectory} trajectory
 * @param {number} rate Hz
 */
export function trajectoryCsv(trajectory, rate) {
	return samplesToCsv(trajectory.sample(rate));
}

/**
 * A run.json the user picked, parsed (resolveRun checks it afterwards).
 * @param {File} file
 * @returns {Promise<Record<string, any>>}
 */
export async function readRunFile(file) {
	const text = await file.text();
	try {
		return JSON.parse(text);
	} catch {
		throw new Error(`${file.name} is not JSON`);
	}
}
