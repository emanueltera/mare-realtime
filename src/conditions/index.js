/**
 * Conditions for simulated flights: named seas, skies and flight patterns, complete
 * run specs, the expansion of job files into runs, and the scene a run is filmed from (docs/ARCHITECTURE.md, "Runs
 * and jobs"). Pure and deterministic, no three.js and no DOM: the batch page, the app
 * and the node scripts all expand a job the same way.
 *
 * ```js
 * import { expandJob, runSummary } from './src/conditions/index.js';
 * for (const run of expandJob(job)) console.log(run.id, runSummary(run));
 * ```
 */
export { seaPresets, lightPresets, patternPresets, describeSea } from './presets.js';
export {
	renderDefaults,
	trajectoryDefaults,
	renderModes,
	seaSeedFor,
	resolveRun,
	runSummary,
	DEFAULT_SEA,
	DEFAULT_LIGHT,
	DEFAULT_PATTERN,
	DEFAULT_SEA_SEED,
	DEFAULT_IMU_RATE
} from './run.js';
export { expandJob, randomRun, WIND_MARGIN } from './job.js';
export { runScene, chaseCamera, onboardCamera } from './views.js';

/** @typedef {import('./run.js').RunSpec} RunSpec */
/** @typedef {import('./run.js').PartialRun} PartialRun */
/** @typedef {import('./run.js').TrajectorySpec} TrajectorySpec */
/** @typedef {import('./run.js').PatternSpec} PatternSpec */
/** @typedef {import('./run.js').RenderSpec} RenderSpec */
/** @typedef {import('./job.js').Job} Job */
