/**
 * The headless page the batch generator drives (batch.html, scripts/generate.mjs). It
 * renders one run at a time with a manual engine and hands each frame back to node:
 *
 * ```js
 * const { frames } = await window.mare.load(run);
 * for (let i = 0; i < frames; i++) {
 *   const { truth, sample, images } = await window.mare.frame(i);   // images: { [mode]: PNG data URL }
 * }
 * window.mare.dispose();
 * ```
 *
 * Time: frame i is the flight at t = i / fps. Before frame 0 the sea, its foam and the
 * camera rigs run for `render.warmup` seconds with the flight held at t = 0 (advance with
 * `flight: false`), then `seek(0)` restarts the rigs settled on the aircraft. So the
 * sea's clock at frame i is `render.seaTime + warmup + t` while the flight's starts at
 * 0, and every frame after the first is exactly one `advance(1 / fps)` from the one
 * before (docs/ARCHITECTURE.md, the manual engine's run sequence).
 *
 * Ground truth is what was rendered: `truth` is the engine's own FrameTruth after the
 * frame was drawn, and `sample` is the trajectory at the t that frame actually drew
 * (FrameTruth.t, a float sum of the steps (and sub-steps), within a few ulps of i / fps).
 *
 * Anything that would make the frames wrong rather than merely slow — no WebGL2, no
 * float targets, a shader that does not compile, a lost context — throws with a message,
 * so the generator records the run as failed instead of writing black frames.
 */
import { resolveRun, runScene } from '../conditions/index.js';
import { OceanEngine } from '../ocean/engine/engine.js';
import { createTrajectory } from '../trajectory/index.js';

/** @typedef {import('../conditions/index.js').RunSpec} RunSpec */
/** @typedef {import('../ocean/engine/engine.js').FrameTruth} FrameTruth */
/** @typedef {import('../ocean/scene.js').RenderMode} RenderMode */
/** @typedef {import('../trajectory/index.js').Trajectory} Trajectory */

/**
 * What `load` reports back: how many frames the run has, and how they will be made.
 * @typedef {object} LoadInfo
 * @property {number} frames one at every 1 / fps from t = 0 up to the duration
 * @property {object} meta
 * @property {string} meta.renderer the GPU as WebGL reports it (unmasked where allowed)
 * @property {string} meta.vendor
 * @property {boolean} meta.software whether that is a software rasteriser (SwiftShader, llvmpipe)
 * @property {number} meta.duration trajectory seconds
 * @property {number} meta.warmupSteps advance() calls before frame 0
 * @property {number} meta.seaTimeAtFrame0 the sea's clock at frame 0, s
 * @property {number} meta.loadSeconds wall-clock seconds to build the engine and warm up
 */

/**
 * One frame, as `frame(i)` returns it.
 * @typedef {object} FrameResult
 * @property {number} index
 * @property {FrameTruth} truth the camera, aircraft, sea and light of the frame drawn
 * @property {import('../trajectory/index.js').TrajectorySample} sample the trajectory at truth.t
 * @property {Record<string, string>} images one PNG data URL per render mode
 */

/** Steps shorter than this are float dust left over from slicing the warm-up. */
const EPS = 1e-9;

/**
 * The run being rendered, or null between runs.
 * @type {null | {
 *   run: RunSpec,
 *   engine: OceanEngine,
 *   root: HTMLElement,
 *   trajectory: Trajectory,
 *   frames: number,
 *   index: number,
 *   busy: boolean,
 *   health: { failure: string | null }
 * }}
 */
let current = null;

/**
 * Which GPU WebGL2 runs on, and whether it is a software rasteriser. Checked on a
 * throwaway canvas before the engine takes the real one, so a missing WebGL2 or float
 * targets fail here with a message rather than somewhere inside three.js.
 */
function probeGpu() {
	const canvas = document.createElement('canvas');
	const gl = canvas.getContext('webgl2');
	if (!gl) throw new Error('WebGL2 is not available in this browser (GPU blocklisted or disabled? try --software)');
	const ext = gl.getExtension('WEBGL_debug_renderer_info');
	const renderer = String(ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER));
	const vendor = String(ext ? gl.getParameter(ext.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR));
	const float = Boolean(gl.getExtension('EXT_color_buffer_float'));
	gl.getExtension('WEBGL_lose_context')?.loseContext();
	if (!float) throw new Error(`WebGL2 on '${renderer}' cannot render to float targets (EXT_color_buffer_float), which the ocean needs`);
	return { renderer, vendor, software: /swiftshader|llvmpipe|softpipe|software/i.test(renderer) };
}

/** @param {Blob} blob @returns {Promise<string>} */
function dataUrl(blob) {
	return new Promise((resolve, reject) => {
		const reader = new FileReader();
		reader.onload = () => resolve(/** @type {string} */ (reader.result));
		reader.onerror = () => reject(reader.error ?? new Error('could not read the captured frame'));
		reader.readAsDataURL(blob);
	});
}

/** Free the current run's engine and its canvas. */
function dispose() {
	const c = current;
	if (!c) return;
	current = null;
	try {
		c.engine.dispose(true);
	} finally {
		c.root.remove();
	}
}

/**
 * Build an engine for the run, run its warm-up and put it at frame 0.
 * @param {import('../conditions/index.js').PartialRun} input a RunSpec (the generator
 *   has resolved it already; it is resolved again so the page stands on its own)
 * @returns {Promise<LoadInfo>}
 */
async function load(input) {
	dispose();
	const started = performance.now();
	const run = resolveRun(input);
	const { width, height, fps, warmup, quality, seaSeed, seaTime } = run.render;
	const gpu = probeGpu();
	const trajectory = createTrajectory(run.trajectory, { sea: run.sea });

	// The stage. The engine sizes the canvas's buffer itself (exactly width × height at
	// pixel ratio 1); the CSS size only matters to someone watching a headed browser.
	const root = document.createElement('div');
	root.style.cssText = `position: relative; width: ${width}px; height: ${height}px; overflow: hidden;`;
	const canvas = document.createElement('canvas');
	const hud = document.createElement('canvas');
	canvas.style.cssText = 'position: absolute; inset: 0; width: 100%; height: 100%;';
	hud.style.cssText = 'display: none;';
	root.append(canvas, hud);
	document.body.append(root);

	// 'fallback' is the engine giving up on WebGL (a shader error, a failed render);
	// 'loading' for a lost context. Either way the frames from here on would be wrong.
	const health = { failure: /** @type {string | null} */ (null) };
	/** @type {OceanEngine} */
	let engine;
	try {
		engine = new OceanEngine(
			{
				canvas,
				hud,
				root,
				onStatus: (status, reason) => {
					if (status === 'fallback' || status === 'error' || reason === 'context lost') health.failure = reason ?? status;
				}
			},
			{
				manual: true,
				quality: /** @type {'high' | 'low'} */ (quality),
				seed: seaSeed,
				size: [width, height],
				preserveDrawingBuffer: false
			}
		);
	} catch (e) {
		root.remove();
		throw new Error(`could not start the ocean engine on '${gpu.renderer}': ${e instanceof Error ? e.message : e}`);
	}
	const frames = Math.floor(trajectory.duration * fps + EPS) + 1;
	current = { run, engine, root, trajectory, frames, index: 0, busy: false, health };

	try {
		engine.setScene(runScene(run, trajectory));
		engine.setSeaTime(seaTime);
		// Warm-up in frame-sized steps (the last one shorter if the warm-up is not a whole
		// number of frames), the flight held at t = 0 while the sea, its foam and the rigs run.
		const step = 1 / fps;
		let left = warmup;
		let warmupSteps = 0;
		while (left > EPS) {
			const dt = Math.min(step, left);
			engine.advance(dt, { flight: false });
			left -= dt;
			warmupSteps++;
		}
		// Frame 0: the flight at t = 0, the rigs restarted there, settled.
		engine.seek(0);
		if (health.failure) throw new Error(`ocean engine failed during warm-up: ${health.failure}`);
		return {
			frames,
			meta: {
				renderer: gpu.renderer,
				vendor: gpu.vendor,
				software: gpu.software,
				duration: trajectory.duration,
				warmupSteps,
				seaTimeAtFrame0: engine.simTime,
				loadSeconds: (performance.now() - started) / 1000
			}
		};
	} catch (e) {
		dispose();
		throw e;
	}
}

/**
 * Render frame i in every mode the run asks for, and return its truth, its trajectory
 * sample and its PNGs. Frames go in order: each is whole 1 / fps steps after the last
 * (one step for the next frame), never back.
 * @param {number} i
 * @returns {Promise<FrameResult>}
 */
async function frame(i) {
	const c = current;
	if (!c) throw new Error('frame() before load()');
	if (c.busy) throw new Error('frame() while another frame is still being captured');
	if (!Number.isInteger(i) || i < c.index || i >= c.frames) {
		throw new Error(`frame(${i}): frames go in order, from ${c.index} to ${c.frames - 1}`);
	}
	if (c.health.failure) throw new Error(`ocean engine failed: ${c.health.failure}`);
	c.busy = true;
	try {
		const { engine, run } = c;
		const step = 1 / run.render.fps;
		for (; c.index < i; c.index++) engine.advance(step);

		/** @type {FrameTruth | null} */
		let truth = null;
		/** @type {Array<[string, Promise<Blob>]>} */
		const captures = [];
		for (const mode of run.render.modes) {
			engine.renderFrame({ mode: /** @type {RenderMode} */ (mode) });
			// The canvas is not preserved once composited: capture it in this same task.
			// toBlob copies the bitmap when it is called, so the next mode can draw at once.
			captures.push([mode, engine.capture()]);
			// Every mode of a frame draws the same camera; the truth is the first one's.
			truth ??= engine.frameTruth();
		}
		if (c.health.failure) throw new Error(`ocean engine failed: ${c.health.failure}`);
		if (!truth) throw new Error('the run has no render modes');

		/** @type {Record<string, string>} */
		const images = {};
		for (const [mode, blob] of captures) images[mode] = await dataUrl(await blob);
		return { index: i, truth, sample: c.trajectory.stateAt(truth.t), images };
	} finally {
		c.busy = false;
	}
}

/** @type {any} */ (window).mare = { ready: true, load, frame, dispose };
