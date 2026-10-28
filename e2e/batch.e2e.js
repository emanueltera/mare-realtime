// @ts-check
/**
 * The batch page (batch.html → src/batch/page.js) in a real browser on the GPU: a tiny
 * run, rendered twice, checked for what the generator relies on — PNGs of the right
 * size with something in them, finite ground truth, a camera whose pose agrees with the
 * gimbal and the chase rig's framing, and the same pixels and truth from the same run.
 */
import { expect, test } from '@playwright/test';
import { resolveRun } from '../src/conditions/index.js';
import { channel } from '../src/ocean/channels.js';
import { projectTruth } from '../src/ocean/engine/truth.js';

/** @typedef {import('../src/batch/page.js').FrameResult} FrameResult */
/** @typedef {import('../src/batch/page.js').LoadInfo} LoadInfo */

const DEG = Math.PI / 180;

/** Three frames (t = 0, 0.2, 0.4 s) at 320×180, a shaded and a label mode. */
const RUN = resolveRun({
	id: 'e2e-tiny',
	sea: 'moderate',
	light: 'morning',
	trajectory: { duration: 0.4, altitude: 120, pattern: 'transit' },
	render: { width: 320, height: 180, fps: 5, warmup: 1, modes: ['shaded', 'height'] }
});

/**
 * Load a run on the page and render every frame.
 * @param {import('@playwright/test').Page} page
 * @param {import('../src/conditions/index.js').RunSpec} run
 * @returns {Promise<{ info: LoadInfo, frames: FrameResult[] }>}
 */
async function render(page, run) {
	return page.evaluate(async (r) => {
		const mare = /** @type {any} */ (window).mare;
		const info = await mare.load(r);
		const frames = [];
		for (let i = 0; i < info.frames; i++) frames.push(await mare.frame(i));
		mare.dispose();
		return { info, frames };
	}, run);
}

/**
 * Decode a PNG data URL in the browser: its size and how much its pixels vary.
 * @param {import('@playwright/test').Page} page
 * @param {string} url
 */
async function decode(page, url) {
	return page.evaluate(async (u) => {
		const bitmap = await createImageBitmap(await (await fetch(u)).blob());
		const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
		const ctx = /** @type {OffscreenCanvasRenderingContext2D} */ (canvas.getContext('2d'));
		ctx.drawImage(bitmap, 0, 0);
		const data = ctx.getImageData(0, 0, bitmap.width, bitmap.height).data;
		let min = 255;
		let max = 0;
		const colours = new Set();
		for (let i = 0; i < data.length; i += 4) {
			const luma = (data[i] * 3 + data[i + 1] * 6 + data[i + 2]) / 10;
			min = Math.min(min, luma);
			max = Math.max(max, luma);
			if (colours.size < 1000) colours.add((data[i] << 16) | (data[i + 1] << 8) | data[i + 2]);
		}
		return { width: bitmap.width, height: bitmap.height, min, max, colours: colours.size };
	}, url);
}

/**
 * Every pixel of a PNG data URL, decoded in the browser, as RGB triples.
 * @param {import('@playwright/test').Page} page
 * @param {string} url
 * @returns {Promise<{ width: number, height: number, rgb: number[] }>}
 */
async function pixels(page, url) {
	return page.evaluate(async (u) => {
		const bitmap = await createImageBitmap(await (await fetch(u)).blob());
		const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
		const ctx = /** @type {OffscreenCanvasRenderingContext2D} */ (canvas.getContext('2d'));
		ctx.drawImage(bitmap, 0, 0);
		const data = ctx.getImageData(0, 0, bitmap.width, bitmap.height).data;
		const rgb = [];
		for (let i = 0; i < data.length; i += 4) rgb.push(data[i], data[i + 1], data[i + 2]);
		return { width: bitmap.width, height: bitmap.height, rgb };
	}, url);
}

/**
 * Read a value back off a label channel's colour: the nearest colour on its ramp
 * (channels.js: evenly spaced stops, linear in sRGB between them).
 * @param {string} id
 */
function rampInverse(id) {
	const { range, stops } = channel(id);
	const rgb = stops.map((h) => [1, 3, 5].map((k) => parseInt(h.slice(k, k + 2), 16)));
	/** @type {Array<[number, number[]]>} */
	const table = [];
	for (let i = 0; i <= 4000; i++) {
		const x = (i / 4000) * (stops.length - 1);
		const k = Math.min(stops.length - 2, Math.floor(x));
		const f = x - k;
		table.push([range[0] + (i / 4000) * (range[1] - range[0]), rgb[k].map((c, j) => c + (rgb[k + 1][j] - c) * f)]);
	}
	/** @param {number[]} c */
	return (c) => {
		let best = table[0];
		let bestD = Infinity;
		for (const row of table) {
			const d = (row[1][0] - c[0]) ** 2 + (row[1][1] - c[1]) ** 2 + (row[1][2] - c[2]) ** 2;
			if (d < bestD) [best, bestD] = [row, d];
		}
		return best[0];
	};
}

/** Every number in a JSON value, with its path. @param {unknown} v @param {string} path @returns {Array<[string, number]>} */
function numbers(v, path = '') {
	if (typeof v === 'number') return [[path, v]];
	if (v && typeof v === 'object') return Object.entries(v).flatMap(([k, x]) => numbers(x, `${path}.${k}`));
	return [];
}

test.beforeEach(async ({ page }) => {
	const errors = /** @type {string[]} */ ([]);
	page.on('pageerror', (e) => errors.push(e.message));
	page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
	await page.goto('batch.html');
	await page.waitForFunction(() => /** @type {any} */ (window).mare?.ready === true);
	test.info().annotations.push({ type: 'renderer', description: await page.evaluate(() => {
		const gl = document.createElement('canvas').getContext('webgl2');
		const ext = gl?.getExtension('WEBGL_debug_renderer_info');
		return ext ? String(gl?.getParameter(ext.UNMASKED_RENDERER_WEBGL)) : 'unknown';
	}) });
	// Checked after each test: the page must not have logged errors.
	/** @type {any} */ (page).__errors = errors;
});

test.afterEach(async ({ page }) => {
	expect(/** @type {any} */ (page).__errors, 'page console errors').toEqual([]);
});

test('onboard: PNGs, truth, the optical axis, and the same run twice', async ({ page }) => {
	const a = await render(page, RUN);
	expect(a.info.frames).toBe(3);
	expect(a.info.meta.seaTimeAtFrame0).toBeCloseTo(RUN.render.seaTime + RUN.render.warmup, 9);
	expect(a.frames).toHaveLength(3);

	for (const [i, f] of a.frames.entries()) {
		// Images: one per mode, decodable, the run's size, not one flat colour.
		expect(Object.keys(f.images).sort()).toEqual(['height', 'shaded']);
		for (const mode of RUN.render.modes) {
			expect(f.images[mode].startsWith('data:image/png;base64,')).toBe(true);
			const img = await decode(page, f.images[mode]);
			expect(img.width, mode).toBe(320);
			expect(img.height, mode).toBe(180);
			expect(img.max - img.min, `${mode} frame ${i} luma range`).toBeGreaterThan(8);
			expect(img.colours, `${mode} frame ${i} colours`).toBeGreaterThan(20);
		}
		expect(f.images.height).not.toBe(f.images.shaded);

		// Truth: every number finite; the frame's time; K centred, square pixels.
		for (const [path, v] of numbers(f.truth)) expect(Number.isFinite(v), `truth${path} = ${v}`).toBe(true);
		expect(f.truth.t).toBeCloseTo(i / RUN.render.fps, 9);
		expect(f.truth.seaTime).toBeCloseTo(RUN.render.seaTime + RUN.render.warmup + i / RUN.render.fps, 9);
		const { camera } = f.truth;
		expect([camera.width, camera.height]).toEqual([320, 180]);
		expect(camera.K[0]).toBeCloseTo(camera.K[4], 6);
		expect(camera.K[2]).toBeCloseTo(160, 6);
		expect(camera.K[5]).toBeCloseTo(90, 6);
		expect(camera.fovY).toBeCloseTo(RUN.trajectory.gimbal?.fov ?? 48, 3);

		// The aircraft is where the trajectory says, and the sample is the one at truth.t.
		expect(f.sample.t).toBe(f.truth.t);
		const air = /** @type {NonNullable<typeof f.truth.aircraft>} */ (f.truth.aircraft);
		expect(air.x).toBe(f.sample.x);
		expect(air.y).toBe(f.sample.y);
		expect(air.altitude).toBe(f.sample.altitude);

		// A point 100 m along the optical axis (R_wc's third column) projects to the centre.
		const R = camera.rotation;
		const p = camera.position;
		const ahead = /** @type {[number, number, number]} */ ([p[0] + 100 * R[2], p[1] + 100 * R[5], p[2] + 100 * R[8]]);
		const [u, v, z] = projectTruth(camera, ahead);
		expect(u).toBeCloseTo(160, 6);
		expect(v).toBeCloseTo(90, 6);
		expect(z).toBeCloseTo(100, 6);

		// The camera rides on the aircraft (the lens is under a metre from its centre) and
		// sits above the sea by about the altitude.
		expect(Math.hypot(p[0] - air.x, p[1] - air.y, p[2] - air.altitude)).toBeLessThan(1);
		expect(f.truth.heightAboveSea).toBeGreaterThan(110);
		expect(f.truth.heightAboveSea).toBeLessThan(130);
	}

	// Frame 0 (wings level): the optical axis is along the nose and about the gimbal's
	// 35° down (a stabilised gimbal takes a share of the airframe's 2° nose-up pitch).
	const f0 = a.frames[0];
	const R = f0.truth.camera.rotation;
	const compass = (Math.atan2(R[2], R[5]) / DEG + 360) % 360;
	const down = -Math.asin(R[8]) / DEG;
	const nose = /** @type {NonNullable<typeof f0.truth.aircraft>} */ (f0.truth.aircraft).heading;
	expect(Math.abs(((compass - nose + 540) % 360) - 180)).toBeLessThan(1);
	expect(Math.abs(down - 35)).toBeLessThan(2);

	// The same run again: the same truth and byte-identical PNGs (same GPU and driver).
	const b = await render(page, RUN);
	for (let i = 0; i < a.frames.length; i++) {
		expect(JSON.stringify(b.frames[i].truth), `truth of frame ${i}`).toBe(JSON.stringify(a.frames[i].truth));
		for (const mode of RUN.render.modes) {
			expect(b.frames[i].images[mode] === a.frames[i].images[mode], `${mode} PNG of frame ${i} identical`).toBe(true);
		}
	}
});

test('chase: the aircraft is centred in the frame', async ({ page }) => {
	const run = resolveRun({ ...RUN, id: 'e2e-chase', render: { ...RUN.render, view: 'chase', modes: ['shaded'] } });
	const { frames } = await render(page, run);
	for (const f of frames) {
		const air = /** @type {NonNullable<typeof f.truth.aircraft>} */ (f.truth.aircraft);
		const [u, v, z] = projectTruth(f.truth.camera, [air.x, air.y, air.altitude]);
		// The rig frames it at the centre, settled from frame 0 (the trajectory's velocity
		// primes it); within a few pixels while the rig's springs follow the turbulence.
		expect(Math.hypot(u - 160, v - 90), `aircraft at (${u.toFixed(1)}, ${v.toFixed(1)})`).toBeLessThan(6);
		expect(z).toBeGreaterThan(15);
		expect(z).toBeLessThan(40);
		const img = await decode(page, f.images.shaded);
		expect(img.max - img.min).toBeGreaterThan(8);
	}
});

test('a bad run fails loudly', async ({ page }) => {
	const message = await page.evaluate(async () => {
		try {
			await /** @type {any} */ (window).mare.frame(0);
			return 'no error';
		} catch (e) {
			return String(/** @type {Error} */ (e).message);
		}
	});
	expect(message).toContain('before load');
	const bad = await page.evaluate(async () => {
		try {
			await /** @type {any} */ (window).mare.load({ sea: 'no-such-sea' });
			return 'no error';
		} catch (e) {
			return String(/** @type {Error} */ (e).message);
		}
	});
	expect(bad).toContain('sea');
});

test('the sea under the camera is the drawn surface, in every sea', async ({ page }) => {
	// Nadir, fully stabilised: the image centre is the water straight below the camera, and
	// the height channel there is the drawn surface's height. A CPU sum of the longest
	// waves missed most of the moderate sea's (RMS error 0.33 m against a 0.32 m sea).
	const height = rampInverse('height');
	for (const sea of ['moderate', 'gale']) {
		const run = resolveRun({
			id: `e2e-under-${sea}`,
			sea,
			trajectory: { duration: 9, altitude: 40, pattern: 'transit', turbulence: 0, gimbal: { pitch: -90, stabilized: 1 } },
			render: { width: 128, height: 72, fps: 1, warmup: 1, modes: ['height'] }
		});
		const { frames } = await render(page, run);
		/** @type {number[]} */
		const drawn = [];
		/** @type {number[]} */
		const truth = [];
		for (const f of frames) {
			const img = await pixels(page, f.images.height);
			// The four pixels round the centre.
			let sum = 0;
			for (const [x, y] of [[63, 35], [64, 35], [63, 36], [64, 36]]) {
				const i = (y * img.width + x) * 3;
				sum += height(img.rgb.slice(i, i + 3));
			}
			// The ramp clips at ±2 m: a gale's crests read 2 there, whatever their height.
			if (Math.abs(sum / 4) > 1.95) continue;
			drawn.push(sum / 4);
			truth.push(f.truth.seaHeightUnderCamera);
			expect(f.truth.heightAboveSea).toBeCloseTo(f.truth.camera.position[2] - f.truth.seaHeightUnderCamera, 9);
		}
		expect(drawn.length).toBeGreaterThan(5);
		const mean = drawn.reduce((a, b) => a + b, 0) / drawn.length;
		const spread = Math.sqrt(drawn.reduce((a, b) => a + (b - mean) ** 2, 0) / drawn.length);
		const rms = Math.sqrt(drawn.reduce((a, d, i) => a + (d - truth[i]) ** 2, 0) / drawn.length);
		expect(spread, `${sea}: the drawn sea moves under the camera`).toBeGreaterThan(0.1);
		expect(rms, `${sea}: truth against the drawn surface (sea std ${spread.toFixed(2)} m)`).toBeLessThan(0.05);
	}
});

test('the frame rate does not change the sea: foam at fps 5 and fps 10 agree', async ({ page }) => {
	// The foam's decay, drift and spread are first-order steps; at fps 5 the engine
	// sub-steps them (each 0.2 s as two of 0.1 s), so the foam at a sea time does not
	// depend on how the run is sliced into frames. Stepped whole, fps 5 differed from
	// fps 10 by 7–8 grey levels on average, against 0.2 between fps 10 and 20.
	/** @param {number} fps */
	const run = (fps) =>
		resolveRun({
			id: `e2e-foam-${fps}`,
			sea: 'gale',
			light: 'midday',
			trajectory: { duration: 2, altitude: 120, pattern: 'transit', turbulence: 0, seed: 100, gimbal: { pitch: -60, stabilized: 1 } },
			render: { width: 240, height: 135, fps, warmup: 4, modes: ['foam'] }
		});
	const at = async (/** @type {number} */ fps) => {
		const { frames } = await render(page, run(fps));
		// t = 0 and t = 2 s.
		return Promise.all([frames[0], frames[frames.length - 1]].map((f) => pixels(page, f.images.foam)));
	};
	const [five, ten] = [await at(5), await at(10)];
	for (let k = 0; k < 2; k++) {
		const a = five[k].rgb;
		const b = ten[k].rgb;
		let d = 0;
		let lit = 0;
		for (let i = 0; i < a.length; i++) {
			d += Math.abs(a[i] - b[i]);
			lit += a[i];
		}
		expect(lit / a.length, 'there is foam to compare').toBeGreaterThan(5);
		expect(d / a.length, `mean |fps 5 − fps 10| at t = ${2 * k} s`).toBeLessThan(0.5);
	}
});

test('a frame bigger than the browser will draw fails instead of being cropped', async ({ page }) => {
	// Chromium caps a drawing buffer near 2²⁵ pixels and hands back a smaller one without
	// saying so; the frame would be a crop that the truth's K does not describe.
	const run = resolveRun({ id: 'e2e-huge', trajectory: { duration: 0.1 }, render: { width: 8192, height: 8192, fps: 10, warmup: 0.1 } });
	const message = await page.evaluate(async (r) => {
		try {
			await /** @type {any} */ (window).mare.load(r);
			return 'no error';
		} catch (e) {
			return String(/** @type {Error} */ (e).message);
		}
	}, run);
	expect(message).toContain('drawing buffer');
});
