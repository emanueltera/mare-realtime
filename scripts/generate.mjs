#!/usr/bin/env node
/**
 * The batch generator: a job (or a single run) in, a dataset on disk out
 * (docs/ARCHITECTURE.md, "Runs and jobs").
 *
 *   node scripts/generate.mjs <job.json | run.json> [--out out] [--url http://localhost:PORT]
 *     [--port 5174] [--only <id-substring>] [--limit N] [--no-frames] [--video]
 *     [--software] [--concurrency 1] [--force]
 *
 * The job is expanded here, in node, by the same pure code the app uses (src/conditions),
 * and so are each run's trajectory and IMU (src/trajectory). Only the frames need a
 * browser: batch.html, driven headless on the real GPU through Playwright, renders each
 * frame with the manual engine and hands back its PNGs and its FrameTruth. Without
 * `--url` the generator runs vite itself on `--port`, without hot reload (scripts/serve.mjs),
 * and stops it however it exits. A server given by `--url` should not hot-reload either:
 * a page reloaded mid-run fails that run (it is retried once on a fresh page).
 *
 * Per run, under <out>/<job>/<run-id>/: run.json, trajectory.csv, imu.csv, frames.csv,
 * frames/000000.png (frames/<mode>/000000.png for label modes), video.mp4 with --video,
 * and meta.json, written last: a run directory whose meta.json says "complete" for the
 * same run.json is skipped next time (unless --force), so an interrupted job resumes.
 * With --no-frames, or a run's `render.frames: false`, no browser is used at all: run.json,
 * trajectory.csv, imu.csv and meta.json only (frames.csv is the rendered camera's truth,
 * so it needs the render).
 *
 * <out>/<job>/index.csv lists every run of the job, done or not. The exit code is 1 if
 * any run failed (they are listed at the end and in index.csv), 2 for bad arguments,
 * 130 when stopped by SIGINT or SIGTERM (the runs cut short stay pending and are redone).
 */
import { spawn, spawnSync } from 'node:child_process';
import { readFileSync, rmSync } from 'node:fs';
import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { basename, extname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { chromium } from '@playwright/test';
import { expandJob, resolveRun, runSummary } from '../src/conditions/index.js';
import { deepEqual, slugify } from '../src/conditions/run.js';
import { seaStats } from '../src/ocean/spectrum.js';
import { createTrajectory, enuToGeodetic, formatNumber, imuToCsv, sampleColumns, samplesToCsv, windAloft } from '../src/trajectory/index.js';
import { launchOptions } from './browser.mjs';
import { serve } from './serve.mjs';

/** @typedef {import('../src/conditions/index.js').RunSpec} RunSpec */
/** @typedef {import('../src/conditions/index.js').Job} Job */
/** @typedef {import('../src/ocean/engine/engine.js').FrameTruth} FrameTruth */
/** @typedef {import('../src/trajectory/index.js').TrajectorySample} TrajectorySample */
/** @typedef {import('../src/batch/page.js').LoadInfo} LoadInfo */
/** @typedef {import('../src/batch/page.js').FrameResult} FrameResult */

const ROOT = fileURLToPath(new URL('..', import.meta.url));
const PKG = JSON.parse(readFileSync(join(ROOT, 'package.json'), 'utf8'));

/** Seconds to wait for the page to build an engine and warm up, and for one frame. */
const LOAD_TIMEOUT = 180;
const FRAME_TIMEOUT = 60;
/** Steps shorter than this are float dust (as in src/batch/page.js). */
const EPS = 1e-9;
/** Errors that mean the page went away under a run rather than the run being bad. */
const LOST_PAGE = /reloaded during the run|Execution context was destroyed|page crashed|Target page, context or browser has been closed/;
/** Set on SIGINT / SIGTERM: the run in progress is abandoned, not failed. */
let stopping = false;
/** ffmpeg processes running now, and the file each writes, to stop and remove on a signal. */
const encoders = new Map();

const USAGE = `usage: node scripts/generate.mjs <job.json | run.json> [options]

  --out <dir>          output root (default: out); runs go in <out>/<job>/<run-id>/
  --url <url>          use this running dev server instead of starting vite
  --port <n>           port for the vite it starts (default 5174, --strictPort)
  --only <text>        only runs whose id contains this
  --limit <n>          at most this many runs (after --only)
  --no-frames          trajectory, IMU, run and meta only: no browser
  --video              also encode frames/ to video.mp4 with ffmpeg
  --software           render on SwiftShader instead of the GPU (slow; for machines without one)
  --concurrency <n>    runs rendered at once, each in its own page (default 1)
  --force              redo runs that are already complete on disk`;

// ------------------------------------------------------------------ arguments

/**
 * @typedef {object} Options
 * @property {string} input
 * @property {string} out
 * @property {string | undefined} url
 * @property {number} port
 * @property {string | undefined} only
 * @property {number} limit
 * @property {boolean} frames
 * @property {boolean} video
 * @property {boolean} software
 * @property {number} concurrency
 * @property {boolean} force
 */

const OPTIONS = /** @type {const} */ ({
	out: { type: 'string', default: 'out' },
	url: { type: 'string' },
	port: { type: 'string', default: '5174' },
	only: { type: 'string' },
	limit: { type: 'string' },
	'no-frames': { type: 'boolean', default: false },
	video: { type: 'boolean', default: false },
	software: { type: 'boolean', default: false },
	concurrency: { type: 'string', default: '1' },
	force: { type: 'boolean', default: false },
	help: { type: 'boolean', short: 'h', default: false }
});

/** @returns {Options} */
function parseOptions() {
	const parse = () => parseArgs({ allowPositionals: true, options: OPTIONS });
	/** @type {ReturnType<typeof parse>} */
	let parsed;
	try {
		parsed = parse();
	} catch (e) {
		// An unknown or malformed flag is a bad argument like any other.
		usage(e instanceof Error ? e.message : String(e));
	}
	const { values, positionals } = parsed;
	if (values.help) {
		console.log(USAGE);
		process.exit(0);
	}
	if (positionals.length !== 1) usage('give exactly one job or run file');
	/** @param {string | undefined} v @param {string} name @param {number} min */
	const int = (v, name, min) => {
		const n = Number(v);
		if (!Number.isInteger(n) || n < min) usage(`--${name} must be an integer >= ${min} (got ${v})`);
		return n;
	};
	return {
		input: resolve(positionals[0]),
		out: resolve(values.out),
		url: values.url,
		port: int(values.port, 'port', 1),
		only: values.only,
		limit: values.limit === undefined ? Infinity : int(values.limit, 'limit', 1),
		frames: !values['no-frames'],
		video: values.video,
		software: values.software,
		concurrency: int(values.concurrency, 'concurrency', 1),
		force: values.force
	};
}

/** @param {string} message @returns {never} */
function usage(message) {
	console.error(`generate: ${message}\n\n${USAGE}`);
	process.exit(2);
}

/**
 * A job file, or a single RunSpec (which is a job of one run, named after its file).
 * @param {string} file
 * @returns {{ name: string, runs: RunSpec[] }}
 */
function readInput(file) {
	/** @type {any} */
	let json;
	try {
		json = JSON.parse(readFileSync(file, 'utf8'));
	} catch (e) {
		usage(`cannot read ${file}: ${e instanceof Error ? e.message : e}`);
	}
	// A RunSpec has none of a job's keys (name, base, sweep, random).
	const isJob = json && typeof json === 'object' && ['name', 'base', 'sweep', 'random'].some((k) => k in json);
	try {
		if (isJob) return { name: slugify(String(json.name ?? basename(file, extname(file)))), runs: expandJob(json) };
		return { name: slugify(basename(file, extname(file))), runs: [resolveRun(json)] };
	} catch (e) {
		usage(e instanceof Error ? e.message : String(e));
	}
}

// ------------------------------------------------------------------ what a run writes

/** Frames in a run: one at every 1 / fps from t = 0 up to the duration (as the page counts them). */
const frameCount = (/** @type {number} */ duration, /** @type {number} */ fps) => Math.floor(duration * fps + EPS) + 1;

/** frames.csv: FrameTruth flattened (camera pose t_wc and R_wc in OpenCV axes → ENU, K, sea, aircraft). */
const FRAME_COLUMNS = [
	'frame', 't', 'seaTime',
	'cam_e', 'cam_n', 'cam_u',
	'R00', 'R01', 'R02', 'R10', 'R11', 'R12', 'R20', 'R21', 'R22',
	'qx', 'qy', 'qz', 'qw',
	'fx', 'fy', 'cx', 'cy', 'width', 'height', 'fovY',
	'heightAboveSea', 'seaHeightUnderCamera',
	'aircraft_x', 'aircraft_y', 'aircraft_altitude', 'aircraft_heading', 'aircraft_pitch', 'aircraft_bank'
];
/** Columns written to 9 decimals: unit vectors, where 1e-6 is a few arc-seconds. */
const FINE = new Set(['R00', 'R01', 'R02', 'R10', 'R11', 'R12', 'R20', 'R21', 'R22', 'qx', 'qy', 'qz', 'qw', 'lat', 'lon', 'cam_lat', 'cam_lon', 'aircraft_lat', 'aircraft_lon']);

/**
 * One frames.csv row.
 * @param {number} frame
 * @param {FrameTruth} truth
 * @returns {Record<string, number>}
 */
function frameRow(frame, truth) {
	const { camera: c, aircraft: a } = truth;
	const [fx, , cx, , fy, cy] = c.K;
	/** @type {Record<string, number>} */
	const row = {
		frame,
		t: truth.t,
		seaTime: truth.seaTime,
		cam_e: c.position[0],
		cam_n: c.position[1],
		cam_u: c.position[2],
		qx: c.quaternion[0],
		qy: c.quaternion[1],
		qz: c.quaternion[2],
		qw: c.quaternion[3],
		fx,
		fy,
		cx,
		cy,
		width: c.width,
		height: c.height,
		fovY: c.fovY,
		heightAboveSea: truth.heightAboveSea,
		seaHeightUnderCamera: truth.seaHeightUnderCamera,
		aircraft_x: a ? a.x : NaN,
		aircraft_y: a ? a.y : NaN,
		aircraft_altitude: a ? a.altitude : NaN,
		aircraft_heading: a ? a.heading : NaN,
		aircraft_pitch: a ? a.pitch : NaN,
		aircraft_bank: a ? a.bank : NaN
	};
	for (let r = 0; r < 3; r++) for (let k = 0; k < 3; k++) row[`R${r}${k}`] = c.rotation[r * 3 + k];
	return row;
}

/**
 * CSV of plain decimals, by column (the trajectory module's number format).
 * @param {Array<Record<string, number>>} rows
 * @param {string[]} columns
 */
function rowsToCsv(rows, columns) {
	const lines = [columns.join(',')];
	for (const row of rows) lines.push(columns.map((c) => formatNumber(row[c], FINE.has(c) ? 9 : 6)).join(','));
	return lines.join('\n') + '\n';
}

/** @param {string | number | boolean | null | undefined} v */
function csvField(v) {
	const s = v === null || v === undefined ? '' : typeof v === 'number' ? formatNumber(v, 4) : String(v);
	return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

/**
 * The wind the trajectory flew in at its cruise altitude, as the trajectory module
 * derives it (compass toward, like everything else).
 * @param {RunSpec} run
 */
function windAt(run) {
	const { wind, altitude } = run.trajectory;
	if (wind === 'none') return { altitude, speed: 0, direction: 0, east: 0, north: 0 };
	if (wind === 'sea' || wind === undefined) {
		const w = windAloft(run.sea, altitude);
		return { altitude, speed: w.speed, direction: w.direction, east: w.east, north: w.north };
	}
	const rad = (wind.direction * Math.PI) / 180;
	return { altitude, speed: wind.speed, direction: wind.direction, east: wind.speed * Math.sin(rad), north: wind.speed * Math.cos(rad) };
}

/** Write a file whole or not at all (a crash mid-write leaves the old one, or none). */
async function writeAtomic(/** @type {string} */ file, /** @type {string | Buffer} */ data) {
	const tmp = `${file}.tmp`;
	await writeFile(tmp, data);
	await rename(tmp, file);
}

/** @param {string} url a data: URL */
const decodeDataUrl = (url) => Buffer.from(url.slice(url.indexOf(',') + 1), 'base64');

// ------------------------------------------------------------------ provenance

/** The repository's commit, and whether the working tree differs from it. */
function gitState() {
	const run = (/** @type {string[]} */ args) => spawnSync('git', ['-C', ROOT, ...args], { encoding: 'utf8' });
	const head = run(['rev-parse', 'HEAD']);
	if (head.status !== 0) return null;
	const status = run(['status', '--porcelain', '--untracked-files=no']);
	return { commit: head.stdout.trim(), dirty: status.status === 0 ? status.stdout.trim().length > 0 : null };
}

/** @param {string} name a package in node_modules */
function versionOf(name) {
	try {
		return JSON.parse(readFileSync(join(ROOT, 'node_modules', name, 'package.json'), 'utf8')).version;
	} catch {
		return null;
	}
}

// ------------------------------------------------------------------ the browser side

/**
 * @template T
 * @param {Promise<T>} promise
 * @param {number} seconds
 * @param {string} what
 * @returns {Promise<T>}
 */
function within(promise, seconds, what) {
	/** @type {ReturnType<typeof setTimeout> | undefined} */
	let timer;
	const timeout = new Promise((_, reject) => {
		timer = setTimeout(() => reject(new Error(`${what} took longer than ${seconds} s`)), seconds * 1000);
	});
	return /** @type {Promise<T>} */ (Promise.race([promise, timeout]).finally(() => clearTimeout(timer)));
}

/**
 * A page with batch.html loaded and `window.mare` ready. Vite may reload a page once
 * while it optimises dependencies on its first request, so loading is retried.
 * @param {import('@playwright/test').Browser} browser
 * @param {string} baseUrl
 */
async function openPage(browser, baseUrl) {
	const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
	/** @type {string[]} */
	const errors = [];
	page.on('console', (m) => {
		if (m.type() === 'error') errors.push(m.text());
	});
	page.on('pageerror', (e) => errors.push(`page error: ${e.message}`));
	page.on('crash', () => errors.push('the page crashed'));
	let lastError = null;
	for (let attempt = 0; attempt < 3; attempt++) {
		try {
			await page.goto(new URL('batch.html', baseUrl).href, { waitUntil: 'load' });
			await page.waitForFunction(() => /** @type {any} */ (window).mare?.ready === true, null, { timeout: 60000 });
			// From here on the page must stay put: a reload (a dev server's hot reload) loses the run.
			page.on('framenavigated', (frame) => {
				if (frame === page.mainFrame()) errors.push('the page reloaded during the run (a dev server with hot reload?)');
			});
			return { page, errors };
		} catch (e) {
			lastError = e;
		}
	}
	await page.close().catch(() => {});
	throw new Error(`batch.html did not load at ${baseUrl}: ${lastError instanceof Error ? lastError.message : lastError}`);
}

// ------------------------------------------------------------------ one run

/**
 * @typedef {object} RunResult
 * @property {RunSpec} run
 * @property {'complete' | 'failed' | 'pending'} status
 * @property {number} frames
 * @property {number} seconds
 * @property {string} [error]
 * @property {boolean} [skipped] already complete on disk
 */

/**
 * Whether a run directory already holds this run, complete (with frames if they are wanted).
 * @param {string} dir
 * @param {RunSpec} run
 * @param {boolean} wantFrames
 */
async function completeOnDisk(dir, run, wantFrames) {
	try {
		const meta = JSON.parse(await readFile(join(dir, 'meta.json'), 'utf8'));
		const onDisk = JSON.parse(await readFile(join(dir, 'run.json'), 'utf8'));
		if (meta.status !== 'complete' || !deepEqual(onDisk, run)) return null;
		if (wantFrames && !meta.frames?.rendered) return null;
		return meta;
	} catch {
		return null;
	}
}

/**
 * Encode the run's frames (the shaded ones, or the first mode's) as an H.264 MP4. It is
 * written beside video.mp4 and moved over it only once ffmpeg has finished well, so an
 * encode cut short (Ctrl-C reaches ffmpeg too) never leaves a truncated video.mp4.
 * @param {string} dir
 * @param {RunSpec} run
 * @returns {Promise<void>}
 */
async function encodeVideo(dir, run) {
	const mode = run.render.modes.includes('shaded') ? 'shaded' : run.render.modes[0];
	const frames = mode === 'shaded' ? join(dir, 'frames') : join(dir, 'frames', mode);
	const out = join(dir, 'video.mp4');
	const tmp = `${out}.tmp`;
	if (stopping) throw new Error('stopped');
	const child = spawn(
		'ffmpeg',
		[
			'-y', '-loglevel', 'error',
			'-framerate', String(run.render.fps),
			'-i', join(frames, '%06d.png'),
			// yuv420p needs even dimensions.
			'-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
			'-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
			'-f', 'mp4', tmp
		],
		{ stdio: ['ignore', 'ignore', 'pipe'] }
	);
	encoders.set(child, tmp);
	let stderr = '';
	child.stderr.setEncoding('utf8');
	child.stderr.on('data', (d) => (stderr += d));
	try {
		const code = await new Promise((resolve, reject) => {
			child.on('error', reject);
			child.on('close', (c, signal) => resolve(signal ? signal : c));
		});
		if (code !== 0) throw new Error(`ffmpeg failed (${code}): ${stderr.trim()}`);
		if (stopping) throw new Error('stopped');
		await rename(tmp, out);
	} catch (e) {
		await rm(tmp, { force: true });
		throw e;
	} finally {
		encoders.delete(child);
	}
}

/**
 * @typedef {object} Context
 * @property {Options} options
 * @property {string} jobName
 * @property {string} jobDir
 * @property {{ commit: string, dirty: boolean | null } | null} git
 * @property {string | null} browserVersion
 * @property {boolean} ffmpeg
 * @property {(line: string) => void} log
 */

/**
 * Render (or just fly) one run and write its directory.
 * @param {RunSpec} run
 * @param {Context} ctx
 * @param {(() => Promise<{ page: import('@playwright/test').Page, errors: string[] }>) | null} getPage
 * @param {string} label e.g. "run 2/14"
 * @returns {Promise<RunResult>}
 */
async function generateRun(run, ctx, getPage, label) {
	const started = performance.now();
	const dir = join(ctx.jobDir, run.id);
	const wantFrames = ctx.options.frames && run.render.frames !== false;
	const { fps } = run.render;
	// A run directory is all one run's: start it clean, so no file of an older attempt survives.
	await rm(dir, { recursive: true, force: true });
	await mkdir(dir, { recursive: true });
	await writeFile(join(dir, 'run.json'), JSON.stringify(run, null, '\t') + '\n');

	const trajectory = createTrajectory(/** @type {any} */ (run.trajectory), { sea: run.sea });
	const imu = trajectory.sample(run.imuRate);
	await writeFile(join(dir, 'imu.csv'), imuToCsv(imu));
	const frames = frameCount(trajectory.duration, fps);
	const toGeo = run.origin ? enuToGeodetic(run.origin) : null;

	/** @type {Array<TrajectorySample & { frame: number }>} */
	const samples = [];
	/** @type {Array<Record<string, number>>} */
	const truthRows = [];
	/** @type {(LoadInfo['meta'] & { console: string[] }) | null} */
	let render = null;
	let renderSeconds = 0;
	/** @type {Error | null} */
	let failure = null;

	if (wantFrames) {
		/** @type {string[]} */
		let errors = [];
		try {
			if (!getPage) throw new Error('internal: frames wanted without a browser');
			const held = await getPage();
			const page = held.page;
			errors = held.errors;
			errors.length = 0;
			const modes = run.render.modes;
			for (const mode of modes) await mkdir(mode === 'shaded' ? join(dir, 'frames') : join(dir, 'frames', mode), { recursive: true });
			const info = /** @type {LoadInfo} */ (
				await within(page.evaluate((r) => /** @type {any} */ (window).mare.load(r), run), LOAD_TIMEOUT, 'loading the run')
			);
			if (info.frames !== frames) throw new Error(`the page counts ${info.frames} frames, node ${frames}`);
			render = { ...info.meta, console: errors };
			if (info.meta.software && !ctx.options.software) {
				ctx.log(`warning: WebGL is running on a software rasteriser ('${info.meta.renderer}'); frames will be slow and differ from GPU ones`);
			}
			const t0 = performance.now();
			let lastReport = t0;
			/** @type {Promise<void>[]} */
			const writes = [];
			for (let i = 0; i < frames; i++) {
				const res = /** @type {FrameResult} */ (
					await within(page.evaluate((k) => /** @type {any} */ (window).mare.frame(k), i), FRAME_TIMEOUT, `frame ${i}`)
				);
				const name = `${String(i).padStart(6, '0')}.png`;
				for (const mode of modes) {
					const file = mode === 'shaded' ? join(dir, 'frames', name) : join(dir, 'frames', mode, name);
					const write = writeFile(file, decodeDataUrl(res.images[mode]));
					// Awaited below; if a later frame fails first, the run has failed anyway.
					write.catch(() => {});
					writes.push(write);
				}
				// The flight at the t this frame drew, from node's own trajectory (the page's
				// agrees to ~1e-12, its Math library not being node's): every CSV of a run
				// then comes from one JS engine.
				samples.push({ ...trajectory.stateAt(res.truth.t), frame: i });
				truthRows.push(frameRow(i, res.truth));
				const now = performance.now();
				if (now - lastReport > 2000 || i === frames - 1) {
					lastReport = now;
					const rate = (i + 1) / ((now - t0) / 1000);
					ctx.log(`${label} ${run.id}  frame ${i + 1}/${frames}  ${rate.toFixed(1)} frames/s`);
				}
			}
			await Promise.all(writes);
			renderSeconds = (performance.now() - t0) / 1000;
			await page.evaluate(() => /** @type {any} */ (window).mare.dispose()).catch(() => {});
		} catch (e) {
			failure = e instanceof Error ? e : new Error(String(e));
			if (errors.length) failure.message += `\n  page console:\n    ${errors.slice(-8).join('\n    ')}`;
		}
	} else {
		// No render: the trajectory at the frame times all the same, for runs that only
		// want the flight (frames.csv, the camera's truth, needs the render).
		for (let i = 0; i < frames; i++) samples.push({ ...trajectory.stateAt(i / fps), frame: i });
	}

	if (!failure) {
		// trajectory.csv: the TrajectorySample of every frame (the t each frame drew).
		const columns = ['frame', ...sampleColumns];
		if (toGeo) {
			for (const s of samples) [/** @type {any} */ (s).lat, /** @type {any} */ (s).lon] = toGeo(s.x, s.y);
			columns.push('lat', 'lon');
			// samplesToCsv writes every column at one precision; latitude needs more.
			await writeFile(join(dir, 'trajectory.csv'), rowsToCsv(/** @type {any} */ (samples), columns));
		} else {
			await writeFile(join(dir, 'trajectory.csv'), samplesToCsv(samples, columns));
		}
		if (wantFrames) {
			const columns = [...FRAME_COLUMNS];
			if (toGeo) {
				for (const row of truthRows) {
					[row.cam_lat, row.cam_lon] = toGeo(row.cam_e, row.cam_n);
					[row.aircraft_lat, row.aircraft_lon] = toGeo(row.aircraft_x, row.aircraft_y);
				}
				columns.push('cam_lat', 'cam_lon', 'aircraft_lat', 'aircraft_lon');
			}
			await writeFile(join(dir, 'frames.csv'), rowsToCsv(truthRows, columns));
		}
	}
	// Cut short by a signal: not a failure. No meta.json, so the run reads as pending and
	// is redone next time.
	if (failure && stopping) return { run, status: 'pending', frames, seconds: (performance.now() - started) / 1000 };
	// The video is a convenience: if ffmpeg fails, the run (its frames) is still complete.
	// Stopped during the encode, the run is complete without one (a later --video makes it).
	/** @type {{ file: string } | { error: string } | null} */
	let video = null;
	if (!failure && wantFrames && ctx.options.video && ctx.ffmpeg) {
		try {
			await encodeVideo(dir, run);
			video = { file: 'video.mp4' };
		} catch (e) {
			if (!stopping) {
				video = { error: e instanceof Error ? e.message : String(e) };
				ctx.log(`${label} ${run.id}  warning: ${video.error}`);
			}
		}
	}
	for (const w of trajectory.warnings) ctx.log(`${label} ${run.id}  warning: ${w}`);

	const seconds = (performance.now() - started) / 1000;
	const stats = seaStats(run.sea);
	const meta = {
		id: run.id,
		job: ctx.jobName,
		summary: runSummary(run),
		status: failure ? 'failed' : 'complete',
		...(failure ? { error: failure.message } : {}),
		frames: {
			count: frames,
			rendered: wantFrames && !failure,
			fps,
			width: run.render.width,
			height: run.render.height,
			modes: run.render.modes,
			view: run.render.view,
			files: wantFrames ? Object.fromEntries(run.render.modes.map((m) => [m, m === 'shaded' ? 'frames/%06d.png' : `frames/${m}/%06d.png`])) : {}
		},
		...(video ? { video } : {}),
		warnings: trajectory.warnings,
		duration: trajectory.duration,
		imu: { rate: run.imuRate, samples: imu.length },
		// The simulator's sea, from the same spectrum the engine draws (spectrum.js).
		seaStats: {
			significantHeight: stats.significantHeight,
			windSeaHeight: stats.windSeaHeight,
			peakPeriod: stats.peakPeriod,
			peakWavelength: stats.peakWavelength,
			peakBand: stats.peakBand,
			beaufort: stats.beaufort
		},
		windAloft: windAt(run),
		quality: run.render.quality,
		seeds: { trajectory: run.trajectory.seed, sea: run.render.seaSeed },
		seaTime: {
			start: run.render.seaTime,
			warmup: run.render.warmup,
			atFrame0: render ? render.seaTimeAtFrame0 : run.render.seaTime + run.render.warmup
		},
		renderer: render
			? { gpu: render.renderer, vendor: render.vendor, software: render.software, browser: ctx.browserVersion, console: render.console }
			: null,
		versions: {
			package: PKG.version,
			git: ctx.git,
			three: versionOf('three'),
			playwright: versionOf('@playwright/test'),
			node: process.version
		},
		timings: {
			seconds,
			loadSeconds: render ? render.loadSeconds : 0,
			renderSeconds,
			framesPerSecond: renderSeconds > 0 ? frames / renderSeconds : null
		},
		generatedAt: new Date().toISOString()
	};
	// Last: a meta.json saying "complete" is what marks the directory done.
	await writeAtomic(join(dir, 'meta.json'), JSON.stringify(meta, null, '\t') + '\n');
	if (failure) return { run, status: 'failed', frames, seconds, error: failure.message };
	return { run, status: 'complete', frames, seconds };
}

// ------------------------------------------------------------------ the job

/**
 * index.csv: every run of the job, done in this invocation, done before, or not yet.
 * @param {string} jobDir
 * @param {RunSpec[]} runs every run of the job
 * @param {Map<string, RunResult>} results this invocation's
 */
async function writeIndex(jobDir, runs, results) {
	const columns = ['id', 'status', 'summary', 'sea_hs', 'sea_tp', 'beaufort', 'wind_speed', 'sun_elevation', 'pattern', 'altitude', 'frames', 'rendered', 'seconds', 'path', 'error'];
	const lines = [columns.join(',')];
	for (const run of runs) {
		const dir = join(jobDir, run.id);
		/** @type {any} */
		let meta = null;
		try {
			meta = JSON.parse(await readFile(join(dir, 'meta.json'), 'utf8'));
		} catch {
			// not generated yet
		}
		const result = results.get(run.id);
		const status = result?.status ?? (meta?.status === 'complete' || meta?.status === 'failed' ? meta.status : 'pending');
		const stats = seaStats(run.sea);
		const row = [
			run.id,
			status,
			runSummary(run),
			stats.significantHeight,
			stats.peakPeriod,
			stats.beaufort,
			run.sea.windSpeed,
			run.light.sunElevation,
			run.trajectory.pattern.type,
			run.trajectory.altitude,
			meta?.frames?.count ?? frameCount(run.trajectory.duration, run.render.fps),
			meta?.frames?.rendered ?? false,
			meta?.timings?.seconds ?? '',
			relative(jobDir, dir),
			// The first line; meta.json has the whole message.
			(result?.error ?? meta?.error ?? '').split('\n')[0]
		];
		lines.push(row.map(csvField).join(','));
	}
	await writeAtomic(join(jobDir, 'index.csv'), lines.join('\n') + '\n');
}

async function main() {
	const options = parseOptions();
	const { name: jobName, runs } = readInput(options.input);
	let selected = options.only ? runs.filter((r) => r.id.includes(/** @type {string} */ (options.only))) : runs;
	selected = selected.slice(0, options.limit);
	const jobDir = join(options.out, jobName);
	await mkdir(jobDir, { recursive: true });
	const tty = process.stderr.isTTY && options.concurrency === 1;
	/** @param {string} line */
	const log = (line) => {
		if (tty && / frame \d+\/\d+ /.test(line)) process.stderr.write(`\r\x1b[2K${line}`);
		else process.stderr.write(`${tty ? '\r\x1b[2K' : ''}${line}\n`);
	};
	log(`generate: ${jobName}: ${selected.length} of ${runs.length} runs → ${relative(process.cwd(), jobDir) || '.'}`);

	const ffmpeg = options.video ? spawnSync('ffmpeg', ['-version']).status === 0 : false;
	if (options.video && !ffmpeg) log('generate: warning: --video given but ffmpeg is not on the PATH; no video.mp4 will be written');

	/** @type {{ url: string, close: () => Promise<void> } | null} */
	let server = null;
	/** @type {import('@playwright/test').Browser | null} */
	let browser = null;
	const cleanup = async () => {
		await browser?.close().catch(() => {});
		await server?.close().catch(() => {});
	};
	// Installed before anything slow (a resumed run's video, vite, the browser). A signal
	// stops ffmpeg and closes the browser, which ends the run in progress; the runs cut
	// short are not failures (no meta.json: pending, redone next time). main then writes
	// index.csv and exits 130. A second signal, or a shutdown that hangs, exits at once.
	for (const signal of /** @type {const} */ (['SIGINT', 'SIGTERM'])) {
		process.on(signal, () => {
			if (stopping) process.exit(130);
			stopping = true;
			log(`generate: ${signal}, stopping (runs not finished are redone next time)`);
			for (const [child, file] of encoders) {
				child.kill('SIGTERM');
				rmSync(file, { force: true });
			}
			setTimeout(() => process.exit(130), 15000).unref();
			browser?.close().catch(() => {});
		});
	}

	/** @type {Map<string, RunResult>} */
	const results = new Map();
	/** @type {RunSpec[]} */
	const todo = [];
	for (const run of selected) {
		if (stopping) break;
		const dir = join(jobDir, run.id);
		const wantFrames = options.frames && run.render.frames !== false;
		const done = options.force ? null : await completeOnDisk(dir, run, wantFrames);
		if (!done) {
			todo.push(run);
			continue;
		}
		/** @type {RunResult} */
		const result = { run, status: 'complete', frames: done.frames?.count ?? 0, seconds: 0, skipped: true };
		// Already there; a video asked for this time and not made last time (or made badly)
		// is made from its frames, and meta.json says so.
		if (options.video && ffmpeg && done.frames?.rendered && (!done.video || 'error' in done.video)) {
			try {
				await encodeVideo(dir, run);
				done.video = { file: 'video.mp4' };
			} catch (e) {
				if (!stopping) {
					done.video = { error: e instanceof Error ? e.message : String(e) };
					log(`${run.id}  warning: ${done.video.error}`);
				}
			}
			if (!stopping) await writeAtomic(join(dir, 'meta.json'), JSON.stringify(done, null, '\t') + '\n');
		}
		results.set(run.id, result);
	}
	const skipped = [...results.values()].filter((r) => r.skipped).length;
	if (skipped) log(`generate: ${skipped} runs already complete on disk (--force to redo)`);

	const needBrowser = !stopping && todo.some((r) => options.frames && r.render.frames !== false);

	try {
		let baseUrl = options.url ?? '';
		if (needBrowser) {
			if (!options.url) {
				// vite in this process, without hot reload (see serve.mjs).
				server = await serve(options.port);
				baseUrl = server.url;
				log(`generate: vite on ${baseUrl}`);
			}
			// The signals are ours (above): Playwright's own handlers would close the browser
			// and exit before index.csv is written.
			if (!stopping) browser = await chromium.launch({ ...launchOptions({ software: options.software }), handleSIGINT: false, handleSIGTERM: false });
			if (stopping) await browser?.close().catch(() => {});
		}
		/** @type {Context} */
		const ctx = {
			options,
			jobName,
			jobDir,
			git: gitState(),
			browserVersion: browser ? browser.version() : null,
			ffmpeg,
			log
		};

		// A pool of workers, each with its own page (and so its own WebGL context).
		let next = 0;
		const worker = async () => {
			const slot = { held: /** @type {{ page: import('@playwright/test').Page, errors: string[] } | null} */ (null) };
			const getPage = async () => {
				if (slot.held && !slot.held.page.isClosed()) return slot.held;
				slot.held = await openPage(/** @type {import('@playwright/test').Browser} */ (browser), baseUrl);
				return slot.held;
			};
			while (next < todo.length) {
				const k = next++;
				const run = todo[k];
				const label = `[${k + 1}/${todo.length}]`;
				/** @type {RunResult | null} */
				let result = null;
				// A page lost under the run (reloaded, crashed) is retried once on a fresh one.
				for (let attempt = 0; attempt < 2 && !stopping; attempt++) {
					try {
						result = await generateRun(run, ctx, browser ? getPage : null, label);
					} catch (e) {
						result = { run, status: 'failed', frames: 0, seconds: 0, error: e instanceof Error ? e.message : String(e) };
					}
					if (stopping || result.status === 'complete' || !LOST_PAGE.test(result.error ?? '')) break;
					log(`${label} ${run.id}  the page was lost (${(result.error ?? '').split('\n')[0]}); retrying on a fresh page`);
					await slot.held?.page.close().catch(() => {});
					slot.held = null;
				}
				// Cut short by a signal (pending), or never started: nothing to record.
				if (!result || result.status === 'pending' || (stopping && result.status === 'failed')) return;
				results.set(run.id, result);
				if (result.status === 'failed') {
					log(`${label} ${run.id}  FAILED: ${result.error}`);
					// Start the next run on a fresh page: this one's WebGL may be gone.
					await slot.held?.page.close().catch(() => {});
					slot.held = null;
				} else {
					const rendered = options.frames && run.render.frames !== false;
					const rate = rendered && result.seconds > 0 ? `, ${(result.frames / result.seconds).toFixed(1)} frames/s overall` : ' (no render)';
					log(`${label} ${run.id}  done: ${result.frames} frames in ${result.seconds.toFixed(1)} s${rate}`);
				}
				if (stopping) return;
			}
			await slot.held?.page.close().catch(() => {});
		};
		await Promise.all(Array.from({ length: Math.min(options.concurrency, Math.max(1, todo.length)) }, worker));
	} finally {
		await writeIndex(jobDir, runs, results).catch((e) => log(`generate: could not write index.csv: ${e.message}`));
		await cleanup();
	}

	const failed = [...results.values()].filter((r) => r.status === 'failed');
	const made = [...results.values()].filter((r) => r.status === 'complete' && !r.skipped);
	log(`generate: ${made.length} runs written, ${skipped} already complete, ${failed.length} failed${stopping ? ', stopped' : ''}; index: ${relative(process.cwd(), join(jobDir, 'index.csv'))}`);
	if (failed.length) {
		log('generate: failed runs:');
		for (const f of failed) log(`  ${f.run.id}: ${(f.error ?? '').split('\n')[0]}`);
		process.exitCode = 1;
	}
	if (stopping) process.exit(130);
}

main().catch((e) => {
	console.error(`generate: ${e instanceof Error ? (process.env.DEBUG ? e.stack : e.message) : e}`);
	process.exitCode = 1;
});
