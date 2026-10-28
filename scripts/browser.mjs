/**
 * How the batch generator and the e2e tests launch Chromium: headless, on the real GPU
 * (WebGL through ANGLE on the platform's native API), or on SwiftShader when asked.
 *
 * Playwright pins one browser build per release. If that build has not been downloaded
 * (`pnpm exec playwright install chromium`), an older build already in Playwright's
 * cache is used instead, newest first; `MARE_CHROMIUM=/path/to/chrome` overrides both.
 */
import { existsSync, readdirSync } from 'node:fs';
import { homedir, platform } from 'node:os';
import { join } from 'node:path';
import { chromium } from '@playwright/test';

/**
 * Chromium flags for WebGL on the GPU. Without them headless Chromium may refuse the
 * GPU (blocklist) and fall back to software, which is slower by an order of magnitude
 * and rasterises differently. Only macOS (Metal) has been verified; the others are
 * ANGLE's usual back ends there.
 * @param {boolean} software SwiftShader instead: slow, but runs anywhere without a GPU
 * @returns {string[]}
 */
export function gpuArgs(software = false) {
	if (software) return ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'];
	const angle = { darwin: 'metal', win32: 'd3d11', linux: 'vulkan' }[/** @type {string} */ (platform())] ?? 'default';
	const args = ['--enable-gpu', '--ignore-gpu-blocklist', `--use-angle=${angle}`];
	if (angle === 'vulkan') args.push('--enable-features=Vulkan');
	return args;
}

/** Where Playwright keeps its browsers on this machine. */
function browsersDir() {
	if (process.env.PLAYWRIGHT_BROWSERS_PATH && process.env.PLAYWRIGHT_BROWSERS_PATH !== '0') return process.env.PLAYWRIGHT_BROWSERS_PATH;
	if (platform() === 'darwin') return join(homedir(), 'Library', 'Caches', 'ms-playwright');
	if (platform() === 'win32') return join(process.env.LOCALAPPDATA ?? homedir(), 'ms-playwright');
	return join(homedir(), '.cache', 'ms-playwright');
}

/**
 * The executables inside one cached browser build, by the layouts Playwright uses.
 * @param {string} dir e.g. …/chromium_headless_shell-1243
 */
function executablesIn(dir) {
	const shell = ['chrome-headless-shell-mac-arm64', 'chrome-headless-shell-mac-x64', 'chrome-headless-shell-linux64', 'chrome-headless-shell-win64'];
	return [
		...shell.map((d) => join(dir, d, d.includes('win') ? 'chrome-headless-shell.exe' : 'chrome-headless-shell')),
		join(dir, 'chrome-linux', 'headless_shell'),
		join(dir, 'chrome-mac-arm64', 'Google Chrome for Testing.app', 'Contents', 'MacOS', 'Google Chrome for Testing'),
		join(dir, 'chrome-mac', 'Chromium.app', 'Contents', 'MacOS', 'Chromium'),
		join(dir, 'chrome-linux', 'chrome'),
		join(dir, 'chrome-win', 'chrome.exe')
	];
}

/**
 * The Chromium executable to launch: MARE_CHROMIUM, else Playwright's own build if it is
 * installed (undefined: let Playwright find it), else the newest cached build.
 * @returns {string | undefined}
 */
export function chromiumExecutable() {
	if (process.env.MARE_CHROMIUM) return process.env.MARE_CHROMIUM;
	const dir = browsersDir();
	// Playwright's headless mode runs the headless shell of its own build, kept beside the
	// full browser: if that is there, Playwright finds it itself.
	const build = /chromium-(\d+)/.exec(chromium.executablePath())?.[1];
	if (build && executablesIn(join(dir, `chromium_headless_shell-${build}`)).some(existsSync)) return undefined;
	if (!existsSync(dir)) return undefined;
	const builds = readdirSync(dir)
		.map((name) => ({ name, m: /^(chromium_headless_shell|chromium)-(\d+)$/.exec(name) }))
		.filter((b) => b.m)
		// Newest build first; at the same build the headless shell before the full browser.
		.sort((a, b) => Number(b.m?.[2]) - Number(a.m?.[2]) || (a.m?.[1] === 'chromium_headless_shell' ? -1 : 1));
	for (const b of builds) {
		const found = executablesIn(join(dir, b.name)).find(existsSync);
		if (found) return found;
	}
	return undefined;
}

/**
 * Options for `chromium.launch` (and the e2e config's `launchOptions`).
 * @param {{ software?: boolean, headless?: boolean }} [options]
 * @returns {import('@playwright/test').LaunchOptions}
 */
export function launchOptions({ software = false, headless = true } = {}) {
	return { headless, executablePath: chromiumExecutable(), args: gpuArgs(software) };
}
