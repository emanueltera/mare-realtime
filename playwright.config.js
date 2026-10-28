// @ts-check
/**
 * Playwright for the e2e tests: the engine in a real browser, on the real GPU (see
 * scripts/browser.mjs for the flags and for which Chromium build is used). The tests
 * start their own vite on MARE_E2E_PORT (default 5175, strict: never another port),
 * without hot reload (scripts/serve.mjs).
 *
 *   pnpm test:e2e                       # or: MARE_E2E_PORT=5190 pnpm test:e2e
 *   MARE_E2E_SOFTWARE=1 pnpm test:e2e   # SwiftShader, on a machine without a GPU
 */
import { defineConfig } from '@playwright/test';
import { launchOptions } from './scripts/browser.mjs';

const port = Number(process.env.MARE_E2E_PORT ?? 5175);
const baseURL = `http://127.0.0.1:${port}/`;

export default defineConfig({
	testDir: 'e2e',
	testMatch: /.*\.e2e\.js$/,
	// One GPU: tests share it, and frame timings are steadier one at a time.
	workers: 1,
	fullyParallel: false,
	timeout: 120_000,
	reporter: [['list']],
	use: {
		baseURL,
		viewport: { width: 1280, height: 720 },
		launchOptions: launchOptions({ software: Boolean(process.env.MARE_E2E_SOFTWARE) })
	},
	webServer: {
		command: `node scripts/serve.mjs --port ${port}`,
		url: `${baseURL}batch.html`,
		reuseExistingServer: false,
		timeout: 60_000
	}
});
