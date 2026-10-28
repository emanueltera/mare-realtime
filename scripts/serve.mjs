/**
 * The dev server the batch generator and the e2e tests render through: vite with the
 * project's config, but without hot reload. A batch page must never reload in the middle
 * of a run, and vite's client reloads every open page when any HTML file under the root
 * changes — a `vite build` writing dist/, say, while a job renders. So HMR is off, and the
 * watcher ignores what the generator and builds write.
 *
 *   node scripts/serve.mjs [--port 5174]   # serve until killed (the e2e config uses this)
 *
 * or, from a script: `const server = await serve(port); …; await server.close();`
 */
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { createServer } from 'vite';

const ROOT = fileURLToPath(new URL('..', import.meta.url));

/**
 * Start vite on 127.0.0.1:port, strictly that port (fail rather than move).
 * @param {number} port
 * @returns {Promise<{ url: string, close: () => Promise<void> }>}
 */
export async function serve(port) {
	// Vite answers SIGTERM by closing itself and exiting the process (128 + 15) as soon as
	// it has, which would cut short the caller's own shutdown (the generator closes its
	// browser, writes index.csv and exits 130). The caller owns the signals: take vite's
	// listener off again.
	const before = new Set(process.listeners('SIGTERM'));
	const server = await createServer({
		root: ROOT,
		logLevel: 'warn',
		clearScreen: false,
		server: {
			host: '127.0.0.1',
			port,
			strictPort: true,
			hmr: false,
			watch: { ignored: ['**/out/**', '**/dist/**', '**/.scratch/**', '**/test-results/**', '**/playwright-report/**'] }
		}
	});
	for (const listener of process.listeners('SIGTERM')) if (!before.has(listener)) process.off('SIGTERM', listener);
	try {
		await server.listen();
	} catch (e) {
		await server.close().catch(() => {});
		throw e;
	}
	return { url: `http://127.0.0.1:${port}/`, close: () => server.close() };
}

// Run directly: serve until killed.
if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
	const { values } = parseArgs({ options: { port: { type: 'string', default: '5174' } } });
	const server = await serve(Number(values.port));
	console.log(`serving ${server.url} (no hot reload)`);
	for (const signal of /** @type {const} */ (['SIGINT', 'SIGTERM'])) {
		process.once(signal, () => {
			server.close().finally(() => process.exit(0));
		});
	}
}
