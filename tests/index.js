/**
 * `node --test tests/` names the directory, and node (≥ 22) takes a directory argument
 * as a module to run rather than a folder to search: it lands here. Load every test
 * file in this folder, so the directory form runs the same tests as
 * `node --test 'tests/*.test.js'` (the `pnpm test` script, one process per file).
 */
import { readdirSync } from 'node:fs';

const here = new URL('./', import.meta.url);
for (const name of readdirSync(here).filter((f) => f.endsWith('.test.js')).sort()) {
	await import(new URL(name, here).href);
}
