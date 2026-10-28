import { svelte } from '@sveltejs/vite-plugin-svelte';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [svelte({ compilerOptions: { runes: true } })],
	build: {
		// The engine chunk is three.js plus the ocean (~750 kB), loaded once and on purpose.
		chunkSizeWarningLimit: 1000,
		rollupOptions: {
			// The interactive app, and the page the batch generator drives headless.
			input: { main: 'index.html', batch: 'batch.html' }
		}
	}
});
