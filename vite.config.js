import { svelte } from '@sveltejs/vite-plugin-svelte';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [svelte({ compilerOptions: { runes: true } })],
	build: {
		rollupOptions: {
			// The interactive app, and the page the batch generator drives headless.
			input: { main: 'index.html', batch: 'batch.html' }
		}
	}
});
