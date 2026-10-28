<script>
	/**
	 * The run as files for the batch generator: run.json (the RunSpec, resolved), the
	 * trajectory as CSV at a chosen rate, and the command that renders the run. A run.json
	 * from anywhere (the batch's output, a job's expansion) can be loaded back in.
	 */
	import { renderModes, resolveRun, runSummary } from '../conditions/index.js';
	import { channels } from '../ocean/channels.js';
	import { download, readRunFile, runJson, trajectoryCsv } from './files.js';
	import Range from './Range.svelte';
	import { cliCommand, toPartial } from './state.js';

	/**
	 * @typedef {object} Props
	 * @property {import('./state.js').AppRun} run
	 * @property {import('../trajectory/index.js').Trajectory | null} trajectory the current build
	 * @property {(partial: Record<string, any>) => void} onimport load a run (throws if invalid)
	 */

	/** @type {Props} */
	let { run = $bindable(), trajectory, onimport } = $props();

	let csvRate = $state(10);
	let message = $state('');
	let bad = $state(false);
	let copied = $state(false);

	/** The run as it would be written, or why it cannot be. */
	let resolved = $derived.by(() => {
		try {
			return { run: resolveRun(toPartial($state.snapshot(run))), error: '' };
		} catch (error) {
			return { run: null, error: error instanceof Error ? error.message : String(error) };
		}
	});
	let frames = $derived(Math.floor(run.trajectory.duration * run.render.fps + 1e-9) + 1);

	/** @param {string} text @param {boolean} [isBad] */
	function say(text, isBad = false) {
		message = text;
		bad = isBad;
	}

	function saveRun() {
		try {
			download('run.json', runJson(toPartial($state.snapshot(run))), 'application/json');
			say('run.json downloaded.');
		} catch (error) {
			say(error instanceof Error ? error.message : String(error), true);
		}
	}

	function saveCsv() {
		if (!trajectory) return say('The trajectory is still being built.', true);
		// The number box can be cleared or set below its minimum; say so rather than throw.
		if (!(csvRate > 0)) return say('The CSV rate must be more than 0 Hz.', true);
		try {
			download('trajectory.csv', trajectoryCsv(trajectory, csvRate), 'text/csv');
			say(`trajectory.csv downloaded: ${Math.floor(trajectory.duration * csvRate + 1e-9) + 1} samples at ${csvRate} Hz.`);
		} catch (error) {
			say(error instanceof Error ? error.message : String(error), true);
		}
	}

	async function copyCommand() {
		try {
			await navigator.clipboard.writeText(cliCommand());
			copied = true;
			setTimeout(() => (copied = false), 1500);
		} catch {
			say('The clipboard is not available here: select the command and copy it.', true);
		}
	}

	/** @param {Event & { currentTarget: HTMLInputElement }} e */
	async function importFile(e) {
		const input = e.currentTarget;
		const file = input.files?.[0];
		input.value = '';
		if (!file) return;
		try {
			onimport(await readRunFile(file));
			say(`Loaded ${file.name}.`);
		} catch (error) {
			say(error instanceof Error ? error.message : String(error), true);
		}
	}

	/** @param {import('../ocean/scene.js').RenderMode} mode */
	function toggleMode(mode) {
		const on = run.render.modes.includes(mode);
		if (on && run.render.modes.length === 1) return;
		run.render.modes = on ? run.render.modes.filter((m) => m !== mode) : renderModes.filter((m) => m === mode || run.render.modes.includes(m));
	}

	/** @param {string} id */
	const modeLabel = (id) => (id === 'shaded' ? 'Shaded' : (channels.find((c) => c.id === id)?.label ?? id));
</script>

<div class="block">
	<label class="field">
		<span>Run id</span>
		<input
			type="text"
			class="mono"
			placeholder={resolved.run && !run.id ? resolved.run.id : 'automatic'}
			value={run.id}
			spellcheck="false"
			onchange={(e) => (run.id = e.currentTarget.value.trim())}
		/>
	</label>
	<p class="hint">{resolved.run ? runSummary(resolved.run) : ''}</p>

	<div class="field">
		<span>View</span>
		<div class="seg">
			<button class:on={run.render.view === 'onboard'} onclick={() => (run.render.view = 'onboard')}>Onboard</button>
			<button class:on={run.render.view === 'chase'} onclick={() => (run.render.view = 'chase')}>Chase</button>
		</div>
	</div>
	<div class="field size">
		<span>Size</span>
		<div class="pair">
			<input class="box mono" type="number" min="16" max="8192" step="1" aria-label="Width" bind:value={run.render.width} />
			<span>×</span>
			<input class="box mono" type="number" min="16" max="8192" step="1" aria-label="Height" bind:value={run.render.height} />
			<span class="unit">px</span>
		</div>
	</div>
	<Range label="Frame rate" bind:value={run.render.fps} min={1} max={60} step={1} unit="fps" />
	<Range label="Warm-up" bind:value={run.render.warmup} min={0} max={60} step={1} unit="s" title="Seconds the sea and foam run before frame 0" />
	<Range label="Sea time" bind:value={run.render.seaTime} min={0} max={600} step={1} unit="s" title="The sea's clock at the start of the warm-up" />
	<div class="field">
		<span>Quality</span>
		<div class="seg">
			<button class:on={run.render.quality === 'high'} onclick={() => (run.render.quality = 'high')}>High</button>
			<button class:on={run.render.quality === 'low'} onclick={() => (run.render.quality = 'low')}>Low</button>
		</div>
	</div>
	<div class="field modes">
		<span>Channels</span>
		<div class="checks">
			{#each renderModes as m (m)}
				<label class="check"><input type="checkbox" checked={run.render.modes.includes(m)} onchange={() => toggleMode(m)} /> {modeLabel(m)}</label>
			{/each}
		</div>
	</div>
	<div class="field">
		<span></span>
		<label class="check"><input type="checkbox" bind:checked={run.render.frames} /> Render frames (off: trajectory and IMU only)</label>
	</div>
	<Range label="IMU rate" bind:value={run.imuRate} min={10} max={1000} step={10} unit="Hz" />
	<p class="hint mono">
		{frames} frames{run.render.frames ? ` × ${run.render.modes.length} channel${run.render.modes.length > 1 ? 's' : ''}` : ' (not rendered)'},
		{run.render.width}×{run.render.height}
	</p>
</div>

<div class="block">
	<div class="actions">
		<button class="btn primary" onclick={saveRun} disabled={!resolved.run} data-testid="download-run">Download run.json</button>
		<label class="btn file">
			Import run.json
			<input type="file" accept=".json,application/json" onchange={importFile} data-testid="import-run" />
		</label>
	</div>
	<div class="actions">
		<button class="btn" onclick={saveCsv} disabled={!trajectory} data-testid="download-csv">Download trajectory.csv</button>
		<label class="rate">
			<input class="box mono" type="number" min="1" max="1000" step="1" bind:value={csvRate} aria-label="CSV rate" />
			<span class="unit">Hz</span>
		</label>
	</div>
	<div class="cmd">
		<code class="mono" data-testid="cli">{cliCommand()}</code>
		<button class="btn" onclick={copyCommand}>{copied ? 'Copied' : 'Copy'}</button>
	</div>
	{#if resolved.error}<p class="hint bad">{resolved.error}</p>{/if}
	{#if message}<p class={['hint', bad && 'bad']} data-testid="export-message">{message}</p>{/if}
</div>

<style>
	.block {
		display: grid;
		/* One column the panel's width: a long line (the command) scrolls, not the panel. */
		grid-template-columns: minmax(0, 1fr);
		gap: 4px;
	}

	.block + .block {
		margin-top: 12px;
		padding-top: 10px;
		border-top: 1px solid var(--line);
	}

	.pair {
		display: flex;
		align-items: center;
		gap: 6px;
		color: var(--ink-faint);
	}

	.pair input {
		width: 0;
		flex: 1;
	}

	.unit {
		color: var(--ink-faint);
		font-family: var(--font-mono);
		font-size: 10.5px;
	}

	.modes {
		align-items: start;
	}

	.checks {
		display: grid;
		gap: 3px 10px;
		grid-template-columns: 1fr 1fr;
		padding-top: 4px;
	}

	.actions {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		margin-bottom: 4px;
	}

	.actions .btn {
		flex: 1 1 auto;
	}

	.btn:disabled {
		opacity: 0.4;
		cursor: default;
	}

	.file {
		position: relative;
		overflow: hidden;
		cursor: pointer;
	}

	.file input {
		position: absolute;
		inset: 0;
		opacity: 0;
		cursor: pointer;
	}

	.rate {
		display: flex;
		align-items: center;
		gap: 4px;
	}

	.rate input {
		width: 64px;
	}

	.cmd {
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 4px 4px 4px 8px;
		border-radius: var(--radius);
		background: var(--plate-strong);
	}

	.cmd code {
		flex: 1;
		min-width: 0;
		overflow-x: auto;
		color: var(--ink-dim);
		font-size: 11px;
		white-space: nowrap;
		user-select: all;
	}

	.bad {
		color: var(--bad);
	}
</style>
