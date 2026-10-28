<script>
	/**
	 * How the preview is looked at — none of it goes into the run (the run's own camera is
	 * the export's `view`). The label channels are the ground truth the batch can write
	 * beside every shaded frame, coloured as channels.js ramps them.
	 */
	import { channels } from '../ocean/channels.js';

	/**
	 * @typedef {object} Props
	 * @property {import('./state.js').ViewState} view
	 * @property {boolean} interactive the user has the camera (drag to look, scroll to zoom)
	 */

	/** @type {Props} */
	let { view = $bindable(), interactive = $bindable() } = $props();

	const cameras = /** @type {const} */ ([
		['chase', 'Chase', 'A chase plane behind the aircraft'],
		['onboard', 'Onboard', "The aircraft's gimbal camera: what the batch renders with view 'onboard'"],
		['overhead', 'Overhead', 'Straight down on the whole flight, high enough to hold it']
	]);
	const overlays = /** @type {const} */ ([
		['grid', 'Grid', 'A metric grid on the sea'],
		['horizon', 'Horizon', 'The true horizon and the camera attitude, as a HUD'],
		['flow', 'Flow', 'Streaks following the simulated orbital velocity']
	]);

	/** @param {import('../ocean/scene.js').OverlayName} name */
	function toggle(name) {
		view.overlays = view.overlays.includes(name) ? view.overlays.filter((o) => o !== name) : [...view.overlays, name];
	}

	let channel = $derived(channels.find((c) => c.id === view.mode));
</script>

<div class="block">
	<div class="field">
		<span>Camera</span>
		<div class="seg" data-testid="camera">
			{#each cameras as [id, label, title] (id)}
				<button class:on={view.camera === id} onclick={() => (view.camera = id)} {title}>{label}</button>
			{/each}
		</div>
	</div>
	<div class="field">
		<span>Overlays</span>
		<div class="seg">
			{#each overlays as [id, label, title] (id)}
				<button class:on={view.overlays.includes(id)} onclick={() => toggle(id)} {title}>{label}</button>
			{/each}
		</div>
	</div>
	<div class="field">
		<span></span>
		<div class="checks">
			<label class="check"><input type="checkbox" bind:checked={view.track} /> Trajectory</label>
			<label class="check"><input type="checkbox" bind:checked={view.footprint} /> Camera footprint</label>
		</div>
	</div>
	<label class="field">
		<span>Render</span>
		<select bind:value={view.mode} data-testid="mode">
			<option value="shaded">Shaded</option>
			{#each channels as c (c.id)}
				<option value={c.id}>{c.label} ({c.id})</option>
			{/each}
		</select>
	</label>
	{#if channel}
		<div class="legend">
			<div class="ramp" style:background="linear-gradient(90deg, {channel.stops.join(', ')})"></div>
			<div class="ends mono">
				<span>{channel.id === 'normals' ? 'RGB = 0.5 + 0.5·n' : `${channel.range[0]} ${channel.unit}`}</span>
				{#if channel.id !== 'normals'}<span>{channel.range[1]} {channel.unit}</span>{/if}
			</div>
			<p class="hint">{channel.description}</p>
		</div>
	{/if}
	<label class="check take">
		<input type="checkbox" bind:checked={interactive} /> Take the camera <kbd>G</kbd>
		<span class="hint">drag to look, scroll to zoom</span>
	</label>
</div>

<style>
	.block {
		display: grid;
		/* One column the panel's width: a long line (the command) scrolls, not the panel. */
		grid-template-columns: minmax(0, 1fr);
		gap: 4px;
	}

	.checks {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 14px;
	}

	.legend {
		margin: 2px 0 4px;
	}

	.ramp {
		height: 6px;
		border-radius: 1px;
	}

	.ends {
		display: flex;
		justify-content: space-between;
		margin-top: 2px;
		color: var(--ink-faint);
		font-size: 10.5px;
	}

	.take {
		flex-wrap: wrap;
		margin-top: 4px;
	}

	.take .hint {
		margin: 0;
	}
</style>
