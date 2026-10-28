<script>
	/**
	 * The sea and the sky: a preset to start from, then every SeaState and Lighting field.
	 * Moving a slider away from a preset's values makes the sea (or sky) "custom"; the
	 * readout is the simulator's own statistics for the sea as set (spectrum.js), the same
	 * numbers the engine reports and the batch writes to meta.json.
	 */
	import { lightPresets, seaPresets } from '../conditions/index.js';
	import { beaufortNames, seaStats } from '../ocean/spectrum.js';
	import { windAloft } from '../trajectory/index.js';
	import Range from './Range.svelte';
	import { lightPresetOf, seaPresetOf } from './state.js';

	/** @type {{ run: import('./state.js').AppRun }} */
	let { run = $bindable() } = $props();

	let seaName = $derived(seaPresetOf(run.sea));
	let lightName = $derived(lightPresetOf(run.light));
	let stats = $derived(seaStats(run.sea));
	let aloft = $derived(windAloft(run.sea, run.trajectory.altitude));
	let seaNote = $derived(seaName ? seaPresets[seaName].description : '');
	let lightNote = $derived(lightName ? lightPresets[lightName].description : '');
</script>

<div class="block">
	<label class="field">
		<span>Sea</span>
		<select
			value={seaName ?? ''}
			onchange={(e) => {
				const p = seaPresets[e.currentTarget.value];
				if (p) run.sea = structuredClone(p.sea);
			}}
		>
			{#if !seaName}<option value="">Custom</option>{/if}
			{#each Object.entries(seaPresets) as [name, p] (name)}
				<option value={name}>{p.label}</option>
			{/each}
		</select>
	</label>
	{#if seaNote}<p class="hint">{seaNote}</p>{/if}

	<dl class="stats" data-testid="sea-stats">
		<div><dt>H<sub>s</sub></dt><dd data-testid="hs">{stats.significantHeight.toFixed(2)}<small>m</small></dd></div>
		<div><dt>T<sub>p</sub></dt><dd>{stats.peakPeriod.toFixed(1)}<small>s</small></dd></div>
		<div><dt>λ<sub>p</sub></dt><dd>{stats.peakWavelength.toFixed(0)}<small>m</small></dd></div>
		<div><dt>Bft</dt><dd title={beaufortNames[stats.beaufort]}>{stats.beaufort}</dd></div>
	</dl>
	<p class="hint">
		Simulated sea, from the spectrum. Wind at {run.trajectory.altitude.toFixed(0)} m:
		<span class="mono">{aloft.speed.toFixed(1)} m/s</span> toward <span class="mono">{aloft.direction.toFixed(0)}°</span>
		(log profile over open water).
	</p>

	<Range label="Wind U10" bind:value={run.sea.windSpeed} min={0} max={30} step={0.1} unit="m/s" title="Wind speed at 10 m; drives the wind-sea spectrum" />
	<Range label="Wind toward" bind:value={run.sea.windDirection} min={0} max={360} step={1} unit="°" title="Compass degrees the wind and its sea travel toward" />
	<Range label="Fetch" bind:value={run.sea.fetch} min={5} max={1500} step={5} unit="km" title="Open water the wind has blown over" />
	<Range label="Spread" bind:value={run.sea.spread} min={1} max={32} step={0.5} title="Directional spreading exponent: higher is more long-crested" />
	<Range label="Swell Hs" bind:value={run.sea.swellHeight} min={0} max={6} step={0.05} unit="m" />
	<Range label="Swell toward" bind:value={run.sea.swellDirection} min={0} max={360} step={1} unit="°" />
	<Range label="Swell period" bind:value={run.sea.swellPeriod} min={4} max={20} step={0.1} unit="s" />
	<Range label="Choppiness" bind:value={run.sea.choppiness} min={0} max={1.6} step={0.01} title="Horizontal displacement: 0 rolling, 1.6 sharp crests" />

	<label class="field">
		<span>Sea seed</span>
		<input
			class="box mono"
			type="number"
			step="1"
			value={run.render.seaSeed}
			onchange={(e) => {
				const v = Math.round(Number(e.currentTarget.value));
				if (Number.isFinite(v)) run.render.seaSeed = v;
			}}
			title="The sea's Gaussian field: the same seed is the same sea. Changing it restarts the engine."
		/>
	</label>
</div>

<div class="block">
	<label class="field">
		<span>Light</span>
		<select
			value={lightName ?? ''}
			onchange={(e) => {
				const p = lightPresets[e.currentTarget.value];
				if (p) run.light = structuredClone(p.light);
			}}
		>
			{#if !lightName}<option value="">Custom</option>{/if}
			{#each Object.entries(lightPresets) as [name, p] (name)}
				<option value={name}>{p.label}</option>
			{/each}
		</select>
	</label>
	{#if lightNote}<p class="hint">{lightNote}</p>{/if}
	<Range label="Sun elevation" bind:value={run.light.sunElevation} min={-8} max={90} step={0.5} unit="°" />
	<Range label="Sun azimuth" bind:value={run.light.sunAzimuth} min={0} max={360} step={1} unit="°" title="Compass bearing toward the sun" />
	<Range label="Overcast" bind:value={run.light.overcast} min={0} max={1} step={0.01} />
	<Range label="Haze" bind:value={run.light.haze} min={0} max={1} step={0.01} />
	<Range label="Exposure" bind:value={run.light.exposure} min={0.25} max={3} step={0.01} unit="×" />
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

	.stats {
		display: grid;
		gap: 1px;
		grid-template-columns: repeat(4, 1fr);
		margin: 6px 0 2px;
		overflow: hidden;
		border-radius: var(--radius);
		background: var(--line);
	}

	.stats div {
		padding: 5px 7px 4px;
		background: var(--plate-strong);
	}

	.stats dt {
		color: var(--ink-faint);
		font-size: 10.5px;
	}

	.stats dd {
		margin: 0;
		color: var(--ink);
		font-family: var(--font-mono);
		font-size: 13px;
	}

	.stats small {
		margin-left: 2px;
		color: var(--ink-faint);
		font-size: 10px;
	}
</style>
