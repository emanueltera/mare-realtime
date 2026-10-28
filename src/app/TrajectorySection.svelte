<script>
	/**
	 * The flight: a pattern (a preset, then its parameters by PatternSpec type), how long
	 * and how high, the airspeed, the wind it flies through, turbulence, the seed of every
	 * random choice in it, and the gimbal camera. The trajectory is rebuilt from these by
	 * the app (debounced), so this section only edits the spec.
	 *
	 * Pattern lengths are metres; waypoints and orbit centres are [right, ahead] of the
	 * start along its heading ([east, north] for a start heading of 0); directions are
	 * compass degrees, toward.
	 */
	import { patternPresets } from '../conditions/index.js';
	import { patternDefaults, windAloft } from '../trajectory/index.js';
	import Range from './Range.svelte';
	import { patternPresetOf } from './state.js';

	/**
	 * @typedef {object} Props
	 * @property {import('./state.js').AppRun} run
	 * @property {string} [status] how the last rebuild went: its time, or the error
	 * @property {boolean} [failed] the last rebuild threw
	 */

	/** @type {Props} */
	let { run = $bindable(), status = '', failed = false } = $props();

	/** @typedef {import('../conditions/index.js').PatternSpec} PatternSpec */

	let t = $derived(run.trajectory);
	let pattern = $derived(/** @type {Record<string, any>} */ (t.pattern));
	let presetName = $derived(patternPresetOf(t.pattern));
	let note = $derived(presetName ? patternPresets[presetName].description : '');
	let windMode = $derived(typeof t.wind === 'string' ? t.wind : 'manual');

	/**
	 * A pattern with every parameter the type has written in (the trajectory module's
	 * defaults where it gave none), so each slider has a value to show.
	 * @param {PatternSpec} p
	 * @returns {PatternSpec}
	 */
	function complete(p) {
		if (p.type === 'straight') return { type: 'straight' };
		// A snapshot, not structuredClone: the pattern is usually a state proxy.
		// The type first, as the presets and resolveRun write it.
		if (p.type === 'waypoints') return { type: p.type, ...patternDefaults.waypoints, ...$state.snapshot(p) };
		return /** @type {PatternSpec} */ ({ type: p.type, ...structuredClone(patternDefaults[p.type]), ...$state.snapshot(p) });
	}

	/** @param {string} name */
	function choosePattern(name) {
		if (name === 'waypoints') {
			run.trajectory.pattern = {
				type: 'waypoints',
				points: [
					[0, 600],
					[500, 900],
					[900, 300],
					[300, -100]
				],
				loop: true
			};
			return;
		}
		const p = patternPresets[name];
		if (p) run.trajectory.pattern = complete(p.pattern);
	}

	// A pattern from a URL or an imported run may leave parameters to the defaults: write
	// them in so the sliders below have something to bind to.
	$effect.pre(() => {
		const p = run.trajectory.pattern;
		const full = complete(p);
		if (Object.keys(full).length !== Object.keys(p).length) run.trajectory.pattern = full;
	});

	let pointsText = $state('');
	let pointsError = $state('');
	// The waypoint text follows the pattern unless it is being edited into something else.
	$effect.pre(() => {
		if (pattern.type !== 'waypoints') return;
		const text = /** @type {number[][]} */ (pattern.points).map((p) => p.join(', ')).join('\n');
		if (parsePoints(pointsText)?.map((p) => p.join(', ')).join('\n') !== text) pointsText = text;
	});

	/**
	 * Waypoints typed one per line, `right, ahead` or `right, ahead, altitude`.
	 * @param {string} text
	 * @returns {Array<[number, number] | [number, number, number]> | null}
	 */
	function parsePoints(text) {
		const rows = text
			.split('\n')
			.map((l) => l.trim())
			.filter(Boolean)
			.map((l) => l.split(/[\s,;]+/).map(Number));
		if (rows.length < 2 || rows.some((r) => (r.length !== 2 && r.length !== 3) || r.some((v) => !Number.isFinite(v)))) return null;
		return /** @type {Array<[number, number] | [number, number, number]>} */ (rows);
	}

	/** @param {string} text */
	function setPoints(text) {
		pointsText = text;
		const points = parsePoints(text);
		pointsError = points ? '' : 'At least two lines of "right, ahead" or "right, ahead, altitude" in metres.';
		if (points && pattern.type === 'waypoints') run.trajectory.pattern = { ...pattern, type: 'waypoints', points };
	}

	/** @param {string} mode */
	function setWind(mode) {
		if (mode === 'sea' || mode === 'none') run.trajectory.wind = mode;
		else {
			// Manual starts from the wind the flight had (the sea's, carried up to cruise
			// altitude), not the sea's 10 m wind: switching how the wind is set should not
			// change it. Rounded to the slider's step.
			const aloft = windAloft(run.sea, run.trajectory.altitude);
			run.trajectory.wind = { speed: Math.round(aloft.speed * 10) / 10, direction: run.sea.windDirection };
		}
	}

	/** @param {'constant' | 'sine' | 'steps'} type */
	function setProfile(type) {
		if (type === 'sine') run.trajectory.altitudeProfile = { type, amplitude: 20, period: 90 };
		else if (type === 'steps') run.trajectory.altitudeProfile = { type, amplitude: 40, interval: 45 };
		else run.trajectory.altitudeProfile = { type };
	}
</script>

<div class="block">
	<label class="field">
		<span>Pattern</span>
		<select value={presetName ?? (pattern.type === 'waypoints' ? 'waypoints' : '')} onchange={(e) => choosePattern(e.currentTarget.value)}>
			{#if !presetName && pattern.type !== 'waypoints'}<option value="">Custom {pattern.type}</option>{/if}
			{#each Object.entries(patternPresets) as [name, p] (name)}
				<option value={name}>{p.label} · {p.pattern.type}</option>
			{/each}
			<option value="waypoints">Waypoints · waypoints</option>
		</select>
	</label>
	{#if note}<p class="hint">{note}</p>{/if}

	{#if pattern.type === 'orbit' || pattern.type === 'racetrack'}
		{#if pattern.type === 'racetrack'}
			<Range label="Length" bind:value={pattern.length} min={100} max={5000} step={10} unit="m" />
		{/if}
		<Range label="Radius" bind:value={pattern.radius} min={30} max={2000} step={5} unit="m" />
		<div class="field">
			<span>Direction</span>
			<div class="seg">
				<button class:on={pattern.direction !== -1} onclick={() => (pattern.direction = 1)}>Clockwise</button>
				<button class:on={pattern.direction === -1} onclick={() => (pattern.direction = -1)}>Anticlockwise</button>
			</div>
		</div>
	{:else if pattern.type === 'figure8'}
		<Range label="Radius" bind:value={pattern.radius} min={30} max={2000} step={5} unit="m" />
	{:else if pattern.type === 'lawnmower'}
		<Range label="Leg length" bind:value={pattern.legLength} min={100} max={5000} step={10} unit="m" />
		<Range label="Spacing" bind:value={pattern.spacing} min={20} max={1000} step={5} unit="m" />
		<Range label="Legs" bind:value={pattern.legs} min={1} max={30} step={1} />
	{:else if pattern.type === 'random'}
		<Range label="Leg min" bind:value={pattern.legLength[0]} min={50} max={pattern.legLength[1]} step={10} unit="m" />
		<Range label="Leg max" bind:value={pattern.legLength[1]} min={pattern.legLength[0]} max={5000} step={10} unit="m" />
		<Range label="Turn min" bind:value={pattern.turn[0]} min={-180} max={pattern.turn[1]} step={1} unit="°" title="Turn between legs, positive right" />
		<Range label="Turn max" bind:value={pattern.turn[1]} min={pattern.turn[0]} max={180} step={1} unit="°" />
	{:else if pattern.type === 'waypoints'}
		<label class="stack">
			<span>Waypoints <em>right, ahead[, altitude] m</em></span>
			<textarea class="mono" rows="4" spellcheck="false" value={pointsText} oninput={(e) => setPoints(e.currentTarget.value)}></textarea>
		</label>
		{#if pointsError}<p class="hint bad">{pointsError}</p>{/if}
		<label class="check"><input type="checkbox" bind:checked={pattern.loop} /> Loop back to the first</label>
	{/if}
</div>

<div class="block">
	<Range label="Duration" bind:value={run.trajectory.duration} min={5} max={900} step={5} unit="s" />
	<Range label="Altitude" bind:value={run.trajectory.altitude} min={10} max={1000} step={1} unit="m" title="Cruise altitude above mean sea level" />
	<div class="field">
		<span>Profile</span>
		<div class="seg">
			{#each /** @type {const} */ (['constant', 'sine', 'steps']) as type (type)}
				<button class:on={t.altitudeProfile.type === type} onclick={() => setProfile(type)}>{type}</button>
			{/each}
		</div>
	</div>
	{#if run.trajectory.altitudeProfile.type === 'sine'}
		<Range label="Amplitude" bind:value={run.trajectory.altitudeProfile.amplitude} min={0} max={200} step={1} unit="m" />
		<Range label="Period" bind:value={run.trajectory.altitudeProfile.period} min={10} max={600} step={5} unit="s" />
	{:else if run.trajectory.altitudeProfile.type === 'steps'}
		<Range label="Amplitude" bind:value={run.trajectory.altitudeProfile.amplitude} min={0} max={200} step={1} unit="m" />
		<Range label="Interval" bind:value={run.trajectory.altitudeProfile.interval} min={5} max={300} step={5} unit="s" />
	{/if}
	<Range label="Airspeed" bind:value={run.trajectory.airspeed} min={10} max={45} step={0.5} unit="m/s" />
	<Range label="Variation" bind:value={run.trajectory.airspeedVariation} min={0} max={8} step={0.1} unit="m/s" title="Size of slow seeded airspeed changes: their RMS is that of a sine of this amplitude, so the peaks of a long flight reach 1.5–2 times it" />
	<Range label="Start heading" bind:value={run.trajectory.start.heading} min={0} max={360} step={1} unit="°" />
	<Range label="Start east" bind:value={run.trajectory.start.x} min={-3000} max={3000} step={10} unit="m" />
	<Range label="Start north" bind:value={run.trajectory.start.y} min={-3000} max={3000} step={10} unit="m" />
</div>

<div class="block">
	<div class="field">
		<span>Wind</span>
		<div class="seg">
			<button class:on={windMode === 'sea'} onclick={() => setWind('sea')} title="The sea's U10 carried up to altitude">Sea</button>
			<button class:on={windMode === 'none'} onclick={() => setWind('none')}>None</button>
			<button class:on={windMode === 'manual'} onclick={() => setWind('manual')}>Manual</button>
		</div>
	</div>
	{#if typeof run.trajectory.wind === 'object'}
		<Range label="Wind aloft" bind:value={run.trajectory.wind.speed} min={0} max={30} step={0.1} unit="m/s" />
		<Range label="Toward" bind:value={run.trajectory.wind.direction} min={0} max={360} step={1} unit="°" />
	{/if}
	<Range label="Turbulence" bind:value={run.trajectory.turbulence} min={0} max={1} step={0.01} title="0 still air, 1 strong" />
	<div class="field">
		<span>Seed</span>
		<div class="seedrow">
			<input
				class="box mono"
				type="number"
				step="1"
				value={run.trajectory.seed}
				onchange={(e) => {
					const v = Math.round(Number(e.currentTarget.value));
					if (Number.isFinite(v)) run.trajectory.seed = v;
				}}
			/>
			<button class="btn" onclick={() => (run.trajectory.seed += 1)} title="Next seed">+1</button>
		</div>
	</div>
</div>

<div class="block">
	<Range label="Gimbal pitch" bind:value={run.trajectory.gimbal.pitch} min={-90} max={30} step={0.5} unit="°" title="Negative looks down" />
	<Range label="Gimbal yaw" bind:value={run.trajectory.gimbal.yaw} min={-180} max={180} step={1} unit="°" title="From the nose, positive right" />
	<Range label="Field of view" bind:value={run.trajectory.gimbal.fov} min={10} max={120} step={0.5} unit="°" title="Vertical" />
	<Range label="Stabilised" bind:value={run.trajectory.gimbal.stabilized} min={0} max={1} step={0.01} title="0 hard-mounted, 1 fully stabilised" />
</div>

{#if status}<p class={['hint', 'status', failed && 'bad']} data-testid="trajectory-status">{status}</p>{/if}

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

	.stack {
		display: grid;
		gap: 4px;
		color: var(--ink-dim);
	}

	.stack em {
		color: var(--ink-faint);
		font-style: normal;
	}

	textarea {
		width: 100%;
		resize: vertical;
		font-size: 11px;
	}

	.seedrow {
		display: flex;
		gap: 6px;
	}

	.seedrow input {
		flex: 1;
		width: 0;
	}

	.status {
		margin-top: 10px;
	}

	.bad {
		color: var(--bad);
	}
</style>
