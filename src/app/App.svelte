<script>
	/**
	 * The interactive app: design one run (sea, sky, flight, camera), watch it flown over
	 * the real-time ocean, and export it for the batch generator (docs/ARCHITECTURE.md).
	 *
	 * The state is a resolved RunSpec (`run`) and how the preview looks at it (`view`);
	 * both live in the URL's hash, so a link reproduces the screen. From them:
	 * - the trajectory, rebuilt by `createTrajectory` a moment after the flight (or, for
	 *   wind 'sea', the sea) stops changing — it integrates the whole flight, which takes a
	 *   noticeable fraction of a second for a long one;
	 * - the sea and sky, handed to the engine live through `oceanControl` while a slider
	 *   moves, and into the scene once they settle;
	 * - the scene (scene.js): the aircraft placed exactly on the trajectory, the trajectory
	 *   drawn on the sea, and a chase, onboard or overhead camera.
	 *
	 * Playback is the app's clock, not the engine's: every animation frame it seeks the
	 * engine to `t`, so pausing, scrubbing and 0.5×–8× playback all go through one path,
	 * and the readouts come from the trajectory sample at `t` (the ground truth), not from
	 * the picture.
	 */
	import { onMount, untrack } from 'svelte';
	import { oceanControl, oceanReadout } from '../ocean/control.js';
	import { createTrajectory } from '../trajectory/index.js';
	import ConditionsSection from './ConditionsSection.svelte';
	import ExportSection from './ExportSection.svelte';
	import OceanStage from './OceanStage.svelte';
	import Timeline from './Timeline.svelte';
	import TrajectorySection from './TrajectorySection.svelte';
	import ViewSection from './ViewSection.svelte';
	import { buildScene, trackOverlay } from './scene.js';
	import { decodeHash, defaultRun, encodeHash, fromPartial, viewDefaults } from './state.js';

	/** @typedef {import('../ocean/engine/engine.js').OceanEngine} OceanEngine */
	/** @typedef {import('../trajectory/index.js').Trajectory} Trajectory */

	/** Narrower than this, the panel is a sheet over the bottom of the sea. */
	const NARROW = 760;
	/** px the bar along the top covers, with its margin. */
	const BAR = 52;

	let linkError = $state('');
	/** @returns {{ run: import('./state.js').AppRun, view: import('./state.js').ViewState }} */
	function fromLocation() {
		try {
			const decoded = decodeHash(location.hash);
			if (decoded) return decoded;
		} catch (error) {
			linkError = `The link's run could not be read (${error instanceof Error ? error.message : error}); showing the default run.`;
		}
		return { run: defaultRun(), view: structuredClone({ ...viewDefaults }) };
	}

	const initial = fromLocation();
	let run = $state(initial.run);
	let view = $state(initial.view);
	let interactive = $state(false);
	let panelOpen = $state(innerWidth > NARROW);

	// ------------------------------------------------------------------ the trajectory

	/** @type {{ trajectory: Trajectory, track: import('../ocean/scene.js').TrackOverlay } | null} */
	let built = $state.raw(null);
	let buildStatus = $state('');
	let buildFailed = $state(false);

	$effect(() => {
		const spec = $state.snapshot(run.trajectory);
		// Only a trajectory flown in the sea's own wind depends on the sea.
		const sea = spec.wind === 'sea' ? $state.snapshot(run.sea) : undefined;
		const first = untrack(() => built === null);
		const timer = setTimeout(() => {
			try {
				const start = performance.now();
				const trajectory = createTrajectory(spec, sea ? { sea } : {});
				const track = trackOverlay(trajectory);
				built = { trajectory, track };
				buildFailed = false;
				buildStatus = [
					`Trajectory: ${trajectory.duration} s integrated at ${trajectory.rate} Hz in ${Math.round(performance.now() - start)} ms.`,
					...trajectory.warnings.map((w) => `Warning: ${w}.`)
				].join(' ');
			} catch (error) {
				buildFailed = true;
				buildStatus = error instanceof Error ? error.message : String(error);
			}
		}, first ? 0 : 250);
		return () => clearTimeout(timer);
	});

	// ------------------------------------------------------------------ sea and sky

	// Live: the engine eases toward whatever is in oceanControl every frame.
	$effect(() => {
		oceanControl.sea = $state.snapshot(run.sea);
		oceanControl.light = $state.snapshot(run.light);
	});

	// Settled: into the scene, so a scene change (or the fallback) carries them too.
	let conditions = $state.raw({ sea: structuredClone(initial.run.sea), light: structuredClone(initial.run.light) });
	$effect(() => {
		const next = { sea: $state.snapshot(run.sea), light: $state.snapshot(run.light) };
		const timer = setTimeout(() => (conditions = next), 400);
		return () => clearTimeout(timer);
	});

	// ------------------------------------------------------------------ layout and scene

	let width = $state(innerWidth);
	let height = $state(innerHeight);
	let panelWidth = $state(0);
	let panelHeight = $state(0);
	let timelineHeight = $state(0);
	let narrow = $derived(width <= NARROW);

	/** What of the window the interface covers, settled (a resize reframes the cameras). */
	let viewport = $state.raw({ width: innerWidth, height: innerHeight, top: 0, right: 0, bottom: 0 });
	$effect(() => {
		const next = {
			width,
			height,
			top: BAR,
			right: !narrow && panelOpen ? panelWidth + 12 : 0,
			bottom: timelineHeight + (narrow && panelOpen ? panelHeight : 0)
		};
		const prev = untrack(() => viewport);
		if (Object.entries(next).every(([k, v]) => prev[/** @type {keyof typeof next} */ (k)] === v)) return;
		const timer = setTimeout(() => (viewport = next), prev.bottom === 0 ? 0 : 250);
		return () => clearTimeout(timer);
	});

	let scene = $derived(
		built
			? buildScene({
					trajectory: built.trajectory,
					track: built.track,
					sea: conditions.sea,
					light: conditions.light,
					view: $state.snapshot(view),
					viewport
				})
			: null
	);

	// ------------------------------------------------------------------ playback

	let t = $state(0);
	let playing = $state(true);
	let speed = $state(1);
	/** @type {OceanEngine | null} */
	let engine = null;

	/** The trajectory sample and the readout, polled a few times a second. */
	/** @type {import('../trajectory/index.js').TrajectorySample | null} */
	let sample = $state.raw(null);
	/** @type {number | null} */
	let heightAboveSea = $state(null);
	let readout = $state.raw({ ...oceanReadout });

	/** @param {number} next */
	function seek(next) {
		const duration = built?.trajectory.duration ?? 0;
		t = Math.min(duration, Math.max(0, next));
		engine?.seek(t);
		poll();
	}

	function poll() {
		readout = { ...oceanReadout };
		const trajectory = built?.trajectory;
		if (!trajectory) return;
		const s = trajectory.stateAt(t);
		sample = s;
		// The sea under the aircraft from the engine's own sampler, at its sea clock.
		heightAboveSea = engine ? s.altitude - engine.sampler.sample(s.x, s.y, engine.simTime).height : null;
	}

	onMount(() => {
		/** @type {number | null} */
		let last = null;
		let raf = requestAnimationFrame(function tick(ms) {
			raf = requestAnimationFrame(tick);
			const dt = last === null ? 0 : Math.min(0.1, (ms - last) / 1000);
			last = ms;
			const trajectory = built?.trajectory;
			if (!trajectory) return;
			const duration = trajectory.duration;
			if (playing && duration > 0) {
				t += dt * speed;
				if (t >= duration) t %= duration;
			}
			if (t > duration) t = duration;
			// The app owns the flight's clock: the engine shows the aircraft at t (held at the
			// ends, never wrapped) instead of stepping it on by its own frame time.
			engine?.seek(t);
		});
		const timer = setInterval(poll, 100);

		/** @param {KeyboardEvent} e */
		function onKey(e) {
			if (e.metaKey || e.ctrlKey || e.altKey || e.repeat) return;
			const el = /** @type {HTMLElement | null} */ (e.target instanceof HTMLElement ? e.target : null);
			const typing =
				el && (el.isContentEditable || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT' || (el.tagName === 'INPUT' && !['range', 'checkbox', 'radio', 'button'].includes(/** @type {HTMLInputElement} */ (el).type)));
			if (typing) return;
			if (e.key === ' ') {
				e.preventDefault();
				playing = !playing;
			} else if (e.key === 'h' || e.key === 'H') {
				panelOpen = !panelOpen;
			} else if (e.key === 'g' || e.key === 'G') {
				interactive = !interactive;
			}
		}

		function onHash() {
			if (location.hash === `#${lastHash}`) return;
			const next = fromLocation();
			run = next.run;
			view = next.view;
		}

		window.addEventListener('keydown', onKey);
		window.addEventListener('hashchange', onHash);
		if (import.meta.env.DEV) {
			/** @type {any} */ (window).__mareApp = {
				run: () => $state.snapshot(run),
				view: () => $state.snapshot(view),
				trajectory: () => built?.trajectory ?? null,
				track: () => built?.track ?? null,
				time: () => t,
				seek
			};
		}
		return () => {
			cancelAnimationFrame(raf);
			clearInterval(timer);
			window.removeEventListener('keydown', onKey);
			window.removeEventListener('hashchange', onHash);
			if (import.meta.env.DEV) delete (/** @type {any} */ (window).__mareApp);
		};
	});

	// ------------------------------------------------------------------ URL

	let lastHash = '';
	$effect(() => {
		const r = $state.snapshot(run);
		const v = $state.snapshot(view);
		const timer = setTimeout(() => {
			try {
				lastHash = encodeHash(r, v);
				history.replaceState(null, '', `#${lastHash}`);
			} catch {
				// A run that does not resolve (a field mid-edit) keeps the last good link.
			}
		}, 400);
		return () => clearTimeout(timer);
	});

	/** @param {Record<string, any>} partial */
	function importRun(partial) {
		run = fromPartial(partial);
		t = 0;
	}

	/** @param {OceanEngine | null} next */
	function onEngine(next) {
		engine = next;
		if (next) next.seek(t);
	}

	let statusText = $derived(
		readout.status === 'running'
			? `${readout.fps} fps`
			: readout.status === 'loading'
				? 'starting the engine…'
				: 'no WebGL2: CSS sea, export still works'
	);
</script>

<svelte:window bind:innerWidth={width} bind:innerHeight={height} />

<main class={['app', narrow && 'narrow', panelOpen && 'panel-open']} data-status={readout.status}>
	<div class="stage">
		{#if scene}
			{#key run.render.seaSeed}
				<OceanStage {scene} {interactive} seed={run.render.seaSeed} onengine={onEngine} />
			{/key}
		{/if}
	</div>

	<header class="bar">
		<span class="name">mare<span>·</span>realtime</span>
		<span class={['status', readout.status]} data-testid="status">
			<span class="dot"></span>{statusText}
		</span>
		{#if interactive}<span class="taken">camera taken · <kbd>G</kbd> to release</span>{/if}
		<button class="btn toggle" onclick={() => (panelOpen = !panelOpen)} aria-expanded={panelOpen} aria-controls="panel" title="Show or hide the panel (H)">
			{panelOpen ? 'Hide panel' : 'Panel'}
		</button>
	</header>

	{#if linkError}
		<div class="banner" role="alert">
			{linkError}
			<button class="btn" onclick={() => (linkError = '')}>Dismiss</button>
		</div>
	{/if}

	<aside id="panel" class="panel" aria-hidden={!panelOpen} inert={!panelOpen} bind:offsetWidth={panelWidth} bind:offsetHeight={panelHeight} style:--dock={`${timelineHeight}px`}>
		<details open>
			<summary>Conditions</summary>
			<ConditionsSection bind:run />
		</details>
		<details open>
			<summary>Trajectory</summary>
			<TrajectorySection bind:run status={buildStatus} failed={buildFailed} />
		</details>
		<details open>
			<summary>View</summary>
			<ViewSection bind:view bind:interactive />
		</details>
		<details open>
			<summary>Export</summary>
			<ExportSection bind:run trajectory={built?.trajectory ?? null} onimport={importRun} />
		</details>
		<p class="hint keys"><kbd>Space</kbd> play / pause · <kbd>H</kbd> panel · <kbd>G</kbd> take the camera</p>
	</aside>

	<footer class="dock" bind:offsetHeight={timelineHeight}>
		<Timeline
			{t}
			duration={built?.trajectory.duration ?? run.trajectory.duration}
			bind:playing
			bind:speed
			{sample}
			{heightAboveSea}
			onseek={seek}
		/>
	</footer>
</main>

<style>
	.app {
		position: fixed;
		inset: 0;
		overflow: hidden;
	}

	.stage {
		position: absolute;
		inset: 0;
	}

	/* ------------------------------------------------------------------ top bar */

	.bar {
		position: absolute;
		top: 12px;
		left: 12px;
		display: flex;
		align-items: center;
		gap: 12px;
		padding: 5px 6px 5px 12px;
		border-radius: var(--radius);
		background: var(--plate);
		pointer-events: auto;
	}

	.name {
		color: var(--ink);
		font-weight: 600;
		letter-spacing: 0.02em;
	}

	.name span {
		margin: 0 1px;
		color: var(--accent);
	}

	.status {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		color: var(--ink-faint);
		font-family: var(--font-mono);
		font-size: 11px;
	}

	.dot {
		width: 6px;
		height: 6px;
		border-radius: 50%;
		background: var(--ink-faint);
	}

	.status.running .dot {
		background: var(--sea);
	}

	.status.fallback .dot,
	.status.error .dot {
		background: var(--warn);
	}

	.taken {
		color: var(--accent);
		font-size: 11px;
	}

	.toggle {
		padding: 3px 9px;
	}

	.banner {
		position: absolute;
		top: 54px;
		left: 12px;
		display: flex;
		align-items: center;
		gap: 10px;
		max-width: min(560px, calc(100vw - 24px));
		padding: 8px 8px 8px 12px;
		border-radius: var(--radius);
		background: var(--plate-strong);
		color: var(--warn);
	}

	/* ------------------------------------------------------------------ panel */

	.panel {
		position: absolute;
		top: 12px;
		right: 12px;
		bottom: calc(var(--dock) + 12px);
		width: var(--panel-w);
		overflow-x: hidden;
		overflow-y: auto;
		overscroll-behavior: contain;
		padding: 4px 14px 12px;
		border-radius: var(--radius);
		background: var(--plate);
		scrollbar-color: var(--line-strong) transparent;
		scrollbar-width: thin;
		transition:
			transform 0.28s ease,
			opacity 0.28s ease;
	}

	.app:not(.panel-open) .panel {
		opacity: 0;
		pointer-events: none;
		transform: translateX(calc(100% + 16px));
	}

	details {
		padding: 8px 0 12px;
	}

	details + details {
		border-top: 1px solid var(--line-strong);
	}

	summary {
		display: flex;
		align-items: center;
		justify-content: space-between;
		margin: 0 0 8px;
		padding: 4px 0;
		color: var(--ink);
		font-size: 11px;
		font-weight: 600;
		letter-spacing: 0.12em;
		text-transform: uppercase;
		cursor: pointer;
		list-style: none;
	}

	summary::-webkit-details-marker {
		display: none;
	}

	summary::after {
		color: var(--ink-faint);
		font-family: var(--font-mono);
		font-weight: 400;
		content: '−';
	}

	details:not([open]) summary {
		margin-bottom: 0;
	}

	details:not([open]) summary::after {
		content: '+';
	}

	.keys {
		margin-top: 8px;
		padding-top: 10px;
		border-top: 1px solid var(--line);
	}

	/* ------------------------------------------------------------------ timeline */

	.dock {
		position: absolute;
		right: 0;
		bottom: 0;
		left: 0;
	}

	/* ------------------------------------------------------------------ narrow */

	.narrow .bar {
		top: 8px;
		right: 8px;
		left: 8px;
	}

	.narrow .toggle {
		margin-left: auto;
	}

	.narrow .panel {
		top: auto;
		right: 0;
		bottom: var(--dock);
		left: 0;
		width: auto;
		max-height: 50dvh;
		padding: 2px 14px 10px;
		border-top: 1px solid var(--line-strong);
		border-radius: 0;
		background: var(--plate-strong);
	}

	.narrow:not(.panel-open) .panel {
		transform: translateY(calc(100% + var(--dock)));
	}

	.narrow .banner {
		top: 50px;
		left: 8px;
	}
</style>
