<script>
	/**
	 * The simulated ocean behind every slide, with the UAV flying over it. The engine
	 * (three.js, GPU FFT) loads client-side only, by dynamic import, so the prerendered
	 * title slide never touches WebGL. Without WebGL2 or float render targets, or after a
	 * shader error or an unrecoverable context loss, a CSS sea stands in and the readout
	 * is filled from the same spectrum formulas, so slides still show their numbers.
	 */
	import { onMount } from 'svelte';
	import { channel } from '../ocean/channels.js';
	import { oceanControl, oceanReadout } from '../ocean/control.js';
	import { DEFAULT_TRANSITION, aircraftDefaults } from '../ocean/scene.js';
	import { seaStats } from '../ocean/spectrum.js';

	/**
	 * @typedef {object} Props
	 * @property {import('../ocean/scene.js').SceneSpec} scene
	 * @property {boolean} [interactive] let the presenter drag to look around
	 */

	/** @type {Props} */
	let { scene, interactive = false } = $props();

	/** @type {HTMLDivElement | undefined} */
	let root = $state();
	/** @type {HTMLCanvasElement | undefined} */
	let canvas = $state();
	/** @type {HTMLCanvasElement | undefined} */
	let hud = $state();

	/** @typedef {'loading' | 'running' | 'fallback' | 'error'} Status */

	let status = $state(/** @type {Status} */ ('loading'));
	let splitShown = $state('');
	/** @type {import('../ocean/engine/engine.js').OceanEngine | null} */
	let engine = $state.raw(null);
	let reducedMotion = $state(false);

	let mode = $derived(scene.mode ?? 'shaded');
	let splitMode = $derived(scene.split || oceanControl.splitMode ? (oceanControl.splitMode ?? scene.split?.mode ?? '') : '');
	let cameraMode = $derived(scene.camera.mode ?? 'free');
	let transition = $derived((scene.transition ?? DEFAULT_TRANSITION) * (reducedMotion ? 0.55 : 1));
	let scrim = $derived(scene.scrim ?? 'none');
	let fallback = $derived(status === 'fallback' || status === 'error');
	let chipLabel = $derived(splitShown ? channel(splitShown).label : '');

	// Start fetching the engine (and three.js) as soon as this module runs in the browser,
	// so the download overlaps hydration rather than waiting for onMount.
	/** @type {Promise<typeof import('../ocean/engine/engine.js')> | null} */
	const engineModule = typeof window === 'undefined' ? null : import('../ocean/engine/engine.js');
	// A failed fetch is handled where it is awaited; this only keeps it from being reported
	// as unhandled before then.
	engineModule?.catch(() => {});

	/** @param {Status} next */
	function setStatus(next) {
		status = next;
		oceanReadout.status = next;
	}

	onMount(() => {
		reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
		let cancelled = false;
		/** @type {ReturnType<typeof setTimeout> | undefined} */
		let lostTimer;

		async function start() {
			if (!canvas || !hud || !root) return;
			try {
				const { OceanEngine } = await (engineModule ?? import('../ocean/engine/engine.js'));
				if (cancelled) return;
				// Let the prerendered slide paint before the engine compiles its shaders,
				// which can hold the main thread for seconds on a cold shader cache.
				await afterPaint();
				if (cancelled || engine) return;
				/** @type {import('../ocean/engine/engine.js').OceanEngine | null} */
				let created = null;
				created = new OceanEngine(
					{
						canvas,
						hud,
						root,
						onStatus: (next) => {
							if (next === 'loading' && engine) {
								// Context lost: give the browser a few seconds to restore it.
								clearTimeout(lostTimer);
								lostTimer = setTimeout(() => toFallback(), 5000);
							}
							if (next === 'fallback' || next === 'error') {
								// A shader error or an exception mid-frame: free the engine (its
								// frame loop, its GPU memory, the camera drag's listeners and cursor), once
								// it is out of the frame or compile callback that reported it.
								queueMicrotask(() => {
									if (created && engine === created) toFallback();
								});
							}
							setStatus(next);
						},
						onSplitMode: (m) => (splitShown = m)
					},
					{ reducedMotion }
				);
				engine = created;
				// three.js listens for the context's restore on this canvas too, and has to
				// rebuild its own state before the old engine is let go (see onRestored), so
				// this listener goes after the renderer's.
				canvas.removeEventListener('webglcontextrestored', onRestored);
				canvas.addEventListener('webglcontextrestored', onRestored);
				created.setScene(scene);
				created.setInteractive(interactive);
			} catch (error) {
				console.warn('Mayfly ocean: falling back to the CSS sea:', error);
				toFallback();
			}
		}

		function toFallback() {
			engine?.dispose();
			engine = null;
			setStatus('fallback');
		}

		/** Two animation frames: the browser has painted what was on the page. */
		function afterPaint() {
			return new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
		}

		function onRestored() {
			clearTimeout(lostTimer);
			// The new engine draws into this same, restored context: keep it. The old one's
			// GPU objects went with the lost context, so it only lets go of its listeners.
			engine?.dispose(false);
			engine = null;
			setStatus('loading');
			void start();
		}

		canvas?.addEventListener('webglcontextrestored', onRestored);
		void start();

		return () => {
			cancelled = true;
			clearTimeout(lostTimer);
			canvas?.removeEventListener('webglcontextrestored', onRestored);
			engine?.dispose();
			engine = null;
		};
	});

	$effect(() => {
		engine?.setScene(scene);
	});

	$effect(() => {
		engine?.setInteractive(interactive);
	});

	// The fallback still answers the slides: the same spectrum formulas on the scene's sea.
	$effect(() => {
		if (!fallback) return;
		const sea = { ...scene.sea, ...oceanControl.sea };
		const stats = seaStats(sea);
		const air = scene.aircraft ? { ...aircraftDefaults, ...scene.aircraft } : null;
		const light = { ...scene.light, ...oceanControl.light };
		const cam = scene.camera;
		const onboard = cameraMode === 'onboard';
		Object.assign(oceanReadout, {
			windSpeed: sea.windSpeed,
			windDirection: sea.windDirection,
			significantHeight: stats.significantHeight,
			peakPeriod: stats.peakPeriod,
			peakWavelength: stats.peakWavelength,
			beaufort: stats.beaufort,
			sunElevation: light.sunElevation,
			haze: light.haze ?? 0.2,
			overcast: light.overcast ?? 0,
			cameraHeight: air && (onboard || !('height' in cam)) ? air.altitude : 'height' in cam ? cam.height : 0,
			fov: air && onboard ? (air.gimbal.fov ?? 48) : 'fov' in cam && cam.fov ? cam.fov : 50,
			roll: 0,
			pitch: air && onboard ? air.gimbal.pitch : 'pitch' in cam ? cam.pitch : 0,
			heave: 0,
			aircraft: Boolean(air),
			aircraftAltitude: air?.altitude ?? 0,
			aircraftSpeed: air?.speed ?? 0,
			aircraftHeading: air?.heading ?? 0,
			aircraftBank: 0,
			aircraftPitch: 0,
			cameraMode,
			trackTime: -1,
			trackGap: -1,
			sample: 0,
			mode,
			fps: 0
		});
		/** @type {any} */ (window).__mayfly = {
			readout: { ...oceanReadout },
			stats: { frames: 0, fps: 0, renderScale: 0, cascades: 0, resolution: 0 }
		};
	});
</script>

<div
	class="ocean-stage"
	bind:this={root}
	data-ocean-status={status}
	data-mode={mode}
	data-split-mode={splitMode}
	data-camera-mode={cameraMode}
	data-interactive={interactive}
	style:--transition="{transition}s"
>
	<canvas class="sea" bind:this={canvas} class:hidden={fallback} aria-hidden="true"></canvas>

	{#if fallback}
		<div class={['fallback', reducedMotion && 'still']} aria-hidden="true">
			<div class="sky"></div>
			<div class="sun"></div>
			<div class="water">
				<div class="swell a"></div>
				<div class="swell b"></div>
				<div class="swell c"></div>
				<div class="glint"></div>
			</div>
			<div class="horizon"></div>
		</div>
	{/if}

	<canvas class="hud" bind:this={hud} aria-hidden="true"></canvas>

	<div class="scrim left" class:on={scrim === 'left'} aria-hidden="true"></div>
	<div class="scrim right" class:on={scrim === 'right'} aria-hidden="true"></div>
	<div class="scrim bottom" class:on={scrim === 'bottom'} aria-hidden="true"></div>
	<div class="dim" style:opacity={scene.dim ?? 0} aria-hidden="true"></div>

	{#if !fallback}
		<div class="split-chip" class:on={splitShown !== ''} aria-hidden="true">
			<span class="dot"></span>{chipLabel}
		</div>
	{/if}
</div>

<style>
	.ocean-stage {
		position: absolute;
		inset: 0;
		overflow: hidden;
		background: linear-gradient(180deg, #24435d 0%, #10293d 48%, #06213a 52%, #020b14 100%);
	}

	canvas {
		position: absolute;
		inset: 0;
		display: block;
		width: 100%;
		height: 100%;
	}

	.sea,
	.hud {
		pointer-events: none;
	}

	.hidden {
		visibility: hidden;
	}

	.scrim,
	.dim {
		position: absolute;
		inset: 0;
		pointer-events: none;
		opacity: 0;
		transition: opacity var(--transition) ease;
	}

	.scrim.on {
		opacity: 1;
	}

	.scrim.left {
		background: linear-gradient(
			90deg,
			rgba(2, 9, 16, 0.78) 0%,
			rgba(2, 9, 16, 0.55) 28%,
			rgba(2, 9, 16, 0.18) 52%,
			rgba(2, 9, 16, 0) 68%
		);
	}

	.scrim.right {
		background: linear-gradient(
			270deg,
			rgba(2, 9, 16, 0.78) 0%,
			rgba(2, 9, 16, 0.55) 28%,
			rgba(2, 9, 16, 0.18) 52%,
			rgba(2, 9, 16, 0) 68%
		);
	}

	.scrim.bottom {
		background: linear-gradient(
			0deg,
			rgba(2, 9, 16, 0.86) 0%,
			rgba(2, 9, 16, 0.68) 28%,
			rgba(2, 9, 16, 0.32) 45%,
			rgba(2, 9, 16, 0) 64%
		);
	}

	.dim {
		background: #02080f;
	}

	.split-chip {
		position: absolute;
		top: clamp(4.6rem, 12vh, 7.4rem);
		left: var(--split-x, 50%);
		display: inline-flex;
		align-items: center;
		gap: 0.55em;
		padding: 0.4em 0.75em 0.34em 0.65em;
		border: 0;
		border-radius: 2px;
		background: var(--panel-strong, rgba(6, 12, 18, 0.96));
		color: var(--ink, #eaeeef);
		font-family: var(--font-label, 'Arial Narrow', sans-serif);
		font-size: var(--label-size, 0.8rem);
		font-weight: 500;
		letter-spacing: 0.09em;
		text-transform: uppercase;
		white-space: nowrap;
		pointer-events: none;
		transform: translateX(0.9rem);
		opacity: 0;
	}

	.split-chip.on {
		opacity: var(--split-o, 0);
	}

	/* On a phone the slide's panel covers the top of the frame, where the chip would sit
	   behind it; the panel names the channel in large type there anyway. */
	@media (max-width: 900px) {
		.split-chip {
			display: none;
		}
	}

	.dot {
		width: 0.5em;
		height: 0.5em;
		border-radius: 1px;
		background: var(--sea, #4fd8e6);
	}

	/* ------------------------------------------------------------ CSS fallback sea */

	.fallback {
		position: absolute;
		inset: 0;
		overflow: hidden;
	}

	.fallback .sky {
		position: absolute;
		inset: 0 0 52% 0;
		background: linear-gradient(180deg, #1d3d58 0%, #3b6688 55%, #9db8c9 100%);
	}

	.fallback .sun {
		position: absolute;
		top: 30%;
		left: 66%;
		width: 26vmin;
		height: 26vmin;
		border-radius: 50%;
		background: radial-gradient(
			circle,
			rgba(255, 236, 200, 0.75) 0%,
			rgba(255, 210, 150, 0.18) 30%,
			transparent 70%
		);
		transform: translate(-50%, -50%);
	}

	.fallback .water {
		position: absolute;
		inset: 48% 0 0 0;
		overflow: hidden;
		background: linear-gradient(180deg, #4a6f88 0%, #1d4561 12%, #0c2c45 45%, #041626 100%);
		perspective: 40vh;
	}

	.fallback .swell {
		position: absolute;
		inset: -20% -50% -60% -50%;
		opacity: 0.55;
		transform: rotateX(72deg);
		transform-origin: 50% 0;
		animation: roll 14s linear infinite;
	}

	.fallback .swell.a {
		background: repeating-linear-gradient(
			178deg,
			rgba(160, 205, 230, 0) 0px,
			rgba(160, 205, 230, 0.16) 22px,
			rgba(160, 205, 230, 0) 46px
		);
	}

	.fallback .swell.b {
		background: repeating-linear-gradient(
			183deg,
			rgba(4, 20, 34, 0) 0px,
			rgba(4, 20, 34, 0.35) 31px,
			rgba(4, 20, 34, 0) 64px
		);
		animation-duration: 21s;
	}

	.fallback .swell.c {
		background: repeating-linear-gradient(
			174deg,
			rgba(200, 230, 245, 0) 0px,
			rgba(200, 230, 245, 0.08) 9px,
			rgba(200, 230, 245, 0) 19px
		);
		animation-duration: 9s;
	}

	.fallback .glint {
		position: absolute;
		top: 0;
		left: 66%;
		width: 34vmin;
		height: 100%;
		background: radial-gradient(ellipse 50% 70% at 50% 0%, rgba(255, 232, 196, 0.35), transparent 70%);
		mix-blend-mode: screen;
		transform: translateX(-50%);
		animation: shimmer 4.5s ease-in-out infinite alternate;
	}

	.fallback .horizon {
		position: absolute;
		top: 48%;
		right: 0;
		left: 0;
		height: 6vh;
		background: linear-gradient(180deg, rgba(190, 215, 228, 0.45), rgba(190, 215, 228, 0));
		filter: blur(6px);
		transform: translateY(-50%);
	}

	.fallback.still .swell,
	.fallback.still .glint {
		animation: none;
	}

	@keyframes roll {
		from {
			background-position: 0 0;
		}

		to {
			background-position: 0 640px;
		}
	}

	@keyframes shimmer {
		from {
			opacity: 0.55;
		}

		to {
			opacity: 1;
		}
	}
</style>
