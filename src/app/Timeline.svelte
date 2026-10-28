<script>
	/**
	 * Playback of the trajectory: play / pause, the scrubber, the speed, and what the
	 * aircraft is doing at the current moment, read from the trajectory's own sample (the
	 * ground truth the batch writes), not from the picture.
	 *
	 * Headings and courses are compass degrees; the crab angle is heading − course,
	 * positive when the nose points right of the track (a wind from the right). Wind aloft
	 * is the air at the aircraft, gusts included, as the speed and the compass direction it
	 * blows toward.
	 */
	import { angleDiff, compass } from './state.js';

	/**
	 * @typedef {object} Props
	 * @property {number} t seconds since the start of the trajectory
	 * @property {number} duration s
	 * @property {boolean} playing
	 * @property {number} speed playback rate, × real time
	 * @property {import('../trajectory/index.js').TrajectorySample | null} sample at the readout's t
	 * @property {number | null} heightAboveSea the aircraft's height above the sea surface under it, m
	 * @property {(t: number) => void} onseek
	 */

	/** @type {Props} */
	let { t, duration, playing = $bindable(), speed = $bindable(), sample, heightAboveSea, onseek } = $props();

	const speeds = [0.5, 1, 2, 4, 8];

	/** @param {number} s */
	function clock(s) {
		const m = Math.floor(s / 60);
		const sec = s - m * 60;
		return `${m}:${sec.toFixed(1).padStart(4, '0')}`;
	}

	let crab = $derived(sample ? angleDiff(sample.heading, sample.course) : 0);
	let windSpeed = $derived(sample ? Math.hypot(sample.windE, sample.windN) : 0);
	let windToward = $derived(sample ? compass((Math.atan2(sample.windE, sample.windN) * 180) / Math.PI) : 0);
</script>

<div class="timeline">
	<div class="transport">
		<button class="play" onclick={() => (playing = !playing)} aria-label={playing ? 'Pause' : 'Play'} title="Play / pause (space)" data-testid="play">
			{#if playing}
				<svg viewBox="0 0 12 12" aria-hidden="true"><rect x="2" y="1.5" width="3" height="9" /><rect x="7" y="1.5" width="3" height="9" /></svg>
			{:else}
				<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M3 1.5v9l7.5-4.5z" /></svg>
			{/if}
		</button>
		<span class="time mono" data-testid="time">{clock(t)}<span class="of">&ensp;/&ensp;{clock(duration)}</span></span>
		<input
			class="scrub"
			type="range"
			min="0"
			max={duration}
			step="0.01"
			value={t}
			oninput={(e) => onseek(Number(e.currentTarget.value))}
			aria-label="Trajectory time"
			data-testid="scrub"
		/>
		<div class="seg speeds" role="group" aria-label="Playback speed">
			{#each speeds as s (s)}
				<button class:on={speed === s} onclick={() => (speed = s)}>{s}×</button>
			{/each}
		</div>
	</div>
	<dl class="readouts mono" data-testid="readouts">
		{#if sample}
			<div><dt>Alt</dt><dd>{sample.altitude.toFixed(1)}<small>m</small></dd></div>
			<div><dt>Above sea</dt><dd>{heightAboveSea === null ? '–' : heightAboveSea.toFixed(1)}<small>m</small></dd></div>
			<div><dt>Airspeed</dt><dd>{sample.airspeed.toFixed(1)}<small>m/s</small></dd></div>
			<div><dt>Ground</dt><dd>{sample.groundSpeed.toFixed(1)}<small>m/s</small></dd></div>
			<div><dt>Heading</dt><dd>{compass(sample.heading).toFixed(0).padStart(3, '0')}<small>°</small></dd></div>
			<div><dt>Course</dt><dd>{compass(sample.course).toFixed(0).padStart(3, '0')}<small>°</small></dd></div>
			<div><dt>Crab</dt><dd>{crab >= 0 ? '+' : ''}{crab.toFixed(1)}<small>°</small></dd></div>
			<div><dt>Bank</dt><dd>{sample.bank >= 0 ? '+' : ''}{sample.bank.toFixed(1)}<small>°</small></dd></div>
			<div><dt>Wind</dt><dd>{windSpeed.toFixed(1)}<small>m/s</small> → {windToward.toFixed(0).padStart(3, '0')}<small>°</small></dd></div>
		{:else}
			<div class="empty">Building the trajectory…</div>
		{/if}
	</dl>
</div>

<style>
	.timeline {
		display: grid;
		gap: 6px;
		padding: 10px 14px 10px;
		background: var(--plate);
		border-top: 1px solid var(--line);
	}

	.transport {
		display: flex;
		align-items: center;
		gap: 12px;
	}

	.play {
		display: grid;
		flex: none;
		width: 30px;
		height: 30px;
		padding: 0;
		border: 0;
		border-radius: 50%;
		background: var(--accent);
		place-items: center;
	}

	.play svg {
		width: 12px;
		height: 12px;
		fill: var(--accent-ink);
	}

	.time {
		flex: none;
		min-width: 92px;
		color: var(--ink);
		font-size: 12px;
	}

	.of {
		color: var(--ink-faint);
	}

	.scrub {
		flex: 1;
		min-width: 60px;
	}

	.speeds {
		flex: none;
	}

	.speeds button {
		padding: 3px 7px;
		font-family: var(--font-mono);
		font-size: 11px;
	}

	.readouts {
		display: flex;
		flex-wrap: wrap;
		gap: 2px 18px;
		margin: 0;
		font-size: 11.5px;
	}

	.readouts div {
		display: flex;
		align-items: baseline;
		gap: 6px;
	}

	.readouts dt {
		color: var(--ink-faint);
		font-family: var(--font-body);
		font-size: 10.5px;
		letter-spacing: 0.04em;
		text-transform: uppercase;
	}

	.readouts dd {
		margin: 0;
		color: var(--ink);
	}

	.readouts small {
		margin-left: 1px;
		color: var(--ink-faint);
		font-size: 10px;
	}

	.empty {
		color: var(--ink-faint);
	}

	@media (max-width: 760px) {
		.timeline {
			padding: 8px 10px;
		}

		.transport {
			flex-wrap: wrap;
			gap: 8px 10px;
		}

		.scrub {
			order: 3;
			flex-basis: 100%;
		}

		.speeds {
			margin-left: auto;
		}

		.readouts {
			gap: 1px 12px;
			font-size: 11px;
		}
	}
</style>
