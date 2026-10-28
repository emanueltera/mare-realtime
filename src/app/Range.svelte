<script>
	/**
	 * One numeric field: a label, a slider and the value as an editable number. The
	 * slider and the number edit the same value; a typed number is clamped to the range
	 * when it is committed (on change), not on every keystroke.
	 */

	/**
	 * @typedef {object} Props
	 * @property {string} label
	 * @property {number} value
	 * @property {number} min
	 * @property {number} max
	 * @property {number} [step]
	 * @property {string} [unit]
	 * @property {string} [title] what the field is, for the tooltip
	 * @property {(value: number) => void} [onchange] called after the value changes
	 */

	/** @type {Props} */
	let { label, value = $bindable(), min, max, step = 1, unit = '', title, onchange } = $props();

	const id = $props.id();
	let digits = $derived(step >= 1 ? 0 : Math.min(3, Math.ceil(-Math.log10(step) - 1e-9)));

	/** @param {number} v */
	function set(v) {
		if (!Number.isFinite(v)) return;
		const next = Math.min(max, Math.max(min, v));
		value = Number(next.toFixed(digits + 2));
		onchange?.(value);
	}
</script>

<div class="range" {title}>
	<label for={id}>{label}</label>
	<input {id} type="range" {min} {max} {step} value={value} oninput={(e) => set(Number(e.currentTarget.value))} />
	<span class="num">
		<input
			type="number"
			aria-label={label}
			{min}
			{max}
			{step}
			value={Number((value ?? min).toFixed(digits))}
			onchange={(e) => set(Number(e.currentTarget.value))}
		/>{#if unit}<span class="unit">{unit}</span>{/if}
	</span>
</div>

<style>
	.range {
		display: grid;
		align-items: center;
		gap: 8px;
		grid-template-columns: var(--label-w, 92px) minmax(0, 1fr) 76px;
		min-height: 26px;
	}

	label {
		overflow: hidden;
		color: var(--ink-dim);
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.num {
		display: flex;
		align-items: baseline;
		justify-content: flex-end;
		gap: 3px;
	}

	.num input {
		width: 100%;
		min-width: 0;
		padding: 2px 0;
		border: 0;
		border-bottom: 1px solid transparent;
		background: transparent;
		color: var(--ink);
		font: inherit;
		font-family: var(--font-mono);
		font-size: 11.5px;
		text-align: right;
		-moz-appearance: textfield;
		appearance: textfield;
	}

	.num input::-webkit-inner-spin-button,
	.num input::-webkit-outer-spin-button {
		margin: 0;
		-webkit-appearance: none;
	}

	.num input:hover,
	.num input:focus {
		border-bottom-color: var(--line-strong);
		outline: none;
	}

	.unit {
		flex: none;
		color: var(--ink-faint);
		font-family: var(--font-mono);
		font-size: 10.5px;
	}
</style>
