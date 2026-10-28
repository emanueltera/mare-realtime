/**
 * Quality presets and the adaptive controller. `?quality=low|high` in the URL pins a
 * preset; otherwise the engine starts high, within a pixel budget (so a 2× screen the
 * size of a 16-inch laptop or a 4K monitor starts a little under 2×), and if frames
 * keep missing 60 Hz for two seconds it steps the render scale down (to 0.6 at least),
 * then the FFT size to 128², then drops a cascade. It never steps back up within a
 * session, so it cannot oscillate.
 */

/**
 * @typedef {object} QualityPreset
 * @property {number} fftSize
 * @property {number} cascades
 * @property {number} renderScale
 * @property {number} maxDpr
 * @property {number} foamSize resolution of each world-space foam field
 */

/** @type {Record<'high' | 'low', QualityPreset>} */
export const PRESETS = {
	high: { fftSize: 256, cascades: 3, renderScale: 1, maxDpr: 2, foamSize: 1024 },
	low: { fftSize: 256, cascades: 2, renderScale: 0.75, maxDpr: 1.5, foamSize: 512 }
};

/** @returns {'high' | 'low' | null} */
export function qualityFromUrl() {
	if (typeof location === 'undefined') return null;
	const q = new URLSearchParams(location.search).get('quality');
	return q === 'low' || q === 'high' ? q : null;
}

const SCALES = [1, 0.85, 0.72, 0.6];

/**
 * Device pixels the unpinned engine draws at most: a full 2× on a laptop screen up to
 * about 1700 × 880 CSS px, a little less on bigger ones (about 1.7× at 1920 × 1080).
 */
const PIXEL_BUDGET = 6e6;

/** A frame longer than this missed a 60 Hz refresh (with room for a 50 Hz projector). */
const LATE_FRAME = 0.022;

/** Share of late frames, over about a second, that counts as stuttering. */
const LATE_SHARE = 0.1;

export class AdaptiveQuality {
	/**
	 * @param {QualityPreset} preset
	 * @param {boolean} pinned when the URL chose the preset, never adapt
	 */
	constructor(preset, pinned) {
		this.preset = { ...preset };
		this.pinned = pinned;
		this.scaleIndex = Math.max(0, SCALES.findIndex((s) => s <= preset.renderScale));
		this.slowFor = 0;
		this.grace = 3;
		this.fps = 60;
		/** Moving share of frames that missed 60 Hz, over about a second. */
		this.late = 0;
	}

	get renderScale() {
		return this.pinned ? this.preset.renderScale : Math.min(this.preset.renderScale, SCALES[this.scaleIndex]);
	}

	/**
	 * Drawing-buffer pixels per CSS pixel for a stage of this size: the screen's, capped
	 * by the preset and (unless pinned) the pixel budget, times the render scale.
	 * @param {number} width CSS px
	 * @param {number} height CSS px
	 */
	pixelRatio(width, height) {
		const screen = typeof window === 'undefined' ? 1 : window.devicePixelRatio || 1;
		let dpr = Math.min(screen, this.preset.maxDpr);
		if (!this.pinned) dpr = Math.min(dpr, Math.max(1, Math.sqrt(PIXEL_BUDGET / Math.max(1, width * height))));
		return dpr * this.renderScale;
	}

	/** Give the frame rate a few seconds to settle, e.g. after a resize or a slide change. */
	settle(seconds = 2.5) {
		this.grace = Math.max(this.grace, seconds);
		this.slowFor = 0;
		this.late = 0;
	}

	/**
	 * Feed a frame. Returns 'scale', 'fft' or 'cascades' when the engine should step down:
	 * first the render scale (to 0.6), then the FFT size (to 128²), then a cascade.
	 *
	 * Slow is frames missing 60 Hz, not only a low average: a deck alternating 60 and
	 * 30 Hz frames averages 45–50 fps and judders on every moving camera.
	 * @param {number} dt seconds (unscaled)
	 * @param {number} fps the mean frame rate (one over the mean frame time)
	 * @returns {'scale' | 'fft' | 'cascades' | null}
	 */
	frame(dt, fps) {
		this.fps = fps;
		if (this.pinned) return null;
		if (this.grace > 0) {
			this.grace -= dt;
			return null;
		}
		this.late += ((dt > LATE_FRAME ? 1 : 0) - this.late) * (1 - Math.exp(-dt));
		const slow = this.late > LATE_SHARE || fps < 45;
		this.slowFor = slow ? this.slowFor + dt : Math.max(0, this.slowFor - dt * 0.5);
		if (this.slowFor < 2) return null;
		this.slowFor = 0;
		this.late = 0;
		this.grace = 1.5;
		if (this.scaleIndex < SCALES.length - 1) {
			// Straight to the scale the frame rate suggests (the cost goes with the pixel
			// count, the square of the scale), so a slow machine stutters through one or two
			// steps, not four.
			const wanted = SCALES[this.scaleIndex] * Math.sqrt(Math.min(1, fps / 60));
			let next = this.scaleIndex + 1;
			while (next < SCALES.length - 1 && SCALES[next] > wanted) next++;
			this.scaleIndex = next;
			return 'scale';
		}
		if (this.preset.fftSize > 128) {
			this.preset.fftSize = 128;
			return 'fft';
		}
		if (this.preset.cascades > 2) {
			this.preset.cascades--;
			return 'cascades';
		}
		return null;
	}
}
