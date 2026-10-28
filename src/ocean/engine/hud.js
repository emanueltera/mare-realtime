/**
 * The `horizon` overlay: a minimal instrument HUD on a 2D canvas over the sea. The
 * horizon line is the true one — the geometric horizon below the camera's actual
 * attitude, dipped by the camera's height above the water (about 0.38° at 140 m) — so
 * it lies on the rendered horizon. Around it: a pitch ladder, a bank scale with its
 * pointer, a fixed miniature-aircraft reference at the boresight and the numbers. The
 * roll and pitch are the camera's (its gimbal is only partly stabilised), labelled so,
 * since the app shows the airframe's own attitude beside them.
 */
import * as THREE from 'three';
import { DEG, clamp } from './util.js';

const ACCENT = '79, 216, 230';
const _v = new THREE.Vector3();
const _qi = new THREE.Quaternion();

/**
 * @typedef {object} HudFrame
 * @property {number} width CSS px
 * @property {number} height CSS px
 * @property {number} dpr
 * @property {number} alpha 0–1
 * @property {THREE.Quaternion} quaternion camera orientation
 * @property {number} fov vertical degrees
 * @property {number} pitch degrees, the camera's
 * @property {number} roll degrees, right side down positive
 * @property {number} heading compass degrees
 * @property {number} dip horizon dip, degrees
 * @property {number} altitude m above the water
 * @property {number} [airspeed] m/s
 */

export class HorizonHud {
	/** @param {HTMLCanvasElement} canvas */
	constructor(canvas) {
		this.canvas = canvas;
		this.ctx = canvas.getContext('2d');
		this.blank = true;
		this.font = getComputedStyle(canvas).getPropertyValue('--font-mono').trim() || 'ui-monospace, monospace';
	}

	clear() {
		if (this.blank || !this.ctx) return;
		this.ctx.setTransform(1, 0, 0, 1, 0, 0);
		this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
		this.blank = true;
	}

	/** @param {HudFrame} f */
	draw(f) {
		const ctx = this.ctx;
		if (!ctx) return;
		const W = Math.round(f.width * f.dpr);
		const H = Math.round(f.height * f.dpr);
		if (this.canvas.width !== W || this.canvas.height !== H) {
			this.canvas.width = W;
			this.canvas.height = H;
		}
		ctx.setTransform(1, 0, 0, 1, 0, 0);
		ctx.clearRect(0, 0, W, H);
		this.blank = false;
		if (f.alpha <= 0.002) return;
		ctx.setTransform(f.dpr, 0, 0, f.dpr, 0, 0);
		const w = f.width;
		const h = f.height;
		const tanV = Math.tan((f.fov * DEG) / 2);
		const aspect = w / h;
		_qi.copy(f.quaternion).invert();

		/**
		 * Screen position of a world direction at compass bearing az and elevation el.
		 * @param {number} az
		 * @param {number} el
		 * @returns {[number, number] | null}
		 */
		const project = (az, el) => {
			const ce = Math.cos(el * DEG);
			_v.set(ce * Math.sin(az * DEG), Math.sin(el * DEG), -ce * Math.cos(az * DEG)).applyQuaternion(_qi);
			if (_v.z > -1e-3) return null;
			const x = _v.x / -_v.z / (tanV * aspect);
			const y = _v.y / -_v.z / tanV;
			return [(x * 0.5 + 0.5) * w, (0.5 - y * 0.5) * h];
		};

		const a = f.alpha;
		const cx = w / 2;
		const cy = h / 2;
		const unit = clamp(Math.min(w, h) / 900, 0.7, 1.6);
		// Text never shrinks below its size on a 900 px tall screen: it has to read on a
		// projector at the back of a room.
		const text = Math.max(1, unit);
		ctx.lineCap = 'round';
		ctx.shadowColor = 'rgba(2, 10, 18, 0.55)';
		ctx.shadowBlur = 3 * unit;

		// Pitch ladder: lines parallel to the horizon every 5°, labelled every 10°.
		const ladder = [-20, -15, -10, -5, 5, 10, 15, 20];
		ctx.font = `${12 * text}px ${this.font}`;
		ctx.textBaseline = 'middle';
		for (const step of ladder) {
			const el = -f.dip + step;
			const p0 = project(f.heading - 4.2, el);
			const p1 = project(f.heading + 4.2, el);
			if (!p0 || !p1) continue;
			const major = step % 10 === 0;
			ctx.strokeStyle = `rgba(${ACCENT}, ${a * (major ? 0.62 : 0.38)})`;
			ctx.lineWidth = unit;
			ctx.setLineDash(step < 0 ? [5 * unit, 4 * unit] : []);
			const dx = p1[0] - p0[0];
			const dy = p1[1] - p0[1];
			const span = major ? 1 : 0.55;
			const mx = (p0[0] + p1[0]) / 2;
			const my = (p0[1] + p1[1]) / 2;
			ctx.beginPath();
			ctx.moveTo(mx - (dx / 2) * span, my - (dy / 2) * span);
			ctx.lineTo(mx - (dx / 2) * span * 0.32, my - (dy / 2) * span * 0.32);
			ctx.moveTo(mx + (dx / 2) * span * 0.32, my + (dy / 2) * span * 0.32);
			ctx.lineTo(mx + (dx / 2) * span, my + (dy / 2) * span);
			ctx.stroke();
			if (major) {
				ctx.fillStyle = `rgba(${ACCENT}, ${a * 0.85})`;
				ctx.textAlign = 'left';
				ctx.fillText(String(Math.abs(step)), mx + (dx / 2) * span + 6 * unit, my + (dy / 2) * span);
			}
		}
		ctx.setLineDash([]);

		// The true horizon, across the frame with a gap around the reference symbol.
		const h0 = project(f.heading - 30, -f.dip);
		const h1 = project(f.heading + 30, -f.dip);
		if (h0 && h1) {
			const dx = h1[0] - h0[0];
			const dy = h1[1] - h0[1];
			const len = Math.hypot(dx, dy) || 1;
			const ux = dx / len;
			const uy = dy / len;
			// The point on the line nearest the frame centre.
			const t = (cx - h0[0]) * ux + (cy - h0[1]) * uy;
			const px = h0[0] + ux * t;
			const py = h0[1] + uy * t;
			const reach = Math.hypot(w, h);
			// Leave room round the reference symbol only when the horizon runs past it.
			const offset = Math.hypot(px - cx, py - cy);
			const gap = 92 * unit * (1 - Math.min(1, Math.max(0, (offset - 30 * unit) / (60 * unit))));
			// Faded out toward the frame edges, and across the left third on a wide
			// frame, which is kept clear for whatever the page lays over it there.
			const edge = ctx.createLinearGradient(0, 0, w, 0);
			const left = w > 900 ? 0.3 : 0.02;
			edge.addColorStop(left, `rgba(${ACCENT}, 0)`);
			edge.addColorStop(Math.min(left + 0.16, 0.5), `rgba(${ACCENT}, ${a * 0.9})`);
			edge.addColorStop(0.88, `rgba(${ACCENT}, ${a * 0.9})`);
			edge.addColorStop(0.99, `rgba(${ACCENT}, 0)`);
			ctx.strokeStyle = edge;
			ctx.lineWidth = 1.4 * unit;
			ctx.beginPath();
			ctx.moveTo(px - ux * reach, py - uy * reach);
			ctx.lineTo(px - ux * gap, py - uy * gap);
			ctx.moveTo(px + ux * gap, py + uy * gap);
			ctx.lineTo(px + ux * reach, py + uy * reach);
			ctx.stroke();
			// Ticks hanging off the horizon every 10° of bearing.
			ctx.lineWidth = unit;
			ctx.globalAlpha = 0.55;
			ctx.strokeStyle = edge;
			for (let b = -60; b <= 60; b += 10) {
				if (b === 0) continue;
				const q = project(f.heading + b, -f.dip);
				if (!q) continue;
				ctx.beginPath();
				ctx.moveTo(q[0] + uy * 0, q[1]);
				ctx.lineTo(q[0] - uy * 7 * unit, q[1] + ux * 7 * unit);
				ctx.stroke();
			}
			ctx.globalAlpha = 1;
		}

		// Fixed miniature aircraft at the boresight.
		ctx.strokeStyle = `rgba(${ACCENT}, ${a})`;
		ctx.lineWidth = 2 * unit;
		ctx.beginPath();
		const s = 1.05 * unit;
		ctx.moveTo(cx - 52 * s, cy);
		ctx.lineTo(cx - 20 * s, cy);
		ctx.lineTo(cx - 10 * s, cy + 9 * s);
		ctx.lineTo(cx, cy);
		ctx.lineTo(cx + 10 * s, cy + 9 * s);
		ctx.lineTo(cx + 20 * s, cy);
		ctx.lineTo(cx + 52 * s, cy);
		ctx.stroke();
		ctx.beginPath();
		ctx.arc(cx, cy, 1.6 * unit, 0, Math.PI * 2);
		ctx.fillStyle = `rgba(${ACCENT}, ${a})`;
		ctx.fill();

		// Bank scale: an arc over the symbol, ticks at 10/20/30/45/60, pointer at the roll.
		const R = 132 * unit;
		const arcCy = cy;
		ctx.lineWidth = unit;
		ctx.strokeStyle = `rgba(${ACCENT}, ${a * 0.55})`;
		ctx.beginPath();
		ctx.arc(cx, arcCy, R, (-90 - 62) * DEG, (-90 + 62) * DEG);
		ctx.stroke();
		for (const t of [-60, -45, -30, -20, -10, 0, 10, 20, 30, 45, 60]) {
			const ang = (-90 + t) * DEG;
			const len = t === 0 ? 11 : Math.abs(t) % 30 === 0 ? 9 : 5;
			ctx.beginPath();
			ctx.moveTo(cx + Math.cos(ang) * R, arcCy + Math.sin(ang) * R);
			ctx.lineTo(cx + Math.cos(ang) * (R - len * unit), arcCy + Math.sin(ang) * (R - len * unit));
			ctx.stroke();
		}
		const rollAng = (-90 - clamp(f.roll, -65, 65)) * DEG;
		const tip = [cx + Math.cos(rollAng) * (R + 2 * unit), arcCy + Math.sin(rollAng) * (R + 2 * unit)];
		const back = R + 11 * unit;
		const side = 4.5 * unit;
		const nx = -Math.sin(rollAng);
		const ny = Math.cos(rollAng);
		ctx.fillStyle = `rgba(${ACCENT}, ${a * 0.95})`;
		ctx.beginPath();
		ctx.moveTo(tip[0], tip[1]);
		ctx.lineTo(cx + Math.cos(rollAng) * back + nx * side, arcCy + Math.sin(rollAng) * back + ny * side);
		ctx.lineTo(cx + Math.cos(rollAng) * back - nx * side, arcCy + Math.sin(rollAng) * back - ny * side);
		ctx.closePath();
		ctx.fill();

		// Numbers, up and right of the boresight so the left of the frame stays clear, on a
		// soft dark backing so they read over whitecaps. On a phone-width screen a panel
		// over the page covers the whole frame and the block would show through it, so it
		// is left out there.
		if (w < 720) return;
		ctx.textAlign = 'left';
		ctx.font = `${14 * text}px ${this.font}`;
		const fmt = (/** @type {number} */ v, /** @type {number} */ d = 1) => {
			const r = Number(v.toFixed(d)) || 0;
			return `${r < 0 ? '−' : r > 0 ? '+' : ' '}${Math.abs(r).toFixed(d)}`;
		};
		const rows = [
			['CAM ROLL', `${fmt(f.roll)}°`],
			['CAM PITCH', `${fmt(f.pitch)}°`],
			['DIP', `${f.dip.toFixed(2)}°`],
			['ALT', `${Math.round(f.altitude)} m`]
		];
		if (f.airspeed !== undefined) rows.push(['IAS', `${f.airspeed.toFixed(0)} m/s`]);
		const labelWidth = Math.max(...rows.map(([label]) => ctx.measureText(label).width));
		const valueWidth = Math.max(...rows.map(([, value]) => ctx.measureText(value).width));
		const gap = 14 * text;
		const line = 19 * text;
		const pad = 10 * text;
		const blockW = labelWidth + gap + valueWidth;
		const blockH = (rows.length - 1) * line;
		const x0 = Math.min(cx + R + 34 * unit, w - blockW - pad - 12);
		// Its last row level with the boresight, so it stays clear of a card below it.
		let y0 = cy - blockH;
		ctx.shadowBlur = 0;
		ctx.fillStyle = `rgba(3, 11, 20, ${a * 0.42})`;
		ctx.beginPath();
		ctx.roundRect(x0 - pad, y0 - line / 2 - pad * 0.4, blockW + pad * 2, blockH + line + pad * 0.8, 6 * text);
		ctx.fill();
		ctx.shadowColor = 'rgba(2, 10, 18, 0.8)';
		ctx.shadowBlur = 4 * text;
		for (const [label, value] of rows) {
			ctx.fillStyle = `rgba(${ACCENT}, ${a * 0.8})`;
			ctx.fillText(label, x0, y0);
			ctx.fillStyle = `rgba(${ACCENT}, ${a})`;
			ctx.fillText(value, x0 + labelWidth + gap, y0);
			y0 += line;
		}
		ctx.shadowBlur = 0;
	}
}
