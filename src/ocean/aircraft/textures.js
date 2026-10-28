/**
 * Small CanvasTextures for the UAV: fuselage panel lines (bump + a whisper of colour),
 * the tail marking, a soft glow for the lights and the propeller's motion-blur
 * disc. Created per model and disposed with it. Client-side only (needs a canvas).
 */
import * as THREE from 'three';

/** @param {number} w @param {number} h */
function canvas(w, h) {
	const c = document.createElement('canvas');
	c.width = w;
	c.height = h;
	const ctx = c.getContext('2d');
	if (!ctx) throw new Error('aircraft: no 2D canvas context');
	return { c, ctx };
}

/**
 * Fuselage maps. UVs: u runs round the body (0 underneath, 0.25 right side, 0.5 on top,
 * 0.75 left side), v along it from nose (0) to tail (1).
 * @param {{ zNose: number, zTail: number }} body
 */
export function fuselageMaps({ zNose, zTail }) {
	const W = 1024;
	const H = 1024;
	/** @param {number} z */
	const Y = (z) => (1 - (z - zNose) / (zTail - zNose)) * H;
	/** @param {number} u */
	const X = (u) => u * W;

	const bump = canvas(W, H);
	const colour = canvas(W, H);
	const b = bump.ctx;
	const k = colour.ctx;
	b.fillStyle = '#808080';
	b.fillRect(0, 0, W, H);
	k.fillStyle = '#ffffff';
	k.fillRect(0, 0, W, H);

	/** @param {(ctx: CanvasRenderingContext2D) => void} draw @param {string} bumpStyle @param {string} colourStyle */
	const both = (draw, bumpStyle, colourStyle) => {
		b.strokeStyle = bumpStyle;
		b.fillStyle = bumpStyle;
		k.strokeStyle = colourStyle;
		k.fillStyle = colourStyle;
		draw(b);
		draw(k);
	};

	// Circumferential joints: nose cone, payload bay, tail cone.
	both(
		(ctx) => {
			ctx.lineWidth = 2.2;
			for (const z of [-0.6, -0.17, 0.24, 0.42]) {
				ctx.beginPath();
				ctx.moveTo(0, Y(z));
				ctx.lineTo(W, Y(z));
				ctx.stroke();
			}
		},
		'#3a3a3a',
		'#c9cdd1'
	);

	// Access hatches: one on top ahead of the wing, one per side aft of it.
	/** @param {number} u0 @param {number} u1 @param {number} z0 @param {number} z1 */
	const hatch = (u0, u1, z0, z1) =>
		both(
			(ctx) => {
				ctx.lineWidth = 2;
				const x = X(u0);
				const y = Y(z1);
				const w = X(u1) - X(u0);
				const h = Y(z0) - Y(z1);
				ctx.beginPath();
				ctx.roundRect(x, y, w, h, 10);
				ctx.stroke();
				// fasteners
				for (const [fx, fy] of [
					[x + 7, y + 7],
					[x + w - 7, y + 7],
					[x + 7, y + h - 7],
					[x + w - 7, y + h - 7]
				]) {
					ctx.beginPath();
					ctx.arc(fx, fy, 2.2, 0, Math.PI * 2);
					ctx.fill();
				}
			},
			'#404040',
			'#c4c8cc'
		);
	hatch(0.42, 0.58, -0.52, -0.24);
	hatch(0.19, 0.31, 0.26, 0.38);
	hatch(0.69, 0.81, 0.26, 0.38);

	// Cooling louvres low on each side near the engine.
	for (const u of [0.17, 0.83]) {
		for (let i = 0; i < 5; i++) {
			const z = 0.3 + i * 0.016;
			b.fillStyle = '#202020';
			b.fillRect(X(u) - 18, Y(z) - 2, 36, 4);
			k.fillStyle = '#4b5056';
			k.fillRect(X(u) - 18, Y(z) - 2, 36, 4);
		}
	}

	// Fastener rows along the payload-bay joints.
	both(
		(ctx) => {
			for (const z of [-0.6, -0.17, 0.24]) {
				for (let u = 0.02; u < 1; u += 0.025) {
					ctx.beginPath();
					ctx.arc(X(u), Y(z) + 7, 1.5, 0, Math.PI * 2);
					ctx.fill();
				}
			}
		},
		'#555555',
		'#d4d7da'
	);

	const bumpMap = new THREE.CanvasTexture(bump.c);
	bumpMap.colorSpace = THREE.NoColorSpace;
	bumpMap.anisotropy = 4;
	const map = new THREE.CanvasTexture(colour.c);
	map.colorSpace = THREE.SRGBColorSpace;
	map.anisotropy = 4;
	return { bumpMap, map };
}

/**
 * The tail marking: a hull code ("UAV-01") in small dark letters on the outer face of a tail panel.
 * UVs on a lifting surface: u runs round the airfoil (0 at the trailing edge over the
 * upper/outer skin to 0.5 at the leading edge), v along the span from the apex (0) to
 * the boom (1).
 * @param {boolean} mirrored the left panel is the mirror image of the right, so its
 *   lettering is drawn mirrored to read correctly from outside
 */
export function tailMarking(mirrored) {
	const W = 1024;
	const H = 512;
	const { c, ctx } = canvas(W, H);
	ctx.fillStyle = '#ffffff';
	ctx.fillRect(0, 0, W, H);
	ctx.save();
	// Lettering reads leading edge → trailing edge, i.e. toward decreasing u, with its
	// baseline toward the boom (larger v).
	const cx = 0.335 * W;
	const cy = (1 - 0.56) * H;
	ctx.translate(cx, cy);
	ctx.scale(mirrored ? -1 : 1, -1);
	ctx.fillStyle = '#4a535c';
	ctx.font = '500 40px "Archivo Narrow Variable", "Arial Narrow", "Inter Variable", system-ui, sans-serif';
	ctx.textAlign = 'center';
	ctx.textBaseline = 'middle';
	if ('letterSpacing' in ctx) /** @type {any} */ (ctx).letterSpacing = '7px';
	ctx.fillText('UAV-01', 0, 0);
	ctx.fillRect(-118, 28, 236, 3);
	ctx.restore();
	const map = new THREE.CanvasTexture(c);
	map.colorSpace = THREE.SRGBColorSpace;
	map.anisotropy = 8;
	return map;
}

/** A soft round glow with a hot core, for the light sprites. */
export function glowTexture() {
	const S = 128;
	const { c, ctx } = canvas(S, S);
	const g = ctx.createRadialGradient(S / 2, S / 2, 0, S / 2, S / 2, S / 2);
	g.addColorStop(0, 'rgba(255,255,255,1)');
	g.addColorStop(0.1, 'rgba(255,255,255,0.92)');
	g.addColorStop(0.28, 'rgba(255,255,255,0.32)');
	g.addColorStop(0.6, 'rgba(255,255,255,0.06)');
	g.addColorStop(1, 'rgba(255,255,255,0)');
	ctx.fillStyle = g;
	ctx.fillRect(0, 0, S, S);
	const map = new THREE.CanvasTexture(c);
	map.colorSpace = THREE.SRGBColorSpace;
	return map;
}

/**
 * The propeller's motion blur: a dark translucent disc, densest where the blades are
 * widest, with the faint ring of the painted tips at the rim. Planar UVs over a disc
 * of the tip radius (as RingGeometry and CircleGeometry give).
 * @param {number} inner hub radius as a fraction of the tip radius
 */
export function propDiscTexture(inner) {
	const S = 256;
	const { c, ctx } = canvas(S, S);
	const image = ctx.createImageData(S, S);
	for (let j = 0; j < S; j++) {
		for (let i = 0; i < S; i++) {
			const r = Math.hypot(i + 0.5 - S / 2, j + 0.5 - S / 2) / (S / 2);
			let a = 0;
			let l = 0;
			if (r > inner && r < 1) {
				// Blade chord distribution, roughly: widest at 40–60 %, thin at root and tip.
				const t = (r - inner) / (1 - inner);
				a = 0.6 * Math.sin(Math.PI * Math.min(1, 0.15 + 0.85 * t)) ** 0.6 + 0.08;
				if (r > 0.91) {
					a = 0.3;
					l = 0.45;
				}
				a *= Math.min(1, (1 - r) / 0.015, (r - inner) / 0.03);
			}
			const k = (j * S + i) * 4;
			// sRGB 140 (linear ≈ 0.26) for the blades, brighter where the painted tips pass.
			const grey = 140 + 115 * l;
			image.data[k] = grey;
			image.data[k + 1] = grey;
			image.data[k + 2] = grey;
			image.data[k + 3] = Math.round(255 * Math.max(0, Math.min(1, a)));
		}
	}
	ctx.putImageData(image, 0, 0);
	const map = new THREE.CanvasTexture(c);
	map.colorSpace = THREE.SRGBColorSpace;
	return map;
}
