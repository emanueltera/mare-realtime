/**
 * The whitecaps' break gate travels downwind: a cell of its pattern seen at one place now
 * is GATE_SPEED · dt further along the wind sea's direction dt later, whatever that
 * direction is. The gate is sampled at `en + uGateOffset` (foam.js, BREAK_GATE_GLSL), so
 * a feature at texture coordinate q sits at en = q − offset.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { GATE_SPEED, gateStep } from '../src/ocean/engine/foam.js';

const DEG = Math.PI / 180;

test('break gate pattern travels toward the wind direction at GATE_SPEED', () => {
	for (const wind of [0, 60, 90, 145, 240, 359]) {
		let offE = 0;
		let offN = 0;
		// One feature, at texture coordinate (0, 0): it sits at en = −offset.
		const dt = 0.05;
		for (let i = 0; i < 200; i++) {
			const [dE, dN] = gateStep(dt, wind);
			offE += dE;
			offN += dN;
		}
		const [e, n] = [-offE, -offN];
		const travelled = Math.hypot(e, n);
		assert.ok(Math.abs(travelled - GATE_SPEED * 10) < 1e-9, `distance ${travelled}`);
		const bearing = (((Math.atan2(e, n) / DEG) % 360) + 360) % 360;
		const off = Math.abs(((bearing - wind + 540) % 360) - 180);
		assert.ok(off < 1e-9, `wind ${wind}: pattern went toward ${bearing}`);
	}
	// The speed of the old fixed scroll (1.7 m/s east, 1.02 m/s north), ~1.98 m/s.
	assert.ok(Math.abs(GATE_SPEED - 1.9825) < 1e-3);
});
