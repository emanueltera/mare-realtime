/**
 * Flight patterns as paths: an endless chain of straight legs and circular arcs for
 * the guidance to follow (guidance.js). Every pattern is laid out relative to the
 * trajectory's start point and start heading, so the same pattern flown from another
 * start or on another heading is the same shape moved and turned.
 *
 * Pattern coordinates: a pair `[a, b]` is `a` metres to the right of the start heading
 * and `b` metres ahead of the start. With `start.heading` 0 (north) these are plain
 * east / north offsets from the start. World positions are ENU metres (x east, y north);
 * bearings are compass radians (0 north, clockwise). "Clockwise" (direction 1) is as
 * seen from above: the aircraft turns right.
 *
 * The path is where the aircraft is asked to fly, not where it is: it starts at the
 * start point on the start heading, at cruise altitude and airspeed, and joins the
 * path by guidance — no pattern teleports it onto its first leg.
 */
import { stream, uniform } from './random.js';

const DEG = Math.PI / 180;
const TAU = 2 * Math.PI;
const G = 9.81;

/** Pattern defaults, m and degrees. */
export const patternDefaults = Object.freeze({
	orbit: Object.freeze({ radius: 180, direction: 1 }),
	racetrack: Object.freeze({ length: 600, radius: 150, direction: 1 }),
	figure8: Object.freeze({ radius: 150 }),
	lawnmower: Object.freeze({ legs: 6, legLength: 800, spacing: 120 }),
	waypoints: Object.freeze({ loop: true }),
	random: Object.freeze({ legLength: Object.freeze([300, 900]), turn: Object.freeze([-120, 120]) })
});

/**
 * One piece of a path. Progress `s` is metres along it from its start; a segment may be
 * endless (`length` Infinity), and `closest` may return progress outside [0, length]
 * (before the start of a leg, past its end).
 * @typedef {object} Segment
 * @property {'line' | 'arc'} kind
 * @property {number} length m
 * @property {number} [radius] m, arcs only
 * @property {number} [altitude] m above mean sea level to hold while flying this segment
 *   (a waypoint's altitude); otherwise the trajectory's altitude profile applies
 * @property {(x: number, y: number, hint: number) => number} closest progress of the
 *   closest point; for an arc, the one nearest `hint` (arcs wrap, so the aircraft's
 *   bearing from the centre is unwrapped against where it was a step ago)
 * @property {(s: number, out: { x: number, y: number }) => { x: number, y: number }} point
 * @property {(x: number, y: number) => number} distance from (x, y) to the segment
 *   within its extent, m
 */

/**
 * @param {number} x0
 * @param {number} y0
 * @param {number} bearing compass radians of travel
 * @param {number} length m, may be Infinity
 * @param {number} [altitude]
 * @returns {Segment}
 */
export function line(x0, y0, bearing, length, altitude) {
	const ux = Math.sin(bearing);
	const uy = Math.cos(bearing);
	return {
		kind: 'line',
		length,
		altitude,
		closest: (x, y) => (x - x0) * ux + (y - y0) * uy,
		point(s, out) {
			out.x = x0 + s * ux;
			out.y = y0 + s * uy;
			return out;
		},
		distance(x, y) {
			const s = Math.max(0, Math.min(length, (x - x0) * ux + (y - y0) * uy));
			return Math.hypot(x - x0 - s * ux, y - y0 - s * uy);
		}
	};
}

/**
 * A straight leg between two world points.
 * @param {number} x0
 * @param {number} y0
 * @param {number} x1
 * @param {number} y1
 * @param {number} [altitude]
 */
export function lineBetween(x0, y0, x1, y1, altitude) {
	return line(x0, y0, Math.atan2(x1 - x0, y1 - y0), Math.hypot(x1 - x0, y1 - y0), altitude);
}

/**
 * An arc of a circle round (cx, cy), starting at compass bearing `start` from the
 * centre and sweeping `sweep` radians clockwise (direction 1) or anticlockwise (−1).
 * @param {number} cx
 * @param {number} cy
 * @param {number} radius m
 * @param {number} start compass radians, from the centre to the arc's first point
 * @param {1 | -1} direction
 * @param {number} sweep radians, may be Infinity (an orbit)
 * @param {number} [altitude]
 * @returns {Segment}
 */
export function arc(cx, cy, radius, start, direction, sweep, altitude) {
	/** Angle swept from the arc's start to the aircraft's bearing, unwrapped near `hint`. @param {number} x @param {number} y @param {number} hint radians */
	const angle = (x, y, hint) => {
		const a = direction * (Math.atan2(x - cx, y - cy) - start);
		return a + TAU * Math.round((hint - a) / TAU);
	};
	/** @param {number} s @param {{ x: number, y: number }} out */
	const point = (s, out) => {
		const b = start + direction * (s / radius);
		out.x = cx + radius * Math.sin(b);
		out.y = cy + radius * Math.cos(b);
		return out;
	};
	return {
		kind: 'arc',
		length: radius * sweep,
		radius,
		altitude,
		closest: (x, y, hint) => radius * angle(x, y, hint / radius),
		point,
		distance(x, y) {
			const d = Math.hypot(x - cx, y - cy);
			if (sweep >= TAU) return Math.abs(d - radius);
			const a = angle(x, y, sweep / 2);
			if (a >= 0 && a <= sweep) return Math.abs(d - radius);
			const p = point(0, { x: 0, y: 0 });
			const d0 = Math.hypot(x - p.x, y - p.y);
			point(radius * sweep, p);
			return Math.min(d0, Math.hypot(x - p.x, y - p.y));
		}
	};
}

/**
 * A path: segment(i) for i = 0, 1, 2, … — endless, generated on demand and cached so
 * the random pattern draws its legs in order, whatever asks for them.
 * @typedef {object} Path
 * @property {(i: number) => Segment} segment
 */

/**
 * A path from a fixed list: `prefix` once, then `cycle` repeated for ever (or, with no
 * cycle, the last segment extended straight on).
 * @param {Segment[]} prefix
 * @param {Segment[]} cycle
 * @returns {Path}
 */
function chain(prefix, cycle) {
	if (cycle.length === 0) {
		const last = prefix[prefix.length - 1];
		const tail = straightOn(last);
		return { segment: (i) => (i < prefix.length ? prefix[i] : tail) };
	}
	return { segment: (i) => (i < prefix.length ? prefix[i] : cycle[(i - prefix.length) % cycle.length]) };
}

/**
 * The endless line that carries on from the end of a segment, along its final direction.
 * @param {Segment} seg
 */
function straightOn(seg) {
	const a = { x: 0, y: 0 };
	const b = { x: 0, y: 0 };
	seg.point(seg.length, a);
	seg.point(seg.length - 1, b);
	return line(a.x, a.y, Math.atan2(a.x - b.x, a.y - b.y), Infinity, seg.altitude);
}

/**
 * @typedef {object} Frame
 * @property {number} x start, m east
 * @property {number} y start, m north
 * @property {number} heading start heading, compass radians
 */

/**
 * World position of a pattern point [right, ahead] of the start.
 * @param {Frame} f
 * @param {number} right
 * @param {number} ahead
 */
function world(f, right, ahead) {
	const s = Math.sin(f.heading);
	const c = Math.cos(f.heading);
	// Ahead is (sin h, cos h); right is ahead turned 90° clockwise, (cos h, −sin h).
	return { x: f.x + right * c + ahead * s, y: f.y - right * s + ahead * c };
}

/**
 * A half turn from the end of a leg onto a parallel leg `spacing` to its side, flying
 * the other way. When a half circle of radius spacing/2 is no tighter than `radius`,
 * that is the turn. Otherwise it is an Ω (a "bulb" turn): out the other way on a circle
 * of `radius`, round the far side of a second, and back in on a third, each touching the
 * next, so the aircraft rolls out on the new leg exactly where it starts.
 *
 * In the leg end's own frame (right of and ahead of `end` along `bearing`, the turn to
 * side `d`), the three centres are at (−d·R, 0), (d·S/2, k) and (d·(S + R), 0), with
 * k = √(4R² − (S/2 + R)²) for the middle circle to touch the outer two. With β the angle
 * of the first centre-to-centre line off the leg direction, the arcs sweep π/2 − β,
 * 2π − 2β and π/2 − β: a net half turn.
 * @param {{ x: number, y: number }} end world point where the leg ends
 * @param {number} bearing compass radians the leg runs along
 * @param {1 | -1} d side the next leg lies on: 1 right, −1 left
 * @param {number} S spacing, m
 * @param {number} R tightest comfortable turn radius, m
 * @returns {Segment[]}
 */
function uTurn(end, bearing, d, S, R) {
	const f = { x: end.x, y: end.y, heading: bearing };
	if (S / 2 >= R) {
		const c = world(f, (d * S) / 2, 0);
		// The leg's end is on the far side of the centre from the turn.
		return [arc(c.x, c.y, S / 2, bearing - (d * Math.PI) / 2, d, Math.PI)];
	}
	const k = Math.sqrt(4 * R * R - (S / 2 + R) ** 2);
	const beta = Math.atan2(S / 2 + R, k);
	const c1 = world(f, -d * R, 0);
	const c2 = world(f, (d * S) / 2, k);
	const c3 = world(f, d * (S + R), 0);
	/** Compass bearing from p to q. @param {{ x: number, y: number }} p @param {{ x: number, y: number }} q */
	const toward = (p, q) => Math.atan2(q.x - p.x, q.y - p.y);
	const turn = /** @type {1 | -1} */ (-d);
	return [
		arc(c1.x, c1.y, R, toward(c1, end), turn, Math.PI / 2 - beta),
		arc(c2.x, c2.y, R, toward(c2, c1), d, 2 * Math.PI - 2 * beta),
		arc(c3.x, c3.y, R, toward(c3, c2), turn, Math.PI / 2 - beta)
	];
}

/**
 * Build the path for a pattern.
 * @param {import('./index.js').PatternSpec} pattern
 * @param {Frame} frame
 * @param {{ seed: number, airspeed: number, windSpeed: number, maxBank: number }} o the
 *   airspeed, the mean wind speed (m/s) and the autopilot's bank limit set the tightest
 *   turn the lawnmower lays out
 * @returns {Path}
 */
export function createPath(pattern, frame, { seed, airspeed, windSpeed, maxBank }) {
	const h = frame.heading;
	switch (pattern.type) {
		case 'straight':
			return chain([line(frame.x, frame.y, h, Infinity)], []);

		case 'orbit': {
			const d = pattern.direction ?? patternDefaults.orbit.direction;
			const r = pattern.radius ?? patternDefaults.orbit.radius;
			// Default centre: abeam the start on the side the orbit turns to, so the start
			// is on the circle and its heading is the circle's tangent.
			const [cr, ca] = pattern.center ?? [d * r, 0];
			const c = world(frame, cr, ca);
			// Begin the circle at the aircraft's bearing from the centre: progress 0 is
			// where it starts, whether or not that is on the circle.
			const start = Math.atan2(frame.x - c.x, frame.y - c.y);
			return chain([arc(c.x, c.y, r, start, d, Infinity)], []);
		}

		case 'racetrack': {
			const d = pattern.direction ?? patternDefaults.racetrack.direction;
			const L = pattern.length ?? patternDefaults.racetrack.length;
			const r = pattern.radius ?? patternDefaults.racetrack.radius;
			// Out along the start heading, a half turn to the `d` side, back parallel, and
			// a half turn home. The circle centres sit `r` to that side of the outbound leg.
			const p1 = world(frame, 0, L);
			const c1 = world(frame, d * r, L);
			const p2 = world(frame, 2 * d * r, L);
			const p3 = world(frame, 2 * d * r, 0);
			const c2 = world(frame, d * r, 0);
			const back = h + Math.PI;
			// Seen from a centre, the outbound leg's end lies on the far side from `d`.
			const side = h - (d * Math.PI) / 2;
			return chain(
				[],
				[
					lineBetween(frame.x, frame.y, p1.x, p1.y),
					arc(c1.x, c1.y, r, side, d, Math.PI),
					line(p2.x, p2.y, back, L),
					arc(c2.x, c2.y, r, side + Math.PI, d, Math.PI)
				]
			);
		}

		case 'figure8': {
			const r = pattern.radius ?? patternDefaults.figure8.radius;
			// Two circles touching at the start, one each side of the start heading: once
			// round the right-hand circle clockwise, once round the left anticlockwise.
			const cR = world(frame, r, 0);
			const cL = world(frame, -r, 0);
			return chain(
				[],
				[arc(cR.x, cR.y, r, h - Math.PI / 2, 1, 2 * Math.PI), arc(cL.x, cL.y, r, h + Math.PI / 2, -1, 2 * Math.PI)]
			);
		}

		case 'lawnmower': {
			const n = Math.max(1, Math.round(pattern.legs ?? patternDefaults.lawnmower.legs));
			const L = pattern.legLength ?? patternDefaults.lawnmower.legLength;
			const S = pattern.spacing ?? patternDefaults.lawnmower.spacing;
			// Boustrophedon: leg i runs ahead (even i) or back (odd i) at i·spacing to the
			// right of the start, joined by turns the aircraft can fly at a comfortable bank
			// (a little inside the limit, to leave room for gusts) at its fastest ground
			// speed, downwind — the way a survey planner sizes its turns for the wind.
			const fastest = airspeed + windSpeed;
			const turnRadius = (fastest * fastest) / (G * Math.tan(Math.max(5, maxBank - 5) * DEG));
			/** @type {Segment[]} */
			const cycle = [];
			for (let i = 0; i < n; i++) {
				const forward = i % 2 === 0;
				const a = world(frame, i * S, forward ? 0 : L);
				const b = world(frame, i * S, forward ? L : 0);
				cycle.push(lineBetween(a.x, a.y, b.x, b.y));
				if (i === n - 1) break;
				// Ahead-running legs turn right onto the next; back-running ones, left.
				cycle.push(...uTurn(b, forward ? h : h + Math.PI, forward ? 1 : -1, S, turnRadius));
			}
			// After the last leg, back to a point two turn radii short of the first leg and in
			// along its line, so the aircraft has rounded the corner and settled by the time
			// the first leg starts again; then round again.
			const end = world(frame, (n - 1) * S, (n - 1) % 2 === 0 ? L : 0);
			const lead = world(frame, 0, -2 * turnRadius);
			cycle.push(lineBetween(end.x, end.y, lead.x, lead.y), lineBetween(lead.x, lead.y, frame.x, frame.y));
			return chain([], cycle);
		}

		case 'waypoints': {
			const loop = pattern.loop ?? patternDefaults.waypoints.loop;
			/** @type {{ x: number, y: number, altitude?: number }[]} */
			const pts = [];
			for (const p of pattern.points ?? []) {
				const w = world(frame, p[0], p[1]);
				const q = { x: w.x, y: w.y, altitude: p.length > 2 ? p[2] : undefined };
				const last = pts[pts.length - 1];
				// Repeated points would make zero-length legs with no direction.
				if (last && Math.hypot(q.x - last.x, q.y - last.y) < 1) continue;
				pts.push(q);
			}
			if (pts.length === 0) return createPath({ type: 'straight' }, frame, { seed, airspeed, windSpeed, maxBank });
			// From the start to the first point once, then point to point; each leg holds
			// the altitude of the point it flies to, if that point gives one.
			/** @type {Segment[]} */
			const prefix = [];
			const p0 = pts[0];
			if (Math.hypot(p0.x - frame.x, p0.y - frame.y) >= 1) prefix.push(lineBetween(frame.x, frame.y, p0.x, p0.y, p0.altitude));
			/** @type {Segment[]} */
			const legs = [];
			for (let i = 1; i < pts.length; i++) {
				const a = pts[i - 1];
				const b = pts[i];
				legs.push(lineBetween(a.x, a.y, b.x, b.y, b.altitude));
			}
			if (loop && pts.length > 1) {
				const a = pts[pts.length - 1];
				if (Math.hypot(a.x - p0.x, a.y - p0.y) >= 1) legs.push(lineBetween(a.x, a.y, p0.x, p0.y, p0.altitude));
				return chain(prefix, legs);
			}
			// Not looping (or a single point): fly on along the last leg when it is done.
			const all = [...prefix, ...legs];
			if (all.length === 0) all.push(line(frame.x, frame.y, h, 1));
			return chain(all, []);
		}

		case 'random': {
			const [lo, hi] = pattern.legLength ?? patternDefaults.random.legLength;
			const [tlo, thi] = pattern.turn ?? patternDefaults.random.turn;
			const random = stream(seed, 'pattern');
			/** @type {Segment[]} */
			const legs = [];
			let x = frame.x;
			let y = frame.y;
			let bearing = h;
			// The first leg runs along the start heading; each later one turns by a seeded
			// angle (positive right) at the end of the last. Legs are drawn only as the
			// guidance reaches them, in order, so the mission is the same however long the
			// flight.
			return {
				segment(i) {
					while (legs.length <= i) {
						if (legs.length > 0) bearing += uniform(random, tlo, thi) * DEG;
						const len = uniform(random, lo, hi);
						const seg = line(x, y, bearing, len);
						legs.push(seg);
						x += len * Math.sin(bearing);
						y += len * Math.cos(bearing);
					}
					return legs[i];
				}
			};
		}

		default:
			throw new TypeError(`trajectory: unknown pattern type '${/** @type {{ type: string }} */ (pattern).type}'`);
	}
}
