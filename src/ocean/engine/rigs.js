/**
 * The cameras that film the UAV, and how they hand over to each other.
 *
 * Every rig produces a CameraPose in absolute east/north/up metres plus a three.js
 * world orientation. During a scene transition the engine evaluates the outgoing and
 * the incoming rig every frame and blends their poses (`blendPoses`): position along a
 * gentle arc with height eased in log space, orientation by slerp, field of view
 * linearly — so a chase camera flies into the aircraft's gimbal, and an onboard view
 * rises smoothly to an overhead one.
 */
import * as THREE from 'three';
import { attitudeQuaternion } from '../aircraft.js';
import { cameraDefaults, chaseDefaults, onboardDefaults } from '../scene.js';
import { DEG, angleDelta, clamp, damp, lerp, wrap360 } from './util.js';

/**
 * @typedef {object} CameraPose
 * @property {number} east absolute metres
 * @property {number} north
 * @property {number} up metres above mean sea level
 * @property {THREE.Quaternion} quaternion world orientation (x east, y up, z south)
 * @property {number} fov vertical, degrees
 * @property {number} heave metres of sea-driven heave (free rig sway)
 */

/**
 * @typedef {object} AbsoluteAircraft
 * @property {import('../aircraft.js').AircraftState} state x/y absolute east/north metres
 * @property {{ pitch: number, yaw: number, fov: number }} gimbal
 * @property {number} scale
 */

/**
 * @typedef {object} RigContext
 * @property {number} dt seconds
 * @property {number} time seconds
 * @property {AbsoluteAircraft | null} aircraft the blended aircraft, when there is one
 * @property {THREE.Vector3} cameraMount gimbal position in the model frame
 * @property {(east: number, north: number) => number} seaHeight instantaneous sea height, m
 * @property {(east: number, north: number) => { pitch: number, roll: number, heave: number }} [seaMotion]
 *   how a small float at a point would pitch, roll and heave, for sway
 * @property {{ yaw: number, pitch: number, zoom: number }} offsets offsets from the presenter taking the camera
 * @property {number} aspect width / height
 * @property {boolean} reducedMotion
 * @property {{ east: number, north: number }} shift where this scene's frame sits in the world
 * @property {(id: string) => { east: number, north: number, up: number } | null} [include]
 *   where a recorded track of this scene is now (absolute metres), for a chase rig's `include`
 */

/** @returns {CameraPose} */
export function emptyPose() {
	return { east: 0, north: 0, up: 10, quaternion: new THREE.Quaternion(), fov: 50, heave: 0 };
}

/**
 * Orientation of a camera with yaw (compass) and pitch, no roll.
 * @param {number} heading compass degrees the camera looks toward
 * @param {number} pitch degrees, negative down
 * @param {number} [roll] degrees, right side down positive
 * @param {THREE.Quaternion} [out]
 */
export function lookQuaternion(heading, pitch, roll = 0, out = new THREE.Quaternion()) {
	return out.setFromEuler(new THREE.Euler(pitch * DEG, -heading * DEG, -roll * DEG, 'YXZ'));
}

const _euler = new THREE.Euler();
const _v = new THREE.Vector3();
const _q = new THREE.Quaternion();
const _q2 = new THREE.Quaternion();

/**
 * Heading, pitch and roll of an orientation (inverse of lookQuaternion).
 * @param {THREE.Quaternion} q
 */
export function attitudeOf(q) {
	_euler.setFromQuaternion(q, 'YXZ');
	return { heading: wrap360(-_euler.y / DEG), pitch: _euler.x / DEG, roll: -_euler.z / DEG };
}

/**
 * A critically damped spring that follows a moving target. Motion the caller predicts
 * (`feedForward`, the target's smoothed velocity) is followed exactly; only the
 * surprise — gusts, the start of a turn — is eased, so a steadily flying aircraft is
 * held in frame without the camera trailing behind it.
 */
class Spring {
	/** @param {number} value */
	constructor(value) {
		this.value = value;
		this.target = value;
		this.err = 0;
		this.errV = 0;
	}

	/**
	 * @param {number} target
	 * @param {number} omega rad/s
	 * @param {number} dt
	 * @param {number} [feedForward] predicted target velocity
	 */
	step(target, omega, dt, feedForward = 0) {
		const surprise = target - this.target - feedForward * dt;
		this.target = target;
		let e = this.err - surprise;
		let ev = this.errV;
		const ex = Math.exp(-omega * dt);
		const tmp = (ev + omega * e) * dt;
		ev = (ev - omega * tmp) * ex;
		e = (e + tmp) * ex;
		this.err = e;
		this.errV = ev;
		this.value = target + e;
		return this.value;
	}
}

/** The free camera: placed in the world, optionally drifting, orbiting and riding the sea. */
export class FreeRig {
	/** @param {import('../scene.js').OceanCamera} spec */
	constructor(spec) {
		this.spec = { ...cameraDefaults, ...spec };
		this.mode = /** @type {const} */ ('free');
		this.orbit = 0;
		this.travelled = 0;
		this.pose = emptyPose();
	}

	/** @param {RigContext} ctx */
	update(ctx) {
		const s = this.spec;
		const motion = ctx.reducedMotion ? 0 : 1;
		this.orbit += s.orbit * motion * ctx.dt;
		this.travelled += s.drift * motion * ctx.dt;
		const heading = s.heading + this.orbit + ctx.offsets.yaw;
		const pitch = clamp(s.pitch + ctx.offsets.pitch, -89, 60);
		const course = s.heading * DEG;
		const east = s.x + ctx.shift.east + Math.sin(course) * this.travelled;
		const north = s.y + ctx.shift.north + Math.cos(course) * this.travelled;
		let up = s.height * ctx.offsets.zoom;
		let roll = 0;
		let dPitch = 0;
		let heave = 0;
		const sway = s.sway * motion;
		if (sway > 0 && ctx.seaMotion) {
			const m = ctx.seaMotion(east, north);
			heave = m.heave * sway;
			up += heave;
			dPitch = m.pitch * sway;
			roll = m.roll * sway;
		}
		up = Math.max(up, ctx.seaHeight(east, north) + 0.6);
		const p = this.pose;
		p.east = east;
		p.north = north;
		p.up = up;
		p.heave = heave;
		p.fov = s.fov;
		lookQuaternion(heading, pitch + dPitch, roll, p.quaternion);
		return p;
	}
}

/** Metres the aircraft can move in one frame before a chase rig takes it as a cut (a looped recording wrapping). */
const CUT_DISTANCE = 300;
/**
 * Seconds a chase rig follows rigidly after a cut, while it measures the aircraft's new
 * velocity, so its springs start settled instead of drifting into place on screen (the
 * aircraft fades back in over about this long).
 */
const CUT_SETTLE = 0.35;

/**
 * A chase camera trailing the aircraft, aimed so it sits at `frame` in the picture. With
 * `include`, it aims at the middle of the aircraft and that track's position and pulls
 * back as far as it needs to keep both in the picture.
 */
export class ChaseRig {
	/** @param {import('../scene.js').ChaseRig} spec */
	constructor(spec) {
		this.spec = { ...chaseDefaults, ...spec };
		this.mode = /** @type {const} */ ('chase');
		this.orbit = 0;
		/** @type {Spring[] | null} */
		this.pos = null;
		/** @type {Spring[] | null} */
		this.aim = null;
		this.heading = new Spring(0);
		this.prev = { e: 0, n: 0, u: 0, t: -1 };
		this.vel = { e: 0, n: 0, u: 0 };
		/** Extra distance pulled back to hold an `include` in frame, m, eased. */
		this.reach = 0;
		/** Seconds left of rigid following after a cut. */
		this.settle = 0;
		this.pose = emptyPose();
	}

	/** @param {RigContext} ctx */
	update(ctx) {
		const p = this.pose;
		const s = this.spec;
		const a = ctx.aircraft;
		if (!a) return p;
		const st = a.state;
		const dt = Math.max(ctx.dt, 1e-4);
		// What the camera follows: the aircraft, or the middle of it and the included track.
		const other = s.include && ctx.include ? ctx.include(s.include) : null;
		const te = other ? (st.x + other.east) / 2 : st.x;
		const tn = other ? (st.y + other.north) / 2 : st.y;
		const tu = other ? (st.altitude + other.up) / 2 : st.altitude;
		// A looped recording jumps back to its start: cut to it rather than fly across.
		const cut = this.pos && Math.hypot(st.x - this.prev.e, st.y - this.prev.n) > CUT_DISTANCE;
		if (!this.pos || !this.aim || cut) {
			this.pos = [new Spring(te), new Spring(tn), new Spring(tu)];
			this.aim = [new Spring(te), new Spring(tn), new Spring(tu)];
			this.heading = new Spring(st.heading);
			this.prev = { e: st.x, n: st.y, u: st.altitude, t: ctx.time };
			this.vel = { e: 0, n: 0, u: 0 };
			this.reach = -1;
			this.settle = cut ? CUT_SETTLE : 0;
		}
		// The aircraft's velocity, smoothed, as feed-forward for the springs; just after a
		// cut, taken raw from frame to frame (the first frame has nothing to measure).
		const settling = this.settle > 0;
		const k = settling ? 1 : 1 - Math.exp(-dt / 0.25);
		const fresh = settling && this.prev.t === ctx.time;
		if (!fresh) {
			this.vel.e += ((st.x - this.prev.e) / dt - this.vel.e) * k;
			this.vel.n += ((st.y - this.prev.n) / dt - this.vel.n) * k;
			this.vel.u += ((st.altitude - this.prev.u) / dt - this.vel.u) * k;
		}
		this.prev = { e: st.x, n: st.y, u: st.altitude, t: ctx.time };
		if (settling) {
			// Follow rigidly: the springs sit on their targets, as if they had been tracking
			// them at the measured velocity all along, so they have no error to work off.
			const targets = [te, tn, tu];
			const vel = [this.vel.e, this.vel.n, this.vel.u];
			for (let i = 0; i < 3; i++) {
				this.pos[i] = new Spring(targets[i]);
				this.aim[i] = new Spring(targets[i]);
				this.pos[i].target -= vel[i] * dt;
				this.aim[i].target -= vel[i] * dt;
			}
			this.heading = new Spring(st.heading);
			this.settle -= ctx.dt;
		}

		const lag = Math.max(0.05, s.lag);
		const omega = 2 / lag;
		const pe = this.pos[0].step(te, omega, dt, this.vel.e);
		const pn = this.pos[1].step(tn, omega, dt, this.vel.n);
		const pu = this.pos[2].step(tu, omega * 0.8, dt, this.vel.u);
		const ae = this.aim[0].step(te, omega * 2.2, dt, this.vel.e);
		const an = this.aim[1].step(tn, omega * 2.2, dt, this.vel.n);
		const au = this.aim[2].step(tu, omega * 1.8, dt, this.vel.u);
		// Unwrap the heading so the spring takes the short way round.
		let target = st.heading;
		while (target - this.heading.value > 180) target -= 360;
		while (target - this.heading.value < -180) target += 360;
		const followed = this.heading.step(target, omega * 0.9, dt);
		// Held on a compass bearing, the rig keeps its side of the aircraft through turns.
		const heading = s.heading ?? followed;

		this.orbit += s.orbit * (ctx.reducedMotion ? 0 : 1) * ctx.dt;
		const az = s.azimuth + this.orbit + ctx.offsets.yaw;
		const el = clamp(s.elevation + ctx.offsets.pitch, -60, 80);
		// A portrait screen sees a narrow slice of the scene's 16:9 framing: pull back
		// so the whole airframe stays in frame, and bring it in toward the middle.
		const portrait = ctx.aspect < 1 ? Math.pow(1 / ctx.aspect, 0.8) : 1;
		let dist = s.distance * portrait * ctx.offsets.zoom * Math.max(1, Math.pow(a.scale, 0.85));
		const bearing = (heading + az) * DEG;
		/** @type {[number, number]} */
		const frame = ctx.aspect < 1 ? [s.frame[0] * 0.4, s.frame[1]] : s.frame;
		if (other) {
			// Pull back until the pair, each end padded by the airframe, fits in the part of
			// the picture on its side of `frame`, with a margin. Seen from a camera at
			// distance d from their middle, an end at depth a toward the camera and lateral
			// offset l sits at l / (d − a) on screen, so d ≥ a + l / room for each end.
			const se = other.east - st.x;
			const sn = other.north - st.y;
			const su = other.up - st.altitude;
			const ce = Math.cos(el * DEG);
			const sne = Math.sin(el * DEG);
			// Camera right, camera up and toward the camera, as components of the separation.
			const right = se * Math.cos(bearing) - sn * Math.sin(bearing);
			const level = se * Math.sin(bearing) + sn * Math.cos(bearing);
			const upward = su * ce - level * sne;
			const toward = level * ce + su * sne;
			const pad = 3 * 0.75 * a.scale;
			const tanV = Math.tan((s.fov * DEG) / 2);
			const roomX = Math.max(0.2, (1 - Math.abs(frame[0])) * 0.8) * tanV * ctx.aspect;
			const roomY = Math.max(0.2, (1 - Math.abs(frame[1])) * 0.72) * tanV;
			const lateral = Math.max((Math.abs(right) / 2 + pad) / roomX, (Math.abs(upward) / 2 + pad) / roomY);
			const need = Math.abs(toward) / 2 + pad + lateral;
			const want = Math.max(0, need - dist);
			// Ease the pull-back so the framing breathes rather than pumps; start settled.
			this.reach = this.reach < 0 || settling ? want : damp(this.reach, want, 1.4, ctx.dt);
			dist += this.reach;
		} else {
			this.reach = 0;
		}
		let east = pe + Math.cos(el * DEG) * Math.sin(bearing) * dist;
		let north = pn + Math.cos(el * DEG) * Math.cos(bearing) * dist;
		let up = pu + Math.sin(el * DEG) * dist;
		up = Math.max(up, ctx.seaHeight(east, north) + 2);

		p.east = east;
		p.north = north;
		p.up = up;
		p.heave = 0;
		p.fov = s.fov;
		aimAt(p, ae - east, an - north, au - up, frame, ctx.aspect);
		return p;
	}
}

/**
 * Orient a pose (no roll) so a point at offset (de, dn, du) projects to `frame`
 * (fractions of the half-width and half-height, +x right, +y up).
 * @param {CameraPose} pose
 * @param {number} de
 * @param {number} dn
 * @param {number} du
 * @param {[number, number]} frame
 * @param {number} aspect
 */
export function aimAt(pose, de, dn, du, frame, aspect) {
	const len = Math.hypot(de, dn, du) || 1;
	// Target direction in three world coordinates.
	const tx = de / len;
	const ty = du / len;
	const tz = -dn / len;
	const tanV = Math.tan((pose.fov * DEG) / 2);
	// The target's direction in camera space (x right, y up, −z forward).
	const cx = frame[0] * tanV * aspect;
	const cy = frame[1] * tanV;
	const cl = Math.hypot(cx, cy, 1);
	const dx = cx / cl;
	const dy = cy / cl;
	const dz = -1 / cl;
	// Pitch θ such that rotating d about x gives the target's elevation: dy·cosθ − dz·sinθ = ty.
	const A = dy;
	const B = -dz;
	const R = Math.hypot(A, B);
	const phi = Math.atan2(B, A);
	const c = clamp(ty / R, -1, 1);
	let theta = phi - Math.acos(c);
	const alt = phi + Math.acos(c);
	if (Math.abs(alt) < Math.abs(theta)) theta = alt;
	theta = clamp(theta, -Math.PI / 2 + 1e-3, Math.PI / 2 - 1e-3);
	// After the pitch, the direction's horizontal part; yaw it onto the target's.
	const px = dx;
	const pz = dy * Math.sin(theta) + dz * Math.cos(theta);
	const yaw = Math.atan2(-tx, -tz) - Math.atan2(-px, -pz);
	pose.quaternion.setFromEuler(new THREE.Euler(theta, yaw, 0, 'YXZ'));
}

/** The aircraft's own gimbal camera. */
export class OnboardRig {
	/** @param {import('../scene.js').OnboardRig} spec */
	constructor(spec) {
		this.spec = { ...onboardDefaults, ...spec };
		this.mode = /** @type {const} */ ('onboard');
		this.pose = emptyPose();
	}

	/** @param {RigContext} ctx */
	update(ctx) {
		const p = this.pose;
		const a = ctx.aircraft;
		if (!a) return p;
		const st = a.state;
		const stab = clamp(this.spec.stabilized, 0, 1);
		const air = attitudeQuaternion(st.heading, st.pitch, st.bank, _q);
		_v.copy(ctx.cameraMount).multiplyScalar(a.scale).applyQuaternion(air);
		p.east = st.x + _v.x;
		p.north = st.y - _v.z;
		p.up = st.altitude + _v.y;
		p.heave = 0;
		const level = attitudeQuaternion(st.heading, 0, 0, _q2);
		p.quaternion.copy(air).slerp(level, stab);
		const gimbal = lookQuaternion(a.gimbal.yaw + ctx.offsets.yaw, clamp(a.gimbal.pitch + ctx.offsets.pitch, -90, 30));
		p.quaternion.multiply(gimbal);
		p.fov = clamp(a.gimbal.fov / ctx.offsets.zoom, 12, 100);
		return p;
	}
}

/**
 * @typedef {FreeRig | ChaseRig | OnboardRig} Rig
 */

/**
 * @param {import('../scene.js').CameraRig} spec
 * @returns {Rig}
 */
export function createRig(spec) {
	if (spec.mode === 'chase') return new ChaseRig(spec);
	if (spec.mode === 'onboard') return new OnboardRig(spec);
	return new FreeRig(/** @type {import('../scene.js').OceanCamera} */ (spec));
}

/** Per blend (keyed by its output pose): the heading change it committed to. */
const blendTurns = new WeakMap();

/**
 * Blend two poses: horizontal position eased, height in log space when it changes a
 * lot, a small rise mid-way when the camera travels far. Orientation is blended as
 * heading, pitch and roll, the way a camera operator pans and tilts, so the horizon
 * stays level through a turn; the direction of the turn is fixed when the blend starts,
 * so two views nearly opposite each other never flip the camera mid-way round.
 * @param {CameraPose} a
 * @param {CameraPose} b
 * @param {number} t 0 → a, 1 → b (already eased)
 * @param {CameraPose} out persists across frames for the same blend
 * @param {number} [tq] separate weight for the orientation; default t
 * @param {boolean} [aimed] blend what the camera looks at instead of where it is (see aimedBlend)
 */
export function blendPoses(a, b, t, out, tq = t, aimed = false) {
	if (aimed) return aimedBlend(a, b, t, out);
	out.east = lerp(a.east, b.east, t);
	out.north = lerp(a.north, b.north, t);
	const ha = Math.max(a.up, 0.5);
	const hb = Math.max(b.up, 0.5);
	const ratio = Math.max(ha, hb) / Math.min(ha, hb);
	const linear = lerp(a.up, b.up, t);
	const logH = Math.exp(lerp(Math.log(ha), Math.log(hb), t));
	const w = clamp((ratio - 1.5) / 2, 0, 1);
	const travel = Math.hypot(b.east - a.east, b.north - a.north);
	out.up = lerp(linear, logH, w) + Math.sin(Math.PI * t) * Math.min(travel * 0.12, 60);
	const from = attitudeOf(a.quaternion);
	const to = attitudeOf(b.quaternion);
	let turn = angleDelta(from.heading, to.heading);
	const committed = blendTurns.get(out);
	if (committed !== undefined) {
		while (turn - committed > 180) turn -= 360;
		while (turn - committed < -180) turn += 360;
	}
	blendTurns.set(out, turn);
	lookQuaternion(
		from.heading + turn * tq,
		lerp(from.pitch, to.pitch, tq),
		from.roll + angleDelta(from.roll, to.roll) * tq,
		out.quaternion
	);
	out.fov = lerp(a.fov, b.fov, t);
	out.heave = lerp(a.heave, b.heave, t);
	return out;
}

/** The shallowest a camera is taken to look down when finding the point it aims at, degrees. */
const MIN_AIM_DIP = 4;

/**
 * The point on the sea a pose looks at and how far away it is, m.
 * @param {CameraPose} pose
 * @param {{ heading: number, pitch: number }} att
 */
function aimOf(pose, att) {
	const h = att.heading * DEG;
	const p = att.pitch * DEG;
	const range = Math.max(pose.up, 0.5) / Math.sin(Math.max(-p, MIN_AIM_DIP * DEG));
	const ground = range * Math.cos(p);
	return { east: pose.east + Math.sin(h) * ground, north: pose.north + Math.cos(h) * ground, range };
}

/**
 * A blend between views far apart in scale, such as a chase camera metres from the
 * aircraft and an overhead view of a whole route from tens of kilometres up. Blending
 * position and height separately sends the camera skimming kilometres across the sea
 * before it climbs. This blends the point the camera looks at and its range to it: the
 * range in log space, and the aim point in step with the range, so the pan is the same
 * fraction of the view all the way. Going up, it pulls back off the aircraft and pans
 * once it is high; coming down, it pans while high and then drops onto the aircraft.
 * @param {CameraPose} a
 * @param {CameraPose} b
 * @param {number} t 0 → a, 1 → b (already eased)
 * @param {CameraPose} out persists across frames for the same blend
 */
function aimedBlend(a, b, t, out) {
	const from = attitudeOf(a.quaternion);
	const to = attitudeOf(b.quaternion);
	const aimA = aimOf(a, from);
	const aimB = aimOf(b, to);
	const range = Math.exp(lerp(Math.log(aimA.range), Math.log(aimB.range), t));
	const span = aimB.range - aimA.range;
	const s = Math.abs(span) > aimA.range * 1e-3 ? (range - aimA.range) / span : t;
	let turn = angleDelta(from.heading, to.heading);
	const committed = blendTurns.get(out);
	if (committed !== undefined) {
		while (turn - committed > 180) turn -= 360;
		while (turn - committed < -180) turn += 360;
	}
	blendTurns.set(out, turn);
	const heading = from.heading + turn * t;
	const pitch = lerp(from.pitch, to.pitch, t);
	// Back off the aim point along the blended line of sight, at the blended range, by
	// the same dip the aim points were found with.
	const h = heading * DEG;
	const dip = Math.max(-pitch, MIN_AIM_DIP) * DEG;
	const ground = range * Math.cos(dip);
	out.east = lerp(aimA.east, aimB.east, s) - Math.sin(h) * ground;
	out.north = lerp(aimA.north, aimB.north, s) - Math.cos(h) * ground;
	out.up = Math.max(range * Math.sin(dip), 0.5);
	lookQuaternion(heading, pitch, from.roll + angleDelta(from.roll, to.roll) * t, out.quaternion);
	out.fov = lerp(a.fov, b.fov, t);
	out.heave = lerp(a.heave, b.heave, t);
	return out;
}
