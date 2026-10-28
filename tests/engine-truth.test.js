/**
 * The engine's exported camera truth against three.js itself: a point projected by the
 * three.js camera that renders (Vector3.project) and by the exported K, R_wc, t_wc must
 * land on the same pixel. Also the conventions the rigs and the trajectory flight hold to
 * (compass headings, ENU, OpenCV axes). No WebGL: three's cameras are plain maths.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import * as THREE from 'three';
import { trajectoryFlight } from '../src/ocean/aircraft/flight.js';
import { ChaseRig, OnboardRig, lookQuaternion } from '../src/ocean/engine/rigs.js';
import { cameraTruth, projectTruth } from '../src/ocean/engine/truth.js';
import { mulberry32 } from '../src/ocean/engine/util.js';

const DEG = Math.PI / 180;

/**
 * A camera placed as the engine places its own: ENU pose on a floating origin.
 * @param {{ east: number, north: number, up: number, quaternion: THREE.Quaternion, fov: number }} pose
 * @param {{ east: number, north: number }} origin
 * @param {number} width
 * @param {number} height
 */
function engineCamera(pose, origin, width, height) {
	const cam = new THREE.PerspectiveCamera(pose.fov, width / height, 0.3, 40000);
	cam.position.set(pose.east - origin.east, pose.up, -(pose.north - origin.north));
	cam.quaternion.copy(pose.quaternion);
	cam.updateProjectionMatrix();
	cam.updateMatrixWorld();
	return cam;
}

/**
 * The pixel three.js puts an ENU point at, top-left origin.
 * @param {THREE.PerspectiveCamera} cam
 * @param {{ east: number, north: number }} origin
 * @param {[number, number, number]} p
 * @param {number} width
 * @param {number} height
 */
function threePixel(cam, origin, p, width, height) {
	const v = new THREE.Vector3(p[0] - origin.east, p[2], -(p[1] - origin.north)).project(cam);
	return [((v.x + 1) / 2) * width, ((1 - v.y) / 2) * height];
}

/**
 * Points in front of a truth camera, spread over (and a little past) its view.
 * @param {import('../src/ocean/engine/truth.js').CameraTruth} truth
 * @param {() => number} random
 * @returns {Array<[number, number, number]>}
 */
function pointsInView(truth, random) {
	const R = truth.rotation;
	/** @type {Array<[number, number, number]>} */
	const out = [];
	for (let i = 0; i < 12; i++) {
		const Z = 2 + random() * 3000;
		const X = (random() * 2.4 - 1.2) * Z * 0.8;
		const Y = (random() * 2.4 - 1.2) * Z * 0.5;
		// Camera frame (OpenCV) to ENU: P = R_wc p_c + t_wc.
		out.push([
			R[0] * X + R[1] * Y + R[2] * Z + truth.position[0],
			R[3] * X + R[4] * Y + R[5] * Z + truth.position[1],
			R[6] * X + R[7] * Y + R[8] * Z + truth.position[2]
		]);
	}
	return out;
}

test('K, R_wc and t_wc project like the three.js camera (< 0.01 px)', () => {
	const random = mulberry32(7);
	let worst = 0;
	for (let n = 0; n < 200; n++) {
		const width = [1280, 640, 1920, 333][n % 4];
		const height = [720, 480, 1080, 517][n % 4];
		const origin = { east: Math.round((random() - 0.5) * 8000), north: Math.round((random() - 0.5) * 8000) };
		const pose = {
			east: origin.east + (random() - 0.5) * 4000,
			north: origin.north + (random() - 0.5) * 4000,
			up: 1 + random() * 400,
			quaternion: lookQuaternion(random() * 360, -89 + random() * 120, (random() - 0.5) * 60),
			fov: 12 + random() * 80
		};
		const cam = engineCamera(pose, origin, width, height);
		const truth = cameraTruth(cam, origin, width, height);
		for (const p of pointsInView(truth, random)) {
			const [u0, v0] = threePixel(cam, origin, p, width, height);
			const [u1, v1, Z] = projectTruth(truth, p);
			assert.ok(Z > 0);
			worst = Math.max(worst, Math.hypot(u1 - u0, v1 - v0));
		}
		// Square pixels, centred principal point, the fov the camera was given.
		assert.ok(Math.abs(truth.K[0] - truth.K[4]) < 1e-9 * truth.K[4]);
		assert.ok(Math.abs(truth.K[2] - width / 2) < 1e-9 && Math.abs(truth.K[5] - height / 2) < 1e-9);
		assert.ok(Math.abs(truth.K[4] - height / 2 / Math.tan((pose.fov * DEG) / 2)) < 1e-6);
		assert.ok(Math.abs(truth.fovY - pose.fov) < 1e-9);
		// t_wc is the pose, the floating origin undone.
		assert.ok(Math.hypot(truth.position[0] - pose.east, truth.position[1] - pose.north, truth.position[2] - pose.up) < 1e-6);
	}
	assert.ok(worst < 0.01, `worst disagreement ${worst} px`);
});

test('a shifted projection (setViewOffset) moves the principal point, and K follows it', () => {
	const random = mulberry32(11);
	const origin = { east: 100, north: -300 };
	const pose = { east: 140, north: -250, up: 90, quaternion: lookQuaternion(30, -40, 5), fov: 48 };
	const width = 1280;
	const height = 720;
	const cam = engineCamera(pose, origin, width, height);
	cam.setViewOffset(1600, 900, 210, 60, width, height);
	cam.updateProjectionMatrix();
	const truth = cameraTruth(cam, origin, width, height);
	assert.ok(Math.abs(truth.K[2] - width / 2) > 1, 'principal point moved');
	let worst = 0;
	for (const p of pointsInView(truth, random)) {
		const [u0, v0] = threePixel(cam, origin, p, width, height);
		const [u1, v1] = projectTruth(truth, p);
		worst = Math.max(worst, Math.hypot(u1 - u0, v1 - v0));
	}
	assert.ok(worst < 0.01, `worst disagreement ${worst} px`);
});

test('R_wc is a rotation, its quaternion matches it, and the axes are OpenCV in ENU', () => {
	// Looking east (compass 90), 30° down, level: optical axis (cos30, 0, −sin30) in ENU,
	// image x (right) toward south, image y (down) below the axis.
	const pose = { east: 0, north: 0, up: 100, quaternion: lookQuaternion(90, -30, 0), fov: 50 };
	const truth = cameraTruth(engineCamera(pose, { east: 0, north: 0 }, 640, 480), { east: 0, north: 0 }, 640, 480);
	const R = truth.rotation;
	const col = (/** @type {number} */ c) => [R[c], R[3 + c], R[6 + c]];
	/** @param {number[]} a @param {number[]} b */
	const near = (a, b) => a.every((x, i) => Math.abs(x - b[i]) < 1e-9);
	assert.ok(near(col(2), [Math.cos(30 * DEG), 0, -Math.sin(30 * DEG)]), `z ${col(2)}`);
	assert.ok(near(col(0), [0, -1, 0]), `x ${col(0)}`);
	assert.ok(near(col(1), [-Math.sin(30 * DEG), 0, -Math.cos(30 * DEG)]), `y ${col(1)}`);
	// det = +1 and the quaternion gives the same matrix.
	const m = new THREE.Matrix4().set(R[0], R[1], R[2], 0, R[3], R[4], R[5], 0, R[6], R[7], R[8], 0, 0, 0, 0, 1);
	assert.ok(Math.abs(m.determinant() - 1) < 1e-9);
	const [x, y, z, w] = truth.quaternion;
	assert.ok(w >= 0);
	const back = new THREE.Matrix4().makeRotationFromQuaternion(new THREE.Quaternion(x, y, z, w)).elements;
	for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) assert.ok(Math.abs(back[c * 4 + r] - R[r * 3 + c]) < 1e-9);
});

/** A straight, level trajectory: 24 m/s at 100 m, heading 45, gimbal 35° down. */
const straight45 = {
	duration: 10,
	/** @param {number} t */
	stateAt(t) {
		const v = 24;
		const h = 45 * DEG;
		return {
			t,
			x: Math.sin(h) * v * t,
			y: Math.cos(h) * v * t,
			altitude: 100,
			vE: Math.sin(h) * v,
			vN: Math.cos(h) * v,
			vU: 0,
			heading: 45,
			pitch: 0,
			bank: 0,
			airspeed: v,
			turnRate: 0,
			gimbalPitch: -35,
			gimbalYaw: 0,
			gimbalFov: 48
		};
	}
};

test('a trajectory flight is placed exactly, held at its ends, or looped', () => {
	const held = trajectoryFlight(straight45);
	held.step(2.5);
	assert.equal(held.state.time, 2.5);
	assert.equal(held.state.x, straight45.stateAt(2.5).x);
	assert.equal(held.state.altitude, 100);
	assert.equal(held.state.speed, 24);
	held.seek(14);
	assert.equal(held.state.time, 10);
	assert.equal(held.sample.gimbalPitch, -35);
	// Looped (the app), playing on wraps past the end, but a seek is held within the
	// flight: a seek to the end shows the end, not the start again.
	const looped = trajectoryFlight(straight45, { loop: true });
	looped.seek(9);
	looped.step(4);
	assert.ok(Math.abs(looped.state.time - 3) < 1e-12);
	looped.seek(10);
	assert.equal(looped.state.time, 10);
	assert.equal(looped.state.x, straight45.stateAt(10).x);
	looped.seek(13);
	assert.equal(looped.state.time, 10);
	looped.seek(10);
	looped.step(0);
	assert.equal(looped.state.time, 10);
});

test('the onboard camera looks along the heading, down by the gimbal pitch', () => {
	const flight = trajectoryFlight(straight45);
	flight.seek(3);
	const rig = new OnboardRig({ mode: 'onboard', stabilized: 0.55 });
	const pose = rig.update({
		dt: 0.1,
		time: 3,
		aircraft: { state: flight.state, gimbal: { pitch: -35, yaw: 0, fov: 48 }, scale: 1 },
		cameraMount: new THREE.Vector3(0, -0.2, -0.4),
		seaHeight: () => 0,
		offsets: { yaw: 0, pitch: 0, zoom: 1 },
		aspect: 16 / 9,
		reducedMotion: false,
		shift: { east: 0, north: 0 }
	});
	const truth = cameraTruth(engineCamera(pose, { east: 0, north: 0 }, 1280, 720), { east: 0, north: 0 }, 1280, 720);
	const R = truth.rotation;
	const c = Math.cos(35 * DEG);
	const s = Math.sin(35 * DEG);
	const want = [Math.sin(45 * DEG) * c, Math.cos(45 * DEG) * c, -s];
	assert.ok(Math.hypot(R[2] - want[0], R[5] - want[1], R[8] - want[2]) < 1e-9, `axis ${[R[2], R[5], R[8]]}`);
	// The lens sits at the mount on the airframe, just below the aircraft's centre.
	assert.ok(Math.abs(truth.position[2] - 99.8) < 1e-9);
	assert.ok(Math.abs(truth.fovY - 48) < 1e-9);
});

test('a chase rig given the trajectory velocity holds its framing from the first frame', () => {
	/** @param {boolean} primed */
	const offsets = (primed) => {
		const flight = trajectoryFlight(straight45);
		const rig = new ChaseRig({ mode: 'chase', distance: 18, azimuth: 150, elevation: 12 });
		const out = [];
		for (let i = 0; i < 20; i++) {
			if (i > 0) flight.step(0.1);
			const sample = flight.sample;
			const pose = rig.update({
				dt: i === 0 ? 0 : 0.1,
				time: flight.state.time,
				aircraft: {
					state: flight.state,
					gimbal: { pitch: -35, yaw: 0, fov: 48 },
					scale: 1,
					velocity: primed ? { e: sample.vE ?? 0, n: sample.vN ?? 0, u: 0 } : null
				},
				cameraMount: new THREE.Vector3(),
				seaHeight: () => 0,
				offsets: { yaw: 0, pitch: 0, zoom: 1 },
				aspect: 16 / 9,
				reducedMotion: false,
				shift: { east: 0, north: 0 }
			});
			out.push([pose.east - flight.state.x, pose.north - flight.state.y]);
		}
		return out;
	};
	const primed = offsets(true);
	const spread = Math.max(...primed.map(([e, n]) => Math.hypot(e - primed[0][0], n - primed[0][1])));
	assert.ok(spread < 1e-3, `primed framing moved ${spread} m`);
	const unprimed = offsets(false);
	const lag = Math.max(...unprimed.map(([e, n]) => Math.hypot(e - unprimed[0][0], n - unprimed[0][1])));
	assert.ok(lag > 0.5, 'without the velocity the rig starts out lagging');
});
