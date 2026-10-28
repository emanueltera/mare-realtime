/**
 * How a run is filmed: the SceneSpec a RunSpec is rendered from, and the two cameras of
 * `render.view`. The batch page renders exactly `runScene(run, trajectory)`, and the
 * app's preview takes its chase and onboard cameras from here, so what the app shows
 * for a run is the shot the batch writes (the app only moves the aircraft within the
 * picture, off the part of the window its panel covers).
 *
 * Pure: SceneSpec is plain data, read by the engine (src/ocean/scene.js); nothing here
 * touches three.js or the DOM.
 */

/** @typedef {import('../ocean/scene.js').SceneSpec} SceneSpec */
/** @typedef {import('../ocean/scene.js').ChaseRig} ChaseRig */
/** @typedef {import('../ocean/scene.js').OnboardRig} OnboardRig */
/** @typedef {import('../ocean/scene.js').TrajectorySource} TrajectorySource */
/** @typedef {import('./run.js').RunSpec} RunSpec */

/**
 * The chase view for data: 24 m from the aircraft, behind its right quarter (azimuth 160°
 * clockwise from the nose: 90° is off the right wing, 180° dead astern) and 12° above it, the aircraft centred so the sea fills the
 * rest of the frame. Looking 12° down through a 42° lens puts the horizon in the upper
 * half of the picture. (The engine's default `frame`, [0.3, -0.05], sets the aircraft
 * off-centre; a dataset has nothing to clear.)
 * @param {[number, number]} [frame] where the aircraft sits in the picture, fractions of
 *   the half-width and half-height from the centre (+x right, +y up); default centred
 * @returns {ChaseRig}
 */
export function chaseCamera(frame = [0, 0]) {
	return { mode: 'chase', distance: 24, azimuth: 160, elevation: 12, lag: 0.5, fov: 42, frame: [frame[0], frame[1]] };
}

/**
 * The aircraft's own gimbal camera, as stabilised as the trajectory's gimbal says
 * (0 rolls and pitches with the airframe, 1 holds the horizon).
 * @param {number} stabilized
 * @returns {OnboardRig}
 */
export function onboardCamera(stabilized) {
	return { mode: 'onboard', stabilized };
}

/**
 * The scene a run is drawn from: its sea and light, the aircraft placed exactly on the
 * trajectory (path 'trajectory': the engine reads the gimbal from its samples), and the
 * camera `render.view` names. No HUD, no overlays, no tracks, entered without a blend.
 * @param {RunSpec} run a resolved run
 * @param {TrajectorySource} trajectory `createTrajectory(run.trajectory, { sea: run.sea })`
 * @returns {SceneSpec}
 */
export function runScene(run, trajectory) {
	const g = run.trajectory.gimbal;
	return {
		sea: { ...run.sea },
		light: { ...run.light },
		camera: run.render.view === 'chase' ? chaseCamera() : onboardCamera(g.stabilized),
		aircraft: {
			path: 'trajectory',
			trajectory,
			// Only where a sample lacks them: the trajectory's samples carry the gimbal.
			gimbal: { pitch: g.pitch, yaw: g.yaw, fov: g.fov }
		},
		mode: 'shaded',
		overlays: [],
		hud: false,
		transition: 0
	};
}
