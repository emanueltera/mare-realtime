/**
 * The ocean engine: the simulator running live behind every slide, with the UAV over
 * it and the cameras that film it. Client-only (three.js, WebGL2); OceanStage.svelte
 * loads it with a dynamic import.
 *
 * A frame:
 *  1. Blend the scene: sea and light ease toward the current scene (or the current
 *     randomised sample) with live slide overrides layered on top.
 *  2. Step every flight still on screen and blend their states, so the aircraft never
 *     teleports; then every camera rig, each filming its own scene's aircraft carried
 *     onto the blended one, and blend their poses into the camera.
 *  3. Simulate the sea (GPU FFT cascades), carry the foam on the water forward
 *     (foam.js) and refresh the sky LUT if the light moved.
 *  4. Render: sky + sea into three targets (radiance and ground-truth channels), the
 *     aircraft into its own anti-aliased layer, overlays into a layer that shares the
 *     sea's depth; lens glare from the radiance (bloom.js); then compose, tone map and
 *     draw the label views in one pass.
 *  5. Draw the HUD, publish the readout a few times a second.
 *
 * Scenes are positioned freely: the sea is the same everywhere, so each scene's frame
 * is shifted on entry to put its aircraft where the current one is (or its free camera
 * where the current camera is). A floating origin keeps shader coordinates small.
 */
import * as THREE from 'three';
import { attitudeQuaternion, blendStates, createAircraftModel, createFlight, emptyState } from '../aircraft.js';
import { oceanControl, oceanReadout } from '../control.js';
import { DEFAULT_TRANSITION } from '../scene.js';
import { seaStats } from '../spectrum.js';
import { computeLight, pathTransmittance, skyRadiance, sunVisible } from './atmosphere.js';
import { Bloom } from './bloom.js';
import { FINE_SHARE, FlowField } from './flow.js';
import { FoamField, GATE_PERIOD } from './foam.js';
import { HorizonHud } from './hud.js';
import { LineSet } from './lines.js';
import { Exposure, PostPass, modeIndex } from './post.js';
import { AdaptiveQuality, PRESETS, qualityFromUrl } from './quality.js';
import { randomSample } from './randomizer.js';
import { advanceReplay, createReplay, replayLines, replayPosition } from './replay.js';
import { isTimed } from '../aircraft/track.js';
import { OnboardRig, attitudeOf, blendPoses, createRig, emptyPose } from './rigs.js';
import { SeaSampler } from './sampler.js';
import { CLOUD_PERIOD, EnvironmentMap, SKY_GLSL, SkyLut, WATER_GLSL, createSkyUniforms, horizonDip } from './sky.js';
import {
	LIGHT_ANGLES,
	SEA_ANGLES,
	approach,
	blendLight,
	blendSea,
	copyLight,
	copySea,
	resolveAircraft,
	resolveLight,
	resolveSea
} from './state.js';
import { LACE_PERIOD, OceanSurface } from './surface.js';
import { createFoamTexture, createNoiseTexture } from './textures.js';
import { DEG, clamp, damp, lerp, parseColor, smootherstep, smoothstep, srgbToLinear } from './util.js';
import { JACOBIAN_WEIGHTS, WaveSimulation } from './waves.js';

/** @typedef {import('../scene.js').SceneSpec} SceneSpec */
/** @typedef {import('../aircraft.js').AircraftState} AircraftState */
/** @typedef {import('./state.js').ResolvedAircraft} ResolvedAircraft */
/** @typedef {import('./rigs.js').CameraPose} CameraPose */

/**
 * One scene on screen: its flight, its camera rig, its tracks, and where its frame sits.
 * @typedef {object} Layer
 * @property {'layer'} kind
 * @property {SceneSpec} spec
 * @property {ResolvedAircraft | null} aircraft
 * @property {import('../aircraft.js').Flight | null} flight
 * @property {{ east: number, north: number }} shift
 * @property {import('./rigs.js').Rig} rig
 * @property {LineSet | null} tracks
 * @property {Array<{ id: string, points: Array<[number, number]>, altitudes?: number[], loopSeconds: number }>} markers
 *   markers looping along untimed tracks
 * @property {import('./replay.js').Replay | null} replay the scene's recorded (timed) tracks and their clock
 * @property {number} elapsed simulation seconds since the layer was created (its flight's time)
 * @property {AircraftState} abs absolute aircraft state, refreshed every frame
 */

/**
 * Two layers mid-blend, frozen at a weight, when a scene changes during a transition.
 * @typedef {object} Composite
 * @property {'composite'} kind
 * @property {Node} a
 * @property {Node} b
 * @property {number} w eased weight of b
 * @property {number} p raw progress of the blend, 0–1
 * @property {CameraPose} pose
 */

/** @typedef {Layer | Composite} Node */

/**
 * @typedef {object} BlendedAircraft
 * @property {AircraftState} state
 * @property {{ pitch: number, yaw: number, fov: number }} gimbal
 * @property {number} scale
 * @property {number} footprint 0–1
 * @property {number} presence 0–1
 * @property {number} turbulence
 */

/**
 * @typedef {object} EngineHost
 * @property {HTMLCanvasElement} canvas
 * @property {HTMLCanvasElement} hud
 * @property {HTMLElement} root sized element the stage fills
 * @property {(status: 'loading' | 'running' | 'fallback' | 'error', reason?: string) => void} onStatus
 * @property {(mode: string) => void} [onSplitMode]
 */

/**
 * @typedef {object} EngineOptions
 * @property {boolean} [reducedMotion]
 */

const ACCENT = parseColor('#4fd8e6');
const _forward = new THREE.Vector3();
const _quat = new THREE.Quaternion();
const _euler = new THREE.Euler();
/** How stabilised the gimbal is taken to be when its camera is not the one on screen. */
const FOOTPRINT_STABILIZED = 0.85;
const ACCENT_LINEAR = /** @type {[number, number, number]} */ ([ACCENT.r, ACCENT.g, ACCENT.b].map(srgbToLinear));

/** Fraction of the light a lens scatters into wide glare. */
const BLOOM_STRENGTH = 0.06;

/** Range, m, beyond which every other drop line under a track fades (see buildTracks). */
const DROP_LOD = 2500;

/** Randomised samples change every this many seconds. */
const SAMPLE_INTERVAL = 3.2;
const SAMPLE_BLEND = 0.7;

export class OceanEngine {
	/**
	 * @param {EngineHost} host
	 * @param {EngineOptions} [options]
	 */
	constructor(host, options = {}) {
		this.host = host;
		this.reducedMotion = Boolean(options.reducedMotion);
		this.disposed = false;
		this.failed = false;

		const renderer = new THREE.WebGLRenderer({
			canvas: host.canvas,
			antialias: false,
			alpha: false,
			depth: false,
			stencil: false,
			powerPreference: 'high-performance',
			preserveDrawingBuffer: false
		});
		if (!renderer.extensions.has('EXT_color_buffer_float')) {
			renderer.dispose();
			throw new Error('float render targets unavailable');
		}
		renderer.autoClear = false;
		renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
		renderer.toneMapping = THREE.NoToneMapping;
		renderer.debug.onShaderError = (gl, program, vertex, fragment) => {
			const log = [gl.getProgramInfoLog(program), gl.getShaderInfoLog(vertex), gl.getShaderInfoLog(fragment)]
				.map((s) => (s ?? '').trim())
				.filter(Boolean)
				.join('\n');
			console.error(`Mayfly ocean: shader error\n${log}`);
			this.fail('shader compile error');
		};
		this.renderer = renderer;

		const urlQuality = qualityFromUrl();
		this.quality = new AdaptiveQuality(PRESETS[urlQuality ?? 'high'], urlQuality !== null);

		// --- simulation ----------------------------------------------------------------
		this.waves = new WaveSimulation(renderer, { size: this.quality.preset.fftSize, cascades: this.quality.preset.cascades });
		this.sampler = new SeaSampler(this.waves.noise, this.waves.size, this.waves.patches[0]);
		this.foamField = new FoamField(renderer, { size: this.quality.preset.foamSize });
		/** Metres the foam (and its lace) has drifted downwind, east and north. */
		this.foamShiftE = 0;
		this.foamShiftN = 0;
		this.gateScroll = 0;

		// --- sky -------------------------------------------------------------------------
		this.noiseTexture = createNoiseTexture();
		this.foamTexture = createFoamTexture();
		this.skyUniforms = createSkyUniforms();
		this.skyUniforms.uNoiseTex.value = this.noiseTexture;
		this.skyUniforms.uFoamTex.value = this.foamTexture;
		this.skyLut = new SkyLut(renderer);
		this.skyUniforms.uSkyLut.value = this.skyLut.target.texture;
		this.environment = new EnvironmentMap(renderer, this.skyUniforms);

		// --- targets ---------------------------------------------------------------------
		this.depthTexture = new THREE.DepthTexture(1, 1);
		this.mainTarget = new THREE.WebGLRenderTarget(1, 1, {
			count: 3,
			type: THREE.HalfFloatType,
			format: THREE.RGBAFormat,
			minFilter: THREE.NearestFilter,
			magFilter: THREE.NearestFilter,
			depthBuffer: true,
			depthTexture: this.depthTexture,
			generateMipmaps: false
		});
		this.uiTarget = new THREE.WebGLRenderTarget(1, 1, {
			type: THREE.HalfFloatType,
			format: THREE.RGBAFormat,
			minFilter: THREE.NearestFilter,
			magFilter: THREE.NearestFilter,
			depthBuffer: true,
			depthTexture: this.depthTexture,
			generateMipmaps: false
		});
		// The aircraft's layer. Multisampled while a chase camera has it large on screen
		// (see airSamples); from overhead or onboard at a high pixel ratio it is small or
		// absent, and four samples of a full-screen half-float target are hundreds of MB.
		this.airTarget = new THREE.WebGLRenderTarget(1, 1, {
			type: THREE.HalfFloatType,
			format: THREE.RGBAFormat,
			minFilter: THREE.LinearFilter,
			magFilter: THREE.LinearFilter,
			depthBuffer: true,
			samples: 4,
			generateMipmaps: false
		});
		/** Whether airTarget's texture is known to hold nothing, so it need not be redrawn. */
		this.airEmpty = false;
		this.emptyScene = new THREE.Scene();

		// --- scenes ----------------------------------------------------------------------
		this.camera = new THREE.PerspectiveCamera(50, 16 / 9, 0.5, 200000);
		this.mainScene = new THREE.Scene();
		this.mainScene.matrixWorldAutoUpdate = true;
		this.surface = new OceanSurface(this.skyUniforms, this.waves.count);
		this.dome = this.createDome();
		this.mainScene.add(this.dome, this.surface.group);

		this.airScene = new THREE.Scene();
		this.airFog = new THREE.FogExp2(0x9fb4c8, 0);
		this.airScene.fog = this.airFog;
		this.model = createAircraftModel();
		this.model.object.visible = false;
		this.airScene.add(this.model.object);
		this.modelScale = 1;
		this.modelOpacity = 0;
		this.lightingKey = '';
		/** Whether warmUp has compiled the programs drawn only later. */
		this.warm = false;

		this.uiScene = new THREE.Scene();
		this.frustum = new LineSet({ capacity: 8, fade: [6000, 12000] });
		this.markers = new LineSet({ capacity: 512, lift: 1.5, fade: [8000, 16000] });
		/** How far the current scene's track lines reach before fading, m (see the track build). */
		this.trackReach = 20000;
		this.flow = new FlowField(renderer);
		this.uiScene.add(this.frustum.mesh, this.markers.mesh, this.flow.mesh);

		this.post = new PostPass();
		this.bloom = new Bloom(renderer);
		this.meter = new Exposure(renderer);
		/** @type {THREE.Vector3} */ (this.post.uniforms.uAccent.value).set(...ACCENT_LINEAR);

		this.hud = new HorizonHud(host.hud);

		// --- state -----------------------------------------------------------------------
		this.simTime = 0;
		this.clock = 0;
		/** @type {number | null} */
		this.last = null;
		this.fps = 60;
		this.frameTime = 1 / 60;
		this.frames = 0;
		this.originE = 0;
		this.originN = 0;
		this.flowShift = new THREE.Vector2();
		this.offsets = { yaw: 0, pitch: 0, zoom: 1 };
		this.interactive = false;

		/** @type {SceneSpec | null} */
		this.spec = null;
		/** @type {Node | null} */
		this.root = null;
		/** @type {Layer | null} */
		this.currentLayer = null;
		this.transition = { start: 0, duration: 0 };
		/** @type {BlendedAircraft | null} */
		this.aircraft = null;
		this.pose = emptyPose();
		this.cameraMode = /** @type {'free' | 'chase' | 'onboard'} */ ('free');

		/** @type {import('./state.js').ResolvedSea} */
		this.sea = resolveSea({ windSpeed: 8, windDirection: 0 });
		this.seaFrom = copySea(this.sea);
		/** @type {import('./state.js').ResolvedLight} */
		this.light = resolveLight({ sunElevation: 30, sunAzimuth: 180 });
		this.lightFrom = copyLight(this.light);
		this.seaKey = '';
		this.stats = seaStats(this.sea);
		/** @type {import('./atmosphere.js').LightState | null} */
		this.lightState = null;
		this.lightKey = '';
		this.exposure = 1;

		this.overlays = { flow: 0, grid: 0, horizon: 0 };
		/** Overlay amounts when the current transition began. */
		this.overlaysFrom = { ...this.overlays };
		this.splitFrom = 0;
		this.modeA = 0;
		this.modeB = 0;
		this.modeMix = 1;
		this.splitA = 1;
		this.splitB = 1;
		this.splitMix = 1;
		this.splitAmount = 0;
		this.splitPos = 0.5;
		this.splitModeName = '';
		this.sweepClock = 0;

		this.sample = 0;
		this.sampleClock = 0;
		/** @type {import('./randomizer.js').Sample | null} */
		this.sampleState = null;

		this.readoutClock = 0;
		this.fogClock = 0;
		/** @type {THREE.PerspectiveCamera | null} */
		this.gimbalCam = null;
		this.footRig = new OnboardRig({ mode: 'onboard', stabilized: FOOTPRINT_STABILIZED });
		this.gimbalAim = { pitch: NaN, yaw: NaN };
		this.status = /** @type {'loading' | 'running' | 'fallback' | 'error'} */ ('loading');

		// --- DOM -------------------------------------------------------------------------
		this.resizeObserver = new ResizeObserver(() => this.resize());
		this.resizeObserver.observe(host.root);
		// A window dragged to a screen with another pixel ratio (a 1× projector, a Retina
		// laptop) keeps its CSS size, so the observer never fires: watch the ratio itself.
		/** @type {MediaQueryList | null} */
		this.dprQuery = null;
		this.onDprChange = () => {
			this.watchDpr();
			this.resize();
		};
		this.watchDpr();
		this.onVisibility = () => {
			this.last = null;
		};
		document.addEventListener('visibilitychange', this.onVisibility);
		this.onContextLost = (/** @type {Event} */ event) => {
			event.preventDefault();
			this.lost = true;
			this.setStatus('loading', 'context lost');
		};
		this.lost = false;
		host.canvas.addEventListener('webglcontextlost', this.onContextLost);
		// Taking the camera: while interactive, a drag anywhere on the stage steers the camera,
		// including over slide copy (the deck lays its slides over the canvas), but never one
		// that starts on a control. Listening on the window is what makes that possible.
		this.drag = /** @type {{ id: number, x: number, y: number } | null} */ (null);
		this.onPointerDown = (/** @type {PointerEvent} */ e) => {
			if (!this.interactive || this.drag) return;
			if (e.pointerType === 'mouse' && e.button !== 0) return;
			if (!this.ownsPointer(e)) return;
			this.drag = { id: e.pointerId, x: e.clientX, y: e.clientY };
			this.setCursor('grabbing');
		};
		this.onPointerMove = (/** @type {PointerEvent} */ e) => {
			if (!this.drag || e.pointerId !== this.drag.id) return;
			const dx = e.clientX - this.drag.x;
			const dy = e.clientY - this.drag.y;
			this.drag.x = e.clientX;
			this.drag.y = e.clientY;
			const k = (this.pose.fov / Math.max(1, host.root.clientHeight)) * 1.1;
			// Free and onboard rigs turn the camera (drag the view); a chase rig swings round
			// the aircraft (drag the world).
			const chase = this.cameraMode === 'chase';
			this.offsets.yaw -= dx * k * (chase ? -1.6 : 1);
			this.offsets.pitch = clamp(this.offsets.pitch + dy * k * (chase ? 1.2 : 1), -70, 70);
		};
		this.onPointerUp = (/** @type {PointerEvent} */ e) => {
			if (this.drag && e.pointerId === this.drag.id) {
				this.drag = null;
				this.setCursor(this.interactive ? 'grab' : '');
			}
		};
		this.onWheel = (/** @type {WheelEvent} */ e) => {
			if (!this.interactive || !this.ownsPointer(e)) return;
			e.preventDefault();
			// Distance (chase), height (free) or lens (onboard), within sane limits.
			this.offsets.zoom = clamp(this.offsets.zoom * Math.exp(e.deltaY * 0.0012), 0.4, 3.5);
		};
		window.addEventListener('pointerdown', this.onPointerDown);
		window.addEventListener('pointermove', this.onPointerMove);
		window.addEventListener('pointerup', this.onPointerUp);
		window.addEventListener('pointercancel', this.onPointerUp);
		window.addEventListener('wheel', this.onWheel, { passive: false });
		/** The element the taken camera's cursor and touch-action go on: the stage and what covers it. */
		this.helmSurface = host.root.parentElement ?? host.root;
		this.setInteractive(false);

		this.resize();
		/** @type {number} */
		this.raf = requestAnimationFrame(this.frame);
		// The engine itself, for poking at uniforms from the console while developing.
		if (import.meta.env?.DEV) /** @type {any} */ (window).__mayflyEngine = this;
	}

	// ------------------------------------------------------------------ public API

	/**
	 * Show a scene, blending in from whatever is on screen.
	 * @param {SceneSpec} spec
	 */
	setScene(spec) {
		if (this.spec === spec) return;
		const first = this.spec === null;
		this.spec = spec;
		this.sample = 0;
		this.sampleClock = 0;
		this.sampleState = null;
		const duration = first ? 0 : (spec.transition ?? DEFAULT_TRANSITION) * (this.reducedMotion ? 0.55 : 1);
		if (spec.randomize) {
			this.sample = 1;
			this.enter(this.sampleScene(spec), duration);
		} else {
			this.enter(spec, duration);
		}
		this.quality.settle(Math.max(2.5, duration + 1));
	}

	/** Listen for the next change of the screen's pixel ratio (a query matches one ratio). */
	watchDpr() {
		this.dprQuery?.removeEventListener('change', this.onDprChange);
		this.dprQuery = null;
		if (this.disposed || typeof matchMedia !== 'function') return;
		this.dprQuery = matchMedia(`(resolution: ${window.devicePixelRatio || 1}dppx)`);
		this.dprQuery.addEventListener('change', this.onDprChange);
	}

	/** @param {boolean} on */
	setInteractive(on) {
		this.interactive = on;
		const c = this.host.canvas;
		c.style.pointerEvents = on ? 'auto' : 'none';
		this.helmSurface.style.touchAction = on ? 'none' : '';
		if (!on) this.drag = null;
		this.setCursor(on ? 'grab' : '');
	}

	/** @param {string} cursor */
	setCursor(cursor) {
		this.helmSurface.style.cursor = cursor;
		this.host.canvas.style.cursor = cursor;
	}

	/**
	 * Whether a pointer or wheel event belongs to the taken camera: inside the stage, and not on
	 * a control a slide put in the presenter's hand.
	 * @param {Event} e
	 */
	ownsPointer(e) {
		const target = e.target instanceof Element ? e.target : null;
		if (!target || !this.helmSurface.contains(target)) return false;
		return !target.closest(
			'input, button, select, textarea, a[href], label, [role="slider"], [role="radio"], [role="button"], [role="dialog"], [contenteditable="true"], [data-no-helm]'
		);
	}

	/**
	 * Free everything: the frame loop, the listeners, every GPU resource and, unless the
	 * canvas is about to get a new engine (after a context restore), the context itself.
	 * @param {boolean} [release] also release the WebGL context now rather than at GC
	 */
	dispose(release = true) {
		if (this.disposed) return;
		this.disposed = true;
		cancelAnimationFrame(this.raf);
		this.resizeObserver.disconnect();
		document.removeEventListener('visibilitychange', this.onVisibility);
		this.dprQuery?.removeEventListener('change', this.onDprChange);
		this.dprQuery = null;
		const c = this.host.canvas;
		c.removeEventListener('webglcontextlost', this.onContextLost);
		window.removeEventListener('pointerdown', this.onPointerDown);
		window.removeEventListener('pointermove', this.onPointerMove);
		window.removeEventListener('pointerup', this.onPointerUp);
		window.removeEventListener('pointercancel', this.onPointerUp);
		window.removeEventListener('wheel', this.onWheel);
		this.drag = null;
		this.setCursor('');
		this.helmSurface.style.touchAction = '';
		c.style.pointerEvents = 'none';
		this.hud.clear();
		if (typeof window !== 'undefined') {
			delete (/** @type {any} */ (window).__mayfly);
			if (/** @type {any} */ (window).__mayflyEngine === this) delete (/** @type {any} */ (window).__mayflyEngine);
		}
		if (this.lost) {
			// The context these were made in is gone and every GPU object went with it;
			// deleting them in a restored context only raises errors. The renderer still
			// lets go of its own listeners on the canvas.
			this.renderer.dispose();
			return;
		}
		this.disposeNode(this.root);
		this.waves.dispose();
		this.foamField.dispose();
		this.skyLut.dispose();
		this.environment.dispose();
		this.noiseTexture.dispose();
		this.foamTexture.dispose();
		this.mainTarget.dispose();
		this.uiTarget.dispose();
		this.airTarget.dispose();
		this.depthTexture.dispose();
		this.surface.dispose();
		this.dome.geometry.dispose();
		/** @type {THREE.Material} */ (this.dome.material).dispose();
		this.model.dispose();
		this.frustum.dispose();
		this.markers.dispose();
		this.flow.dispose();
		this.post.dispose();
		this.bloom.dispose();
		this.meter.dispose();
		this.renderer.dispose();
		if (release && !this.renderer.getContext().isContextLost()) this.renderer.forceContextLoss();
	}

	// ------------------------------------------------------------------ scene layers

	/**
	 * @param {SceneSpec} spec
	 * @param {number} duration seconds
	 */
	enter(spec, duration) {
		const layer = this.createLayer(spec);
		if (this.root && duration > 0) {
			const previous = this.root;
			// Collapse anything older than one blend so rapid key presses stay cheap: the
			// oldest blend (frozen where it was interrupted) becomes whichever of its two
			// scenes it was showing more of, so what is on screen barely moves.
			if (previous.kind === 'composite' && previous.a.kind === 'composite') {
				const oldest = previous.a;
				const keep = oldest.w < 0.5 ? oldest.a : oldest.b;
				this.disposeNode(oldest.w < 0.5 ? oldest.b : oldest.a);
				previous.a = keep;
			}
			/** @type {Composite} */
			const composite = { kind: 'composite', a: previous, b: layer, w: 0, p: 0, pose: emptyPose() };
			this.root = composite;
			this.transition = { start: this.clock, duration };
			this.seaFrom = copySea(this.sea);
			this.lightFrom = copyLight(this.light);
			this.overlaysFrom = { ...this.overlays };
			this.splitFrom = this.splitAmount;
		} else {
			this.disposeNode(this.root);
			this.root = layer;
			this.transition = { start: this.clock, duration: 0 };
		}
		this.currentLayer = layer;
		this.cameraMode = layer.rig.mode;
		this.updateAirSamples();
	}

	/**
	 * Multisample the aircraft's layer only when it needs it: while a chase camera is on
	 * screen (the aircraft fills a good part of the frame), or on a low pixel ratio
	 * screen, where a small aircraft's edges would stair-step. Changing it reallocates
	 * the target, so it happens only as a scene arrives or leaves.
	 */
	updateAirSamples() {
		let chase = false;
		this.forEachLayer(this.root, (layer) => {
			if (layer.rig.mode === 'chase' && layer.aircraft) chase = true;
		});
		const root = this.host.root;
		const dpr = this.quality.pixelRatio(Math.max(1, root.clientWidth), Math.max(1, root.clientHeight));
		const samples = chase || dpr < 1.5 ? 4 : 0;
		if (samples === this.airTarget.samples) return;
		this.airTarget.samples = samples;
		// three sets a render target up on first use; disposing it makes the next use
		// allocate it again with the new sample count.
		this.airTarget.dispose();
		this.airEmpty = false;
	}

	/**
	 * Build a layer, shifting its frame so its aircraft (or free camera) starts where
	 * the current one is.
	 * @param {SceneSpec} spec
	 * @returns {Layer}
	 */
	createLayer(spec) {
		const aircraft = spec.aircraft ? resolveAircraft(spec.aircraft) : null;
		let flight = null;
		if (aircraft && spec.aircraft) {
			flight = createFlight(
				{ ...spec.aircraft, turbulence: aircraft.turbulence * (this.reducedMotion ? 0.35 : 1) },
				spec.tracks ?? [],
				spec.trackClock
			);
			flight.step(1e-3);
			// Join an orbit where its heading matches the aircraft's, so it doesn't swing round.
			if (aircraft.path === 'orbit' && this.aircraft) {
				const period = (2 * Math.PI * aircraft.orbitRadius) / Math.max(aircraft.speed, 1);
				let best = 0;
				let bestErr = Infinity;
				const probe = createFlight(spec.aircraft, spec.tracks ?? [], spec.trackClock);
				const steps = 72;
				for (let i = 0; i <= steps; i++) {
					const s = probe.step(i === 0 ? 1e-3 : period / steps);
					const err = Math.abs(((s.heading - this.aircraft.state.heading + 540) % 360) - 180);
					if (err < bestErr) {
						bestErr = err;
						best = i;
					}
				}
				for (let i = 0; i < best; i++) flight.step(period / steps);
			}
		}
		/** @type {{ east: number, north: number }} */
		const shift = { east: 0, north: 0 };
		if (this.root) {
			if (flight && this.aircraft && this.aircraft.presence > 0.05) {
				shift.east = this.aircraft.state.x - flight.state.x;
				shift.north = this.aircraft.state.y - flight.state.y;
			} else if (flight) {
				// No aircraft yet: put it ahead of the camera.
				shift.east = this.pose.east - flight.state.x;
				shift.north = this.pose.north - flight.state.y;
			} else if (spec.camera.mode !== 'chase' && spec.camera.mode !== 'onboard') {
				const cam = /** @type {import('../scene.js').OceanCamera} */ (spec.camera);
				shift.east = this.pose.east - (cam.x ?? 0);
				shift.north = this.pose.north - (cam.y ?? 0);
			}
		}
		const rig = createRig(spec.camera);
		/** @type {Layer['markers']} */
		const markers = [];
		const flownId = aircraft?.path === 'track' ? aircraft.track : undefined;
		for (const t of spec.tracks ?? []) {
			// A recorded track's marker is placed by the scene's clock (see replay.js).
			if (t.marker && t.id !== flownId && !isTimed(t)) {
				markers.push({ id: t.id, points: t.points, altitudes: t.altitudes, loopSeconds: t.loopSeconds ?? 18 });
			}
		}
		const replay = createReplay(spec, flownId);
		const elapsed = flight ? flight.state.time : 0;
		if (replay) advanceReplay(replay, elapsed);
		const tracks = this.buildTracks(spec, replay);
		return {
			kind: 'layer',
			spec,
			aircraft,
			flight,
			shift,
			rig,
			tracks,
			markers,
			replay,
			elapsed,
			abs: emptyState()
		};
	}

	/**
	 * The tracks of a scene: each line at altitude, its ground track on the sea and
	 * faint drop lines between them so the height reads.
	 * A recorded track is drawn along the same curve the aircraft and the markers follow
	 * (its interpolation in time, subdivided where it turns), with the recording's times
	 * on every vertex so the overlay can show a window of it round the clock.
	 * @param {SceneSpec} spec
	 * @param {import('./replay.js').Replay | null} replay
	 */
	buildTracks(spec, replay) {
		const tracks = spec.tracks ?? [];
		if (tracks.length === 0) return null;
		/** @type {import('./lines.js').Polyline[]} */
		const lines = [];
		const lift = 2.5;
		for (const t of tracks) {
			const c = parseColor(t.color);
			/** @type {[number, number, number]} */
			const color = [srgbToLinear(c.r), srgbToLinear(c.g), srgbToLinear(c.b)];
			const width = t.width ?? 3;
			const dash = t.dashed ? 34 : 0;
			const timed = replay?.tracks.get(t.id)?.timed;
			const path = timed ? curveOf(timed) : null;
			const pts = path ? path.points : t.points;
			const times = path ? path.times : undefined;
			const alts = path ? path.altitudes : t.altitudes;
			const air = pts.map(([e, n], i) => new THREE.Vector3(e, alts ? alts[i] : lift, -n));
			if (alts) {
				const ground = pts.map(([e, n]) => new THREE.Vector3(e, lift, -n));
				lines.push({ points: ground, color, alpha: c.a * 0.42, width: Math.max(1.2, width * 0.6), dash, times });
				// Drop lines every ~200 m along the path, thinning out with distance: every
				// other one fades beyond DROP_LOD m, every fourth beyond twice that, and so
				// on, so a long track seen from near the aircraft is not a picket fence.
				let acc = 0;
				let next = 0;
				let k = 0;
				for (let i = 0; i < pts.length; i++) {
					if (i > 0) acc += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
					const last = i === pts.length - 1;
					if (acc >= next || last) {
						let level = 0;
						while (level < 8 && k > 0 && (k >> level) % 2 === 0) level++;
						const lod = k === 0 || last ? 0 : DROP_LOD * 2 ** level;
						const at = times ? [times[i], times[i]] : undefined;
						lines.push({ points: [air[i], ground[i]], color, alpha: c.a * 0.3, width: 1, lod, times: at });
						next = acc + 200;
						k++;
					}
				}
			}
			lines.push({ points: air, color, alpha: c.a, width, dash, times });
		}
		// Sized to the scene: a recorded flight is thousands of segments, drawn once.
		// A ground track 2.5 m up is cut by crests seen at a grazing angle from afar; a depth
		// bias that grows with range keeps it whole without moving it on screen.
		// The lines fade out and are culled with range. From a chase camera a short reach
		// keeps the other passes of a long flight out of the frame; a whole route framed
		// from tens of kilometres up needs a reach to match, so it scales with the extent
		// of what is drawn (the marker lines follow it, see the replay update).
		let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
		for (const line of lines) {
			for (const q of line.points) {
				if (q.x < minX) minX = q.x;
				if (q.x > maxX) maxX = q.x;
				if (q.z < minZ) minZ = q.z;
				if (q.z > maxZ) maxZ = q.z;
			}
		}
		const extent = lines.length ? Math.hypot(maxX - minX, maxZ - minZ) : 0;
		const reach = Math.max(20000, extent * 3.5);
		this.trackReach = reach;
		const set = new LineSet({ capacity: LineSet.segments(lines), lift: 1, liftRange: 0.006, fade: [reach * 0.45, reach] });
		set.set(lines);
		set.setOpacity(0);
		set.setResolution(this.post.uniforms.uResolution.value.x, this.post.uniforms.uResolution.value.y);
		this.uiScene.add(set.mesh);
		return set;
	}

	/**
	 * @param {Node | null} node
	 * @param {(layer: Layer, weight: number) => void} fn
	 * @param {number} [weight]
	 */
	forEachLayer(node, fn, weight = 1) {
		if (!node) return;
		if (node.kind === 'layer') {
			fn(node, weight);
			return;
		}
		this.forEachLayer(node.a, fn, weight * (1 - node.w));
		this.forEachLayer(node.b, fn, weight * node.w);
	}

	/**
	 * The blended aircraft of a node, or null when none of its layers has one.
	 * @param {Node} node
	 * @returns {BlendedAircraft | null}
	 */
	aircraftOf(node) {
		if (node.kind === 'layer') {
			if (!node.aircraft || !node.flight) return null;
			return {
				state: node.abs,
				gimbal: node.aircraft.gimbal,
				scale: node.aircraft.scale,
				footprint: node.aircraft.footprint ? 1 : 0,
				presence: 1,
				turbulence: node.aircraft.turbulence
			};
		}
		const a = this.aircraftOf(node.a);
		const b = this.aircraftOf(node.b);
		if (!a && !b) return null;
		if (!a || !b) {
			const only = /** @type {BlendedAircraft} */ (a ?? b);
			const w = a ? 1 - node.w : node.w;
			return { ...only, presence: only.presence * w, footprint: only.footprint * w };
		}
		const t = node.w;
		return {
			state: blendStates(a.state, b.state, t),
			gimbal: {
				pitch: lerp(a.gimbal.pitch, b.gimbal.pitch, t),
				yaw: lerp(a.gimbal.yaw, b.gimbal.yaw, t),
				fov: lerp(a.gimbal.fov, b.gimbal.fov, t)
			},
			scale: Math.exp(lerp(Math.log(a.scale), Math.log(b.scale), t)),
			footprint: lerp(a.footprint, b.footprint, t),
			presence: lerp(a.presence, b.presence, t),
			turbulence: lerp(a.turbulence, b.turbulence, t)
		};
	}

	/**
	 * @param {Node} node
	 * @param {import('./rigs.js').RigContext} ctx
	 * @returns {CameraPose}
	 */
	poseOf(node, ctx) {
		if (node.kind === 'layer') {
			ctx.shift = node.shift;
			const replay = node.replay;
			ctx.include = replay ? (id) => replayPosition(replay, id, node.shift) : undefined;
			const blended = ctx.aircraft;
			if (node.rig.mode === 'free' || !blended || !node.flight || !node.aircraft) return node.rig.update(ctx);
			// A rig films its own scene's aircraft, so its smoothing only ever sees that
			// flight; the pose is then carried onto the aircraft actually on screen (the
			// blend of every scene's), keeping the framing however far a transition moves it.
			ctx.aircraft = { state: node.abs, gimbal: node.aircraft.gimbal, scale: node.aircraft.scale };
			const pose = node.rig.update(ctx);
			ctx.aircraft = blended;
			pose.east += blended.state.x - node.abs.x;
			pose.north += blended.state.y - node.abs.y;
			pose.up += blended.state.altitude - node.abs.altitude;
			if (node.rig.mode === 'chase') pose.up = Math.max(pose.up, ctx.seaHeight(pose.east, pose.north) + 2);
			return pose;
		}
		const a = this.poseOf(node.a, ctx);
		const b = this.poseOf(node.b, ctx);
		const from = this.modeOf(node.a);
		const to = this.modeOf(node.b);
		// Into the gimbal: the camera closes on the aircraft along its line of sight in the
		// first part of the transition, holding its aim so the aircraft stays where it was
		// framed as it grows, turns to the gimbal's view only once it is inside the fading
		// airframe, and rides the aircraft for the rest. Out of the gimbal is the reverse.
		if (from === 'chase' && to === 'onboard') {
			const t = smootherstep(node.p / 0.7);
			return blendPoses(a, b, t, node.pose, smootherstep((t - 0.5) / 0.5));
		}
		if (from === 'onboard' && to === 'chase') {
			const t = smootherstep((node.p - 0.1) / 0.9);
			return blendPoses(a, b, t, node.pose, smootherstep(t / 0.5));
		}
		// To or from a camera placed in the world (an overhead view of a whole route, tens
		// of kilometres up), blend what it looks at so it pulls back rather than skims.
		const aimed = from === 'free' || to === 'free';
		return blendPoses(a, b, node.w, node.pose, node.w, aimed);
	}

	/**
	 * The rig mode a node mostly shows.
	 * @param {Node} node
	 * @returns {'free' | 'chase' | 'onboard'}
	 */
	modeOf(node) {
		if (node.kind === 'layer') return node.rig.mode;
		return this.modeOf(node.w >= 0.5 ? node.b : node.a);
	}

	// ------------------------------------------------------------------ frame

	/** @param {number} ms */
	frame = (ms) => {
		if (this.disposed) return;
		this.raf = requestAnimationFrame(this.frame);
		if (this.failed || this.lost) return;
		if (document.hidden) {
			this.last = null;
			return;
		}
		const now = ms / 1000;
		const rawDt = this.last === null ? 1 / 60 : clamp(now - this.last, 0, 0.1);
		this.last = now;
		// The mean frame time, not the mean of the rates: a 60/30 Hz mix averages 45 fps, and
		// averaging rates would read it as 50.
		if (rawDt > 0) {
			this.frameTime += (Math.max(rawDt, 1 / 240) - this.frameTime) * 0.06;
			this.fps = 1 / this.frameTime;
		}
		this.frames++;
		const timeScale = this.reducedMotion ? 0.35 : 1;
		const dt = rawDt * timeScale;
		this.clock += rawDt;
		this.simTime += dt;

		try {
			this.update(rawDt, dt);
			this.render(rawDt);
		} catch (error) {
			console.error(error);
			this.fail(error instanceof Error ? error.message : String(error));
			return;
		}

		const step = this.quality.frame(rawDt, this.fps);
		if (step === 'scale') this.resize(false);
		if (step === 'fft' || step === 'cascades') this.rebuildWaves(this.quality.preset.cascades);

		if (this.status !== 'running' && this.frames > 2) this.setStatus('running');
	};

	/**
	 * @param {number} rawDt wall-clock seconds
	 * @param {number} dt simulation seconds
	 */
	update(rawDt, dt) {
		const spec = this.spec;
		if (!spec || !this.root || !this.currentLayer) return;

		// Randomised samples: a new one every few seconds, blended in quickly.
		if (spec.randomize) {
			this.sampleClock += rawDt;
			const interval = SAMPLE_INTERVAL * (this.reducedMotion ? 2 : 1);
			if (this.sampleClock >= interval) {
				this.sampleClock = 0;
				this.sample++;
				const derived = this.sampleScene(spec);
				this.enter(derived, SAMPLE_BLEND * (this.reducedMotion ? 1.4 : 1));
				this.seaFrom = copySea(this.sea);
				this.lightFrom = copyLight(this.light);
			}
		}

		// Transition weight.
		const tr = this.transition;
		const p = tr.duration > 0 ? clamp((this.clock - tr.start) / tr.duration, 0, 1) : 1;
		const w = smootherstep(p);
		if (this.root.kind === 'composite') {
			this.root.w = w;
			this.root.p = p;
			if (p >= 1) {
				this.disposeNode(this.root.a);
				this.root = this.currentLayer;
				this.updateAirSamples();
			}
		}

		// Sea and light.
		const sceneSea = this.sampleState ? this.sampleState.sea : resolveSea(spec.sea);
		const sceneLight = this.sampleState ? this.sampleState.light : resolveLight(spec.light);
		const seaTarget = resolveSea(sceneSea, oceanControl.sea);
		const lightTarget = resolveLight(sceneLight, oceanControl.light);
		if (p < 1) {
			this.sea = blendSea(this.seaFrom, seaTarget, w);
			// The light leads the camera, settling by two thirds of the way: a camera still
			// high and looking down while the sun swings round to it at a middling height
			// sees the sun's whole glitter patch, blown out, before the new view arrives.
			this.light = blendLight(this.lightFrom, lightTarget, smootherstep(p / 0.65));
		} else {
			const k = 1 - Math.exp(-rawDt / 0.3);
			approach(this.sea, seaTarget, k, SEA_ANGLES);
			approach(this.light, lightTarget, k, LIGHT_ANGLES);
		}

		// Flights and recorded-track clocks, then the blended aircraft every rig films.
		// Each scene keeps its own clock through a transition.
		this.forEachLayer(this.root, (layer) => {
			if (layer.flight) {
				const s = layer.flight.step(dt);
				Object.assign(layer.abs, s);
				layer.abs.x = s.x + layer.shift.east;
				layer.abs.y = s.y + layer.shift.north;
				layer.elapsed = s.time;
			} else {
				layer.elapsed += dt;
			}
			if (layer.replay) advanceReplay(layer.replay, layer.elapsed);
		});
		this.aircraft = this.aircraftOf(this.root);
		if (this.aircraft) this.aimGimbal(this.aircraft);

		// The presenter's camera offsets ease home when the presenter lets go.
		if (!this.interactive) {
			this.offsets.yaw = damp(this.offsets.yaw, 0, 0.6, rawDt);
			this.offsets.pitch = damp(this.offsets.pitch, 0, 0.6, rawDt);
			this.offsets.zoom = Math.exp(damp(Math.log(this.offsets.zoom), 0, 0.6, rawDt));
		}

		this.sampler.update(this.sea, this.clock);
		const seaHs = this.stats.significantHeight;
		/** @type {import('./rigs.js').RigContext} */
		const ctx = {
			dt,
			time: this.simTime,
			aircraft: this.aircraft
				? { state: this.aircraft.state, gimbal: this.aircraft.gimbal, scale: this.aircraft.scale }
				: null,
			cameraMount: this.model.cameraMount,
			seaHeight: (e, n) => this.sampler.sample(e, n, this.simTime).height + 0.35 * seaHs,
			seaMotion: (e, n) => {
				const cam = /** @type {import('../scene.js').OceanCamera} */ (spec.camera);
				return this.sampler.motion(e, n, this.simTime, cam.heading ?? 0);
			},
			offsets: this.offsets,
			aspect: this.camera.aspect,
			reducedMotion: this.reducedMotion,
			shift: { east: 0, north: 0 }
		};
		const pose = this.poseOf(this.root, ctx);
		this.pose.east = pose.east;
		this.pose.north = pose.north;
		this.pose.up = pose.up;
		this.pose.quaternion.copy(pose.quaternion);
		this.pose.fov = pose.fov;
		this.pose.heave = pose.heave;

		// Overlays and the split view fade with the scene: what leaves goes in the first two
		// thirds of a transition, what arrives comes in over the last two thirds. Changes
		// outside a transition (a slide's channel picker) ease on their own clock.
		const kOverlay = 1 - Math.exp(-rawDt / 0.32);
		const overlays = new Set(spec.overlays ?? []);
		/** @param {number} from @param {number} to @param {number} current */
		const fade = (from, to, current) => {
			if (p >= 1) return current + (to - current) * kOverlay;
			const k = to > from ? smoothstep(0.33, 1, p) : smoothstep(0, 0.67, p);
			return from + (to - from) * k;
		};
		for (const name of /** @type {const} */ (['flow', 'grid', 'horizon'])) {
			this.overlays[name] = fade(this.overlaysFrom[name], overlays.has(name) ? 1 : 0, this.overlays[name]);
		}

		const mode = modeIndex(spec.mode ?? 'shaded');
		if (mode !== this.modeB) {
			this.modeA = this.modeMix < 0.5 ? this.modeA : this.modeB;
			this.modeB = mode;
			this.modeMix = 0;
		}
		this.modeMix = Math.min(1, this.modeMix + rawDt / 0.6);

		const splitModeName = oceanControl.splitMode ?? spec.split?.mode ?? null;
		const hasSplit = Boolean(spec.split || oceanControl.splitMode);
		if (splitModeName) {
			const idx = modeIndex(splitModeName);
			if (idx !== this.splitB) {
				this.splitA = this.splitAmount < 0.02 ? idx : this.splitMix < 0.5 ? this.splitA : this.splitB;
				this.splitB = idx;
				this.splitMix = this.splitAmount < 0.02 ? 1 : 0;
			}
		}
		this.splitMix = Math.min(1, this.splitMix + rawDt / 0.6);
		this.splitAmount = fade(this.splitFrom, hasSplit ? 1 : 0, this.splitAmount);
		const shownSplit = hasSplit && splitModeName ? splitModeName : '';
		if (shownSplit !== this.splitModeName) {
			this.splitModeName = shownSplit;
			this.host.onSplitMode?.(shownSplit);
		}
		// A split arriving starts its sweep from its resting position.
		if (hasSplit && this.splitAmount < 0.02) this.sweepClock = 0;
		let targetPos = spec.split?.position ?? 0.5;
		if (!hasSplit) {
			// A split on its way out holds still while it fades.
			targetPos = this.splitPos;
		} else if (oceanControl.splitPosition !== null) {
			targetPos = oceanControl.splitPosition;
		} else if (spec.split?.sweep) {
			this.sweepClock += rawDt * (this.reducedMotion ? 0.4 : 1);
			// Ease back and forth across the middle right of the frame, eight seconds each
			// way: clear of the copy column on the left (the labels slide's card ends near
			// 0.47), and never so far right that the label channel is a sliver: it always
			// fills at least the right 28% of the frame.
			const lo = 0.52;
			const span = 0.2;
			const phase = (this.sweepClock / 8) % 2;
			const tri = phase < 1 ? phase : 2 - phase;
			// Start from the scene's resting position.
			const rest = (clamp(spec.split.position ?? 0.62, lo, lo + span) - lo) / span;
			const u = (tri + rest) % 2;
			targetPos = lo + span * smootherstep(u < 1 ? u : 2 - u);
		}
		this.splitPos = this.splitAmount < 0.02 ? targetPos : damp(this.splitPos, targetPos, 0.25, rawDt);
	}

	/**
	 * The scene for the current randomised sample: the slide's scene with the sample's
	 * sea and sky, and its camera — the aircraft's altitude, speed, gusts, gimbal and
	 * lens for an onboard rig, height, pitch and lens for a free one.
	 * @param {SceneSpec} spec
	 * @returns {SceneSpec}
	 */
	sampleScene(spec) {
		const sample = randomSample(this.sample, resolveSea(spec.sea), resolveLight(spec.light));
		this.sampleState = sample;
		/** @type {SceneSpec} */
		const derived = { ...spec };
		if (spec.camera.mode === 'onboard' && spec.aircraft) {
			const o = sample.onboard;
			derived.aircraft = {
				...spec.aircraft,
				altitude: o.altitude,
				speed: o.speed,
				turbulence: o.turbulence,
				gimbal: { ...spec.aircraft.gimbal, pitch: o.gimbalPitch, fov: o.fov }
			};
			derived.camera = { mode: 'onboard', stabilized: o.stabilized };
		} else if (spec.camera.mode !== 'chase' && spec.camera.mode !== 'onboard') {
			const f = sample.free;
			derived.camera = { ...spec.camera, height: f.height, pitch: f.pitch, fov: f.fov };
		}
		return derived;
	}

	/**
	 * Release a node's GPU resources (its tracks) and take them out of the overlay scene.
	 * @param {Node | null} node
	 */
	disposeNode(node) {
		this.forEachLayer(node, (layer) => {
			if (!layer.tracks) return;
			this.uiScene.remove(layer.tracks.mesh);
			layer.tracks.dispose();
			layer.tracks = null;
		});
	}

	/** @param {number} rawDt */
	render(rawDt) {
		const renderer = this.renderer;
		const spec = this.spec;
		if (!spec) return;

		// --- the sea state ---------------------------------------------------------------
		const seaKey = Object.values(this.sea)
			.map((v) => v.toFixed(3))
			.join(',');
		if (seaKey !== this.seaKey) {
			this.seaKey = seaKey;
			this.stats = seaStats(this.sea);
			this.skyUniforms.uHs.value = Math.max(0.05, this.stats.significantHeight);
		}
		this.waves.setSea(this.sea);

		// --- camera and floating origin ---------------------------------------------
		const pose = this.pose;
		if (Math.abs(pose.east - this.originE) > 2000 || Math.abs(pose.north - this.originN) > 2000) {
			const ne = Math.round(pose.east);
			const nn = Math.round(pose.north);
			this.flowShift.set(this.flowShift.x + (ne - this.originE), this.flowShift.y + (nn - this.originN));
			this.originE = ne;
			this.originN = nn;
		}
		const cam = this.camera;
		cam.position.copy(this.toWorld(pose.east, pose.north, pose.up));
		cam.quaternion.copy(pose.quaternion);
		cam.fov = pose.fov;
		const seaUnder = this.sampler.sample(pose.east, pose.north, this.simTime).height;
		const heightAbove = Math.max(0.3, pose.up - seaUnder);
		let near = clamp(heightAbove * 0.35, 0.3, 60);
		const air = this.aircraft;
		if (air && air.presence > 0.01) {
			const da = Math.hypot(air.state.x - pose.east, air.state.y - pose.north, air.state.altitude - pose.up);
			near = Math.min(near, Math.max(0.05, da - this.model.span * air.scale * 1.2));
		}
		cam.near = near;
		cam.far = Math.max(30000, Math.sqrt(2 * 6.36e6 * (heightAbove + 10)) * 2.2);
		cam.updateProjectionMatrix();
		cam.updateMatrixWorld();

		// --- light and sky -------------------------------------------------------------
		const altitude = Math.max(1, pose.up);
		const lightKey = `${Object.values(this.light).map((v) => v.toFixed(3)).join(',')}|${Math.log(altitude).toFixed(2)}`;
		if (lightKey !== this.lightKey || !this.lightState) {
			this.lightKey = lightKey;
			this.lightState = computeLight(this.light, altitude);
		}
		const ls = this.lightState;
		const su = this.skyUniforms;
		const dip = horizonDip(heightAbove);
		su.uSunDir.value.set(ls.sunDir[0], ls.sunDir[1], -ls.sunDir[2]);
		su.uDip.value = dip;
		const cover = ls.cloudCover;
		// The deck's shadow is applied where the light lands (cloudShadow); uSunVisible is its
		// average, for the far sea and the aircraft's environment.
		su.uSunE.value.set(...ls.sunIrradianceClear);
		su.uSunVisible.value = sunVisible(cover);
		su.uSunEClear.value.set(...ls.sunIrradianceClear);
		su.uSkyE.value.set(...ls.skyIrradiance);
		su.uCloudBase.value.set(...ls.cloudRadiance);
		su.uCloudCover.value = cover;
		su.uHazeB.value = ls.hazeBeta;
		su.uCamAlt.value = altitude;
		su.uSunDiscRadiance.value.set(...ls.sunIrradianceClear).multiplyScalar(1 / 6.8e-5);
		const wind = this.sea.windDirection * DEG;
		const drift = this.simTime * (2 + this.sea.windSpeed * 0.6);
		// The deck is read at (position + shift), so the drift is subtracted for it to move
		// downwind, with the sea and the foam.
		su.uCloudShift.value.set(
			mod(this.originE - Math.sin(wind) * drift, CLOUD_PERIOD),
			mod(this.originN - Math.cos(wind) * drift, CLOUD_PERIOD)
		);
		const mss = this.waves.meanSquareSlope;
		su.uFarVariance.value.set(mss * 0.5, mss * 0.5, 0);
		this.skyLut.update(ls.sunDir[1] > -1 ? Math.asin(clamp(ls.sunDir[1], -1, 1)) : 0, altitude, ls.hazeBeta, dip);
		// The meter does the rest (see Exposure); the scene's own setting is a bias on it.
		this.exposure = this.light.exposure;

		// --- simulation ------------------------------------------------------------------
		const simDt = rawDt * (this.reducedMotion ? 0.35 : 1);
		this.waves.update(this.simTime);
		// The break gate drifts slowly so the same water does not always break.
		this.gateScroll = (this.gateScroll + simDt * 1.7) % GATE_PERIOD;
		this.surface.uniforms.uGateOffset.value.set(
			mod(this.originE + this.gateScroll, GATE_PERIOD),
			mod(this.originN + this.gateScroll * 0.6, GATE_PERIOD)
		);
		const su2 = this.surface.uniforms;
		this.waves.offsets(this.originE, this.originN, [su2.uOffset0.value, su2.uOffset1.value, su2.uOffset2.value, su2.uOffset3.value]);
		// Foam: whitecaps where the crests break, and what they leave on the water.
		const fwd = _forward.set(0, 0, -1).applyQuaternion(pose.quaternion);
		const flat = Math.hypot(fwd.x, fwd.z) || 1;
		this.foamField.update({
			dt: simDt,
			camEast: pose.east,
			camNorth: pose.north,
			camHeight: heightAbove,
			forwardEast: fwd.x / flat,
			forwardNorth: -fwd.z / flat,
			forwardDown: -fwd.y,
			originEast: this.originE,
			originNorth: this.originN,
			waves: this.waves,
			offsets: [su2.uOffset0.value, su2.uOffset1.value, su2.uOffset2.value, su2.uOffset3.value],
			jacobianWeights: JACOBIAN_WEIGHTS,
			foamTexture: this.foamTexture,
			gateOffset: su2.uGateOffset.value
		});
		su2.uFoamField0.value = this.foamField.texture(0);
		su2.uFoamField1.value = this.foamField.texture(1);
		su2.uFoamRect0.value.copy(this.foamField.rects[0]);
		su2.uFoamRect1.value.copy(this.foamField.rects[1]);
		this.foamShiftE += this.waves.foamDrift.x * simDt;
		this.foamShiftN += this.waves.foamDrift.y * simDt;
		const patches = this.waves.patches;
		su2.uPatch.value.set(patches[0], patches[1] ?? 1, patches[2] ?? 1, patches[3] ?? 1);
		const L = (/** @type {number} */ i) => patches[i] ?? 1;
		su2.uTexel.value.set(L(0), L(1), L(2), L(3)).multiplyScalar(1 / this.waves.size);
		su2.uFadeNear.value.set(L(0), L(1), L(2), L(3)).multiplyScalar(12);
		su2.uFadeFar.value.set(L(0), L(1), L(2), L(3)).multiplyScalar(48);
		su2.uTailVariance.value = this.waves.tailSlopeVariance;
		su2.uCamEN.value.set(pose.east - this.originE, pose.north - this.originN);
		// The foam lace drifts with the foam it details.
		const we = Math.sin(wind);
		const wn = Math.cos(wind);
		su2.uWindEN.value.set(we, wn);
		// The camera's place in the lace, in double precision: its position less the foam's
		// drift, turned into the wind frame and wrapped.
		const le = pose.east - this.foamShiftE;
		const ln = pose.north - this.foamShiftN;
		su2.uLaceCam.value.set(pose.east - this.originE, pose.north - this.originN);
		su2.uLaceOffset.value.set(mod(le * we + ln * wn, LACE_PERIOD), mod(-le * wn + ln * we, LACE_PERIOD));
		for (let c = 0; c < 4; c++) {
			const t = c < this.waves.count ? this.waves.textures(c) : this.waves.textures(this.waves.count - 1);
			const u = /** @type {Record<string, THREE.IUniform>} */ (/** @type {unknown} */ (su2));
			u[`uDisp${c}`].value = t.disp;
			u[`uSlope${c}`].value = t.slope;
			u[`uJac${c}`].value = t.jac;
		}
		su2.uActiveFoam.value.set(this.waves.activeFoam.bias, this.waves.activeFoam.gain);
		this.surface.layout(pose.east, pose.north, this.originE, this.originN, heightAbove);

		// --- the aircraft ----------------------------------------------------------------
		const model = this.model;
		if (air && air.presence > 0.002) {
			if (Math.abs(air.scale - this.modelScale) > 1e-4) {
				this.modelScale = air.scale;
				model.setScale(air.scale);
			}
			model.pose(air.state, this.toWorld);
			const da = Math.hypot(air.state.x - pose.east, air.state.y - pose.north, air.state.altitude - pose.up);
			const span = model.span * air.scale;
			const proximity = smoothstep(span * 0.9, span * 2, da);
			const opacity = air.presence * proximity * clamp(air.state.visible, 0, 1);
			model.setOpacity(opacity);
			this.modelOpacity = opacity;
			this.updateAircraftLighting(da, air);
			if (!this.warm && this.environment.texture) this.warmUp(cam);
		} else {
			if (this.modelOpacity !== 0) model.setOpacity(0);
			this.modelOpacity = 0;
			model.object.visible = false;
		}

		// --- overlays --------------------------------------------------------------------
		this.updateOverlays(rawDt);

		// --- draw --------------------------------------------------------------------------
		const domeU = /** @type {THREE.ShaderMaterial} */ (this.dome.material).uniforms;
		const tanHalf = Math.tan((cam.fov * DEG) / 2);
		domeU.uTanHalf.value.set(tanHalf * cam.aspect, tanHalf);
		domeU.uCameraWorld.value.copy(cam.matrixWorld);

		renderer.setRenderTarget(this.mainTarget);
		renderer.setClearColor(0x000000, 0);
		renderer.clear(true, true, false);
		renderer.render(this.mainScene, cam);

		const drawAir = this.modelOpacity > 0.002;
		if (drawAir || !this.airEmpty) {
			renderer.setRenderTarget(this.airTarget);
			renderer.setClearColor(0x000000, 0);
			renderer.clear(true, true, false);
			// Rendered even when empty: a multisampled target resolves into the texture the
			// post pass reads only at the end of a render, so a bare clear would leave the
			// last aircraft drawn composited, frozen, for as long as it stays hidden. Once
			// it is empty, it stays so without being touched.
			renderer.render(drawAir ? this.airScene : this.emptyScene, cam);
			this.airEmpty = !drawAir;
		}

		renderer.setRenderTarget(this.uiTarget);
		renderer.setClearColor(0x000000, 0);
		renderer.clear(true, false, false);
		renderer.render(this.uiScene, cam);

		const pu = this.post.uniforms;
		pu.uAdapted.value = this.meter.update(this.mainTarget.textures[0], rawDt);
		pu.uColor.value = this.mainTarget.textures[0];
		pu.uData0.value = this.mainTarget.textures[1];
		pu.uData1.value = this.mainTarget.textures[2];
		pu.uAir.value = this.airTarget.texture;
		pu.uUi.value = this.uiTarget.texture;
		pu.uBloom.value = this.bloom.render(this.mainTarget.textures[0], this.airTarget.texture);
		// A lens's glare: a small fraction of the light spread wide.
		pu.uBloomStrength.value = BLOOM_STRENGTH * this.bloom.normalization;
		// A white airframe in full sun sits four stops over dark water: exposed for the sea it
		// clips flat. Pull it down a little in hard light, as a grade would, so its shape
		// reads; at dusk or under cloud it needs nothing.
		const hardLight = smoothstep(6, 35, this.light.sunElevation) * (1 - 0.75 * this.light.overcast);
		pu.uAirExposure.value = 1 - 0.42 * hardLight;
		pu.uExposure.value = this.exposure;
		pu.uModeA.value = this.modeA;
		pu.uModeB.value = this.modeB;
		pu.uModeMix.value = smootherstep(this.modeMix);
		pu.uSplitA.value = this.splitA;
		pu.uSplitB.value = this.splitB;
		pu.uSplitModeMix.value = smootherstep(this.splitMix);
		pu.uSplitPos.value = this.splitPos;
		pu.uSplitAmount.value = this.splitAmount;
		pu.uTime.value = this.clock;
		const tanV = Math.tan((cam.fov * DEG) / 2);
		pu.uTanHalf.value.set(tanV * cam.aspect, tanV);
		pu.uCameraWorld.value.copy(cam.matrixWorld);
		pu.uCameraPos.value.copy(cam.position);
		pu.uGridAmount.value = this.overlays.grid;
		const shift = this.currentLayer?.shift ?? { east: 0, north: 0 };
		pu.uGridShift.value.set(mod(this.originE - shift.east, 100), mod(this.originN - shift.north, 100));
		this.post.render(renderer);

		this.host.root.style.setProperty('--split-x', `${(this.splitPos * 100).toFixed(3)}%`);
		this.host.root.style.setProperty('--split-o', this.splitAmount.toFixed(3));

		this.drawHud();
		this.publish(rawDt);
	}

	/**
	 * Compile every program the frame can need before it is first drawn: the aircraft
	 * opaque and faded (three keys its programs on transparency), its shadow pass and nav
	 * lights, and the overlays and their passes. Otherwise the first time the aircraft
	 * fades, or the footprint and flow first appear, a frame stalls mid-transition.
	 * Runs once, on the first frame the aircraft is lit.
	 * @param {THREE.Camera} camera
	 */
	warmUp(camera) {
		this.warm = true;
		const r = this.renderer;
		const previous = r.getRenderTarget();
		r.setRenderTarget(this.airTarget);
		for (const opacity of [1, 0.5]) {
			this.model.setOpacity(opacity);
			r.compile(this.airScene, camera);
			// The shadow pass and the sprites compile only when drawn.
			r.render(this.airScene, camera);
		}
		r.clear(true, true, false);
		// Resolve the clear too (see render), or the half-faded airframe stays in the layer.
		r.render(this.emptyScene, camera);
		this.airEmpty = false;
		this.model.setOpacity(this.modelOpacity);
		r.compile(this.uiScene, camera);
		r.compile(this.flow.update.mesh, camera);
		r.setRenderTarget(previous);
	}

	/**
	 * Light the aircraft like the sea, and haze it like the sea at its distance.
	 * @param {number} distance camera to aircraft, m
	 * @param {BlendedAircraft} air
	 */
	updateAircraftLighting(distance, air) {
		const ls = this.lightState;
		if (!ls) return;
		const envKey = `${this.light.sunElevation.toFixed(1)}|${this.light.sunAzimuth.toFixed(0)}|${this.light.overcast.toFixed(2)}|${this.light.haze.toFixed(2)}`;
		const envChanged = this.environment.update(envKey, this.clock);
		const key = `${this.lightKey}|${envChanged}`;
		if (key !== this.lightingKey || envChanged) {
			this.lightingKey = key;
			const sd = ls.sunDir;
			this.model.setLighting({
				environment: this.environment.texture,
				sunDirection: new THREE.Vector3(sd[0], sd[1], -sd[2]).normalize(),
				sunColor: new THREE.Color(...ls.sunIrradiance),
				ambient: new THREE.Color(...ls.skyIrradiance),
				night: ls.night
			});
		}
		// Haze: exact at the aircraft's distance (three's FogExp2 is a squared exponential).
		const T = pathTransmittance(distance, Math.max(0, (this.pose.up + air.state.altitude) / 2), ls.hazeBeta);
		this.airFog.density = Math.sqrt(Math.max(0, -Math.log(Math.max(T, 1e-6)))) / Math.max(distance, 1);
		this.fogClock -= 1;
		if (this.fogClock <= 0) {
			this.fogClock = 12;
			const dx = air.state.x - this.pose.east;
			const dn = air.state.y - this.pose.north;
			const horiz = Math.hypot(dx, dn) || 1;
			const sun = /** @type {[number, number, number]} */ ([ls.sunDir[0], ls.sunDir[1], ls.sunDir[2]]);
			const { radiance } = skyRadiance([dx / horiz, 0.02, dn / horiz], sun, Math.max(1, this.pose.up), ls.hazeBeta, 12);
			this.airFog.color.setRGB(radiance[0], radiance[1], radiance[2], THREE.LinearSRGBColorSpace);
		}
	}

	/** @param {number} rawDt */
	updateOverlays(rawDt) {
		const res = this.post.uniforms.uResolution.value;
		// Tracks: the current scene's fade in as the previous scene's fade out.
		this.forEachLayer(this.root, (layer, weight) => {
			if (!layer.tracks) return;
			layer.tracks.mesh.position.copy(this.toWorld(layer.shift.east, layer.shift.north, 0));
			layer.tracks.setResolution(res.x, res.y);
			layer.tracks.setOpacity(weight);
			// A recorded track shown over a window round the clock moves with it, so it dips
			// with the aircraft as the loop wraps; shown whole, it just stays.
			const replay = layer.replay;
			const window = layer.spec.trackClock?.window;
			if (replay && window) layer.tracks.setWindow(replay.time, window[0], window[1], replay.visible);
			else layer.tracks.setWindow(replay?.time ?? 0, 1e9, 1e9, 1);
			layer.tracks.setClear(replay ? (layer.spec.trackClock?.clear ?? 0) : 0);
			// A replay's tracks wander wherever the flight went: keep them faint behind the copy.
			layer.tracks.setScrim(layer.spec.scrim ?? 'none', replay ? 0.9 : 0);
		});

		// Markers on tracks the aircraft isn't flying.
		/** @type {import('./lines.js').Polyline[]} */
		const markerLines = [];
		this.forEachLayer(this.root, (layer, weight) => {
			for (const m of layer.markers) {
				const pt = pointAlong(m.points, m.altitudes, (this.simTime % m.loopSeconds) / m.loopSeconds);
				const p = this.toWorld(pt[0] + layer.shift.east, pt[1] + layer.shift.north, pt[2] + 2);
				markerLines.push({ points: [p, p.clone()], color: ACCENT_LINEAR, alpha: weight, width: 11 });
			}
		});
		// Recorded tracks: the estimate's marker and the gap to the reference, at the clock.
		/** @type {import('./replay.js').ReplayView} */
		const view = { camera: this.camera, toWorld: this.toWorld, width: res.x, height: res.y };
		this.forEachLayer(this.root, (layer, weight) => {
			if (layer.replay) replayLines(layer.replay, layer.shift, weight, view, markerLines);
		});
		this.markers.set(markerLines);
		this.markers.setResolution(res.x, res.y);
		this.markers.setFade((this.trackReach ?? 20000) * 0.4, this.trackReach ?? 16000);
		this.markers.setOpacity(markerLines.length ? 1 : 0);

		// The gimbal camera's frustum and footprint.
		const air = this.aircraft;
		const footAmount = air ? air.footprint * air.presence * clamp(air.state.visible, 0, 1) : 0;
		/** @type {THREE.Vector2[] | null} */
		let quad = null;
		if (air && footAmount > 0.002) {
			const gimbalCam = this.gimbalCamera(air);
			const origin = gimbalCam.position.clone();
			/** @type {THREE.Vector3[]} */
			const hits = [];
			const tanV = Math.tan((gimbalCam.fov * DEG) / 2);
			const tanH = tanV * gimbalCam.aspect;
			for (const [sx, sy] of [
				[-1, -1],
				[1, -1],
				[1, 1],
				[-1, 1]
			]) {
				const dir = new THREE.Vector3(sx * tanH, sy * tanV, -1).normalize().applyQuaternion(gimbalCam.quaternion);
				const far = 1500;
				let t = far;
				if (dir.y < -1e-3) t = Math.min(far, origin.y / -dir.y);
				hits.push(origin.clone().addScaledVector(dir, t));
			}
			/** @type {[number, number, number]} */
			const col = ACCENT_LINEAR;
			this.frustum.set(hits.map((h) => ({ points: [origin, h], color: col, alpha: 0.75, width: 1.4 })));
			quad = hits.map((h) => new THREE.Vector2(h.x, -h.z));
			const pu = this.post.uniforms;
			pu.uFootViewProj.value.multiplyMatrices(gimbalCam.projectionMatrix, gimbalCam.matrixWorldInverse);
			pu.uFootScan.value = ((this.clock * 0.32) % 1.2) * 2 - 1.1;
		}
		this.frustum.setResolution(res.x, res.y);
		this.frustum.setOpacity(footAmount);
		this.post.uniforms.uFootAmount.value = footAmount;

		// Flow particles.
		const flowAmount = this.overlays.flow;
		if (flowAmount > 0.002) {
			const center = new THREE.Vector2();
			let radius = 120;
			if (quad) {
				for (const q of quad) center.add(q);
				center.multiplyScalar(0.25);
				radius = Math.max(60, ...quad.map((q) => q.distanceTo(center))) * 1.3;
			} else {
				// Around where the camera looks at the water.
				const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(this.camera.quaternion);
				const t = fwd.y < -0.05 ? Math.min(600, this.camera.position.y / -fwd.y) : 200;
				center.set(this.camera.position.x + fwd.x * t, -(this.camera.position.z + fwd.z * t));
				radius = Math.max(40, t * 0.6);
			}
			const wind = this.sea.windDirection * DEG;
			const driftSpeed = 0.25 + 0.025 * this.sea.windSpeed;
			this.flow.step({
				dt: rawDt,
				time: this.simTime,
				drift: new THREE.Vector2(Math.sin(wind) * driftSpeed, Math.cos(wind) * driftSpeed),
				quad,
				center,
				radius,
				shift: this.flowShift
			});
			this.flowShift.set(0, 0);
			const fu = this.flow.material.uniforms;
			const t0 = this.waves.textures(0);
			const t1 = this.waves.textures(Math.min(1, this.waves.count - 1));
			fu.uDisp0.value = t0.disp;
			fu.uDisp1.value = t1.disp;
			fu.uMotion0.value = t0.motion;
			fu.uMotion1.value = t1.motion;
			fu.uPatch.value.copy(this.surface.uniforms.uPatch.value);
			fu.uOffset0.value.copy(this.surface.uniforms.uOffset0.value);
			fu.uOffset1.value.copy(this.surface.uniforms.uOffset1.value);
			fu.uResolution.value.copy(res);
			const range = this.camera.position.y;
			fu.uFade.value.set(Math.max(250, range * 4), Math.max(900, range * 12));
			fu.uHasFoot.value = quad ? 1 : 0;
			const bands = this.waves.bandStats;
			const drawn = bands[0].speedVariance + (bands[1] ? FINE_SHARE ** 2 * bands[1].speedVariance : 0);
			fu.uSpeedScale.value = 1 / Math.max(0.05, 2 * Math.sqrt(drawn));
			fu.uFootVP.value.copy(this.post.uniforms.uFootViewProj.value);
		} else {
			this.flow.needsReset = true;
		}
		this.flow.setOpacity(flowAmount);
	}

	/**
	 * The aircraft's gimbal camera as a three.js camera, for the footprint.
	 * @param {BlendedAircraft} air
	 */
	gimbalCamera(air) {
		if (!this.gimbalCam) this.gimbalCam = new THREE.PerspectiveCamera(48, 16 / 9, 1, 5000);
		const g = this.gimbalCam;
		const pose = this.gimbalPose(air);
		g.position.copy(this.toWorld(pose.east, pose.north, pose.up));
		g.quaternion.copy(pose.quaternion);
		g.fov = pose.fov;
		g.updateProjectionMatrix();
		g.updateMatrixWorld();
		g.matrixWorldInverse.copy(g.matrixWorld).invert();
		return g;
	}

	/**
	 * The aircraft's gimbal camera pose as its own rig computes it (no presenter offsets),
	 * stabilised as the current onboard rig is, or as a typical gimbal when no onboard
	 * view is on screen.
	 * @param {BlendedAircraft} air
	 */
	gimbalPose(air) {
		const layer = this.currentLayer;
		const stabilized = layer && layer.rig instanceof OnboardRig ? layer.rig.spec.stabilized : FOOTPRINT_STABILIZED;
		const rig = this.footRig;
		rig.spec.stabilized = stabilized;
		return rig.update({
			dt: 0,
			time: this.simTime,
			aircraft: { state: air.state, gimbal: air.gimbal, scale: air.scale },
			cameraMount: this.model.cameraMount,
			seaHeight: () => 0,
			offsets: { yaw: 0, pitch: 0, zoom: 1 },
			aspect: 16 / 9,
			reducedMotion: this.reducedMotion,
			shift: { east: 0, north: 0 }
		});
	}

	/**
	 * Point the model's gimbal ball where its camera looks: the camera's orientation
	 * relative to the airframe, as gimbal pitch and yaw.
	 * @param {BlendedAircraft} air
	 */
	aimGimbal(air) {
		if (!this.model.setGimbal) return;
		const camera = this.gimbalPose(air).quaternion;
		const st = air.state;
		const relative = attitudeQuaternion(st.heading, st.pitch, st.bank, _quat).invert().multiply(camera);
		_euler.setFromQuaternion(relative, 'YXZ');
		const pitch = _euler.x / DEG;
		const yaw = -_euler.y / DEG;
		if (Math.abs(pitch - this.gimbalAim.pitch) + Math.abs(yaw - this.gimbalAim.yaw) < 0.02) return;
		this.gimbalAim = { pitch, yaw };
		this.model.setGimbal(pitch, yaw);
	}

	drawHud() {
		const amount = this.overlays.horizon;
		if (amount < 0.002) {
			this.hud.clear();
			return;
		}
		const root = this.host.root;
		const att = attitudeOf(this.pose.quaternion);
		const seaUnder = this.sampler.sample(this.pose.east, this.pose.north, this.simTime).height;
		const height = Math.max(0.3, this.pose.up - seaUnder);
		this.hud.draw({
			width: root.clientWidth,
			height: root.clientHeight,
			dpr: Math.min(window.devicePixelRatio || 1, 2),
			alpha: amount,
			quaternion: this.pose.quaternion,
			fov: this.pose.fov,
			pitch: att.pitch,
			roll: att.roll,
			heading: att.heading,
			dip: horizonDip(height) / DEG,
			altitude: this.aircraft && this.cameraMode === 'onboard' ? this.aircraft.state.altitude : height,
			airspeed: this.aircraft && this.cameraMode === 'onboard' ? this.aircraft.state.speed : undefined
		});
	}

	/**
	 * Publish what is on screen: to the slides about 8 times a second, and to
	 * window.__mayfly for tests.
	 * @param {number} rawDt
	 */
	publish(rawDt) {
		this.readoutClock -= rawDt;
		if (this.readoutClock > 0) return;
		this.readoutClock = 0.125;
		const att = attitudeOf(this.pose.quaternion);
		const air = this.aircraft;
		const onboard = this.cameraMode === 'onboard';
		const seaUnder = this.sampler.sample(this.pose.east, this.pose.north, this.simTime).height;
		const airSea = air ? this.sampler.sample(air.state.x, air.state.y, this.simTime).height : 0;
		const r = oceanReadout;
		const st = this.stats;
		r.status = this.status;
		r.windSpeed = this.sea.windSpeed;
		r.windDirection = this.sea.windDirection;
		r.significantHeight = st.significantHeight;
		r.peakPeriod = st.peakPeriod;
		r.peakWavelength = st.peakWavelength;
		r.beaufort = st.beaufort;
		r.sunElevation = this.light.sunElevation;
		r.haze = this.light.haze;
		r.overcast = this.light.overcast;
		r.cameraHeight = onboard && air ? air.state.altitude - airSea : this.pose.up - seaUnder;
		r.fov = this.pose.fov;
		r.roll = att.roll;
		r.pitch = att.pitch;
		r.heave = this.pose.heave;
		r.aircraft = Boolean(air && air.presence > 0.5);
		r.aircraftAltitude = air ? air.state.altitude : 0;
		r.aircraftSpeed = air ? air.state.speed : 0;
		r.aircraftHeading = air ? air.state.heading : 0;
		r.aircraftBank = air ? air.state.bank : 0;
		r.aircraftPitch = air ? air.state.pitch : 0;
		r.cameraMode = this.cameraMode;
		const replay = this.currentLayer?.replay ?? null;
		r.trackTime = replay ? replay.time : -1;
		r.trackGap = replay ? replay.gap : -1;
		r.sample = this.spec?.randomize ? this.sample : 0;
		r.mode = this.spec?.mode ?? 'shaded';
		r.fps = Math.round(this.fps);
		if (typeof window !== 'undefined') {
			/** @type {any} */ (window).__mayfly = {
				readout: { ...r },
				stats: {
					frames: this.frames,
					fps: Math.round(this.fps * 10) / 10,
					renderScale: this.quality.renderScale,
					cascades: this.waves.count,
					resolution: this.waves.size
				}
			};
		}
	}

	// ------------------------------------------------------------------ plumbing

	/**
	 * Absolute east/north/up metres to three.js world coordinates (floating origin).
	 * @param {number} east
	 * @param {number} north
	 * @param {number} up
	 */
	toWorld = (east, north, up) => new THREE.Vector3(east - this.originE, up, -(north - this.originN));

	/** @param {boolean} [settle] give the frame rate time to settle (not after a quality step) */
	resize(settle = true) {
		if (this.disposed) return;
		const root = this.host.root;
		const w = Math.max(1, root.clientWidth);
		const h = Math.max(1, root.clientHeight);
		const dpr = this.quality.pixelRatio(w, h);
		this.renderer.setPixelRatio(dpr);
		this.renderer.setSize(w, h, false);
		const bw = Math.max(1, Math.round(w * dpr));
		const bh = Math.max(1, Math.round(h * dpr));
		this.mainTarget.setSize(bw, bh);
		this.uiTarget.setSize(bw, bh);
		this.airTarget.setSize(bw, bh);
		this.airEmpty = false;
		this.bloom.setSize(bw, bh);
		this.camera.aspect = w / h;
		this.camera.updateProjectionMatrix();
		this.post.uniforms.uResolution.value.set(bw, bh);
		this.updateAirSamples();
		if (settle) this.quality.settle(2);
	}

	/** @param {number} cascades */
	rebuildWaves(cascades) {
		this.waves.dispose();
		this.waves = new WaveSimulation(this.renderer, { size: this.quality.preset.fftSize, cascades });
		this.sampler = new SeaSampler(this.waves.noise, this.waves.size, this.waves.patches[0]);
		this.surface.build(cascades);
		this.seaKey = '';
	}

	createDome() {
		const material = new THREE.ShaderMaterial({
			glslVersion: THREE.GLSL3,
			vertexShader: /* glsl */ `
				uniform vec2 uTanHalf;
				uniform mat4 uCameraWorld;
				out vec3 vDir;
				void main() {
					vDir = mat3(uCameraWorld) * vec3(position.xy * uTanHalf, -1.0);
					gl_Position = vec4(position.xy, 0.0, 1.0);
				}`,
			fragmentShader: /* glsl */ `
				${SKY_GLSL}
				${WATER_GLSL}
				in vec3 vDir;
				layout(location = 0) out vec4 outColor;
				layout(location = 1) out vec4 outData0;
				layout(location = 2) out vec4 outData1;
				void main() {
					vec3 dir = normalize(vDir);
					vec3 origin = cameraPosition;
					if (dir.y >= -sin(uDip)) {
						vec4 lut = skyLut(dir);
						vec4 cl = cloudLayer(origin, dir, 0.0);
						vec3 col = mix(lut.rgb, cl.rgb, cl.a) + sunDisc(dir) * (1.0 - cl.a);
						outColor = vec4(col, 1.0);
						outData0 = vec4(0.0, 0.0, 1.0, 0.0);
						outData1 = vec4(0.0, 0.0, 60000.0, 0.0);
					} else {
						// The sea beyond the mesh: a flat, rough mirror on the curved earth.
						float r0 = R_GROUND + max(uCamAlt, 0.5);
						float b = r0 * dir.y;
						float disc = b * b - (r0 * r0 - R_GROUND * R_GROUND);
						float d = -b - sqrt(max(disc, 0.0));
						vec3 pPlanet = vec3(0.0, r0, 0.0) + dir * d;
						vec3 N = normalize(pPlanet);
						vec3 P = origin + dir * d;
						vec3 sea = shadeWater(P, -dir, N, uFarVariance, 0.0, 0.0, uSunVisible);
						vec4 lut = skyLut(dir);
						outColor = vec4(sea * lut.a + lut.rgb, 1.0);
						outData0 = vec4(N.x, -N.z, N.y, 0.0);
						outData1 = vec4(0.0, 0.0, min(d, 60000.0), 1.0);
					}
				}`,
			uniforms: {
				...this.skyUniforms,
				uTanHalf: { value: new THREE.Vector2(1, 1) },
				uCameraWorld: { value: new THREE.Matrix4() }
			},
			depthTest: false,
			depthWrite: false
		});
		const geometry = new THREE.BufferGeometry();
		geometry.setAttribute('position', new THREE.Float32BufferAttribute([-1, -1, 0, 3, -1, 0, -1, 3, 0], 3));
		const mesh = new THREE.Mesh(geometry, material);
		mesh.frustumCulled = false;
		mesh.renderOrder = -100;
		return mesh;
	}

	/**
	 * @param {'loading' | 'running' | 'fallback' | 'error'} status
	 * @param {string} [reason]
	 */
	setStatus(status, reason) {
		if (this.status === status) return;
		this.status = status;
		oceanReadout.status = status;
		this.host.onStatus(status, reason);
	}

	/** @param {string} reason */
	fail(reason) {
		if (this.failed) return;
		this.failed = true;
		this.setStatus('fallback', reason);
	}
}

/** @param {number} x @param {number} m */
function mod(x, m) {
	return ((x % m) + m) % m;
}

/**
 * A point a fraction u of the way along a polyline, by arc length.
 * @param {Array<[number, number]>} points
 * @param {number[] | undefined} altitudes
 * @param {number} u
 * @returns {[number, number, number]}
 */
function pointAlong(points, altitudes, u) {
	if (points.length === 1) return [points[0][0], points[0][1], altitudes?.[0] ?? 0];
	const lengths = [0];
	for (let i = 1; i < points.length; i++) {
		lengths.push(lengths[i - 1] + Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]));
	}
	const target = u * lengths[lengths.length - 1];
	let i = 1;
	while (i < lengths.length - 1 && lengths[i] < target) i++;
	const seg = lengths[i] - lengths[i - 1] || 1;
	const f = (target - lengths[i - 1]) / seg;
	const a = points[i - 1];
	const b = points[i];
	const alt = altitudes ? altitudes[i - 1] + (altitudes[i] - altitudes[i - 1]) * f : 0;
	return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, alt];
}

/** Degrees of turn per drawn segment along a recorded track. */
const CURVE_STEP = 1.5;

/**
 * A recorded track as the polyline to draw: its samples, with extra vertices on its
 * time interpolation wherever it turns, so the line is the curve the aircraft flies.
 * @param {import('../aircraft/track.js').TimedTrack} timed
 */
function curveOf(timed) {
	/** @type {Array<[number, number]>} */
	const points = [];
	/** @type {number[]} */
	const altitudes = [];
	/** @type {number[]} */
	const times = [];
	const p = { x: 0, y: 0, z: 0 };
	for (let i = 0; i < timed.n; i++) {
		points.push([timed.X[i], timed.Y[i]]);
		altitudes.push(timed.Z[i]);
		times.push(timed.T[i]);
		if (i === timed.n - 1) break;
		const h = timed.T[i + 1] - timed.T[i];
		const turned = (Math.max(Math.abs(timed.turn[i]), Math.abs(timed.turn[i + 1])) * h) / DEG;
		const pieces = Math.min(8, Math.max(1, Math.ceil(turned / CURVE_STEP)));
		for (let k = 1; k < pieces; k++) {
			const t = timed.T[i] + (h * k) / pieces;
			timed.position(t, p);
			points.push([p.x, p.y]);
			altitudes.push(p.z);
			times.push(t);
		}
	}
	return { points, altitudes, times };
}
