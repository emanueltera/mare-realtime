/**
 * The deck's fixed-wing UAV: a generic small airframe drawn for the presentation (not
 * a particular aircraft), about a 3.1 m span and 1.75 m long. A slender pod fuselage with an
 * EO/IR gimbal ball under the nose, a high-aspect-ratio tapered wing (NACA 2412) with
 * dihedral, upturned winglets, flaps and ailerons, twin tail booms joined by an
 * inverted-V tail with ruddervators, and a two-blade pusher propeller between the
 * booms. Procedural geometry only.
 *
 * Model frame: metres, origin at the centre of gravity, nose toward −Z, right wing
 * toward +X, up +Y.
 */
import * as THREE from 'three';
import { attitudeQuaternion } from './attitude.js';
import {
	bodyLoft,
	controlLoop,
	fullLoop,
	hingePoint,
	latheZ,
	loft,
	mainLoop,
	merge,
	mirrorX,
	naca4,
	pchip,
	prepare
} from './geometry.js';
import { fuselageMaps, glowTexture, propDiscTexture, tailMarking } from './textures.js';

const DEG = Math.PI / 180;
const X_AXIS = new THREE.Vector3(1, 0, 0);

// ------------------------------------------------------------------ dimensions

/**
 * Fuselage pod: an ogive nose (the gimbal ball hangs under it), a saddle that rises to
 * meet the wing, and a tail cone that sweeps up to the propeller shaft.
 */
const BODY = { zNose: -0.84, zTail: 0.47, noseLength: 0.34, noseSharpness: 0.68 };
const BODY_Z = [-0.84, -0.7, -0.55, -0.38, -0.2, -0.05, 0.1, 0.22, 0.33, 0.42, 0.47];
const BODY_W = [0.094, 0.104, 0.112, 0.116, 0.118, 0.118, 0.112, 0.098, 0.077, 0.056, 0.046];
const BODY_C = [-0.028, -0.016, -0.007, 0.0, 0.004, 0.008, 0.013, 0.02, 0.029, 0.038, 0.042];
const BODY_A = [0.086, 0.1, 0.112, 0.121, 0.134, 0.141, 0.139, 0.118, 0.088, 0.062, 0.046];
const BODY_B = [0.082, 0.098, 0.108, 0.114, 0.117, 0.116, 0.108, 0.094, 0.074, 0.056, 0.046];

/** Wing: chord line height at the root, quarter-chord station, chords, kink at the booms. */
const WING = {
	y: 0.142,
	zQuarter: -0.03,
	rootChord: 0.3,
	tipChord: 0.17,
	kink: 0.4,
	tip: 1.5,
	dihedral: 3,
	washout: -2,
	hinge: 0.72
};
/** Winglet: bend radius, cant from horizontal, straight height, top chord, sweep of its quarter chord. */
const WINGLET = { radius: 0.045, cant: 76, height: 0.075, topChord: 0.07, sweep: 0.075 };

/** Booms and the inverted-V tail. */
const BOOM = { x: 0.4, y: 0.118, r: 0.024, zFront: -0.24, zBack: 0.95 };
const TAIL = { angle: 33, rootChord: 0.175, apexChord: 0.135, rootLE: 0.745, apexLE: 0.79, hinge: 0.66 };
const TAIL_ROOT_Y = BOOM.y + 0.012;
const TAIL_APEX_Y = TAIL_ROOT_Y + BOOM.x * Math.tan(TAIL.angle * DEG);

/** Pusher propeller. */
const PROP = { z: 0.505, radius: 0.25, pitch: 0.22, hub: 0.035 };

/** Gimbal ball under the nose. */
const GIMBAL = { z: -0.56, y: -0.163, r: 0.054, defaultPitch: -35 };

/** Lighting gains: three's lights scale radiance by 1/π on a Lambertian surface. */
/**
 * Lighting is physical: `sunColor` is the sun's irradiance on a surface facing it and
 * `ambient` the sky's irradiance on a horizontal surface, in whatever linear units the
 * sea is lit in. three's lights already apply albedo/π, so the gains are 1; with an
 * environment map the hemisphere light is only a weak fill (the map supplies the sky).
 */
const SUN_GAIN = 1;
const HEMI_GAIN = 1;
const HEMI_GAIN_WITH_ENV = 0.3;

// ------------------------------------------------------------------ geometry builders

/**
 * A wing station at spanwise position sx on the right wing.
 * @param {number} sx
 * @param {boolean} outerPanel use the outer panel's dihedral at the kink
 * @returns {import('./geometry.js').Station}
 */
function wingStation(sx, outerPanel) {
	const outer = outerPanel || sx > WING.kink + 1e-9;
	const f = Math.max(0, (sx - WING.kink) / (WING.tip - WING.kink));
	const c = WING.rootChord + (WING.tipChord - WING.rootChord) * f;
	const g = (outer ? WING.dihedral : 0) * DEG;
	const y = WING.y + Math.max(0, sx - WING.kink) * Math.tan(WING.dihedral * DEG);
	const T = new THREE.Vector3(Math.cos(g), Math.sin(g), 0);
	const twist = WING.washout * f * DEG;
	const d = new THREE.Vector3(0, 0, 1).applyAxisAngle(T, twist);
	const u = new THREE.Vector3().crossVectors(d, T).normalize();
	const q = new THREE.Vector3(sx, y, WING.zQuarter);
	return { o: q.addScaledVector(d, -c / 4), c, d, u, v: sx / WING.tip };
}

/**
 * Stations evenly spaced over [a, b] of the right wing.
 * @param {number} a
 * @param {number} b
 * @param {number} n
 */
function wingStations(a, b, n) {
	return Array.from({ length: n }, (_, i) => wingStation(a + ((b - a) * i) / (n - 1), a >= WING.kink - 1e-9));
}

/**
 * The winglet's stations, continuing from the wing tip: a bend up through `cant`
 * degrees, then a straight, tapering, swept blade.
 */
function wingletStations() {
	const tip = wingStation(WING.tip, true);
	const g0 = WING.dihedral * DEG;
	const g1 = WINGLET.cant * DEG;
	const qTip = tip.o.clone().addScaledVector(tip.d, tip.c / 4);
	const centre = qTip.clone().add(new THREE.Vector3(-Math.sin(g0), Math.cos(g0), 0).multiplyScalar(WINGLET.radius));
	const arcLength = WINGLET.radius * (g1 - g0);
	const total = arcLength + WINGLET.height;
	/** @type {import('./geometry.js').Station[]} */
	const stations = [];
	const arcSteps = 7;
	const straightSteps = 4;
	for (let k = 1; k <= arcSteps + straightSteps; k++) {
		let g;
		let q;
		let s;
		if (k <= arcSteps) {
			g = g0 + ((g1 - g0) * k) / arcSteps;
			q = centre.clone().add(new THREE.Vector3(Math.sin(g), -Math.cos(g), 0).multiplyScalar(WINGLET.radius));
			s = (WINGLET.radius * (g - g0)) / total;
		} else {
			g = g1;
			const h = (WINGLET.height * (k - arcSteps)) / straightSteps;
			q = centre
				.clone()
				.add(new THREE.Vector3(Math.sin(g), -Math.cos(g), 0).multiplyScalar(WINGLET.radius))
				.add(new THREE.Vector3(Math.cos(g), Math.sin(g), 0).multiplyScalar(h));
			s = (arcLength + h) / total;
		}
		const T = new THREE.Vector3(Math.cos(g), Math.sin(g), 0);
		const d = new THREE.Vector3(0, 0, 1).applyAxisAngle(T, WING.washout * DEG);
		const u = new THREE.Vector3().crossVectors(d, T).normalize();
		const c = WING.tipChord + (WINGLET.topChord - WING.tipChord) * s ** 0.9;
		q.z += WINGLET.sweep * s;
		stations.push({ o: q.addScaledVector(d, -c / 4), c, d, u, v: 1 });
	}
	return stations;
}

/**
 * A tail-panel station at s along the right panel, 0 at the apex and 1 at the boom.
 * @param {number} s
 * @returns {import('./geometry.js').Station}
 */
function tailStation(s) {
	const a = TAIL.angle * DEG;
	const T = new THREE.Vector3(Math.cos(a), -Math.sin(a), 0);
	const d = new THREE.Vector3(0, 0, 1);
	const u = new THREE.Vector3().crossVectors(d, T).normalize();
	const c = TAIL.apexChord + (TAIL.rootChord - TAIL.apexChord) * s;
	const z = TAIL.apexLE + (TAIL.rootLE - TAIL.apexLE) * s;
	const o = new THREE.Vector3(BOOM.x * s, TAIL_APEX_Y + (TAIL_ROOT_Y - TAIL_APEX_Y) * s, z);
	return { o, c, d, u, v: s };
}

/**
 * @param {number} a
 * @param {number} b
 * @param {number} n
 */
function tailStations(a, b, n) {
	return Array.from({ length: n }, (_, i) => tailStation(a + ((b - a) * i) / (n - 1)));
}

/**
 * A hinged control surface: the geometry is moved into a pivot whose local +X is the
 * hinge line (pointing toward +x in the model), so rotating the pivot about X by a
 * positive angle moves the trailing edge down (toward the surface's lower side).
 * @param {THREE.BufferGeometry} geometry in the model frame
 * @param {THREE.Vector3} h0
 * @param {THREE.Vector3} h1
 * @param {THREE.Material} material
 */
function hinged(geometry, h0, h1, material) {
	const [a, b] = h1.x >= h0.x ? [h0, h1] : [h1, h0];
	const axis = b.clone().sub(a).normalize();
	const pivot = new THREE.Group();
	pivot.position.copy(a);
	pivot.quaternion.setFromUnitVectors(X_AXIS, axis);
	pivot.updateMatrix();
	geometry.applyMatrix4(pivot.matrix.clone().invert());
	const mesh = new THREE.Mesh(geometry, material);
	pivot.add(mesh);
	const base = pivot.quaternion.clone();
	const q = new THREE.Quaternion();
	return {
		pivot,
		mesh,
		/** @param {number} deg */
		set(deg) {
			pivot.quaternion.copy(base).multiply(q.setFromAxisAngle(X_AXIS, deg * DEG));
		}
	};
}

/** @param {THREE.Vector3} p */
const mirrored = (p) => new THREE.Vector3(-p.x, p.y, p.z);

// ------------------------------------------------------------------ the model

/** @returns {import('../aircraft.js').AircraftModel} */
export function createAircraftModel() {
	const object = new THREE.Group();
	object.name = 'mayfly-uav';

	/** @type {THREE.BufferGeometry[]} */
	const geometries = [];
	/** @type {THREE.Texture[]} */
	const textures = [];
	/** @template {THREE.BufferGeometry} G @param {G} g */
	const keep = (g) => (geometries.push(g), g);

	// -------------------------------------------------------------- materials
	const maps = fuselageMaps(BODY);
	const markR = tailMarking(false);
	const markL = tailMarking(true);
	const glowMap = glowTexture();
	const discMap = propDiscTexture(PROP.hub / PROP.radius);
	textures.push(maps.bumpMap, maps.map, markR, markL, glowMap, discMap);

	const composite = {
		color: 0xc8ced3,
		roughness: 0.46,
		metalness: 0,
		clearcoat: 0.32,
		clearcoatRoughness: 0.32,
		vertexColors: true
	};
	const body = new THREE.MeshPhysicalMaterial(composite);
	const fuselageMat = new THREE.MeshPhysicalMaterial({
		...composite,
		map: maps.map,
		bumpMap: maps.bumpMap,
		bumpScale: 0.6
	});
	const tailMatR = new THREE.MeshPhysicalMaterial({ ...composite, map: markR });
	const tailMatL = new THREE.MeshPhysicalMaterial({ ...composite, map: markL });
	const dark = new THREE.MeshPhysicalMaterial({
		color: 0x3b4249,
		roughness: 0.5,
		metalness: 0.1,
		clearcoat: 0.2,
		clearcoatRoughness: 0.4,
		vertexColors: true
	});
	const metal = new THREE.MeshStandardMaterial({ color: 0xa9b1b8, roughness: 0.28, metalness: 1, vertexColors: true });
	const propMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.62, metalness: 0, vertexColors: true });
	const glass = new THREE.MeshPhysicalMaterial({
		color: 0x06080b,
		roughness: 0.05,
		metalness: 0,
		clearcoat: 1,
		clearcoatRoughness: 0.03,
		iridescence: 0.8,
		iridescenceIOR: 1.7,
		iridescenceThicknessRange: [260, 420],
		ior: 1.6,
		specularIntensity: 1
	});
	// The blur of matte black blades mostly hides what is behind it, so it is drawn as
	// an unlit dark veil: lit, its faint sheen of bright sky reads as a pale glowing disc.
	const discMat = new THREE.MeshBasicMaterial({
		color: 0x0c0d0f,
		map: discMap,
		transparent: true,
		depthWrite: false,
		side: THREE.DoubleSide
	});
	const lensRed = new THREE.MeshBasicMaterial({ color: 0xff2a1a, toneMapped: false });
	const lensGreen = new THREE.MeshBasicMaterial({ color: 0x19ff6a, toneMapped: false });
	const lensWhite = new THREE.MeshBasicMaterial({ color: 0xfff6e8, toneMapped: false });
	const lensStrobe = new THREE.MeshBasicMaterial({ color: 0xe8eeff, toneMapped: false });

	/** Opaque materials that fade together with setOpacity. */
	const solids = [body, fuselageMat, tailMatR, tailMatL, dark, metal, glass, lensRed, lensGreen, lensWhite, lensStrobe];
	/** Materials that take the sky as their environment map. */
	const lit = [body, fuselageMat, tailMatR, tailMatL, dark, metal, propMat, glass];
	const envIntensity = new Map([
		[body, 0.9],
		[fuselageMat, 0.9],
		[tailMatR, 0.9],
		[tailMatL, 0.9],
		[dark, 0.8],
		[metal, 1],
		[propMat, 0.6],
		[glass, 1.2]
	]);

	/**
	 * @param {THREE.BufferGeometry} geometry
	 * @param {THREE.Material} material
	 * @param {THREE.Object3D} [parent]
	 */
	const add = (geometry, material, parent = object) => {
		const mesh = new THREE.Mesh(keep(geometry), material);
		mesh.castShadow = true;
		mesh.receiveShadow = true;
		parent.add(mesh);
		return mesh;
	};

	// -------------------------------------------------------------- fuselage
	add(
		bodyLoft(
			{
				...BODY,
				halfWidth: pchip(BODY_Z, BODY_W),
				centre: pchip(BODY_Z, BODY_C),
				top: pchip(BODY_Z, BODY_A),
				bottom: pchip(BODY_Z, BODY_B)
			},
			{ around: 72, along: 70, nose: 18 }
		),
		fuselageMat
	);

	// -------------------------------------------------------------- wing
	const wingFoil = naca4(0.02, 0.4, 0.12);
	const leStrip = { strip: 0.8 };
	/** @type {THREE.BufferGeometry[]} */
	const wingRight = [];
	/** @type {Array<{ geometry: THREE.BufferGeometry, h0: THREE.Vector3, h1: THREE.Vector3, kind: 'flap' | 'aileron' }>} */
	const surfacesRight = [];
	const cut = WING.hinge - 0.004;
	wingRight.push(loft(wingStations(0, WING.kink, 4), fullLoop(wingFoil, leStrip), { capStart: false }));
	wingRight.push(loft(wingStations(WING.kink, 0.43, 2), fullLoop(wingFoil, leStrip)));
	for (const [a, b, n, kind] of /** @type {Array<[number, number, number, 'flap' | 'aileron']>} */ ([
		[0.43, 0.82, 6, 'flap'],
		[0.86, 1.38, 8, 'aileron']
	])) {
		const st = wingStations(a, b, n);
		wingRight.push(loft(st, mainLoop(wingFoil, cut, leStrip)));
		const inset = [wingStation(a + 0.004, true), wingStation(b - 0.004, true)];
		const surface = loft(
			wingStations(a + 0.004, b - 0.004, n),
			controlLoop(wingFoil, WING.hinge)
		);
		surfacesRight.push({
			geometry: surface,
			h0: hingePoint(inset[0], wingFoil, WING.hinge),
			h1: hingePoint(inset[1], wingFoil, WING.hinge),
			kind
		});
	}
	wingRight.push(loft(wingStations(0.82, 0.86, 2), fullLoop(wingFoil, leStrip)));
	const tipStations = [...wingStations(1.38, WING.tip, 3), ...wingletStations()];
	wingRight.push(loft(tipStations, fullLoop(wingFoil, leStrip)));
	const wingR = merge(wingRight);
	add(merge([wingR.clone(), mirrorX(wingR)]), body);
	wingR.dispose();

	/** @type {Array<ReturnType<typeof hinged>>} */
	const ailerons = [];
	for (const { geometry, h0, h1, kind } of surfacesRight) {
		const left = mirrorX(geometry);
		const r = hinged(geometry, h0, h1, body);
		const l = hinged(left, mirrored(h0), mirrored(h1), body);
		for (const h of [r, l]) {
			keep(h.mesh.geometry);
			h.mesh.castShadow = true;
			h.mesh.receiveShadow = true;
			object.add(h.pivot);
		}
		if (kind === 'aileron') ailerons.push(r, l);
		else {
			// Flaps stay faired; a degree of droop reads as a separate surface.
			r.set(1);
			l.set(1);
		}
	}

	// -------------------------------------------------------------- booms and tail
	const boomProfile = /** @type {Array<[number, number]>} */ ([
		[0, BOOM.zFront],
		[0.011, BOOM.zFront + 0.008],
		[0.018, BOOM.zFront + 0.024],
		[0.022, BOOM.zFront + 0.045],
		[BOOM.r, BOOM.zFront + 0.075],
		[BOOM.r, 0.8],
		[0.019, 0.9],
		[0.016, BOOM.zBack - 0.012],
		[0.009, BOOM.zBack - 0.002],
		[0, BOOM.zBack]
	]);
	const boom = latheZ(boomProfile, 28);
	add(
		merge([boom.clone().translate(BOOM.x, BOOM.y, 0), boom.clone().translate(-BOOM.x, BOOM.y, 0)]),
		body
	);
	boom.dispose();

	const tailFoil = naca4(0, 0.4, 0.1);
	const tailFixedR = merge([
		loft(tailStations(0, 0.08, 2), fullLoop(tailFoil)),
		loft(tailStations(0.93, 1.06, 2), fullLoop(tailFoil))
	]);
	add(merge([tailFixedR.clone(), mirrorX(tailFixedR)]), body);
	tailFixedR.dispose();
	const tailMainR = loft(tailStations(0.08, 0.93, 6), mainLoop(tailFoil, TAIL.hinge - 0.004));
	add(mirrorX(tailMainR), tailMatL);
	add(tailMainR, tailMatR);
	const rvGeometry = loft(tailStations(0.084, 0.926, 6), controlLoop(tailFoil, TAIL.hinge));
	const rvH0 = hingePoint(tailStation(0.084), tailFoil, TAIL.hinge);
	const rvH1 = hingePoint(tailStation(0.926), tailFoil, TAIL.hinge);
	const rvLeftGeometry = mirrorX(rvGeometry);
	const ruddervatorR = hinged(rvGeometry, rvH0, rvH1, body);
	const ruddervatorL = hinged(rvLeftGeometry, mirrored(rvH0), mirrored(rvH1), body);
	for (const h of [ruddervatorR, ruddervatorL]) {
		keep(h.mesh.geometry);
		h.mesh.castShadow = true;
		h.mesh.receiveShadow = true;
		object.add(h.pivot);
	}

	// Apex fairing where the two tail panels meet; it carries the white tail light.
	const apex = latheZ(
		[
			[0, TAIL.apexLE - 0.035],
			[0.01, TAIL.apexLE - 0.025],
			[0.017, TAIL.apexLE],
			[0.019, TAIL.apexLE + 0.06],
			[0.016, 0.9],
			[0.01, 0.94],
			[0, 0.955]
		],
		20
	);
	apex.translate(0, TAIL_APEX_Y, 0);
	add(apex, body);

	// -------------------------------------------------------------- propeller
	const spinner = latheZ(
		[
			[0, 0.455],
			[0.047, 0.455],
			[0.047, 0.48],
			[0.044, 0.52],
			[0.036, 0.556],
			[0.023, 0.585],
			[0.009, 0.604],
			[0, 0.608]
		],
		28
	);
	add(spinner, dark);

	const propFoil = naca4(-0.03, 0.4, 0.1);
	const bladeChord = pchip([0.12, 0.25, 0.45, 0.7, 0.88, 0.96, 1], [0.026, 0.04, 0.044, 0.036, 0.026, 0.019, 0.01]);
	/** @param {number} rho @param {number} shade @returns {import('./geometry.js').Station} */
	const bladeStation = (rho, shade) => {
		const r = rho * PROP.radius;
		const beta = Math.atan(PROP.pitch / (2 * Math.PI * Math.max(r, 0.02)));
		const c = bladeChord(rho);
		const d = new THREE.Vector3(Math.cos(beta), 0, Math.sin(beta));
		const T = new THREE.Vector3(0, 1, 0);
		const u = new THREE.Vector3().crossVectors(d, T).normalize();
		// Blade stacked on its 35 % chord line.
		const o = new THREE.Vector3(0, r, 0).addScaledVector(d, -0.35 * c);
		return { o, c, d, u, v: rho, shade };
	};
	const BLADE = 0.035;
	const TIP = 0.85;
	const bladeStations = [
		...[0.12, 0.2, 0.3, 0.42, 0.55, 0.68, 0.8, 0.9].map((r) => bladeStation(r, BLADE)),
		bladeStation(0.9, TIP),
		...[0.95, 0.985, 1].map((r) => bladeStation(r, TIP))
	];
	const blade = loft(bladeStations, fullLoop(propFoil, { n: 16 }));
	const propeller = new THREE.Group();
	propeller.position.set(0, BODY_C[BODY_C.length - 1], PROP.z);
	const blades = add(merge([blade.clone(), blade.clone().rotateZ(Math.PI)]), propMat, propeller);
	blade.dispose();
	const disc = new THREE.Mesh(keep(new THREE.CircleGeometry(PROP.radius, 64)), discMat);
	disc.renderOrder = 2;
	propeller.add(disc);
	object.add(propeller);

	// -------------------------------------------------------------- gimbal turret
	const gimbal = new THREE.Group();
	gimbal.position.set(0, GIMBAL.y, GIMBAL.z);
	object.add(gimbal);
	const yoke = latheZ(
		[
			[0, -0.016],
			[0.046, -0.016],
			[0.05, -0.008],
			[0.05, 0.03],
			[0, 0.03]
		],
		32
	);
	yoke.rotateX(-Math.PI / 2); // axis vertical: z → y
	yoke.translate(0, GIMBAL.r * 0.55 + 0.012, 0);
	add(yoke, dark, gimbal);
	const head = new THREE.Group(); // pitches and yaws with the gimbal
	gimbal.add(head);
	add(prepare(new THREE.SphereGeometry(GIMBAL.r, 40, 28)), dark, head);
	// Flat window face with the EO lens and the smaller IR lens.
	const bezel = latheZ(
		[
			[0, 0],
			[0.034, 0],
			[0.036, 0.004],
			[0.036, 0.012],
			[0, 0.012]
		],
		32
	);
	bezel.translate(0, 0, -GIMBAL.r - 0.003);
	add(bezel, dark, head);
	const eo = latheZ(
		[
			[0, -0.006],
			[0.012, -0.0045],
			[0.019, -0.001],
			[0.021, 0.004],
			[0, 0.004]
		],
		28
	);
	eo.translate(0.007, 0.004, -GIMBAL.r - 0.0035);
	add(eo, glass, head);
	const ir = latheZ(
		[
			[0, -0.002],
			[0.008, -0.0012],
			[0.0095, 0.004],
			[0, 0.004]
		],
		20
	);
	ir.translate(-0.022, -0.012, -GIMBAL.r - 0.004);
	add(ir, glass, head);
	const cameraMount = new THREE.Vector3();
	/** @param {number} pitch @param {number} yaw */
	const aimGimbal = (pitch, yaw) => {
		head.rotation.set(pitch * DEG, -yaw * DEG, 0, 'YXZ');
		cameraMount
			.set(0.007, 0.004, -GIMBAL.r - 0.008)
			.applyEuler(head.rotation)
			.add(gimbal.position);
	};
	aimGimbal(GIMBAL.defaultPitch, 0);

	// -------------------------------------------------------------- small details
	const details = [];
	// Blade antenna on the spine aft of the wing.
	const finShape = new THREE.Shape();
	finShape.moveTo(0, 0);
	finShape.lineTo(0.085, 0);
	finShape.quadraticCurveTo(0.075, 0.03, 0.082, 0.072);
	finShape.lineTo(0.064, 0.074);
	finShape.quadraticCurveTo(0.03, 0.03, 0, 0);
	const fin = new THREE.ExtrudeGeometry(finShape, {
		depth: 0.004,
		bevelEnabled: true,
		bevelThickness: 0.0015,
		bevelSize: 0.0015,
		bevelSegments: 2,
		curveSegments: 8
	});
	// Shape x → body z, shape y → up, extrusion → across.
	fin.applyMatrix4(new THREE.Matrix4().makeBasis(new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 1, 0), new THREE.Vector3(1, 0, 0)));
	details.push(prepare(fin.clone().translate(-0.002, 0.127, 0.25)));
	// A smaller one under the belly.
	details.push(prepare(fin.clone().scale(1, -0.6, 0.7).translate(-0.002, -0.105, -0.3)));
	fin.dispose();
	// Whip antenna under the tail cone, raked back.
	const whip = new THREE.CylinderGeometry(0.0025, 0.0035, 0.16, 8);
	whip.translate(0, -0.08, 0);
	whip.rotateX(-28 * DEG);
	details.push(prepare(whip.translate(0, -0.084, 0.12)));
	const whipBase = new THREE.CylinderGeometry(0.008, 0.01, 0.012, 12);
	details.push(prepare(whipBase.translate(0, -0.09, 0.12)));
	add(merge(details), dark);

	// GNSS puck on the nose and a satcom-style dome behind it (body colour).
	const puck = latheZ(
		[
			[0, -0.004],
			[0.03, -0.004],
			[0.032, 0.0],
			[0.03, 0.005],
			[0, 0.008]
		],
		24
	);
	puck.rotateX(-Math.PI / 2);
	puck.translate(0, 0.116, -0.42);
	add(puck, body);

	// Pitot probe under the left wing.
	const pitotParts = [];
	const strut = new THREE.CylinderGeometry(0.004, 0.006, 0.05, 8);
	strut.translate(0, -0.025, 0);
	pitotParts.push(prepare(strut));
	const tube = new THREE.CylinderGeometry(0.0035, 0.0035, 0.16, 10);
	tube.rotateX(Math.PI / 2);
	tube.translate(0, -0.05, -0.05);
	pitotParts.push(prepare(tube));
	const pitotTip = new THREE.ConeGeometry(0.0035, 0.012, 10);
	pitotTip.rotateX(-Math.PI / 2);
	pitotTip.translate(0, -0.05, -0.136);
	pitotParts.push(prepare(pitotTip));
	const pitotStation = wingStation(1.0, true);
	const pitotAt = pitotStation.o.clone().addScaledVector(pitotStation.d, 0.3 * pitotStation.c);
	pitotAt.y -= 0.012;
	const pitot = merge(pitotParts).translate(-pitotAt.x, pitotAt.y, pitotAt.z);
	add(pitot, metal);

	// -------------------------------------------------------------- lights
	// Navigation lights at the winglet bends: red port (left), green starboard (right);
	// anti-collision strobes just aft on each tip; white tail light at the tail apex.
	const bend = wingletStations()[3];
	const navRight = bend.o.clone().addScaledVector(bend.d, 0.018).addScaledVector(bend.u, -0.01);
	const strobeRight = bend.o.clone().addScaledVector(bend.d, bend.c * 0.92).addScaledVector(bend.u, -0.006);
	const tailLight = new THREE.Vector3(0, TAIL_APEX_Y, 0.957);
	const lensGeometry = keep(prepare(new THREE.SphereGeometry(0.011, 14, 10)));
	const strobeGeometry = keep(prepare(new THREE.SphereGeometry(0.008, 12, 8)));
	/** @param {THREE.Vector3} at @param {THREE.Material} material @param {THREE.BufferGeometry} geometry */
	const lamp = (at, material, geometry) => {
		const mesh = new THREE.Mesh(geometry, material);
		mesh.position.copy(at);
		object.add(mesh);
		return mesh;
	};
	lamp(navRight, lensGreen, lensGeometry);
	lamp(mirrored(navRight), lensRed, lensGeometry);
	lamp(tailLight, lensWhite, lensGeometry);
	lamp(strobeRight, lensStrobe, strobeGeometry);
	lamp(mirrored(strobeRight), lensStrobe, strobeGeometry);

	/**
	 * @typedef {object} Glow
	 * @property {THREE.Sprite} sprite
	 * @property {THREE.SpriteMaterial} material
	 * @property {number} size metres at scale 1
	 * @property {number} minPixels smallest on-screen size, CSS px
	 * @property {number} level brightness set by setLights, before the facing fade
	 * @property {boolean} aft a tail light: shines only into the rear sector
	 */
	/** @type {Glow[]} */
	const glows = [];
	const worldPos = new THREE.Vector3();
	const camPos = new THREE.Vector3();
	const parentScale = new THREE.Vector3();
	const localCam = new THREE.Vector3();
	/**
	 * @param {THREE.Vector3} at
	 * @param {number} color
	 * @param {number} size
	 * @param {number} minPixels
	 * @param {boolean} [aft]
	 */
	const glow = (at, color, size, minPixels, aft = false) => {
		const material = new THREE.SpriteMaterial({
			map: glowMap,
			color,
			// Additive; it also adds α² to alpha, which keeps the glow alive when the engine
			// draws the aircraft into its own layer and un-premultiplies it by alpha.
			blending: THREE.AdditiveBlending,
			transparent: true,
			depthWrite: false,
			toneMapped: false
		});
		const sprite = new THREE.Sprite(material);
		sprite.position.copy(at);
		sprite.renderOrder = 3;
		/** @type {Glow} */
		const g = { sprite, material, size, minPixels, level: 0, aft };
		// Keep the glow at least a few pixels across however far away the camera is,
		// and don't let the overhead-view exaggeration blow it up in proportion.
		sprite.onBeforeRender = (renderer, _scene, camera) => {
			if (!sprite.parent) return;
			sprite.getWorldPosition(worldPos);
			camera.getWorldPosition(camPos);
			const distance = worldPos.distanceTo(camPos);
			const height = Math.max(1, renderer.domElement.clientHeight || renderer.domElement.height);
			let worldPerPixel = 0;
			if (camera instanceof THREE.PerspectiveCamera) {
				worldPerPixel = (2 * distance * Math.tan((camera.fov * DEG) / 2)) / camera.zoom / height;
			} else if (camera instanceof THREE.OrthographicCamera) {
				worldPerPixel = (camera.top - camera.bottom) / camera.zoom / height;
			}
			sprite.parent.getWorldScale(parentScale);
			const scale = parentScale.x || 1;
			const want = Math.max(g.size * Math.sqrt(scale), g.minPixels * worldPerPixel);
			sprite.scale.setScalar(want / scale);
			// A tail light covers the 140° sector behind the aircraft: fade it out as the
			// camera comes round the side, rather than glowing through the airframe.
			let facing = 1;
			if (g.aft) {
				localCam.copy(camPos);
				sprite.parent.worldToLocal(localCam).sub(sprite.position).normalize();
				facing = THREE.MathUtils.smoothstep(localCam.z, 0.1, 0.45);
			}
			material.opacity = g.level * facing;
			sprite.updateMatrix();
			sprite.matrixWorld.multiplyMatrices(sprite.parent.matrixWorld, sprite.matrix);
		};
		object.add(sprite);
		glows.push(g);
		return g;
	};
	const navOut = 0.012;
	const glowGreen = glow(navRight.clone().add(new THREE.Vector3(navOut, 0, -0.004)), 0x33ff77, 0.42, 14);
	const glowRed = glow(mirrored(navRight).add(new THREE.Vector3(-navOut, 0, -0.004)), 0xff3322, 0.42, 14);
	// Small and dim: the tail glow sits at the boom apex just above the propeller, where
	// a wide sprite reads as a pale halo through bloom rather than as a lamp.
	const glowTail = glow(tailLight.clone().add(new THREE.Vector3(0, 0, 0.01)), 0xfff2dd, 0.14, 4, true);
	const glowStrobeR = glow(strobeRight.clone().add(new THREE.Vector3(0.01, 0, 0.004)), 0xf0f4ff, 1.1, 26);
	const glowStrobeL = glow(mirrored(strobeRight).add(new THREE.Vector3(-0.01, 0, 0.004)), 0xf0f4ff, 1.1, 26);

	// -------------------------------------------------------------- lighting
	const sun = new THREE.DirectionalLight(0xffffff, SUN_GAIN);
	sun.castShadow = true;
	sun.shadow.mapSize.set(1024, 1024);
	sun.shadow.bias = -0.0004;
	sun.shadow.normalBias = 0.012;
	object.add(sun, sun.target);
	const hemi = new THREE.HemisphereLight(0xb8cde0, 0x24384a, HEMI_GAIN);
	object.add(hemi);
	const sunDirection = new THREE.Vector3(0.3, 0.8, 0.5).normalize();
	const inverse = new THREE.Quaternion();
	const inverseMatrix = new THREE.Matrix4();
	let night = 0;
	const sunTint = new THREE.Color();
	/** 0 with the sun well up – ~0.85 with it on the horizon. */
	let twilight = 0;
	let scale = 1;

	const updateLightRig = () => {
		// The sun's direction is fixed in the world; the rig rides the rotating model.
		inverse.copy(object.quaternion).invert();
		sun.position.copy(sunDirection).applyQuaternion(inverse).multiplyScalar(8);
		const extent = 2 * scale;
		const cam = sun.shadow.camera;
		cam.left = -extent;
		cam.right = extent;
		cam.top = extent;
		cam.bottom = -extent;
		cam.near = 0.5 * scale;
		cam.far = 16 * scale;
		cam.updateProjectionMatrix();
		// A hemisphere light's "up" is its world position from the origin, so park it
		// straight above the world origin whatever the model's own transform is.
		object.updateMatrixWorld();
		inverseMatrix.copy(object.matrixWorld).invert();
		hemi.position.set(0, 1, 0).applyMatrix4(inverseMatrix);
	};

	// -------------------------------------------------------------- state
	let userOpacity = 1;
	let fade = 1;
	let lastTime = Number.NaN;
	let lastBank = 0;
	let lastPitch = 0;
	let rollRate = 0;
	let pitchRate = 0;

	/** @param {number} k */
	const applySolidOpacity = (k) => {
		for (const m of solids) {
			const transparent = k < 0.999;
			if (m.transparent !== transparent) {
				m.transparent = transparent;
				m.needsUpdate = true;
			}
			m.opacity = k;
		}
	};

	/** @param {number} rps propeller revolutions per second */
	const setPropBlur = (rps) => {
		const blur = Math.min(1, Math.max(0, (rps - 8) / 22));
		const k = userOpacity * fade;
		const bladeOpacity = (1 - 0.86 * blur) * k;
		const transparent = bladeOpacity < 0.999;
		if (propMat.transparent !== transparent) {
			propMat.transparent = transparent;
			propMat.depthWrite = !transparent;
			propMat.needsUpdate = true;
		}
		propMat.opacity = bladeOpacity;
		discMat.opacity = blur * 0.7 * k;
		disc.visible = discMat.opacity > 0.01;
		blades.castShadow = blur < 0.5;
		return blur;
	};

	const applyFade = () => {
		const k = userOpacity * fade;
		applySolidOpacity(k);
		object.visible = k > 0.01;
	};

	/** @param {number} time */
	const setLights = (time) => {
		const k = userOpacity * fade;
		// Lights read once the sun is low, before the sky is properly dark.
		const dim = Math.max(night, twilight);
		const navLevel = 0.16 + 0.84 * dim;
		// Anti-collision strobe: a double flash once a second.
		const phase = ((time % 1) + 1) % 1;
		const flash = phase < 0.05 || (phase > 0.14 && phase < 0.19) ? 1 : 0;
		const strobeLevel = flash * (0.45 + 0.55 * dim);
		glowGreen.level = navLevel * k;
		glowRed.level = navLevel * k;
		glowTail.level = navLevel * 0.35 * k;
		glowStrobeR.level = strobeLevel * k;
		glowStrobeL.level = strobeLevel * k;
		// Strobes flash smaller by day, when they have less dark around them to bloom into.
		glowStrobeR.size = glowStrobeL.size = 1.1 * (0.55 + 0.45 * dim);
		for (const g of glows) {
			g.material.opacity = g.level;
			g.sprite.visible = g.level > 0.004;
		}
		const lensLevel = 0.55 + 0.6 * dim;
		lensRed.color.setRGB(1 * lensLevel, 0.16 * lensLevel, 0.1 * lensLevel);
		lensGreen.color.setRGB(0.1 * lensLevel, 1 * lensLevel, 0.42 * lensLevel);
		lensWhite.color.setRGB(lensLevel, 0.96 * lensLevel, 0.9 * lensLevel);
		const s = 0.35 + 0.65 * flash;
		lensStrobe.color.setRGB(0.9 * s, 0.93 * s, s);
	};

	setPropBlur(0);
	setLights(0);
	updateLightRig();

	return {
		object,
		cameraMount,
		span: 2 * (navRight.x + 0.02),

		pose(state, toWorld) {
			object.position.copy(toWorld(state.x, state.y, state.altitude));
			attitudeQuaternion(state.heading, state.pitch, state.bank, object.quaternion);
			const visible = Math.max(0, Math.min(1, state.visible));
			if (Math.abs(visible - fade) > 1e-4) {
				fade = visible;
				applyFade();
			}

			// Rates for the control surfaces, smoothed.
			const dt = state.time - lastTime;
			if (Number.isFinite(dt) && dt > 1e-4 && dt < 0.5) {
				const a = Math.min(1, dt / 0.18);
				rollRate += ((state.bank - lastBank) / dt - rollRate) * a;
				pitchRate += ((state.pitch - lastPitch) / dt - pitchRate) * a;
			} else if (!(Number.isFinite(dt) && Math.abs(dt) <= 1e-4)) {
				rollRate = 0;
				pitchRate = 0;
			}
			lastTime = state.time;
			lastBank = state.bank;
			lastPitch = state.pitch;

			// Ailerons: lead the roll, plus a little opposite trim held in the turn.
			const clamp = (/** @type {number} */ v, /** @type {number} */ m) => Math.max(-m, Math.min(m, v));
			const aileron = clamp(0.8 * rollRate - 0.05 * state.bank, 18);
			ailerons[0].set(-aileron); // right: trailing edge up to roll right
			ailerons[1].set(aileron);
			// Ruddervators: pull (trailing edges up) with pitch rate and turn load, plus trim;
			// the yaw component differs side to side.
			const load = 1 / Math.max(0.3, Math.cos(state.bank * DEG)) - 1;
			const elevator = clamp(-(1.5 + 0.35 * pitchRate + 14 * load), 15);
			const rudder = clamp(0.8 * state.turnRate, 8);
			ruddervatorR.set(elevator + rudder);
			ruddervatorL.set(elevator - rudder);

			// Propeller: about 4 revolutions per metre of airspeed per second (≈ 5800 rpm at
			// 24 m/s). Above ~10 rev/s a camera would never see the blades frozen, so draw
			// them faint, turning slowly, over the blurred disc.
			const rps = Math.max(0, state.speed) * 4;
			const blur = setPropBlur(rps);
			const visualRps = rps < 10 ? rps : 10 - 8.7 * blur;
			propeller.rotation.z = -2 * Math.PI * ((state.time * visualRps) % 1);

			setLights(state.time);
			updateLightRig();
		},

		setOpacity(opacity) {
			userOpacity = Math.max(0, Math.min(1, opacity));
			applyFade();
		},

		setLighting(lighting) {
			const exposure = lighting.exposure ?? 1;
			sunDirection.copy(lighting.sunDirection).normalize();
			const up = Math.min(1, Math.max(0, (sunDirection.y + 0.02) / 0.08));
			sun.color.copy(lighting.sunColor);
			sun.intensity = SUN_GAIN * up * exposure;
			const env = lighting.environment;
			hemi.color.copy(lighting.ambient);
			// The sea under the aircraft returns a little dim blue sky light.
			hemi.groundColor.copy(lighting.ambient).multiply(new THREE.Color(0.32, 0.4, 0.48));
			hemi.intensity = (env ? HEMI_GAIN_WITH_ENV : HEMI_GAIN) * exposure;
			// The blur disc is unlit, so give it the radiance of matte black blades in this
			// light: a few percent of the ambient and the sun.
			discMat.color
				.copy(lighting.ambient)
				.multiplyScalar(0.15)
				.add(sunTint.copy(lighting.sunColor).multiplyScalar(0.06 * up))
				.multiplyScalar(exposure / Math.PI);
			night = Math.max(0, Math.min(1, lighting.night));
			const t = Math.min(1, Math.max(0, (sunDirection.y + 0.02) / 0.2));
			twilight = 0.85 * (1 - t * t * (3 - 2 * t));
			for (const m of lit) {
				if (m.envMap !== env) {
					m.envMap = env;
					m.needsUpdate = true;
				}
				// At dusk the airframe should read as a silhouette, not glow with the sky.
				m.envMapIntensity = (envIntensity.get(m) ?? 1) * (1 - 0.55 * night) * exposure;
			}
			setLights(Number.isFinite(lastTime) ? lastTime : 0);
			updateLightRig();
		},

		setScale(s) {
			scale = Math.max(1e-3, s);
			object.scale.setScalar(scale);
			updateLightRig();
		},

		setGimbal(pitch, yaw = 0) {
			aimGimbal(pitch, yaw);
		},

		dispose() {
			for (const g of geometries) g.dispose();
			for (const t of textures) t.dispose();
			const materials = new Set([...solids, ...lit, propMat, discMat, ...glows.map((g) => g.material)]);
			for (const m of materials) m.dispose();
			sun.shadow.map?.dispose();
			sun.dispose();
			hemi.dispose();
			object.clear();
		}
	};
}
