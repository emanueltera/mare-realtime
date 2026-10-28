/**
 * Thin helpers over three.js for the engine's compute-style passes: a full-screen
 * triangle that runs a GLSL ES 3.0 fragment shader into a render target, and a
 * factory for float render targets.
 */
import * as THREE from 'three';

const FULLSCREEN_VERTEX = /* glsl */ `
precision highp float;
in vec3 position;
out vec2 vUv;
void main() {
	vUv = position.xy * 0.5 + 0.5;
	gl_Position = vec4(position.xy, 0.0, 1.0);
}`;

/** One triangle that covers clip space. */
function fullscreenGeometry() {
	const geometry = new THREE.BufferGeometry();
	geometry.setAttribute('position', new THREE.Float32BufferAttribute([-1, -1, 0, 3, -1, 0, -1, 3, 0], 3));
	return geometry;
}

const sharedCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

/**
 * A fragment shader run over a whole render target.
 */
export class FullscreenPass {
	/**
	 * @param {string} fragmentShader GLSL ES 3.0 body, without the #version line
	 * @param {Record<string, THREE.IUniform>} uniforms
	 * @param {Partial<THREE.ShaderMaterialParameters>} [options]
	 */
	constructor(fragmentShader, uniforms, options = {}) {
		this.material = new THREE.RawShaderMaterial({
			glslVersion: THREE.GLSL3,
			vertexShader: FULLSCREEN_VERTEX,
			fragmentShader: `precision highp float;\nprecision highp int;\nprecision highp sampler2D;\n${fragmentShader}`,
			uniforms,
			depthTest: false,
			depthWrite: false,
			...options
		});
		this.mesh = new THREE.Mesh(fullscreenGeometry(), this.material);
		this.mesh.frustumCulled = false;
		this.uniforms = uniforms;
	}

	/**
	 * @param {THREE.WebGLRenderer} renderer
	 * @param {THREE.WebGLRenderTarget | null} target
	 */
	render(renderer, target) {
		renderer.setRenderTarget(target);
		renderer.render(this.mesh, sharedCamera);
	}

	dispose() {
		this.mesh.geometry.dispose();
		this.material.dispose();
	}
}

/**
 * @param {number} width
 * @param {number} height
 * @param {THREE.RenderTargetOptions} options
 */
export function floatTarget(width, height, options = {}) {
	return new THREE.WebGLRenderTarget(width, height, {
		type: THREE.FloatType,
		format: THREE.RGBAFormat,
		minFilter: THREE.NearestFilter,
		magFilter: THREE.NearestFilter,
		wrapS: THREE.ClampToEdgeWrapping,
		wrapT: THREE.ClampToEdgeWrapping,
		depthBuffer: false,
		stencilBuffer: false,
		generateMipmaps: false,
		...options
	});
}
