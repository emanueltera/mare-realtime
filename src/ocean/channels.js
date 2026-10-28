/**
 * The ground-truth channels the simulator renders alongside every frame. One table
 * feeds both sides: the shader builds its colour ramps from `stops`, and any legend
 * is drawn from the same stops, so a legend can never disagree with the
 * picture it describes.
 *
 * `range` is the value at the first and last stop. Ramps are sampled linearly in
 * sRGB between evenly spaced stops.
 */

/**
 * @typedef {object} Channel
 * @property {Exclude<import('./scene.js').RenderMode, 'shaded'>} id
 * @property {string} label short name, for buttons and legends
 * @property {string} unit
 * @property {[number, number]} range value at the first and last colour stop
 * @property {string[]} stops hex colours, evenly spaced across `range`
 * @property {string} description one line on what the channel is and why a model wants it
 */

/** @type {Channel[]} */
export const channels = [
	{
		id: 'height',
		label: 'Elevation',
		unit: 'm',
		range: [-2, 2],
		// Diverging: troughs cold, crests warm, mean sea level neutral.
		stops: ['#0b1f4d', '#1f5fa8', '#7fb8d8', '#eef2ee', '#f2c27a', '#d9672e', '#7a1f12'],
		description: 'Surface height above mean sea level at every pixel.'
	},
	{
		id: 'velocity',
		label: 'Heave rate',
		unit: 'm/s',
		range: [-1.5, 1.5],
		// Diverging on vertical velocity: water falling blue, rising magenta.
		stops: ['#1b2a6b', '#2f6fd6', '#8fd3f2', '#f4f4f4', '#f5a3c7', '#c8358a', '#5c0a4a'],
		description: 'Vertical orbital velocity of the surface: how the water is moving, not just where it is.'
	},
	{
		id: 'normals',
		label: 'Normals',
		unit: 'xyz',
		range: [-1, 1],
		// Not a ramp: normals are drawn as RGB = 0.5 + 0.5·n. These stops only seed a legend swatch.
		stops: ['#7f7fff', '#ff7f7f', '#7fff7f'],
		description: 'Surface orientation, which sets where the sky and sun are reflected.'
	},
	{
		id: 'foam',
		label: 'Whitecaps',
		unit: 'mask',
		range: [0, 1],
		stops: ['#031018', '#0f3a4a', '#7fd6e0', '#ffffff'],
		description: 'Where crests fold over and break, from the Jacobian of the surface displacement.'
	},
	{
		id: 'depth',
		label: 'Range',
		unit: 'm',
		range: [0, 400],
		// Sequential, near bright to far dark.
		stops: ['#fff3b0', '#f6b15a', '#e0603a', '#9c2f5c', '#4b1d6b', '#120a2e'],
		description: 'Distance from the camera to the water along each ray.'
	}
];

/** @param {string} id */
export function channel(id) {
	const found = channels.find((c) => c.id === id);
	if (!found) throw new Error(`unknown channel ${id}`);
	return found;
}
