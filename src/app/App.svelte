<script>
	// Scaffold smoke test: the extracted engine with the UAV in a chase view.
	import OceanStage from './OceanStage.svelte';

	/** @type {import('../ocean/scene.js').SceneSpec} */
	const scene = {
		sea: { windSpeed: 9, windDirection: 60, swellHeight: 1, swellDirection: 30 },
		light: { sunElevation: 25, sunAzimuth: 200 },
		camera: { mode: 'chase', distance: 18, azimuth: 150, elevation: 12 },
		aircraft: { altitude: 80, path: 'weave' }
	};
</script>

<div style="position: fixed; inset: 0;"><OceanStage {scene} /></div>
