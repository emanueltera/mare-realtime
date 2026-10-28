## Project Configuration

- **Language**: JavaScript (JSDoc, checked with `pnpm check`)
- **Package Manager**: pnpm
- **Add-ons**: playwright, three.js, Svelte 5 (the app UI only; the engine is framework-free)

## What this is

A real-time spectral ocean for generating simulated UAV trajectories over different ocean
conditions: the flight and its IMU, the onboard camera's view, and per-frame ground truth.
The engine in `src/ocean/` was extracted from Tera AI's `mayfly-demo` deck
(`../mayfly-demo/src/lib/ocean`); its offline, path-traced sibling is `../mare`.
`docs/ARCHITECTURE.md` is the contract between the modules: read it first, and keep it
true when an interface changes.

## Rules

- `src/trajectory/` and `src/conditions/` are pure and deterministic: no three.js, no
  DOM, no `Math.random` or `Date.now` — seeded PRNGs only. They must run in node.
- Ground truth is what was rendered: frame truth comes from the engine's own camera after
  the frame is drawn, never re-derived from the spec.
- Never name the onboard camera's maker or model, in code, comments, file names or docs.
- Numbers in docs and UI describe the simulator (spectrum parameters, sea state, grid
  sizes); do not present them as measurements of a real sea or a real aircraft.
- Write code that reads like the surrounding code: its comment density, naming and idiom.
