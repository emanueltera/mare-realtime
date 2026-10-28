import { mount } from 'svelte';
import App from './app/App.svelte';

mount(App, { target: /** @type {HTMLElement} */ (document.getElementById('app')) });
