import { mount } from 'svelte';
import App from './app/App.svelte';
import './app/app.css';

mount(App, { target: /** @type {HTMLElement} */ (document.getElementById('app')) });
