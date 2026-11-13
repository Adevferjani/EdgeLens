// Full screens (welcome and loading) and the loading-screen controller.

import { $, h } from './util.js';

// Screen rule: a full screen is only replaced by the screen that really comes
// next, never by an intermediate one. Whenever someone waits for the models
// (Get started, Start camera, a photo), the loading screen appears at once and
// stays at least MIN_SHOWN_MS, so it never flashes; a background preload shows
// nothing. Full screens fade in through CSS and fade out here.
const MIN_SHOWN_MS = 600;

export function showScreen(el) {
  clearTimeout(el._leave);
  el.classList.remove('leaving');
  el.hidden = false;
  if (el.id === 'loadingScreen') document.body.dataset.loadingScreen = '';
}
export function hideScreen(el) {
  if (el.hidden) return;
  el.classList.add('leaving');
  clearTimeout(el._leave);
  el._leave = setTimeout(() => {
    el.hidden = true;
    el.classList.remove('leaving');
    if (el.id === 'loadingScreen') delete document.body.dataset.loadingScreen;
  }, matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : 260);
}

export const loading = {
  steps: [],
  active: false, // a model load is running
  held: 0, // callers that keep the screen up after the load (camera start, Get started)
  shownAt: 0,
  hideTimer: 0,
  stepsFor(c, label) {
    const mb = (v) => (v == null ? '' : ` · ${v.toFixed(2)} MB`);
    const prec = label.endsWith('INT8') ? 'int8' : 'fp32';
    return [
      { id: 'detector', name: 'Face detector', meta: `YUNET${mb(0.23)}` },
      { id: 'age_gender', name: 'Age and gender', meta: `MOBILENETV3${c ? mb(c.age_gender.sizes?.[`${prec}_mb`]) : ''}` },
      { id: 'expression', name: 'Expression', meta: `MOBILENETV3${c ? mb(c.expression.sizes?.[`${prec}_mb`]) : ''}` },
      { id: 'warmup', name: 'Warm-up', meta: label.toUpperCase() },
    ].map((st) => ({ ...st, state: 'queued' }));
  },
  /** A load starts; silent = background preload (no screen until someone waits). */
  begin(c, label, { silent = false } = {}) {
    this.steps = this.stepsFor(c, label);
    this.render();
    this.active = true;
    if (!silent) this.show();
  },
  /** Someone waits for a running load: show the screen now. */
  reveal() {
    if (this.active) this.show();
  },
  show() {
    clearTimeout(this.hideTimer);
    const el = $('#loadingScreen');
    if (!el.hidden && !el.classList.contains('leaving')) return;
    if (!this.steps.length) {
      this.steps = this.stepsFor(null, 'WASM');
      this.render();
    }
    $('#loadingTitle').textContent = 'Preparing the models';
    $('#loadingText').textContent = 'They run on this phone. Nothing is uploaded.';
    $('#loadingKind').textContent = navigator.serviceWorker?.controller ? 'FROM THIS DEVICE' : 'FIRST START';
    this.shownAt = performance.now();
    showScreen(el);
  },
  /** Keep the screen up after the load until release() (only if it is shown). */
  hold() {
    this.held++;
  },
  release() {
    this.held = Math.max(0, this.held - 1);
    this.hideWhenDone();
  },
  /** The models are ready and the camera starts behind the screen. */
  cameraPhase() {
    if ($('#loadingScreen').hidden) return;
    $('#loadingTitle').textContent = 'Starting the camera';
    $('#loadingText').textContent = 'The models are ready.';
  },
  step(id, state) {
    const st = this.steps.find((x) => x.id === id);
    if (st) st.state = state;
    this.render();
  },
  render() {
    const label = { queued: 'QUEUED', loading: 'LOADING', ready: 'READY' };
    $('#loadingSteps').replaceChildren(...this.steps.map((st) => h('li', { 'data-state': st.state },
      h('span', { class: 'name' }, st.name),
      h('span', { class: 'state' }, label[st.state]),
      h('span', { class: 'meta' }, st.meta),
      st.state === 'loading' ? h('span', { class: 'line' }, h('i')) : null)));
    const done = this.steps.filter((st) => st.state === 'ready').length;
    [...$('#loadingSegments').children].forEach((seg, i) => seg.classList.toggle('on', i < done));
  },
  /** The load finished (or failed). */
  end() {
    this.active = false;
    this.hideWhenDone();
  },
  hideWhenDone() {
    if (this.active || this.held) return;
    const el = $('#loadingScreen');
    if (el.hidden) return;
    clearTimeout(this.hideTimer);
    this.hideTimer = setTimeout(() => hideScreen(el), Math.max(0, this.shownAt + MIN_SHOWN_MS - performance.now()));
  },
};
