// Lifecycle of the inference engine (detector + two classifiers): loading for
// the current backend/precision settings, background preload, release before
// benchmarks, and the loading-screen steps.

import { configureRuntime, Engine, loadCards, resolveBackend, runtimeInfo } from './inference.js';
import { crumb, logError } from './diagnostics.js';
import { saveSettings, settings } from './settings.js';
import { setEngineChip, toast } from './feedback.js';
import { loading } from './screens.js';

export let cards = null;
export let engine = null;
let enginePromise = null;
let engineKey = '';

const engineListeners = [];
/** Called with the engine every time a model set has finished loading. */
export function onEngineLoaded(fn) {
  engineListeners.push(fn);
}

export async function getCards() {
  cards ??= await loadCards();
  return cards;
}

export function invalidateEngine() {
  engineKey = '';
}

/** Models for the current settings are loaded (no loading screen needed). */
export const engineReady = () => !!engine && engineKey !== '' && !enginePromise;

/**
 * The engine for the current settings, loading it if needed. silent: a
 * background preload (at app start) that shows no loading screen; a later
 * normal call while it is still loading reveals the screen.
 */
export async function getEngine({ silent = false } = {}) {
  const c = await getCards();
  const { backend, precision, notes } = await resolveBackend(settings, c);
  const key = `${backend}|${precision}`;
  if (engineKey === key && (engine || enginePromise)) {
    if (!engine && !silent) loading.reveal();
    return engine || enginePromise;
  }
  const old = engine;
  engine = null;
  engineKey = key;
  setEngineChip('Loading…');
  enginePromise = (async () => {
    await old?.release();
    configureRuntime(settings.threads);
    crumb('engine:start', { engine: `${backend} ${precision}`, threads: runtimeInfo().wasmThreads });
    loading.begin(c, `${backend === 'webgpu' ? 'WebGPU' : 'WASM'} ${precision.toUpperCase()}`, { silent });
    try {
      const e = await Engine.create({ backend, precision, cards: c, onStep: (id, st) => loading.step(id, st) });
      loading.step('warmup', 'loading');
      await e.warmup(settings.detectorSize, Math.ceil((settings.detectorSize * 9) / 16 / 32) * 32);
      loading.step('warmup', 'ready');
      crumb('engine:ready', { engine: e.label, loadMs: Math.round(Object.values(e.loadMs).reduce((a, b) => a + b, 0)), warmupMs: Math.round(e.warmupMs) });
      if (engineKey === key) engine = e;
      setEngineChip(e.label);
      notes.forEach((n) => toast(n));
      engineListeners.forEach((fn) => fn(e));
      return e;
    } catch (err) {
      engineKey = '';
      setEngineChip('Models not loaded');
      const threads = runtimeInfo().wasmThreads;
      logError('engine', err?.message || err, { engine: `${backend} ${precision}`, threads });
      if (threads > 1 && settings.threads !== '1') {
        settings.threads = '1';
        saveSettings();
        toast('Loading failed with several CPU threads. Reloading with one thread…');
        setTimeout(() => location.reload(), 1500);
      }
      const wrapped = new Error(`The models could not be loaded: ${err?.message || err}`);
      wrapped.logged = true;
      throw wrapped;
    } finally {
      enginePromise = null;
      loading.end();
    }
  })();
  return enginePromise;
}

export async function releaseEngine() {
  const e = engine || (enginePromise && (await enginePromise.catch(() => null)));
  engine = null;
  engineKey = '';
  await e?.release();
  setEngineChip('Models idle');
}
