// Crash recorder. iOS ends a page that uses too much memory without any
// event, so the app keeps a short log of what it is doing in localStorage.
// If the previous session was still marked "open" at the next start, it
// ended unexpectedly, and its log is kept for About → Diagnostics.

import { storageGet, storageSet } from './util.js';

const KEY = 'agescope.diag.v1';
const MAX_STEPS = 40;
const MAX_CRASHES = 5;
const MAX_ERRORS = 10;

let data = storageGet(KEY, { crashes: [], current: null });
const save = () => storageSet(KEY, data);

/** Returns the crashed previous session, if any. */
export function startDiagnostics(info) {
  const prev = data.current;
  let crashed = null;
  if (prev?.open) {
    crashed = { ...prev, open: false };
    data.crashes = [crashed, ...(data.crashes || [])].slice(0, MAX_CRASHES);
  }
  data.current = { started: new Date().toISOString(), open: true, info, steps: [] };
  save();

  const setOpen = (open) => {
    if (!data.current) return;
    data.current.open = open;
    save();
  };
  addEventListener('pagehide', () => setOpen(false));
  document.addEventListener('visibilitychange', () => setOpen(document.visibilityState !== 'hidden'));
  addEventListener('error', (e) => logError('page', e.message || e));
  addEventListener('unhandledrejection', (e) => logError('promise', e.reason?.message || e.reason));
  return crashed;
}

/** Errors are kept separately: they survive reloads that are not crashes. */
export function logError(source, message, extra = {}) {
  const text = String(message).slice(0, 400);
  crumb('error', { source, message: text.slice(0, 200) });
  data.errors = [{ time: new Date().toISOString(), source, message: text, ...extra }, ...(data.errors || [])].slice(0, MAX_ERRORS);
  save();
}

export const errors = () => data.errors || [];

export function crumb(event, detail = {}) {
  const c = data.current;
  if (!c) return;
  c.steps.push({ t: Math.round(performance.now() / 100) / 10, event, ...detail });
  if (c.steps.length > MAX_STEPS) c.steps.splice(0, c.steps.length - MAX_STEPS);
  save();
}

export const crashes = () => data.crashes || [];

export function clearCrashes() {
  data.crashes = [];
  data.errors = [];
  save();
}

/** One-line, human readable summary of what a session was doing last. */
export function describe(session) {
  const last = [...session.steps].reverse().find((s) => s.event !== 'error') || session.steps.at(-1);
  if (!last) return 'right after start';
  const names = {
    'engine:start': 'loading the models',
    'engine:ready': 'models loaded',
    'live:start': 'starting the camera',
    'live:frames': 'live camera running',
    'live:stop': 'live camera stopped',
    'photo:start': 'analysing a photo',
    'photo:done': 'photo analysed',
    'bench:stage': 'running the benchmark',
    'bench:done': 'benchmark finished',
  };
  const what = names[last.event] || last.event;
  const engine = [...session.steps].reverse().find((s) => s.engine)?.engine;
  return `${what}${engine ? ` (${engine})` : ''}, ${Math.round(last.t)} s after opening`;
}
