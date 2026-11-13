// User settings: defaults, persistence in localStorage and the one-time
// migration of settings saved by earlier versions. The settings form lives in
// settings-panel.js; every module reads the same `settings` object.

import { storageGet, storageSet } from './util.js';

// Storage keys and the cache name keep the project's earlier working name
// ("agescope"), so settings and offline data saved by installed versions stay valid.
export const SETTINGS_KEY = 'agescope.settings.v1';
export const DEFAULTS = {
  backend: 'auto',
  precision: 'auto',
  threads: 'auto',
  scoreThreshold: 0.6,
  detectorSize: 320,
  maxFaces: 5,
  smoothing: true,
  showLandmarks: false,
  blurFaces: false,
  devMode: true,
  showTimings: true,
};

export const settings = { ...DEFAULTS, ...storageGet(SETTINGS_KEY, {}) };
export const saveSettings = () => storageSet(SETTINGS_KEY, settings);
// One-time update of settings saved by earlier versions: developer mode and
// timings on, 5 faces at once (unless a different number was chosen).
if ((settings.settingsVersion ?? 1) < 2) {
  settings.devMode = true;
  settings.showTimings = true;
  if (settings.maxFaces === 3) settings.maxFaces = 5;
  settings.settingsVersion = 2;
  saveSettings();
}

export const changedSettings = () => Object.keys(DEFAULTS).filter((k) => settings[k] !== DEFAULTS[k]);
