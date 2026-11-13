// Settings form: binding to the `settings` object, reset to defaults (with
// Undo), and the side effects of individual settings.

import { $ } from './util.js';
import { changedSettings, DEFAULTS, saveSettings, settings } from './settings.js';
import { toast } from './feedback.js';
import { invalidateEngine } from './engine-manager.js';
import { checkForUpdate, installUpdate, postponeUpdate, restartApp } from './updates.js';

let live = null; // the LiveView, set by bindSettings()

export function fillSettingsForm() {
  const form = $('#settingsForm');
  for (const [key, value] of Object.entries(settings)) {
    const field = form.elements[key];
    if (!field) continue;
    if (field.type === 'checkbox') field.checked = value;
    else field.value = String(value);
  }
}

/** Apply a complete set of values (defaults, or the previous ones for Undo). */
export function setAllSettings(values) {
  const before = { ...settings };
  for (const k of Object.keys(DEFAULTS)) settings[k] = values[k];
  saveSettings();
  fillSettingsForm();
  if (before.backend !== settings.backend || before.precision !== settings.precision) {
    invalidateEngine();
    if (live.running) live.stop();
  }
  applySettings();
  return before;
}

export function resetSettings() {
  if (!changedSettings().length) {
    toast('All settings already have their default values.');
    return;
  }
  const before = setAllSettings(DEFAULTS);
  const restart = before.threads !== settings.threads;
  toast('Settings reset to their defaults.', [
    { label: 'Undo', run: () => setAllSettings(before) },
    restart && { label: 'Restart', quiet: true, run: () => restartApp('settings') },
  ]);
}

export function bindSettings({ live: liveView }) {
  live = liveView;
  const form = $('#settingsForm');
  fillSettingsForm();
  $('#resetTop').addEventListener('click', resetSettings);
  $('#resetBottom').addEventListener('click', resetSettings);
  form.addEventListener('change', (ev) => {
    const f = ev.target;
    const key = f.name;
    if (!(key in DEFAULTS)) return;
    settings[key] = f.type === 'checkbox' ? f.checked : typeof DEFAULTS[key] === 'number' ? Number(f.value) : f.value;
    saveSettings();
    if (key === 'threads') {
      toast('The thread setting applies after a restart.', { label: 'Restart', run: () => restartApp('settings') });
    }
    if (['backend', 'precision'].includes(key)) {
      invalidateEngine();
      if (live.running) live.stop();
    }
    if (key === 'devMode') {
      toast(settings.devMode ? 'Developer mode on: LAB is now next to LIVE and PHOTO.' : 'Developer mode off.');
      if (!settings.devMode && (settings.backend !== 'auto' || settings.precision !== 'auto')) {
        // without developer mode the processor settings are hidden, so use the defaults again
        settings.backend = 'auto';
        settings.precision = 'auto';
        form.elements.backend.value = 'auto';
        form.elements.precision.value = 'auto';
        saveSettings();
        invalidateEngine();
      }
    }
    applySettings();
  });
  applySettings();
  $('#reloadApp').addEventListener('click', checkForUpdate);
  $('#installUpdate').addEventListener('click', installUpdate);
  $('#laterUpdate').addEventListener('click', postponeUpdate);
}

export function applySettings() {
  const n = changedSettings().length;
  $('#resetBar').hidden = n === 0;
  $('#resetCount').textContent = `${n} SETTING${n === 1 ? '' : 'S'} CHANGED FROM DEFAULT`;
  $('#thresholdValue').textContent = settings.scoreThreshold.toFixed(2);
  $('#maxFacesText').textContent = String(settings.maxFaces);
  document.body.dataset.blur = String(settings.blurFaces);
  document.body.dataset.dev = String(settings.devMode);
  document.body.dataset.timings = String(settings.devMode && settings.showTimings);
}

/** Switch developer mode on (used by the "Details" action after a crash). */
export function enableDevMode() {
  if (!settings.devMode) {
    settings.devMode = true;
    saveSettings();
    $('#settingsForm').elements.devMode.checked = true;
    applySettings();
  }
}
