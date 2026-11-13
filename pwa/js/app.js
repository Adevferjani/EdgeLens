// App shell and entry point: creates the views (Live, Photo, Lab evaluation,
// detail sheet), switches between modes, runs the first-launch flow and starts
// the remaining parts (settings, updates, Lab tools, model preload).
//
// Module overview (all in js/):
//   settings.js / settings-panel.js  settings object and the settings form
//   engine-manager.js                model loading for the current settings
//   screens.js                       welcome and loading screens
//   feedback.js                      toasts and error messages
//   updates.js                       service worker, offline chip, app updates
//   app-info.js                      model table and runtime information
//   lab-benchmark.js, lab-testset.js, lab-diagnostics.js   Lab tools
//   viewport.js                      iPhone height fix and bar size
//   live.js, photo.js, evaluation.js, sheet.js   the views themselves

import { $, $$ } from './util.js';
import { LiveView } from './live.js';
import { EvaluationView } from './evaluation.js';
import { PhotoView } from './photo.js';
import { FaceSheet } from './sheet.js';
import { settings } from './settings.js';
import { bindSettings, enableDevMode } from './settings-panel.js';
import { showError, toast } from './feedback.js';
import { hideScreen, loading, showScreen } from './screens.js';
import { engineReady, getCards, getEngine, onEngineLoaded } from './engine-manager.js';
import { installHint, setupServiceWorker } from './updates.js';
import { renderModels, renderRuntime } from './app-info.js';
import { setupBenchmark } from './lab-benchmark.js';
import { setupTestset } from './lab-testset.js';
import { setupDiagnostics } from './lab-diagnostics.js';
import './viewport.js';

const ONBOARDED_KEY = 'edgelens.onboarded.v1';

// keep the runtime information current after every model load
onEngineLoaded(() => renderRuntime());

// ------------------------------------------------------------------ views
const sheet = new FaceSheet({ root: $('#sheet'), backdrop: $('#sheetBackdrop') });

const live = new LiveView({
  video: $('#video'),
  display: $('#display'),
  overlay: $('#overlay'),
  telemetry: $('#telemetry'),
  getEngine,
  getSettings: () => settings,
  onError: showError,
  // the loading screen (if shown) stays up until the camera picture is ready
  onStartPhase: (phase) => {
    if (phase === 'begin') loading.hold();
    else if (phase === 'camera') loading.cameraPhase();
    else loading.release();
  },
  onState: (state) => {
    document.body.dataset.live = state;
    $('#liveStatus').textContent = { idle: '', starting: '', switching: 'SWITCHING CAMERA…', running: '', frozen: 'FROZEN' }[state];
    $('#startCam').disabled = state === 'starting';
    $('#startCam').textContent = state === 'starting' ? 'Starting camera…' : 'Start camera';
  },
  onFaces: (n) => {
    $('#faceCount').textContent = n ? `${n} FACE${n > 1 ? 'S' : ''}` : 'NO FACE';
  },
  onInspect: (faces, index) => {
    const c = live.engine.cards;
    sheet.open({
      faces,
      index,
      labels: c.expression.labels,
      elderlyMin: c.age_gender.elderly_min_age ?? 60,
      actionLabel: 'Resume live',
      onClose: () => live.resume(),
    });
  },
});
$('#startCam').addEventListener('click', () => live.start());

// Timing bar in Live: tap to expand into a labelled panel (state remembered).
const TELE_KEY = 'edgelens.telemetryOpen';
function setTelemetryOpen(open, focus = false) {
  $('#telePanel').hidden = !open;
  $('#teleOpen').hidden = open;
  $('#teleOpen').setAttribute('aria-expanded', String(open));
  try {
    localStorage.setItem(TELE_KEY, open ? '1' : '0');
  } catch {
    /* not remembered */
  }
  if (focus) (open ? $('#teleClose') : $('#teleOpen')).focus();
}
$('#teleOpen').addEventListener('click', () => setTelemetryOpen(true, true));
$('#teleClose').addEventListener('click', () => setTelemetryOpen(false, true));
try {
  setTelemetryOpen(localStorage.getItem(TELE_KEY) === '1');
} catch {
  setTelemetryOpen(false);
}
$('#flipCam').addEventListener('click', () => live.flip());

// Release the camera while the app is in the background (and before the
// phone's own camera opens for "Take photo"); Live restarts it when shown again.
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'hidden') {
    if (live.stream || live.starting) live.stop();
  } else if (document.body.dataset.view === 'live') {
    live.wake();
  }
});
$('#photoTake').addEventListener('click', () => {
  if (live.stream || live.starting) live.stop();
});
$('#freezeBtn').addEventListener('click', () => {
  if (!live.freeze(0)) toast('No face in view yet.');
});

// The Photo and Lab evaluation views register their own event listeners.
new PhotoView({
  getEngine,
  getSettings: () => settings,
  onError: showError,
  onInspect: (opts) => sheet.open(opts),
});

new EvaluationView({
  root: $('#lab-eval'),
  getEngine,
  getSettings: () => settings,
  onError: showError,
  toast,
});

// ------------------------------------------------------------------ modes
let lastView = 'live';

function showView(name) {
  if (name === 'lab' && !settings.devMode) name = 'live';
  if (!$(`#view-${name}`)) name = 'live';
  sheet.close();
  for (const v of $$('main > .view')) v.hidden = v.id !== `view-${name}`;
  for (const b of $$('.modes [data-view]')) {
    if (b.dataset.view === name) b.setAttribute('aria-current', 'page');
    else b.removeAttribute('aria-current');
  }
  // Live keeps its camera open (paused) while another view is shown and
  // continues when it is shown again.
  if (name === 'live') live.wake();
  else live.sleep();
  if (name !== 'settings') lastView = name;
  document.body.dataset.view = name;
  $('main').scrollTop = 0;
  if (location.hash !== `#${name}`) history.replaceState(null, '', `#${name}`);
}
for (const b of $$('.modes [data-view]')) b.addEventListener('click', () => showView(b.dataset.view));
$('#openSettings').addEventListener('click', () => showView('settings'));
$('#closeSettings').addEventListener('click', () => showView(lastView));
window.addEventListener('hashchange', () => showView(location.hash.slice(1)));

function showLab(panel) {
  for (const b of $$('[data-lab]')) b.setAttribute('aria-selected', String(b.dataset.lab === panel));
  for (const p of $$('.lab-panel')) p.hidden = p.id !== `lab-${panel}`;
}
for (const b of $$('[data-lab]')) b.addEventListener('click', () => showLab(b.dataset.lab));

// ------------------------------------------------------------------ first launch
function setupOnboarding() {
  let seen = false;
  try {
    seen = localStorage.getItem(ONBOARDED_KEY) === '1';
  } catch {
    seen = true; // no storage: do not show it on every start
  }
  if (seen) {
    $('#onboarding').hidden = true;
    return;
  }
  const done = (view) => {
    try {
      localStorage.setItem(ONBOARDED_KEY, '1');
    } catch {
      /* not saved (e.g. storage full): the welcome screen may appear again */
    }
    hideScreen($('#onboarding'));
    showView(view);
  };
  // Get started: load the models (loading screen if they are not ready yet),
  // then the Live start screen asks for the camera as the last step.
  $('#onbStart').addEventListener('click', () => {
    if (engineReady()) {
      done('live'); // models already loaded in the background: straight to the camera start screen
      return;
    }
    // The loading screen fades in on top of the welcome screen, which is then
    // removed underneath it; the Live start screen appears only when the
    // models are ready. No screen in between.
    try {
      localStorage.setItem(ONBOARDED_KEY, '1');
    } catch {
      /* not saved (e.g. storage full): the welcome screen may appear again */
    }
    loading.hold();
    loading.show();
    showView('live');
    setTimeout(() => {
      $('#onboarding').hidden = true;
    }, 260);
    getEngine().catch(showError).finally(() => loading.release());
  });
  $('#onbPhoto').addEventListener('click', () => done('photo'));
  showScreen($('#onboarding')); // already visible from the inline script in index.html
}

// ------------------------------------------------------------------ start
async function main() {
  let tab = location.hash.slice(1);
  try {
    tab ||= sessionStorage.getItem('coiTab') || '';
    sessionStorage.removeItem('coiTab');
  } catch {
    /* storage unavailable: start on the view in the URL, or Live */
  }
  bindSettings({ live });
  showView(tab || 'live');
  setupOnboarding();
  setupDiagnostics({
    openLogs: () => {
      enableDevMode();
      showView('lab');
      showLab('logs');
    },
  });
  installHint();
  setupServiceWorker();
  setupBenchmark({ live });
  setupTestset({ live });
  try {
    await getCards();
    renderModels();
    // load the models in the background, so the camera starts without waiting;
    // a failure is logged, and the next regular load retries and reports it
    getEngine({ silent: true }).catch(() => {});
  } catch (e) {
    showError(e);
  }
  renderRuntime();
}

main();
