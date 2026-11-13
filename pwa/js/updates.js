// App updates and offline support: service worker registration, the
// consent-based update flow ("Install and restart"), the offline status chip and
// the install hint.

import { $ } from './util.js';
import { toast } from './feedback.js';

// A new version is downloaded in the background but only used after the user
// agrees ("Install and restart"), or after EdgeLens is fully closed and reopened.
let installRequested = false;
let updateToast = null;
const shortVersion = (v) => String(v || '').slice(0, 7).toUpperCase();
const clock = () => new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

/** Load the page again through the service worker, on the given view. */
export function restartApp(view = document.body.dataset.view) {
  try {
    sessionStorage.setItem('coiTab', view);
  } catch {
    /* storage unavailable: the app restarts on Live */
  }
  location.replace(location.href.split('#')[0]);
}

function askVersion(worker) {
  return new Promise((resolve) => {
    const ch = new MessageChannel();
    ch.port1.onmessage = (e) => resolve(e.data?.version || null);
    worker.postMessage({ type: 'version' }, [ch.port2]);
    setTimeout(() => resolve(null), 1500);
  });
}

function waitInstalled(worker, timeout = 120000) {
  return new Promise((resolve) => {
    const done = () => ['installed', 'activating', 'activated', 'redundant'].includes(worker.state);
    if (done()) return resolve();
    worker.addEventListener('statechange', () => done() && resolve());
    setTimeout(resolve, timeout);
  });
}

export async function showUpdateReady(worker) {
  const v = await askVersion(worker);
  $('#updateVersion').textContent = v ? shortVersion(v) : '';
  $('#updateReady').hidden = false;
  $('#updateStatus').textContent = '';
  updateOfflineChip();
  if (!updateToast?.isConnected && document.body.dataset.view !== 'settings') {
    updateToast = toast('A new version of EdgeLens is ready.', [
      { label: 'Install', run: installUpdate },
      { label: 'Later', quiet: true, run: postponeUpdate },
    ], { sticky: true });
  }
}

export async function checkForUpdate() {
  const btn = $('#reloadApp');
  const status = $('#updateStatus');
  btn.disabled = true;
  btn.textContent = 'Checking…';
  status.textContent = '';
  try {
    const reg = await navigator.serviceWorker?.getRegistration();
    if (!reg) {
      status.textContent = 'Updates are not available in this browser.';
      return;
    }
    if (!reg.waiting) {
      if (navigator.onLine === false) throw new Error('offline');
      await reg.update();
    }
    const sw = reg.installing || reg.waiting;
    if (!sw) {
      status.textContent = `You have the latest version (${shortVersion(await appVersion())}). Checked ${clock()}.`;
      return;
    }
    if (sw.state !== 'installed') {
      btn.textContent = 'Downloading update…';
      await waitInstalled(sw);
    }
    if (reg.waiting) await showUpdateReady(reg.waiting);
    else status.textContent = 'The update could not be downloaded completely. Try again later.';
  } catch {
    status.textContent = "Couldn't check: no connection. The installed version keeps working offline.";
  } finally {
    btn.disabled = false;
    btn.textContent = 'Check for update';
  }
}

export async function installUpdate() {
  const reg = await navigator.serviceWorker?.getRegistration();
  if (!reg?.waiting) {
    restartApp();
    return;
  }
  installRequested = true;
  $('#installUpdate').disabled = true;
  $('#installUpdate').textContent = 'Restarting…';
  reg.waiting.postMessage({ type: 'skipWaiting' }); // -> controllerchange -> restart
  setTimeout(() => restartApp(), 4000); // fallback
}

export function postponeUpdate() {
  updateToast?.remove();
  $('#updateStatus').textContent = 'The new version will be installed the next time EdgeLens is fully closed and reopened.';
}

export async function appVersion() {
  // Ask the active service worker; older versions do not answer this message, so
  // after a short timeout the version is read from the cache name instead.
  const ctl = navigator.serviceWorker?.controller;
  if (ctl) {
    const v = await new Promise((resolve) => {
      const ch = new MessageChannel();
      ch.port1.onmessage = (e) => resolve(e.data?.version);
      ctl.postMessage({ type: 'version' }, [ch.port2]);
      setTimeout(() => resolve(null), 1500);
    });
    if (v) return v;
  }
  try {
    const key = (await caches.keys()).find((k) => k.startsWith('agescope-'));
    return key ? key.slice('agescope-'.length) : 'not cached yet';
  } catch {
    return 'unknown';
  }
}

// ------------------------------------------------------------------ offline
export async function updateOfflineChip() {
  const chip = $('#offlineChip');
  try {
    const keys = await caches.keys();
    const ready = navigator.serviceWorker?.controller && keys.some((k) => k.startsWith('agescope-'));
    chip.dataset.state = ready ? 'ready' : 'pending';
    chip.title = ready ? 'All processing happens on this device; the app also works offline' : 'All processing happens on this device';
    $('#offlineValue').textContent = ready ? 'READY' : 'SAVING…';
    $('#offlineStatus').textContent = ready
      ? 'Models and app files are stored on this device; EdgeLens works in airplane mode.'
      : 'Still saving the app files for offline use.';
  } catch {
    $('#offlineValue').textContent = 'ONLINE ONLY';
  }
}

// A new service worker asks open pages whether they support "Install and restart".
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.onmessage = (e) => {
    if (e.data?.type === 'update-handshake') e.ports[0]?.postMessage({ consent: true });
  };
}

export async function setupServiceWorker() {
  if (!('serviceWorker' in navigator)) {
    $('#offlineValue').textContent = 'ONLINE ONLY';
    return;
  }
  const hadController = !!navigator.serviceWorker.controller;
  // No automatic reload for cross-origin isolation (needed for multi-threaded
  // WebAssembly): a reload would replace the screen while the user is using it.
  // The page is isolated from the next start on, when the service worker serves
  // it, and runs single-threaded until then. On the tested iPhone 15, four threads
  // were only slightly faster than one for these small models.
  navigator.serviceWorker.addEventListener('controllerchange', () => {
    if (installRequested) {
      restartApp();
      return;
    }
    // installed without asking (the page did not answer, e.g. it was in the background)
    if (hadController) {
      toast('A new version of EdgeLens was installed.', [
        { label: 'Restart', run: () => restartApp() },
        { label: 'Later', quiet: true, run: () => {} },
      ], { sticky: true });
    }
    updateOfflineChip();
  });
  try {
    const reg = await navigator.serviceWorker.register('./sw.js');
    watchUpdates(reg);
    // downloaded earlier but not installed yet ("Not now")
    if (reg.waiting && navigator.serviceWorker.controller) showUpdateReady(reg.waiting);
    await navigator.serviceWorker.ready;
  } catch (e) {
    console.warn('Service worker registration failed', e);
  }
  updateOfflineChip();
  setTimeout(updateOfflineChip, 4000);
}

// An app started from the home screen is often only resumed, not reloaded, so the
// browser never looks for a new version. Check whenever the app comes back to the
// foreground (at most once a minute) and show the download in the status chip.
function watchUpdates(reg) {
  let last = Date.now();
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState !== 'visible' || Date.now() - last < 60000) return;
    last = Date.now();
    reg.update().catch(() => {});
  });
  reg.addEventListener('updatefound', () => {
    const sw = reg.installing;
    if (!sw || !navigator.serviceWorker.controller) return;
    $('#offlineChip').dataset.state = 'updating';
    $('#offlineValue').textContent = 'DOWNLOADING UPDATE…';
    sw.addEventListener('statechange', () => {
      if (sw.state === 'installed' && reg.waiting === sw) showUpdateReady(sw);
      if (sw.state === 'redundant') updateOfflineChip();
    });
  });
}

export function installHint() {
  const standalone = window.matchMedia('(display-mode: standalone)').matches || navigator.standalone === true;
  const ios = /iPhone|iPad|iPod/.test(navigator.userAgent);
  $('#installHint').textContent = standalone
    ? 'Running as an installed app.'
    : ios
      ? 'To install: tap Share in Safari, then Add to Home Screen.'
      : 'To install: use the install button in the address bar, or Add to Home Screen on a phone.';
}
