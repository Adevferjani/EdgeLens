// Lab → Logs: unexpected closures (crash breadcrumbs) and recent errors, with a
// report that can be copied or downloaded.

import { $, download, h, timestamp } from './util.js';
import { runtimeInfo } from './inference.js';
import { clearCrashes, crashes, describe, errors, startDiagnostics } from './diagnostics.js';
import { settings } from './settings.js';
import { toast } from './feedback.js';
import { appVersion } from './updates.js';

function renderDiagnostics() {
  const list = crashes();
  const errs = errors();
  const nodes = list.map((s) =>
    h('details', { class: 'diag' },
      h('summary', {}, `Closed unexpectedly ${new Date(s.started).toLocaleString()}: ${describe(s)}`),
      h('pre', {}, s.steps.slice(-12).map((st) => JSON.stringify(st)).join('\n'))));
  if (errs.length) {
    nodes.push(h('details', { class: 'diag' },
      h('summary', {}, `${errs.length} recent error${errs.length > 1 ? 's' : ''}`),
      h('pre', {}, errs.map((e) => `${e.time} [${e.source}${e.engine ? `, ${e.engine}, ${e.threads} thread(s)` : ''}] ${e.message}`).join('\n\n'))));
  }
  if (!nodes.length) nodes.push(h('p', { class: 'note' }, 'No unexpected closures or errors recorded.'));
  $('#diagList').replaceChildren(...nodes);
  $('#diagCopy').disabled = $('#diagClear').disabled = list.length + errs.length === 0;
}

export function setupDiagnostics({ openLogs }) {
  const crashed = startDiagnostics({
    ua: navigator.userAgent,
    coi: !!self.crossOriginIsolated,
    cores: navigator.hardwareConcurrency || null,
    settings: { backend: settings.backend, precision: settings.precision, threads: settings.threads, detectorSize: settings.detectorSize },
  });
  if (crashed) {
    toast(`EdgeLens closed unexpectedly last time while ${describe(crashed)}.`, {
      label: 'Details',
      run: openLogs,
    });
  }
  $('#diagCopy').addEventListener('click', async () => {
    const report = JSON.stringify({ app: await appVersion(), runtime: runtimeInfo(), settings, errors: errors(), crashes: crashes() }, null, 1);
    try {
      await navigator.clipboard.writeText(report);
      toast('Report copied to the clipboard.');
    } catch {
      download(`edgelens-diagnostics-${timestamp()}.json`, report, 'application/json');
    }
  });
  $('#diagClear').addEventListener('click', () => {
    clearCrashes();
    renderDiagnostics();
  });
  renderDiagnostics();
}
