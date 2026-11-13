// Lab → Speed: latency benchmark of the detector and both classifiers for the
// selected backends and precisions, with CSV/JSON export.

import { $, download, h, ms, timestamp, toCsv } from './util.js';
import { configureRuntime, getWebGpuAdapter, isAppleMobile } from './inference.js';
import { deviceInfo, runBenchmark } from './benchmark.js';
import { logError } from './diagnostics.js';
import { settings } from './settings.js';
import { showError, toast } from './feedback.js';
import { getCards, releaseEngine } from './engine-manager.js';

let live = null; // the LiveView, set by setupBenchmark()
let benchRows = [];
let benchDevice = null;

export async function setupBenchmark({ live: liveView }) {
  live = liveView;
  const gpu = await getWebGpuAdapter();
  const gpuBox = $('#benchWebgpu');
  gpuBox.disabled = !gpu;
  gpuBox.checked = !!gpu && !isAppleMobile(); // opt-in on iPhone: run it separately
  $('#benchWebgpuNote').textContent = !gpu ? 'Not available in this browser' : isAppleMobile() ? 'Run separately on iPhone' : '';
  $('#runBench').addEventListener('click', runBench);
  $('#benchCsv').addEventListener('click', () => download(`edgelens-benchmark-${timestamp()}.csv`, toCsv(benchRows.map((r) => ({ ...r, ...flatDevice() }))), 'text/csv'));
  $('#benchJson').addEventListener('click', () => download(`edgelens-benchmark-${timestamp()}.json`, JSON.stringify({ device: benchDevice, results: benchRows }, null, 2), 'application/json'));
}

function flatDevice() {
  const d = benchDevice || {};
  return { device_user_agent: d.userAgent, wasm_threads: d.wasmThreads, cross_origin_isolated: d.crossOriginIsolated, ort_web: d.onnxruntimeWeb };
}

async function runBench() {
  const backends = [['webgpu', $('#benchWebgpu')], ['wasm', $('#benchWasm')]].filter(([, b]) => b.checked && !b.disabled).map(([n]) => n);
  const precisions = [['fp32', $('#benchFp32')], ['int8', $('#benchInt8')]].filter(([, b]) => b.checked).map(([n]) => n);
  if (!backends.length || !precisions.length) {
    toast('Choose at least one backend and one precision.');
    return;
  }
  const btn = $('#runBench');
  btn.disabled = true;
  live.stop();
  await releaseEngine();
  try {
    configureRuntime(settings.threads);
    const c = await getCards();
    benchDevice = await deviceInfo();
    renderDevice();
    const order = $('#benchReverse').checked ? [...precisions].reverse() : precisions;
    const combos = backends.flatMap((backend) => order.map((precision) => ({ backend, precision })));
    benchRows = await runBenchmark({
      cards: c,
      combos,
      iterations: Number($('#benchIters').value),
      warmup: 10,
      detectorSize: settings.detectorSize,
      onProgress: (text, frac) => {
        $('#benchProgress').textContent = text;
        $('#benchBar').value = frac;
      },
    });
    for (const r of benchRows.filter((x) => x.error)) logError('benchmark', `${r.stage}: ${r.error}`, { engine: `${r.backend} ${r.precision}` });
    renderBench();
    $('#benchProgress').textContent = `Finished ${new Date().toLocaleTimeString()}.`;
  } catch (e) {
    showError(e);
  } finally {
    btn.disabled = false;
  }
}

function renderBench() {
  const body = $('#benchTable tbody');
  body.replaceChildren(...benchRows.map((r) =>
    h('tr', { class: r.error ? 'failed' : '' },
      h('th', { scope: 'row' }, r.stage, h('small', {}, r.input)),
      h('td', {}, r.backend === 'webgpu' ? 'WebGPU' : 'WASM', h('small', {}, r.precision.toUpperCase())),
      ...(r.error
        ? [h('td', { colspan: 5, class: 'err' }, r.error)]
        : [h('td', {}, ms(r.mean_ms)), h('td', { class: 'col-wide' }, ms(r.p50_ms)), h('td', {}, ms(r.p95_ms)),
          h('td', {}, String(r.fps)), h('td', { class: 'col-wide' }, String(r.load_ms))]))));
  $('#benchResults').hidden = benchRows.length === 0;
  $('#benchCsv').disabled = $('#benchJson').disabled = benchRows.length === 0;
}

function renderDevice() {
  const d = benchDevice;
  if (!d) return;
  const items = [
    ['Browser', d.userAgent],
    ['CPU threads available', d.logicalCores ?? 'not reported'],
    ['WASM threads used', `${d.wasmThreads}${d.crossOriginIsolated ? '' : ' (page not cross-origin isolated)'}`],
    ['WebGPU adapter', d.webgpu ? [d.webgpu.vendor, d.webgpu.architecture, d.webgpu.description].filter(Boolean).join(' ') || 'available' : 'not available'],
    ['Screen', d.screen],
    ['Installed app', d.standalone ? 'yes' : 'no (running in a browser tab)'],
    ['ONNX Runtime Web', d.onnxruntimeWeb],
  ];
  $('#deviceInfo').replaceChildren(...items.flatMap(([k, v]) => [h('dt', {}, k), h('dd', {}, String(v))]));
}
