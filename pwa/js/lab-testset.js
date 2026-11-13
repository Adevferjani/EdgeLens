// Lab → Test sets: runs the deployed models on a test-set package exported on
// the PC (training/tools/export_testset.py) and compares accuracy and per-image
// agreement with the PC results.

import { $, download, h, ms, pct, timestamp, toCsv } from './util.js';
import { configureRuntime, getWebGpuAdapter } from './inference.js';
import { deviceInfo } from './benchmark.js';
import { openTestset, runTestset } from './testset.js';
import { logError } from './diagnostics.js';
import { settings } from './settings.js';
import { showError, toast } from './feedback.js';
import { getCards, releaseEngine } from './engine-manager.js';

let live = null; // the LiveView, set by setupTestset()
let testsetResult = null;
let testsetStop = false;

const AG_ROWS = [
  ['Age MAE (years)', 'age_mae', 'num'],
  ['Age MAE, 18+ (years)', 'age_mae_18plus', 'num'],
  ['Adult / 60+ accuracy (18+)', 'age_group_accuracy_18plus', 'pct'],
  ['60+ recall', 'elderly_recall_18plus', 'pct'],
  ['Balanced adult / 60+ accuracy', 'age_group_balanced_accuracy_18plus', 'pct'],
  ['Adult / 60+ macro-F1', 'age_group_macro_f1_18plus', 'f1'],
  ['Gender accuracy', 'gender_accuracy', 'pct'],
  ['Gender macro-F1', 'gender_macro_f1', 'f1'],
];
const EX_ROWS = [
  ['Accuracy (7 classes)', 'accuracy', 'pct'],
  ['Macro-F1', 'macro_f1', 'f1'],
  ['Happy vs sad, forced choice', 'happy_sad_accuracy_2way', 'pct'],
  ['Happy vs sad, top of 7', 'happy_sad_accuracy_7way', 'pct'],
];
const TIME_ROWS = [
  ['Decode + resize (ms/image)', 'decode'],
  ['Age + gender model (ms/image)', 'age_gender'],
  ['Expression model (ms/image)', 'expression'],
];
const fmt = (v, kind) => (v == null ? '–' : kind === 'pct' ? pct(v, 1) : kind === 'f1' ? v.toFixed(3) : v.toFixed(2));

export function setupTestset({ live: liveView }) {
  live = liveView;
  const input = $('#testsetInput');
  input.addEventListener('change', () => {
    const file = input.files[0];
    input.value = '';
    if (file) runTestsetUi(file);
  });
  $('#testsetStop').addEventListener('click', () => {
    testsetStop = true;
  });
  getWebGpuAdapter().then((gpu) => {
    $('#testsetWebgpu').disabled = !gpu;
  });
  $('#testsetCsv').addEventListener('click', () => download(`edgelens-testset-${timestamp()}.csv`, toCsv(testsetRows()), 'text/csv'));
  $('#testsetJson').addEventListener('click', () => download(`edgelens-testset-${timestamp()}.json`, JSON.stringify(testsetResult, null, 1), 'application/json'));
}

async function runTestsetUi(file) {
  const precisions = [['fp32', $('#testsetFp32')], ['int8', $('#testsetInt8')]].filter(([, b]) => b.checked).map(([n]) => n);
  if (!precisions.length) {
    toast('Choose at least one precision.');
    return;
  }
  const backend = $('input[name="testsetBackend"]:checked').value;
  const progress = (text, frac) => {
    $('#testsetProgress').textContent = text;
    $('#testsetBar').value = frac;
  };
  const pick = $('#testsetPick');
  pick.classList.add('disabled');
  pick.setAttribute('aria-disabled', 'true');
  $('#testsetInput').disabled = true;
  $('#testsetStop').hidden = false;
  testsetStop = false;
  let wakeLock = null;
  live.stop();
  await releaseEngine();
  try {
    wakeLock = await navigator.wakeLock?.request('screen').catch(() => null);
    progress(`Reading ${file.name}…`, 0);
    const testset = await openTestset(file);
    configureRuntime(settings.threads);
    const c = await getCards();
    const res = await runTestset({ testset, cards: c, backend, precisions, onProgress: progress, shouldStop: () => testsetStop });
    const m = testset.manifest;
    testsetResult = {
      exported: new Date().toISOString(),
      device: await deviceInfo(),
      testset: { file: testset.name, created: m.created, pc: m.pc, models: m.models,
        utkface_n: m.age_gender.n, utkface_subset: m.age_gender.subset, fer2013_n: m.expression.n, fer2013_subset: m.expression.subset },
      pc_metrics: { age_gender: m.age_gender.pc_metrics, expression: m.expression.pc_metrics },
      stopped: res.stopped,
      runs: res.runs,
    };
    if (res.runs.length) renderTestset();
  } catch (e) {
    progress('', 0);
    logError('testset', e?.message || e);
    showError(e);
  } finally {
    wakeLock?.release().catch(() => {});
    pick.classList.remove('disabled');
    pick.removeAttribute('aria-disabled');
    $('#testsetInput').disabled = false;
    $('#testsetStop').hidden = true;
  }
}

function testsetColumns() {
  const r = testsetResult;
  const pcVariants = Object.keys(r.pc_metrics.age_gender || r.pc_metrics.expression || {});
  return [
    ...pcVariants.map((v) => ({ head: 'PC', sub: `ONNX Runtime ${v.toUpperCase()}`, key: `pc_${v}`,
      metrics: { age_gender: r.pc_metrics.age_gender?.[v], expression: r.pc_metrics.expression?.[v] }, latency: null })),
    ...r.runs.map((run) => ({ head: 'This device', sub: run.label, key: `device_${run.backend}_${run.precision}`, metrics: run.metrics, latency: run.latency })),
  ];
}

function renderTestset() {
  const r = testsetResult;
  const cols = testsetColumns();
  const section = (text) => h('tr', { class: 'section' }, h('th', { scope: 'rowgroup', colspan: cols.length + 1 }, text));
  const rows = (task, defs) => defs.map(([name, key, kind]) => h('tr', {}, h('th', { scope: 'row' }, name),
    ...cols.map((c) => h('td', {}, fmt(c.metrics[task]?.[key], kind)))));
  $('#testsetTable').replaceChildren(
    h('caption', {}, 'Test-set accuracy: PC vs this device'),
    h('thead', {}, h('tr', {}, h('th', {}, 'Metric'), ...cols.map((c) => h('th', {}, c.head, h('small', {}, c.sub))))),
    h('tbody', {},
      section(`Age + gender · UTKFace test images (n = ${r.testset.utkface_n})`), ...rows('age_gender', AG_ROWS),
      section(`Expression · FER2013 test images (n = ${r.testset.fer2013_n})`), ...rows('expression', EX_ROWS),
      section('Time on this device'),
      ...TIME_ROWS.map(([name, key]) => h('tr', {}, h('th', { scope: 'row' }, name),
        ...cols.map((c) => h('td', {}, c.latency?.[key] ? ms(c.latency[key].mean_ms) : '–'))))));

  const agree = (run, task, key, kind) => {
    if (!run.hashMatch[task]) return 'other model file';
    const v = run.agreement[task]?.[key];
    return v == null ? '–' : kind === 'pct' ? pct(v, 2) : v === 0 ? '0' : kind === 'yrs' && v >= 0.01 ? v.toFixed(3) : v.toExponential(1);
  };
  const A_ROWS = [
    ['Age: mean difference (years)', 'age_gender', 'age_mean_abs_diff', 'yrs'],
    ['Age: largest difference (years)', 'age_gender', 'age_max_abs_diff', 'yrs'],
    ['Gender: same decision', 'age_gender', 'gender_agreement', 'pct'],
    ['p(female): largest difference', 'age_gender', 'p_female_max_abs_diff', 'exp'],
    ['Expression: same top class', 'expression', 'top1_agreement', 'pct'],
    ['Probabilities: largest difference', 'expression', 'prob_max_abs_diff', 'exp'],
  ];
  $('#testsetAgree').replaceChildren(
    h('caption', {}, 'Same images, same model file: this device vs PC'),
    h('thead', {}, h('tr', {}, h('th', {}, ''), ...r.runs.map((run) => h('th', {}, run.label)))),
    h('tbody', {}, A_ROWS.map(([name, task, key, kind]) => h('tr', {}, h('th', { scope: 'row' }, name),
      ...r.runs.map((run) => h('td', {}, agree(run, task, key, kind)))))));

  const subset = r.testset.utkface_subset || r.testset.fer2013_subset ? ' Random subset of the test splits.' : '';
  $('#testsetNote').textContent = `PC: ${r.testset.pc?.cpu ?? 'unknown CPU'}, ONNX Runtime ${r.testset.pc?.onnxruntime ?? ''} (file created ${r.testset.created}).`
    + `${subset}${r.stopped ? ' The run was stopped early; only completed precisions are shown.' : ''}`
    + ' Times are per image on this device; the PC times are in runs/*/evaluation.json.';
  $('#testsetResults').hidden = false;
}

function testsetRows() {
  const cols = testsetColumns();
  const line = (section, metric, get) => ({ section, metric, ...Object.fromEntries(cols.map((c) => [c.key, get(c) ?? ''])) });
  const d = testsetResult.device;
  return [
    ...AG_ROWS.map(([name, key]) => line('age_gender', name, (c) => c.metrics.age_gender?.[key])),
    ...EX_ROWS.map(([name, key]) => line('expression', name, (c) => c.metrics.expression?.[key])),
    ...TIME_ROWS.map(([name, key]) => line('time', name, (c) => c.latency?.[key]?.mean_ms)),
  ].map((row) => ({ ...row, device_user_agent: d.userAgent, wasm_threads: d.wasmThreads }));
}
