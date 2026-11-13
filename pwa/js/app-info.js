// Model table and runtime information (Settings → App, Lab → Logs).

import { $, h } from './util.js';
import { getWebGpuAdapter, runtimeInfo } from './inference.js';
import { cards, engine } from './engine-manager.js';
import { appVersion } from './updates.js';
import { viewportInfo } from './viewport.js';

export function renderModels() {
  if (!cards) return;
  const placeholder = cards.age_gender.placeholder || cards.expression.placeholder;
  $('#banner').hidden = !placeholder;
  const px = (card) => `${card.input.shape[2]} px`;
  const mb = (v) => (v == null ? '–' : v.toFixed(1));
  const size = (card) => [`${mb(card.sizes.fp32_mb)} MB FP32`, h('small', {}, `${mb(card.sizes.int8_mb)} MB INT8`)];
  const status = (card) => (card.placeholder ? 'Placeholder' : [`Trained`, h('small', {}, `best epoch ${card.trained_epoch}`)]);
  const row = (name, ...cells) => h('tr', {}, h('th', { scope: 'row' }, name), ...cells.map((c) => h('td', {}, c)));
  $('#modelTable tbody').replaceChildren(
    row('Face detection', 'any', '0.2 MB FP32', ['Pre-trained', h('small', {}, 'YuNet')]),
    row('Age + gender', px(cards.age_gender), size(cards.age_gender), status(cards.age_gender)),
    row('Expression', px(cards.expression), size(cards.expression), status(cards.expression)),
  );
}

export async function renderRuntime() {
  const rt = runtimeInfo();
  const gpu = await getWebGpuAdapter();
  const items = [
    ['App version', await appVersion()],
    ['ONNX Runtime Web', rt.ort],
    ['WebGPU', gpu ? 'available' : 'not available'],
    ['Cross-origin isolated', rt.crossOriginIsolated ? 'yes (multi-threaded CPU inference possible)' : 'no (single CPU thread)'],
    ['Active models', engine ? engine.label : 'not loaded yet'],
    ['Viewport', viewportInfo ? `${viewportInfo.dvh} px (100dvh) · inner ${viewportInfo.innerHeight} · screen ${viewportInfo.screen}${viewportInfo.standalone ? ' · home-screen app' : ''}${viewportInfo.corrected ? ` · corrected +${viewportInfo.corrected} px` : ''}` : '–'],
  ];
  $('#runtimeInfo').replaceChildren(...items.flatMap(([k, v]) => [h('dt', {}, k), h('dd', {}, v)]));
  $('#versionValue').textContent = String(items[0][1]).slice(0, 12).toUpperCase();
}
