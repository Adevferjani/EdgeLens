// On-device benchmark: latency of every model on each backend and precision,
// plus the full per-frame pipeline. Results feed the "edge device" columns of
// the performance tables in docs/results.

import { decodeDetections, prepareDetectorInput } from './detector.js';
import { cropFace } from './facecrop.js';
import { benchSession, Engine, makeTensor, ORT_VERSION, runtimeInfo, webgpuInfo } from './inference.js';
import { crumb } from './diagnostics.js';
import { createCanvases, freeCanvases } from './pipeline.js';
import { nextFrame, stats } from './util.js';

const FRAME = { w: 1280, h: 720 };

function randomData(n, max = 1) {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = Math.random() * max;
  return a;
}

// Warm-up runs at least `warmup` times and for at least `warmupMs`. A fixed count
// alone is very short for these fast models, and on the phone the timings only
// settle after some seconds of continuous load; otherwise the configuration
// measured first looks slower.
async function time(fn, warmup, iterations, warmupMs = 2000) {
  const t0 = performance.now();
  for (let i = 0; i < warmup || performance.now() - t0 < warmupMs; i++) await fn();
  const times = [];
  for (let i = 0; i < iterations; i++) {
    const t = performance.now();
    await fn();
    times.push(performance.now() - t);
  }
  return stats(times);
}

function noiseFrame() {
  const c = document.createElement('canvas');
  c.width = FRAME.w;
  c.height = FRAME.h;
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(FRAME.w, FRAME.h);
  for (let i = 0; i < img.data.length; i++) img.data[i] = (i & 3) === 3 ? 255 : (Math.random() * 255) | 0;
  ctx.putImageData(img, 0, 0);
  return c;
}

let order = 0;
const row = (o, s) => ({
  ...o,
  run_order: ++order,
  mean_ms: +s.mean.toFixed(2),
  p50_ms: +s.p50.toFixed(2),
  p95_ms: +s.p95.toFixed(2),
  fps: +(1000 / s.mean).toFixed(1),
});

export async function deviceInfo() {
  const rt = runtimeInfo();
  return {
    timestamp: new Date().toISOString(),
    userAgent: navigator.userAgent,
    platform: navigator.userAgentData?.platform || navigator.platform || '',
    logicalCores: navigator.hardwareConcurrency || null,
    deviceMemoryGb: navigator.deviceMemory ?? null,
    screen: `${screen.width}x${screen.height} @${window.devicePixelRatio}x`,
    standalone: window.matchMedia('(display-mode: standalone)').matches || navigator.standalone === true,
    onnxruntimeWeb: ORT_VERSION,
    crossOriginIsolated: rt.crossOriginIsolated,
    wasmThreads: rt.wasmThreads,
    webgpu: await webgpuInfo(),
  };
}

/**
 * combos: [{backend, precision}], returns rows. onProgress(text, fraction).
 */
export async function runBenchmark({ cards, combos, iterations = 100, warmup = 10, warmupMs = 2000, detectorSize = 320, onProgress }) {
  const rows = [];
  order = 0;
  const dh = Math.ceil((detectorSize * FRAME.h) / FRAME.w / 32) * 32;
  const stages = [
    { stage: 'Face detection', file: () => cards.detector.file, dims: [1, 3, dh, detectorSize], max: 255, fixedPrecision: 'fp32' },
    { stage: 'Age + gender', file: (p) => cards.age_gender.files[p], dims: cards.age_gender.input.shape, max: 1 },
    { stage: 'Expression', file: (p) => cards.expression.files[p], dims: cards.expression.input.shape, max: 1 },
  ];
  const doneDetector = new Set();
  const total = combos.length * (stages.length + 1);
  let step = 0;
  const progress = (text) => onProgress?.(text, step / total);

  for (const { backend, precision } of combos) {
    for (const st of stages) {
      step++;
      const prec = st.fixedPrecision || precision;
      const key = `${st.stage}|${backend}|${prec}`;
      if (doneDetector.has(key)) continue;
      doneDetector.add(key);
      const base = { stage: st.stage, backend, precision: prec, input: st.dims.slice(2).join('x') };
      progress(`${st.stage} on ${backend.toUpperCase()} ${prec.toUpperCase()}…`);
      crumb('bench:stage', { stage: st.stage, engine: `${backend} ${prec}` });
      await nextFrame();
      let session;
      try {
        const s = await benchSession(st.file(precision), backend);
        session = s.session;
        const feeds = { input: makeTensor(randomData(st.dims.reduce((a, b) => a * b, 1), st.max), st.dims) };
        const r = await time(() => session.run(feeds), warmup, iterations, warmupMs);
        rows.push(row({ ...base, load_ms: +s.loadMs.toFixed(0) }, r));
      } catch (e) {
        rows.push({ ...base, error: String(e.message || e).slice(0, 160) });
      } finally {
        await session?.release().catch(() => {});
      }
    }

    // Full pipeline for one face: frame -> detector -> crop x2 -> both models.
    step++;
    progress(`Full pipeline on ${backend.toUpperCase()} ${precision.toUpperCase()}…`);
    crumb('bench:stage', { stage: 'Full pipeline', engine: `${backend} ${precision}` });
    await nextFrame();
    let engine;
    const frame = noiseFrame();
    const canvases = createCanvases();
    try {
      engine = await Engine.create({ backend, precision, cards });
      const face = {
        x: 540, y: 250, w: 200, h: 220, score: 1,
        landmarks: [[595, 330], [685, 330], [640, 385], [605, 425], [675, 425]],
      };
      const r = await time(async () => {
        const inp = prepareDetectorInput(frame, FRAME.w, FRAME.h, detectorSize, canvases.det);
        const out = await engine.detect(inp.data, inp.height, inp.width);
        decodeDetections(out, inp.width, inp.height, { scale: inp.scale });
        const a = cropFace(frame, face, cards.age_gender.crop, engine.agSize, canvases.ag);
        const b = cropFace(frame, face, cards.expression.crop, engine.exSize, canvases.ex);
        await engine.ageGender(a);
        await engine.expression(b);
      }, Math.max(3, warmup >> 1), Math.max(10, iterations >> 1), warmupMs);
      const load = Object.values(engine.loadMs).reduce((x, y) => x + y, 0);
      rows.push(row({ stage: 'Full pipeline (1 face)', backend, precision, input: `${FRAME.w}x${FRAME.h} frame`, load_ms: +load.toFixed(0) }, r));
    } catch (e) {
      rows.push({ stage: 'Full pipeline (1 face)', backend, precision, input: `${FRAME.w}x${FRAME.h} frame`, error: String(e.message || e).slice(0, 160) });
    } finally {
      await engine?.release();
      freeCanvases({ ...canvases, frame });
    }
  }
  crumb('bench:done', { rows: rows.length });
  onProgress?.('Done', 1);
  return rows;
}
