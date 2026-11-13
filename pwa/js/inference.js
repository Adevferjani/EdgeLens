// ONNX Runtime Web engine: loads the detector and both attribute models on
// the chosen backend (WebGPU or WebAssembly) and precision (FP32 or INT8).

import * as ort from '../vendor/ort/ort.webgpu.min.mjs';

const ROOT = new URL('../', import.meta.url);
export const ORT_VERSION = ort.env.versions?.web ?? 'unknown';

// Fallback framing if a model card has no calibrated crop (see facecrop.js).
const DEFAULT_CROP = {
  age_gender: { scale: 1.5, shift_x: 0, shift_y: 0 },
  expression: { scale: 1.1, shift_x: 0, shift_y: 0 },
};

let runtimeReady = false;

/** Must run once, before the first session is created. */
export function configureRuntime(threadsSetting = 'auto') {
  if (runtimeReady) return;
  ort.env.wasm.wasmPaths = new URL('vendor/ort/', ROOT).href;
  const cores = navigator.hardwareConcurrency || 1;
  const threads = threadsSetting === '1' || !self.crossOriginIsolated ? 1 : Math.max(1, Math.min(4, cores));
  ort.env.wasm.numThreads = threads;
  ort.env.logLevel = 'error';
  runtimeReady = true;
}

export function runtimeInfo() {
  return {
    ort: ORT_VERSION,
    wasmThreads: ort.env.wasm.numThreads,
    crossOriginIsolated: !!self.crossOriginIsolated,
    webgpuApi: 'gpu' in navigator,
  };
}

let adapterPromise;
export function getWebGpuAdapter() {
  if (!('gpu' in navigator)) return Promise.resolve(null);
  adapterPromise ??= navigator.gpu.requestAdapter().catch(() => null);
  return adapterPromise;
}

export async function webgpuInfo() {
  const a = await getWebGpuAdapter();
  if (!a) return null;
  const i = a.info || {};
  return { vendor: i.vendor || '', architecture: i.architecture || '', description: i.description || '' };
}

export const modelUrl = (file) => new URL(`models/${file}`, ROOT).href;

export async function loadCards() {
  const get = async (name) => {
    const r = await fetch(modelUrl(`${name}.json`), { cache: 'no-cache' });
    if (!r.ok) throw new Error(`models/${name}.json is missing (HTTP ${r.status}). Export the models first.`);
    return r.json();
  };
  const [detector, age_gender, expression] = await Promise.all([
    get('detector'),
    get('age_gender'),
    get('expression'),
  ]);
  for (const [task, card] of Object.entries({ age_gender, expression })) {
    card.crop = { ...DEFAULT_CROP[task], ...(card.crop || {}) };
  }
  return { detector, age_gender, expression };
}

/** iPhone / iPad (iPadOS reports itself as a Mac with touch). */
export const isAppleMobile = () =>
  /iPhone|iPad|iPod/.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);

/** Turn "auto" settings into a concrete backend and precision. */
export async function resolveBackend(settings, cards) {
  const notes = [];
  const gpu = await getWebGpuAdapter();
  let backend = settings.backend;
  // Sustained WebGPU inference next to the camera crashed Safari on iPhone
  // (see docs/architecture.md, section 5), so "auto" uses the CPU there.
  if (backend === 'auto') backend = gpu && !isAppleMobile() ? 'webgpu' : 'wasm';
  if (backend === 'webgpu' && !gpu) {
    backend = 'wasm';
    notes.push('WebGPU is not available on this browser, so the CPU (WebAssembly) backend is used.');
  }
  let precision = settings.precision;
  // INT8 gave no speed-up on the iPhone (docs/results/quantization.md), so "auto"
  // uses FP32.
  if (precision === 'auto') precision = 'fp32';
  const hasInt8 = cards.age_gender.files.int8 && cards.expression.files.int8;
  if (precision === 'int8' && !hasInt8) {
    precision = 'fp32';
    notes.push('INT8 models are not exported yet, so FP32 is used.');
  }
  return { backend, precision, notes };
}

async function createSession(file, backend) {
  const t0 = performance.now();
  const options = {
    executionProviders: [backend],
    graphOptimizationLevel: 'all',
    logSeverityLevel: 3, // errors only (hides harmless optimiser notices)
  };
  // Weight-only INT8 models: let ONNX Runtime turn the INT8 weights back into
  // FP32 once at load time (otherwise every inference dequantises them again).
  if (file.includes('int8')) options.extra = { session: { disable_quant_qdq: '1' } };
  const session = await ort.InferenceSession.create(modelUrl(file), options);
  return { session, loadMs: performance.now() - t0 };
}

export class Engine {
  /** onStep(name, state) reports progress: name = detector | age_gender | expression. */
  static async create({ backend, precision, cards, onStep = () => {} }) {
    const e = new Engine();
    e.backend = backend;
    e.precision = precision;
    e.cards = cards;
    e.loadMs = {};
    // Sequential creation: WebGPU initialises its device once, on the first session.
    onStep('detector', 'loading');
    const det = await createSession(cards.detector.file, backend);
    onStep('detector', 'ready');
    onStep('age_gender', 'loading');
    const ag = await createSession(cards.age_gender.files[precision], backend);
    onStep('age_gender', 'ready');
    onStep('expression', 'loading');
    const ex = await createSession(cards.expression.files[precision], backend);
    onStep('expression', 'ready');
    e.det = det.session;
    e.ag = ag.session;
    e.ex = ex.session;
    e.loadMs = { detector: det.loadMs, age_gender: ag.loadMs, expression: ex.loadMs };
    e.agSize = cards.age_gender.input.shape[2];
    e.exSize = cards.expression.input.shape[2];
    return e;
  }

  /** One run per model, so one-time setup (e.g. WebGPU shader compilation) is not timed later. */
  async warmup(detectorW = 320, detectorH = 192) {
    const t0 = performance.now();
    await this.detect(new Float32Array(3 * detectorW * detectorH), detectorH, detectorW);
    await this.ageGender(new Float32Array(3 * this.agSize * this.agSize));
    await this.expression(new Float32Array(3 * this.exSize * this.exSize));
    this.warmupMs = performance.now() - t0;
  }

  get label() {
    return `${this.backend === 'webgpu' ? 'WebGPU' : 'WASM'} ${this.precision.toUpperCase()}`;
  }

  async detect(data, height, width) {
    const out = await this.det.run({ input: new ort.Tensor('float32', data, [1, 3, height, width]) });
    const flat = {};
    for (const [k, t] of Object.entries(out)) {
      flat[k] = t.data;
      t.dispose?.();
    }
    return flat;
  }

  async ageGender(tensor) {
    const s = this.agSize;
    const out = await this.ag.run({ input: new ort.Tensor('float32', tensor, [1, 3, s, s]) });
    const res = { age: out.age.data[0], pFemale: out.p_female.data[0] };
    Object.values(out).forEach((t) => t.dispose?.());
    return res;
  }

  async expression(tensor) {
    const s = this.exSize;
    const out = await this.ex.run({ input: new ort.Tensor('float32', tensor, [1, 3, s, s]) });
    const probs = Float32Array.from(out.probs.data);
    Object.values(out).forEach((t) => t.dispose?.());
    return probs;
  }

  async release() {
    for (const s of [this.det, this.ag, this.ex]) {
      try {
        await s?.release();
      } catch {
        /* already released */
      }
    }
  }
}

/** Single-model session used by the benchmark. */
export async function benchSession(file, backend) {
  return createSession(file, backend);
}

export function makeTensor(data, dims) {
  return new ort.Tensor('float32', data, dims);
}
