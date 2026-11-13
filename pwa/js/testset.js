// On-device accuracy on the UTKFace / FER2013 test sets (file made on the PC with
// training/tools/export_testset.py). The images are decoded by the browser, resized
// exactly like Pillow does on the PC, and run through the deployed models. The
// manifest holds the PC predictions of the same model files for every image, so the
// app reports both the accuracy and the per-image agreement with the PC.

import { unzipSync } from '../vendor/fflate.mjs';
import { crumb } from './diagnostics.js';
import { Engine, modelUrl } from './inference.js';
import { accuracy, EXPRESSIONS, macroF1, perClassF1, recall, round } from './metrics.js';
import { resizeRGBA, toCHW } from './resize.js';
import { stats } from './util.js';

const HAPPY = EXPRESSIONS.indexOf('happy');
const SAD = EXPRESSIONS.indexOf('sad');
const yieldUi = () => new Promise((r) => setTimeout(r, 0));

export async function openTestset(file) {
  let files;
  try {
    files = unzipSync(new Uint8Array(await file.arrayBuffer()));
  } catch (e) {
    throw new Error(`${file.name} is not a readable ZIP file (${e.message}).`);
  }
  if (!files['manifest.json']) throw new Error(`${file.name} has no manifest.json. Create the file with training/tools/export_testset.py.`);
  const manifest = JSON.parse(new TextDecoder().decode(files['manifest.json']));
  if (manifest.format !== 'agescope-testset-1') throw new Error(`Unknown test-set format "${manifest.format}".`);
  return { manifest, files, name: file.name, mb: file.size / 1e6 };
}

async function sha256(url) {
  const buf = await (await fetch(url)).arrayBuffer();
  const d = await crypto.subtle.digest('SHA-256', buf);
  return [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

// ------------------------------------------------------------------ metrics (as training/metrics.py)
export function ageGenderMetrics(ageTrue, agePred, genderTrue, pFemale, elderlyMin = 60) {
  const n = ageTrue.length;
  const err = ageTrue.map((a, i) => Math.abs(agePred[i] - a));
  const mean = (xs) => (xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : null);
  const gPred = pFemale.map((p) => (p >= 0.5 ? 1 : 0));
  const adult = ageTrue.map((a, i) => (a >= 18 ? i : -1)).filter((i) => i >= 0);
  const grp = (a) => (a >= elderlyMin ? 1 : 0);
  const gt = adult.map((i) => grp(ageTrue[i]));
  const gp = adult.map((i) => grp(agePred[i]));
  const recAdult = recall(gt, gp, 0);
  const recElderly = recall(gt, gp, 1);
  const rec = [recAdult, recElderly].filter((v) => v != null);
  return {
    n,
    age_mae: round(mean(err)),
    gender_accuracy: round(accuracy(genderTrue, gPred)),
    gender_macro_f1: round(macroF1(genderTrue, gPred, [0, 1], { allLabels: true })),
    n_18plus: adult.length,
    age_mae_18plus: round(mean(adult.map((i) => err[i]))),
    age_group_accuracy_18plus: round(accuracy(gt, gp)),
    adult_recall_18plus: round(recAdult),
    elderly_recall_18plus: round(recElderly),
    age_group_balanced_accuracy_18plus: round(rec.length ? rec.reduce((a, b) => a + b, 0) / rec.length : null),
    age_group_macro_f1_18plus: round(macroF1(gt, gp, [0, 1], { allLabels: true })),
    age_mae_elderly: round(mean(err.filter((_, i) => ageTrue[i] >= elderlyMin))),
  };
}

const argmax = (p) => {
  let t = 0;
  for (let i = 1; i < p.length; i++) if (p[i] > p[t]) t = i;
  return t;
};

export function expressionMetrics(yTrue, probs) {
  const labels = EXPRESSIONS.map((_, i) => i);
  const yPred = probs.map(argmax);
  const hs = yTrue.map((y, i) => (y === HAPPY || y === SAD ? i : -1)).filter((i) => i >= 0);
  const twoWay = (i) => (probs[i][HAPPY] >= probs[i][SAD] ? HAPPY : SAD);
  const pc = perClassF1(yTrue, yPred, labels);
  return {
    n: yTrue.length,
    accuracy: round(accuracy(yTrue, yPred)),
    macro_f1: round(macroF1(yTrue, yPred, labels, { allLabels: true })),
    per_class_f1: Object.fromEntries(EXPRESSIONS.map((c, i) => [c, round(pc[i] ?? 0)])),
    happy_sad_n: hs.length,
    happy_sad_accuracy_7way: hs.length ? round(hs.filter((i) => yPred[i] === yTrue[i]).length / hs.length) : null,
    happy_sad_accuracy_2way: hs.length ? round(hs.filter((i) => twoWay(i) === yTrue[i]).length / hs.length) : null,
  };
}

function agreement(dev, pc, task) {
  if (!pc) return null;
  const n = task === 'age_gender' ? dev.age.length : dev.probs.length;
  if (task === 'age_gender') {
    let sum = 0;
    let max = 0;
    let same = 0;
    let maxP = 0;
    for (let i = 0; i < n; i++) {
      const d = Math.abs(dev.age[i] - pc.age[i]);
      sum += d;
      max = Math.max(max, d);
      if ((dev.pFemale[i] >= 0.5) === (pc.p_female[i] >= 0.5)) same++;
      maxP = Math.max(maxP, Math.abs(dev.pFemale[i] - pc.p_female[i]));
    }
    return { n, age_mean_abs_diff: round(sum / n, 5), age_max_abs_diff: round(max, 5), gender_agreement: round(same / n), p_female_max_abs_diff: round(maxP, 6) };
  }
  let same = 0;
  let maxP = 0;
  for (let i = 0; i < n; i++) {
    if (argmax(dev.probs[i]) === argmax(pc.probs[i])) same++;
    for (let c = 0; c < dev.probs[i].length; c++) maxP = Math.max(maxP, Math.abs(dev.probs[i][c] - pc.probs[i][c]));
  }
  return { n, top1_agreement: round(same / n), prob_max_abs_diff: round(maxP, 6) };
}

// ------------------------------------------------------------------ run
function makeDecoder() {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  return {
    async pixels(bytes, type) {
      const bmp = await createImageBitmap(new Blob([bytes], { type }), { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
      if (canvas.width !== bmp.width || canvas.height !== bmp.height) {
        canvas.width = bmp.width;
        canvas.height = bmp.height;
      } else {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
      ctx.drawImage(bmp, 0, 0);
      bmp.close?.();
      return { data: ctx.getImageData(0, 0, canvas.width, canvas.height).data, w: canvas.width, h: canvas.height };
    },
    free() {
      canvas.width = 0;
      canvas.height = 0;
    },
  };
}

/**
 * Runs every selected precision on the test images. onProgress(text, fraction);
 * shouldStop() is polled between images. Returns {runs: [...], stopped}.
 */
export async function runTestset({ testset, cards, backend, precisions, onProgress, shouldStop = () => false }) {
  const { manifest: m, files } = testset;
  const ag = m.age_gender;
  const ex = m.expression;
  const total = precisions.length * (ag.n + ex.n);
  let done = 0;
  const runs = [];
  const decoder = makeDecoder();
  const tensors = { ag: new Float32Array(3 * ag.img_size ** 2), ex: new Float32Array(3 * ex.img_size ** 2) };
  let stopped = false;

  try {
    for (const precision of precisions) {
      const label = `${backend === 'webgpu' ? 'WebGPU' : 'WASM'} ${precision.toUpperCase()}`;
      onProgress(`Loading ${label}…`, done / total);
      crumb('testset:start', { engine: label, n: ag.n + ex.n });
      const hash = {
        age_gender: m.models?.age_gender?.[precision]?.sha256 === (await sha256(modelUrl(cards.age_gender.files[precision]))),
        expression: m.models?.expression?.[precision]?.sha256 === (await sha256(modelUrl(cards.expression.files[precision]))),
      };
      const engine = await Engine.create({ backend, precision, cards });
      const run = { backend, precision, label, hashMatch: hash, times: { decode: [], age_gender: [], expression: [] } };
      try {
        await engine.warmup();
        if (engine.agSize !== ag.img_size || engine.exSize !== ex.img_size) {
          throw new Error(`The test-set file was made for input sizes ${ag.img_size}/${ex.img_size}, the deployed models use ${engine.agSize}/${engine.exSize}.`);
        }

        // ---- age + gender (UTKFace) ----
        const dev = { age: [], pFemale: [] };
        for (const [i, item] of ag.items.entries()) {
          if (shouldStop()) throw Object.assign(new Error('stopped'), { stopped: true });
          let t = performance.now();
          const px = await decoder.pixels(files[item.file], 'image/jpeg');
          toCHW(resizeRGBA(px.data, px.w, px.h, ag.img_size, ag.img_size), ag.img_size, ag.img_size, tensors.ag);
          run.times.decode.push(performance.now() - t);
          t = performance.now();
          const out = await engine.ageGender(tensors.ag);
          run.times.age_gender.push(performance.now() - t);
          dev.age.push(out.age);
          dev.pFemale.push(out.pFemale);
          if (++done % 25 === 0) {
            onProgress(`${label}: UTKFace image ${i + 1} of ${ag.n}`, done / total);
            await yieldUi();
          }
        }

        // ---- expression (FER2013) ----
        const devEx = { probs: [] };
        for (const [i, item] of ex.items.entries()) {
          if (shouldStop()) throw Object.assign(new Error('stopped'), { stopped: true });
          let t = performance.now();
          const px = await decoder.pixels(files[item.file], 'image/png');
          toCHW(resizeRGBA(px.data, px.w, px.h, ex.img_size, ex.img_size), ex.img_size, ex.img_size, tensors.ex);
          run.times.decode.push(performance.now() - t);
          t = performance.now();
          devEx.probs.push(Array.from(await engine.expression(tensors.ex)));
          run.times.expression.push(performance.now() - t);
          if (++done % 25 === 0) {
            onProgress(`${label}: FER2013 image ${i + 1} of ${ex.n}`, done / total);
            await yieldUi();
          }
        }

        run.metrics = {
          age_gender: ag.n ? ageGenderMetrics(ag.items.map((r) => r.age), dev.age, ag.items.map((r) => r.gender), dev.pFemale, ag.elderly_min_age) : null,
          expression: ex.n ? expressionMetrics(ex.items.map((r) => r.label), devEx.probs) : null,
        };
        run.agreement = {
          age_gender: hash.age_gender && ag.n ? agreement(dev, ag.pc?.[precision], 'age_gender') : null,
          expression: hash.expression && ex.n ? agreement(devEx, ex.pc?.[precision], 'expression') : null,
        };
        run.latency = Object.fromEntries(Object.entries(run.times).map(([k, v]) => {
          const s = stats(v);
          return [k, v.length ? { mean_ms: +s.mean.toFixed(3), p50_ms: +s.p50.toFixed(3), p95_ms: +s.p95.toFixed(3) } : null];
        }));
        run.predictions = {
          age_gender: { age: dev.age.map((a) => +a.toFixed(3)), p_female: dev.pFemale.map((p) => +p.toFixed(5)) },
          expression: { top: devEx.probs.map(argmax), probs: devEx.probs.map((p) => p.map((v) => +v.toFixed(5))) },
        };
        delete run.times;
        runs.push(run);
        crumb('testset:done', { engine: label });
      } finally {
        await engine.release();
      }
    }
  } catch (e) {
    if (!e.stopped) throw e;
    stopped = true;
  } finally {
    decoder.free();
  }
  onProgress(stopped ? 'Stopped.' : `Finished ${new Date().toLocaleTimeString()}.`, stopped ? done / total : 1);
  return { runs, stopped };
}
