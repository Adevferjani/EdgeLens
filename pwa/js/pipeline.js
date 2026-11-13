// Full on-device pipeline for one image/frame:
// detect faces -> align + crop -> age/gender model + expression model.

import { decodeDetections, prepareDetectorInput } from './detector.js';
import { cropFace } from './facecrop.js';

export const EXPRESSIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'];
const HAPPY = 3;
const SAD = 4;

export function createCanvases() {
  const make = () => document.createElement('canvas');
  return { det: make(), ag: make(), ex: make() };
}

/** Give canvas memory back immediately (Safari frees it lazily otherwise). */
export function freeCanvases(canvases) {
  for (const c of Object.values(canvases)) {
    c.width = 0;
    c.height = 0;
    c._buf = null;
  }
}

// Raw model outputs -> displayed results: age group from the threshold in the
// model card, gender from P(female) >= 0.5, the most likely expression, and
// happy vs. sad as a forced choice between these two probabilities only
// (confidence renormalised over the pair).
export function interpret(raw, cards) {
  const elderlyMin = cards.age_gender.elderly_min_age ?? 60;
  const probs = raw.probs;
  let top = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[top]) top = i;
  const labels = cards.expression.labels || EXPRESSIONS;
  return {
    age: raw.age,
    ageGroup: raw.age >= elderlyMin ? 'elderly' : 'adult',
    ageBand: raw.age < 18 ? 'under 18' : raw.age >= elderlyMin ? 'elderly' : 'adult',
    pFemale: raw.pFemale,
    gender: raw.pFemale >= 0.5 ? 'female' : 'male',
    genderConf: Math.max(raw.pFemale, 1 - raw.pFemale),
    probs,
    expression: labels[top],
    expressionConf: probs[top],
    happySad: probs[HAPPY] >= probs[SAD] ? 'happy' : 'sad',
    happySadConf: Math.max(probs[HAPPY], probs[SAD]) / Math.max(1e-6, probs[HAPPY] + probs[SAD]),
  };
}

/**
 * Analyse `source` (canvas, video frame canvas or ImageBitmap) of size srcW x srcH.
 * Returns {faces: [{box, landmarks, score, raw, ...interpretation}], timings}.
 */
export async function analyze(engine, source, srcW, srcH, canvases, opts = {}) {
  const { detectorMaxSide = 320, scoreThreshold = 0.6, maxFaces = 3, withCrops = false } = opts;
  const timings = { preprocess: 0, detect: 0, crop: 0, ageGender: 0, expression: 0, total: 0 };
  const t0 = performance.now();
  const inp = prepareDetectorInput(source, srcW, srcH, detectorMaxSide, canvases.det);
  const t1 = performance.now();
  const out = await engine.detect(inp.data, inp.height, inp.width);
  let dets = decodeDetections(out, inp.width, inp.height, { scoreThreshold, scale: inp.scale });
  dets.sort((a, b) => b.w * b.h - a.w * a.h);
  dets = dets.slice(0, maxFaces);
  const t2 = performance.now();
  timings.preprocess = t1 - t0;
  timings.detect = t2 - t1;

  const faces = [];
  for (const d of dets) {
    let t = performance.now();
    const agT = cropFace(source, d, engine.cards.age_gender.crop, engine.agSize, canvases.ag);
    const exT = cropFace(source, d, engine.cards.expression.crop, engine.exSize, canvases.ex);
    const crops = withCrops ? { ag: canvases.ag.toDataURL('image/jpeg', 0.85), ex: canvases.ex.toDataURL('image/jpeg', 0.85) } : null;
    timings.crop += performance.now() - t;

    t = performance.now();
    const ag = await engine.ageGender(agT);
    timings.ageGender += performance.now() - t;

    t = performance.now();
    const probs = await engine.expression(exT);
    timings.expression += performance.now() - t;

    const raw = { age: ag.age, pFemale: ag.pFemale, probs };
    faces.push({
      box: { x: d.x, y: d.y, w: d.w, h: d.h },
      landmarks: d.landmarks,
      score: d.score,
      raw,
      crops,
      ...interpret(raw, engine.cards),
    });
  }
  timings.total = performance.now() - t0;
  return { faces, timings, detectorInput: [inp.width, inp.height] };
}
