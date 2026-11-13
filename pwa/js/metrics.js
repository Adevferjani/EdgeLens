// Evaluation metrics shared by the photo evaluation and the test-set check.
// Definitions match training/metrics.py (scikit-learn): macro-F1 averages the
// per-class F1 over the classes that occur in the ground truth or predictions.

export const EXPRESSIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'];
export const AGE_GROUPS = ['under18', 'adult', 'elderly'];
export const AGE_GROUP_LABEL = { under18: 'Under 18', adult: 'Adult', elderly: '60+' };

export const ageGroupOf = (age, elderlyMin = 60) => (age < 18 ? 'under18' : age >= elderlyMin ? 'elderly' : 'adult');

export function accuracy(truth, pred) {
  if (!truth.length) return null;
  let ok = 0;
  for (let i = 0; i < truth.length; i++) if (truth[i] === pred[i]) ok++;
  return ok / truth.length;
}

export function perClassF1(truth, pred, labels) {
  const out = {};
  for (const c of labels) {
    let tp = 0;
    let fp = 0;
    let fn = 0;
    for (let i = 0; i < truth.length; i++) {
      if (pred[i] === c && truth[i] === c) tp++;
      else if (pred[i] === c) fp++;
      else if (truth[i] === c) fn++;
    }
    out[c] = tp + fp + fn === 0 ? null : (2 * tp) / (2 * tp + fp + fn);
  }
  return out;
}

/**
 * Macro-F1. Default: over the classes present in truth or predictions (scikit-learn
 * without `labels`). allLabels: over all `labels`, absent ones count as 0
 * (scikit-learn with labels=... and zero_division=0, as in training/metrics.py).
 */
export function macroF1(truth, pred, labels, { allLabels = false } = {}) {
  if (!truth.length) return null;
  const f = Object.values(perClassF1(truth, pred, labels))
    .map((v) => (v == null && allLabels ? 0 : v))
    .filter((v) => v != null);
  return f.length ? f.reduce((a, b) => a + b, 0) / f.length : null;
}

export function recall(truth, pred, cls) {
  const idx = truth.map((t, i) => (t === cls ? i : -1)).filter((i) => i >= 0);
  return idx.length ? idx.filter((i) => pred[i] === cls).length / idx.length : null;
}

export function mae(truth, pred) {
  if (!truth.length) return null;
  let s = 0;
  for (let i = 0; i < truth.length; i++) s += Math.abs(truth[i] - pred[i]);
  return s / truth.length;
}

export const round = (v, d = 4) => (v == null ? null : Math.round(v * 10 ** d) / 10 ** d);
