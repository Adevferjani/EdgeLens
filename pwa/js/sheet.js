// Detail sheet: every probability for one face, ordered (opened from a frozen
// Live frame or from a face in Photo mode). The live overlay only shows the
// top result per attribute, to stay readable.

import { h } from './util.js';

// Typical age error of the deployed model on the UTKFace test split
// (ONNX FP32 MAE 5.08 years, docs/results/evaluation-table.md).
export const AGE_TYPICAL_ERROR = 5;

const EXPR_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'];
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const pct1 = (p) => (p < 0.001 ? '<0.1 %' : `${(p * 100).toFixed(1)} %`);

export function ageRange(age) {
  return [Math.max(0, Math.round(age - AGE_TYPICAL_ERROR)), Math.round(age + AGE_TYPICAL_ERROR)];
}

export function ageGroupLabel(age, elderlyMin = 60) {
  return age < 18 ? 'Under 18' : age >= elderlyMin ? `${elderlyMin}+` : `Adult (18–${elderlyMin - 1})`;
}

export class FaceSheet {
  constructor({ root, backdrop }) {
    this.root = root;
    this.backdrop = backdrop;
    this.backdrop.addEventListener('click', () => this.close());
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && !this.root.hidden) this.close();
    });
  }

  get isOpen() {
    return !this.root.hidden;
  }

  /**
   * faces: [{age, pFemale, probs, score}], labels: expression labels,
   * actionLabel: text of the closing button, meta(face) -> small text in the header.
   */
  open({ faces, index = 0, labels = EXPR_LABELS, elderlyMin = 60, actionLabel = 'Close', meta, onClose }) {
    this.faces = faces;
    this.index = Math.min(index, faces.length - 1);
    this.opts = { labels, elderlyMin, actionLabel, meta, onClose };
    this.returnFocus = document.activeElement;
    this.render();
    this.backdrop.hidden = false;
    this.root.hidden = false;
    this.root.querySelector('h2')?.focus();
  }

  close() {
    if (this.root.hidden) return;
    this.root.hidden = true;
    this.backdrop.hidden = true;
    this.root.replaceChildren();
    const cb = this.opts?.onClose;
    this.opts = null;
    this.returnFocus?.focus?.();
    cb?.();
  }

  render() {
    const f = this.faces[this.index];
    const { labels, elderlyMin, actionLabel, meta } = this.opts;
    const [lo, hi] = ageRange(f.age);
    const MAX = 90;
    const pos = (v) => `${(Math.min(MAX, Math.max(0, v)) / MAX) * 100}%`;

    const genders = [['female', f.pFemale], ['male', 1 - f.pFemale]].sort((a, b) => b[1] - a[1]);
    const exprs = Array.from(f.probs, (p, i) => [labels[i] || EXPR_LABELS[i], p]).sort((a, b) => b[1] - a[1]);
    const probRows = (rows) => rows.flatMap(([name, p], i) => {
      const top = i === 0 ? ' top' : '';
      return [
        h('span', { class: `lbl${top}` }, name.toUpperCase()),
        h('span', { class: 'bar', 'aria-hidden': 'true' }, h('i', { class: top.trim(), style: `width:${(p * 100).toFixed(1)}%` })),
        h('span', { class: `val${top}` }, pct1(p)),
      ];
    });

    const tabs = this.faces.length > 1
      ? h('div', { class: 'face-tabs', role: 'group', 'aria-label': 'Face' }, this.faces.map((_, i) => h('button', {
        type: 'button',
        'aria-pressed': String(i === this.index),
        onclick: () => {
          this.index = i;
          this.render();
        },
      }, String(i + 1).padStart(2, '0'))))
      : null;

    this.root.replaceChildren(...[
      h('div', { class: 'grab', 'aria-hidden': 'true' }),
      h('div', { class: 'sheet-head' },
        h('h2', { id: 'sheetTitle', tabindex: '-1' }, `Face ${String(this.index + 1).padStart(2, '0')}`),
        h('span', { class: 'mono' }, meta ? meta(f) : f.score != null ? `DETECTOR ${f.score.toFixed(2)}` : '')),
      tabs,
      h('div', { class: 'sheet-block' },
        h('span', { class: 'k' }, 'AGE'),
        h('div', { class: 'age-row' },
          h('div', { class: 'age-big' }, h('b', {}, String(Math.round(f.age))), h('span', {}, 'years')),
          h('div', { class: 'age-side' }, `Likely ${lo}–${hi}`, h('br'), ageGroupLabel(f.age, elderlyMin))),
        h('div', { class: 'scale', 'aria-hidden': 'true' },
          h('span', { class: 'band', style: `left:${pos(lo)};width:calc(${pos(hi)} - ${pos(lo)})` }),
          h('span', { class: 'tick', style: `left:${pos(18)}` }),
          h('span', { class: 'tick', style: `left:${pos(elderlyMin)}` }),
          h('span', { class: 'mark-line', style: `left:${pos(f.age)}` })),
        h('div', { class: 'scale-labels', 'aria-hidden': 'true' },
          h('span', { style: 'left:0' }, '0'),
          h('span', { style: `left:${pos(18)}` }, '18'),
          h('span', { style: `left:${pos(elderlyMin)}` }, String(elderlyMin)),
          h('span', { style: 'left:100%' }, String(MAX)))),
      h('div', { class: 'sheet-block' },
        h('span', { class: 'k' }, 'GENDER'),
        h('div', { class: 'probs' }, probRows(genders))),
      h('div', { class: 'sheet-block' },
        h('span', { class: 'k' }, 'EXPRESSION · 7 CLASSES'),
        h('div', { class: 'probs' }, probRows(exprs))),
      h('p', { class: 'fine' }, `Model outputs, not certainties. Typical age error on the test set: ±${AGE_TYPICAL_ERROR} years. Top result: ${cap(exprs[0][0])}.`),
      h('button', { class: 'btn primary wide', type: 'button', onclick: () => this.close() }, actionLabel),
    ].filter(Boolean));
  }
}
