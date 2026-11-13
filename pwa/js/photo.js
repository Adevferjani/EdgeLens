// Photo mode: analyse one photo (library or camera) for up to 10 faces.
// The photo is kept in memory only; nothing is stored or uploaded.

import { crumb } from './diagnostics.js';
import { decode } from './image.js';
import { analyze, createCanvases, freeCanvases } from './pipeline.js';
import { ageGroupLabel, ageRange } from './sheet.js';
import { $, h } from './util.js';

export const PHOTO_MAX_FACES = 10;
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const pct0 = (p) => `${Math.round(p * 100)} %`;
const signal = () => getComputedStyle(document.documentElement).getPropertyValue('--signal').trim() || '#cff24a';

export class PhotoView {
  constructor({ getEngine, getSettings, onError, onInspect }) {
    Object.assign(this, { getEngine, getSettings, onError, onInspect });
    this.img = document.createElement('canvas');
    this.faces = [];
    for (const id of ['#photoPick', '#photoTake']) {
      const input = $(id);
      input.addEventListener('change', () => {
        const file = input.files[0];
        input.value = '';
        if (file) this.analyse(file);
      });
    }
  }

  async analyse(file) {
    const meta = $('#photoMeta');
    meta.hidden = false;
    meta.replaceChildren(h('span', {}, 'ANALYSING…'));
    let engine;
    try {
      engine = await this.getEngine();
    } catch (e) {
      meta.hidden = true;
      this.onError(e);
      return;
    }
    const canvases = createCanvases();
    try {
      crumb('photo:start', { mb: +(file.size / 1e6).toFixed(1), engine: engine.label });
      const img = await decode(file, this.img);
      const res = await analyze(engine, img, img.width, img.height, canvases, {
        detectorMaxSide: 640,
        scoreThreshold: this.getSettings().scoreThreshold,
        maxFaces: PHOTO_MAX_FACES,
      });
      // number the faces from left to right
      this.faces = res.faces.sort((a, b) => a.box.x - b.box.x);
      this.ms = res.timings.total;
      this.elderlyMin = engine.cards.age_gender.elderly_min_age ?? 60;
      this.labels = engine.cards.expression.labels;
      crumb('photo:done', { faces: this.faces.length, px: `${img.width}x${img.height}` });
      this.render();
    } catch (e) {
      meta.hidden = true;
      this.onError(e);
    } finally {
      freeCanvases(canvases);
    }
  }

  render() {
    const cv = $('#photoCanvas');
    const img = this.img;
    cv.width = img.width;
    cv.height = img.height;
    const ctx = cv.getContext('2d');
    ctx.drawImage(img, 0, 0);
    const lw = Math.max(2, img.width / 220);
    const color = signal();
    this.faces.forEach((f, i) => {
      const { x, y, w, h: hh } = f.box;
      const len = Math.min(w, hh) * 0.24;
      ctx.strokeStyle = color;
      ctx.lineWidth = lw;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(x, y + len); ctx.lineTo(x, y); ctx.lineTo(x + len, y);
      ctx.moveTo(x + w - len, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + len);
      ctx.moveTo(x + w, y + hh - len); ctx.lineTo(x + w, y + hh); ctx.lineTo(x + w - len, y + hh);
      ctx.moveTo(x + len, y + hh); ctx.lineTo(x, y + hh); ctx.lineTo(x, y + hh - len);
      ctx.stroke();
      const fs = Math.max(11, img.width / 60);
      ctx.font = `500 ${fs}px "Martian Mono", ui-monospace, monospace`;
      const label = String(i + 1);
      const tw = ctx.measureText(label).width + fs * 0.8;
      ctx.fillStyle = color;
      ctx.fillRect(x, y - fs * 1.5, tw, fs * 1.4);
      ctx.fillStyle = '#0b0e0f';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, x + fs * 0.4, y - fs * 0.8);
    });
    cv.hidden = false;
    $('#photoEmpty').hidden = true;

    const n = this.faces.length;
    $('#photoMeta').replaceChildren(
      h('span', {}, n ? `${n} FACE${n > 1 ? 'S' : ''}${n === PHOTO_MAX_FACES ? ' (MAX)' : ''}` : 'NO FACE FOUND'),
      h('span', {}, `${Math.round(this.ms)} MS · ON DEVICE`),
    );
    const list = $('#faceList');
    if (!n) {
      list.replaceChildren(h('li', { class: 'empty' }, 'No face was found. Try a photo where faces are larger, sharper or more frontal.'));
      return;
    }
    list.replaceChildren(...this.faces.map((f, i) => {
      const [lo, hi] = ageRange(f.age);
      const group = f.age < 18 || f.age >= this.elderlyMin ? ` · ${ageGroupLabel(f.age, this.elderlyMin)}` : '';
      return h('li', {},
        h('button', { type: 'button', class: 'face-row', onclick: () => this.inspect(i), 'aria-label': `Face ${i + 1}: all probabilities` },
          h('span', { class: 'idx' }, String(i + 1).padStart(2, '0')),
          h('span', { class: 'metric-grid' },
            h('span', { class: 'metric' }, h('span', { class: 'k' }, 'AGE'), h('span', { class: 'v num' }, String(Math.round(f.age))), h('span', { class: 's' }, `${lo}–${hi}${group}`)),
            h('span', { class: 'metric' }, h('span', { class: 'k' }, 'GENDER'), h('span', { class: 'v' }, cap(f.gender)), h('span', { class: 's' }, pct0(f.genderConf))),
            h('span', { class: 'metric' }, h('span', { class: 'k' }, 'EXPRESSION'), h('span', { class: 'v' }, cap(f.expression)), h('span', { class: 's' }, pct0(f.expressionConf)))),
          h('span', { class: 'chev', 'aria-hidden': 'true' }, '›')));
    }));
  }

  inspect(index) {
    this.onInspect({
      faces: this.faces.map((f) => ({ age: f.age, pFemale: f.pFemale, probs: f.probs, score: f.score })),
      index,
      labels: this.labels,
      elderlyMin: this.elderlyMin,
      actionLabel: 'Close',
    });
  }
}
