// Lab → Evaluate (developer mode): labelled test photos evaluated on the device
// (Table 2 of the evaluation).
// Ground truth per photo: age group (under 18 / adult / 60+) with optional exact
// age, gender, and one of the 7 FER2013 expressions. Labels come from the file
// name (e.g. "elderly_female_happy_age67.jpg") or are set on each photo.
// Everything stays on the device; exports contain anonymous IDs only.

import { crumb } from './diagnostics.js';
import {
  accuracy, AGE_GROUP_LABEL, AGE_GROUPS, ageGroupOf, EXPRESSIONS, mae, macroF1, round,
} from './metrics.js';
import { decode } from './image.js';
import { analyze, createCanvases, freeCanvases } from './pipeline.js';
import { $, download, h, pct, sleep, storageGet, storageSet, timestamp, toCsv } from './util.js';

const KEY = 'agescope.photos.v1';
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const HAPPY = 3;
const SAD = 4;

export function truthFromName(name) {
  const base = name.toLowerCase().replace(/\.[a-z0-9]+$/, '');
  const t = base.split(/[^a-z0-9]+/).filter(Boolean);
  const has = (...w) => w.some((x) => t.includes(x));
  const ageTok = base.match(/age[-_ ]?(\d{1,3})/);
  const age = ageTok ? Number(ageTok[1]) : null;
  let group = has('elderly', 'old', 'senior', 'older') ? 'elderly'
    : has('child', 'kid', 'teen', 'minor', 'under18') ? 'under18'
      : has('adult', 'young', 'younger') ? 'adult' : null;
  if (age != null) group = ageGroupOf(age);
  const exprAlias = { smile: 'happy', smiling: 'happy', unhappy: 'sad', surprised: 'surprise', angry: 'angry', anger: 'angry', disgusted: 'disgust', fearful: 'fear', scared: 'fear' };
  let expr = EXPRESSIONS.find((e) => t.includes(e)) || null;
  if (!expr) for (const [k, v] of Object.entries(exprAlias)) if (t.includes(k)) expr = v;
  return {
    age,
    group,
    gender: has('female', 'woman', 'f', 'w') ? 'female' : has('male', 'man', 'm') ? 'male' : null,
    expr,
  };
}

// Predicted values derived from the stored raw outputs (also works for records
// saved by earlier app versions).
function pred(rec) {
  if (!rec.faceFound) return null;
  const p = rec.pred;
  const probs = p.probs;
  let top = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[top]) top = i;
  const hs = probs[HAPPY] + probs[SAD];
  return {
    age: p.age,
    group: ageGroupOf(p.age),
    gender: p.pFemale >= 0.5 ? 'female' : 'male',
    genderConf: Math.max(p.pFemale, 1 - p.pFemale),
    expr: EXPRESSIONS[top],
    exprConf: probs[top],
    happySad: probs[HAPPY] >= probs[SAD] ? 'happy' : 'sad',
    happySadConf: Math.max(probs[HAPPY], probs[SAD]) / Math.max(1e-6, hs),
  };
}

function verdicts(rec) {
  const p = pred(rec);
  const t = rec.truth;
  const v = (truth, predicted) => (truth == null ? null : p ? predicted === truth : false);
  return {
    group: v(t.group, p?.group),
    gender: v(t.gender, p?.gender),
    expr: v(t.expr, p?.expr),
    happySad: t.expr === 'happy' || t.expr === 'sad' ? v(t.expr, p?.happySad) : null,
  };
}

export class EvaluationView {
  constructor({ root, getEngine, getSettings, onError, toast }) {
    Object.assign(this, { root, getEngine, getSettings, onError, toast });
    this.records = storageGet(KEY, []).map((r) => ({ ...r, truth: { age: null, group: null, gender: null, expr: null, ...r.truth } }));
    this.input = $('#photoInput', root);
    this.input.addEventListener('change', () => this.addFiles([...this.input.files]));
    $('#exportCsv', root).addEventListener('click', () => this.exportCsv());
    $('#exportJson', root).addEventListener('click', () => this.exportJson());
    $('#clearPhotos', root).addEventListener('click', () => this.clear());
    this.render();
  }

  save() {
    storageSet(KEY, this.records);
  }

  nextId() {
    const n = this.records.reduce((m, r) => Math.max(m, Number(r.id.slice(1))), 0) + 1;
    return `P${String(n).padStart(2, '0')}`;
  }

  async addFiles(files) {
    this.input.value = '';
    if (!files.length) return;
    const status = $('#photoStatus', this.root);
    let engine;
    try {
      status.textContent = 'Loading models…';
      engine = await this.getEngine();
    } catch (e) {
      status.textContent = '';
      this.onError(e);
      return;
    }
    const canvases = { ...createCanvases(), img: document.createElement('canvas') };
    for (const [i, file] of files.entries()) {
      status.textContent = `Analysing photo ${i + 1} of ${files.length}…`;
      crumb('photo:start', { n: i + 1, of: files.length, mb: +(file.size / 1e6).toFixed(1), engine: engine.label });
      try {
        const img = await decode(file, canvases.img);
        const res = await analyze(engine, img, img.width, img.height, canvases, {
          detectorMaxSide: 640,
          scoreThreshold: this.getSettings().scoreThreshold,
          maxFaces: 1,
          withCrops: true,
        });
        const f = res.faces[0];
        this.records.push({
          id: this.nextId(),
          truth: truthFromName(file.name),
          faceFound: !!f,
          faceScore: f ? +f.score.toFixed(3) : null,
          pred: f
            ? { age: +f.age.toFixed(1), pFemale: +f.pFemale.toFixed(4), probs: Array.from(f.probs, (p) => +p.toFixed(4)) }
            : null,
          thumb: f ? f.crops.ag : null,
          latencyMs: +res.timings.total.toFixed(1),
          engine: engine.label,
        });
        this.save();
        this.render();
        crumb('photo:done', { n: i + 1, px: `${img.width}x${img.height}` });
      } catch (e) {
        this.onError(e);
      }
      await sleep(30); // let the browser release memory between photos
    }
    freeCanvases(canvases);
    status.textContent = '';
    this.toast(`${files.length} photo${files.length > 1 ? 's' : ''} analysed on this device.`);
  }

  clear() {
    if (!this.records.length) return;
    if (!confirm('Remove all photos and results from this device?')) return;
    this.records = [];
    this.save();
    this.render();
  }

  setTruth(id, key, value) {
    const rec = this.records.find((r) => r.id === id);
    if (key === 'age') {
      rec.truth.age = value === '' || value == null ? null : Math.max(0, Math.min(120, Number(value)));
      if (rec.truth.age != null) rec.truth.group = ageGroupOf(rec.truth.age);
    } else {
      rec.truth[key] = rec.truth[key] === value ? null : value;
      if (key === 'group' && rec.truth.age != null && ageGroupOf(rec.truth.age) !== rec.truth.group) rec.truth.age = null;
    }
    this.save();
    this.render();
  }

  remove(id) {
    this.records = this.records.filter((r) => r.id !== id);
    this.save();
    this.render();
  }

  // ---------------------------------------------------------------- metrics
  metrics(records) {
    const pairs = (key, predKey = key, filter = () => true) => {
      const rs = records.filter((r) => r.truth[key] != null && filter(r));
      return { truth: rs.map((r) => r.truth[key]), pred: rs.map((r) => pred(r)?.[predKey] ?? 'none') };
    };
    const g = pairs('group');
    const s = pairs('gender');
    const e = pairs('expr');
    const hs = pairs('expr', 'happySad', (r) => r.truth.expr === 'happy' || r.truth.expr === 'sad');
    const aged = records.filter((r) => r.truth.age != null && r.faceFound);
    const res = (p, labels) => ({ n: p.truth.length, acc: accuracy(p.truth, p.pred), f1: macroF1(p.truth, p.pred, labels) });
    return {
      n: records.length,
      faces: records.filter((r) => r.faceFound).length,
      group: res(g, AGE_GROUPS),
      gender: res(s, ['male', 'female']),
      expr: res(e, EXPRESSIONS),
      happySad: res(hs, ['happy', 'sad']),
      ageMae: { n: aged.length, mae: mae(aged.map((r) => r.truth.age), aged.map((r) => r.pred.age)) },
    };
  }

  subsets() {
    const R = this.records;
    return [
      ['All photos', R],
      ['Under 18', R.filter((r) => r.truth.group === 'under18')],
      ['Adult', R.filter((r) => r.truth.group === 'adult')],
      ['60+', R.filter((r) => r.truth.group === 'elderly')],
      ['Male', R.filter((r) => r.truth.gender === 'male')],
      ['Female', R.filter((r) => r.truth.gender === 'female')],
    ].filter(([, rs]) => rs.length);
  }

  composition() {
    const count = (key, v) => this.records.filter((r) => r.truth[key] === v).length;
    return {
      ageGroup: Object.fromEntries(AGE_GROUPS.map((g) => [g, count('group', g)])),
      gender: { male: count('gender', 'male'), female: count('gender', 'female') },
      expression: Object.fromEntries(EXPRESSIONS.map((x) => [x, count('expr', x)])),
      exactAgeGiven: this.records.filter((r) => r.truth.age != null).length,
    };
  }

  // ---------------------------------------------------------------- render
  render() {
    this.renderComposition();
    this.renderSummary();
    this.renderList();
    const has = this.records.length > 0;
    for (const id of ['#exportCsv', '#exportJson', '#clearPhotos']) $(id, this.root).disabled = !has;
    $('#photoCount', this.root).textContent = has ? `${this.records.length} PHOTO${this.records.length > 1 ? 'S' : ''}` : 'NO PHOTOS YET';
  }

  renderComposition() {
    if (!this.records.length) {
      $('#balance', this.root).replaceChildren();
      return;
    }
    const c = this.composition();
    const unlabelled = this.records.filter((r) => !r.truth.group || !r.truth.gender || !r.truth.expr).length;
    const row = (name, obj, labels) => h('tr', {}, h('th', { scope: 'row' }, name),
      h('td', { class: 'comp' }, Object.entries(obj).filter(([, n]) => n).map(([k, n]) => `${labels?.[k] ?? cap(k)} ${n}`).join(', ') || '–'));
    const nodes = [h('table', { class: 'grid composition-table' },
      h('caption', {}, 'Sample composition (true labels)'),
      h('tbody', {},
        row('Age group', c.ageGroup, AGE_GROUP_LABEL),
        row('Gender', c.gender),
        row('Expression', c.expression),
        h('tr', {}, h('th', { scope: 'row' }, 'Exact age'), h('td', { class: 'comp' }, `${c.exactAgeGiven} of ${this.records.length}`))))];
    if (unlabelled) nodes.push(h('p', { class: 'note' }, `${unlabelled} photo${unlabelled > 1 ? 's are' : ' is'} not fully labelled yet.`));
    $('#balance', this.root).replaceChildren(...nodes);
  }

  renderSummary() {
    const box = $('#summary', this.root);
    if (!this.records.length) {
      box.replaceChildren();
      return;
    }
    const cell = (m) => (m.n ? [pct(m.acc), h('small', {}, `F1 ${m.f1 == null ? '–' : m.f1.toFixed(2)} · n=${m.n}`)] : '–');
    box.replaceChildren(
      h('table', { class: 'grid summary-table' },
        h('caption', {}, 'Accuracy on this device'),
        h('thead', {}, h('tr', {},
          h('th', {}, ''), h('th', {}, 'Faces'),
          h('th', {}, 'Age group'), h('th', {}, 'Age MAE'),
          h('th', {}, 'Gender'),
          h('th', {}, 'Expression (7)'), h('th', {}, 'Happy/sad'))),
        h('tbody', {}, this.subsets().map(([name, rs]) => {
          const m = this.metrics(rs);
          return h('tr', {},
            h('th', { scope: 'row' }, name, h('small', {}, `n=${m.n}`)),
            h('td', {}, `${m.faces}/${m.n}`),
            h('td', {}, cell(m.group)),
            h('td', {}, m.ageMae.n ? [m.ageMae.mae.toFixed(1), h('small', {}, `years · n=${m.ageMae.n}`)] : '–'),
            h('td', {}, cell(m.gender)),
            h('td', {}, cell(m.expr)),
            h('td', {}, cell(m.happySad)));
        }))),
      h('p', { class: 'note' }, 'Expression (7): the top class out of 7 must match. Happy/sad: forced choice between happy and sad, only for photos labelled happy or sad. F1 = macro-F1. A photo without a detected face counts as wrong.'),
    );
  }

  renderList() {
    $('#photoList', this.root).replaceChildren(...this.records.map((r) => this.card(r)));
  }

  card(r) {
    const p = pred(r);
    const v = verdicts(r);
    const verdict = (ok) => (ok == null ? h('span', { class: 'verdict' }, '') : h('span', { class: `verdict ${ok ? 'ok' : 'bad'}` }, ok ? 'Correct' : 'Wrong'));
    const seg = (key, options, labels = {}) => h('div', { class: 'truth', role: 'group', 'aria-label': `True ${key} of ${r.id}` },
      options.map((o) => h('button', {
        type: 'button', class: 'seg', 'aria-pressed': String(r.truth[key] === o), onclick: () => this.setTruth(r.id, key, o),
      }, labels[o] ?? cap(o))));
    const ageInput = h('input', {
      type: 'number', min: '0', max: '120', inputmode: 'numeric', class: 'age-input', placeholder: 'age',
      'aria-label': `True age of ${r.id} (optional)`, value: r.truth.age ?? '',
      onchange: (ev) => this.setTruth(r.id, 'age', ev.target.value),
    });
    const exprSelect = h('select', { class: 'expr-select', 'aria-label': `True expression of ${r.id}`, onchange: (ev) => this.setTruth(r.id, 'expr', ev.target.value || null) },
      h('option', { value: '' }, 'True expression…'),
      EXPRESSIONS.map((x) => h('option', { value: x, selected: r.truth.expr === x }, cap(x))));
    const agePred = p ? [`${Math.round(p.age)} years`, `${AGE_GROUP_LABEL[p.group]}${r.truth.age != null ? ` · error ${Math.abs(p.age - r.truth.age).toFixed(0)} y` : ''}`] : ['No face found', ''];
    const exprPred = p
      ? [`${cap(p.expr)} ${pct(p.exprConf)}`, `happy/sad: ${p.happySad} ${pct(p.happySadConf)}`]
      : ['–', ''];

    return h('li', { class: 'photo-card' },
      h('div', { class: 'photo-thumb' },
        r.thumb ? h('img', { src: r.thumb, alt: `Face crop of ${r.id}` }) : h('div', { class: 'no-face' }, 'No face'),
        h('span', { class: 'photo-id' }, r.id)),
      h('div', { class: 'photo-rows' },
        h('div', { class: 'photo-row' },
          h('span', { class: 'k' }, 'AGE'),
          h('div', { class: 'pred' }, h('strong', {}, agePred[0]), h('span', {}, agePred[1])),
          h('div', { class: 'truth-wrap' }, seg('group', AGE_GROUPS, { under18: '<18', adult: 'Adult', elderly: '60+' }), ageInput),
          verdict(v.group)),
        h('div', { class: 'photo-row' },
          h('span', { class: 'k' }, 'GENDER'),
          h('div', { class: 'pred' }, h('strong', {}, p ? cap(p.gender) : '–'), h('span', {}, p ? pct(p.genderConf) : '')),
          seg('gender', ['male', 'female']),
          verdict(v.gender)),
        h('div', { class: 'photo-row' },
          h('span', { class: 'k' }, 'EXPRESSION'),
          h('div', { class: 'pred' }, h('strong', {}, exprPred[0]), h('span', {}, exprPred[1])),
          exprSelect,
          h('div', { class: 'verdict-stack' }, verdict(v.expr),
            v.happySad == null ? null : h('small', { class: v.happySad ? 'ok' : 'bad' }, `happy/sad ${v.happySad ? '✓' : '✗'}`)))),
      h('button', { type: 'button', class: 'icon-btn remove', 'aria-label': `Remove ${r.id}`, onclick: () => this.remove(r.id) }, '×'));
  }

  // ---------------------------------------------------------------- export
  rows() {
    return this.records.map((r) => {
      const p = pred(r);
      const v = verdicts(r);
      return {
        photo_id: r.id,
        true_age: r.truth.age ?? '',
        true_age_group: r.truth.group ?? '',
        pred_age: p ? +p.age.toFixed(1) : '',
        pred_age_group: p?.group ?? '',
        age_group_correct: v.group ?? '',
        abs_age_error: p && r.truth.age != null ? +Math.abs(p.age - r.truth.age).toFixed(1) : '',
        true_gender: r.truth.gender ?? '',
        p_female: r.pred?.pFemale ?? '',
        pred_gender: p?.gender ?? '',
        gender_correct: v.gender ?? '',
        true_expression: r.truth.expr ?? '',
        pred_expression: p?.expr ?? '',
        pred_expression_conf: p ? +p.exprConf.toFixed(4) : '',
        expression_correct: v.expr ?? '',
        pred_happy_vs_sad: p?.happySad ?? '',
        happy_vs_sad_correct: v.happySad ?? '',
        ...Object.fromEntries(EXPRESSIONS.map((x, i) => [`p_${x}`, r.pred?.probs[i] ?? ''])),
        face_found: r.faceFound,
        face_score: r.faceScore ?? '',
        latency_ms: r.latencyMs,
        engine: r.engine,
      };
    });
  }

  summaryJson() {
    return this.subsets().map(([name, rs]) => {
      const m = this.metrics(rs);
      const r = (x) => ({ n: x.n, accuracy: round(x.acc), macro_f1: round(x.f1) });
      return { subset: name, n: m.n, faces_found: m.faces, age_group: r(m.group), gender: r(m.gender),
        expression_7class: r(m.expr), happy_vs_sad: r(m.happySad), age_mae: { n: m.ageMae.n, mae: round(m.ageMae.mae, 2) } };
    });
  }

  exportCsv() {
    download(`edgelens-evaluation-${timestamp()}.csv`, toCsv(this.rows()), 'text/csv');
  }

  exportJson() {
    const data = {
      exported: new Date().toISOString(),
      userAgent: navigator.userAgent,
      composition: this.composition(),
      summary: this.summaryJson(),
      photos: this.rows(),
    };
    download(`edgelens-evaluation-${timestamp()}.json`, JSON.stringify(data, null, 2), 'application/json');
  }
}
