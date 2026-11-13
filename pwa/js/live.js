// Live camera mode: continuous detection + recognition with an overlay.
// The overlay shows the top result per attribute; tapping a face (or the
// shutter button) freezes the frame and opens all probabilities.

import { crumb } from './diagnostics.js';
import { runtimeInfo } from './inference.js';
import { analyze, createCanvases, freeCanvases, interpret } from './pipeline.js';
import { ageRange } from './sheet.js';
import { ms, nextFrame } from './util.js';

const MONO = '"Martian Mono", ui-monospace, monospace';
const C = { bone: '#e9ece6', muted: '#8c9691', ink: 'rgba(11,14,15,0.88)', rule: '#3a4347', track: '#2a3135' };
const signal = () => getComputedStyle(document.documentElement).getPropertyValue('--signal').trim() || '#cff24a';

// Phones fill the viewfinder (object-fit: cover). On a computer (mouse or
// trackpad) the viewfinder is much wider than the webcam picture, so cover
// would cut off the top and bottom; there the whole frame is shown instead.
const WHOLE_FRAME = window.matchMedia('(hover: hover) and (pointer: fine)');
const fitScale = (cw, ch, vw, vh) => (WHOLE_FRAME.matches ? Math.min(cw / vw, ch / vh) : Math.max(cw / vw, ch / vh));

function iou(a, b) {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w);
  const y2 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return inter / (a.w * a.h + b.w * b.h - inter);
}

export class LiveView {
  constructor({ video, display, overlay, telemetry, getEngine, getSettings, onState, onError, onStartPhase, onFaces, onInspect }) {
    Object.assign(this, { video, display, overlay, telemetry, getEngine, getSettings, onState, onError, onStartPhase, onFaces, onInspect });
    this.facing = 'user';
    this.running = false;
    this.paused = false;
    this.wanted = false; // the user started the camera: restart it automatically when Live is shown again
    this.away = false; // another view (Photo, Lab, Settings) is shown
    this.gen = 0; // incremented by stop(); a start() from an older generation gives up
    this.startingGen = -1;
    this.frame = document.createElement('canvas');
    this.canvases = createCanvases();
    this.tracks = [];
    this.hits = [];
    this.avg = null;
    this.fps = 0;
    this.lastDone = 0;
    this.color = signal();
    window.addEventListener('resize', () => {
      this.paintFrame();
      this.draw();
    });
    overlay.addEventListener('click', (e) => this.tap(e));
  }

  /**
   * Paint the current camera frame onto #display, filling it like
   * object-fit: cover, or contain on a computer (mirrored for the front camera). iPhone Safari sometimes
   * draws a new camera stream as a small window for about a second; painting
   * the frames ourselves avoids that and uses exactly the mapping of the overlay.
   */
  paintFrame() {
    const v = this.video;
    const cv = this.display;
    if (!this.stream || !v.videoWidth) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const cw = cv.clientWidth;
    const ch = cv.clientHeight;
    const w = Math.round(cw * dpr);
    const hh = Math.round(ch * dpr);
    if (cv.width !== w) cv.width = w;
    if (cv.height !== hh) cv.height = hh;
    const ctx = cv.getContext('2d');
    const sc = fitScale(w, hh, v.videoWidth, v.videoHeight);
    const dw = v.videoWidth * sc;
    const dh = v.videoHeight * sc;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    if (dw < w || dh < hh) ctx.clearRect(0, 0, w, hh);
    if (this.facing === 'user') ctx.setTransform(-1, 0, 0, 1, w, 0);
    ctx.drawImage(v, (w - dw) / 2, (hh - dh) / 2, dw, dh);
  }

  /** Repaint on every new camera frame (independent of the analysis speed). */
  paintLoop(stream) {
    const v = this.video;
    const next = () => {
      if (this.stream !== stream) return;
      if (!this.paused) this.paintFrame();
      if ('requestVideoFrameCallback' in v) v.requestVideoFrameCallback(next);
      else requestAnimationFrame(next);
    };
    next();
  }

  clearDisplay() {
    const cv = this.display;
    cv.getContext('2d').clearRect(0, 0, cv.width, cv.height);
  }

  /**
   * Resolves once the camera delivers frames of a stable size (three frames in
   * a row with the same, non-zero size) or after `timeout` ms. iPhone Safari
   * first shows a new stream as a small window; the image stays hidden until then.
   */
  async waitForStableVideo(timeout = 1500) {
    const v = this.video;
    const t0 = performance.now();
    let last = '';
    let same = 0;
    while (performance.now() - t0 < timeout) {
      await new Promise((resolve) => {
        if ('requestVideoFrameCallback' in v) {
          const timer = setTimeout(resolve, 100);
          v.requestVideoFrameCallback(() => {
            clearTimeout(timer);
            resolve();
          });
        } else {
          setTimeout(resolve, 50);
        }
      });
      const size = `${v.videoWidth}x${v.videoHeight}`;
      if (v.videoWidth && size === last) {
        if (++same >= 2) return true;
      } else {
        same = 0;
      }
      last = size;
    }
    return false;
  }

  get frozen() {
    return this.running && this.paused;
  }

  /**
   * Models first, then the camera: the camera (and the iPhone's camera
   * indicator) only turns on once the models are ready, and the picture fades
   * in when its size is final.
   */
  async start() {
    if (this.running || this.starting) return;
    const gen = this.gen;
    this.startingGen = gen;
    this.wanted = true;
    this.onStartPhase?.('begin');
    let ok = false;
    try {
      ok = await this.startCamera(gen);
    } finally {
      if (this.startingGen === gen) this.startingGen = -1;
      // on success, let the picture finish fading in before a loading screen fades out over it
      if (ok) setTimeout(() => this.onStartPhase?.('end'), 250);
      else this.onStartPhase?.('end');
    }
  }

  /** A start of the current generation is in progress (stop() cancels it). */
  get starting() {
    return this.startingGen === this.gen;
  }

  async startCamera(gen) {
    const current = () => gen === this.gen;
    this.onState(this.switching ? 'switching' : 'starting');
    try {
      this.engine = await this.getEngine();
    } catch (e) {
      if (!current()) return;
      this.wanted = false;
      this.onState('idle');
      this.onError(e);
      return;
    }
    if (!current()) return;
    if (this.away) {
      this.onState('idle'); // the user left Live while the models loaded; wake() starts again
      return;
    }
    this.onStartPhase?.('camera');
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: this.facing, width: { ideal: 1280 }, height: { ideal: 720 } },
      });
    } catch (e) {
      if (!current()) return;
      this.wanted = false;
      this.onState('idle');
      const msg = e.name === 'NotAllowedError'
        ? 'Camera access was denied. Allow the camera for this site in the browser settings, then try again.'
        : `The camera could not be started (${e.name}).`;
      this.onError(new Error(msg));
      return;
    }
    if (!current()) {
      stream.getTracks().forEach((t) => t.stop()); // stopped while the permission prompt was open
      return;
    }
    this.stream = stream;
    this.display.classList.add('warming'); // hidden until the size is final
    this.video.srcObject = stream;
    await this.video.play().catch(() => {});
    const stable = await this.waitForStableVideo();
    if (!current() || this.stream !== stream) return;
    if (this.away) {
      this.stop(); // left Live while the camera started; wake() starts again
      return;
    }
    this.paintFrame();
    this.paintLoop(stream);
    crumb('live:camera', { size: `${this.video.videoWidth}x${this.video.videoHeight}`, stable });
    this.running = true;
    this.paused = false;
    this.tracks = [];
    this.avg = null;
    this.frames = 0;
    this.color = signal();
    crumb('live:start', { engine: this.engine.label, facing: this.facing, video: `${this.video.videoWidth}x${this.video.videoHeight}` });
    this.display.classList.remove('warming'); // fades in
    this.onState('running');
    this.loop();
    return true;
  }

  /**
   * Live is hidden (Photo, Lab or Settings shown): keep the camera open but
   * pause it, so coming back is instant. After `keepMs` without Live the camera
   * is switched off; it restarts automatically when Live is shown again.
   */
  sleep(keepMs = 60000) {
    if (this.away) return;
    this.away = true;
    if (!this.stream) return;
    this.video.pause();
    this.tracks = [];
    this.clearOverlay();
    this.onFaces?.(0);
    clearTimeout(this.sleepTimer);
    this.sleepTimer = setTimeout(() => {
      crumb('live:sleep-off', { ms: keepMs });
      this.stop();
    }, keepMs);
  }

  /** Live is shown again: continue with the open camera, or restart it. */
  wake() {
    this.away = false;
    clearTimeout(this.sleepTimer);
    const track = this.stream?.getVideoTracks()[0];
    if (this.running && track?.readyState === 'live') {
      this.paused = false;
      this.video.play().catch(() => {});
      this.onState('running');
      return;
    }
    if (this.wanted && !this.starting) {
      if (this.stream) this.stop({ keepState: true });
      this.start();
    }
  }

  stop({ keepState = false } = {}) {
    this.gen++;
    clearTimeout(this.sleepTimer);
    if (this.running) crumb('live:stop', { frames: this.frames });
    this.running = false;
    this.paused = false;
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.video.srcObject = null;
    this.tracks = [];
    this.hits = [];
    this.display.classList.add('warming');
    this.clearDisplay();
    this.clearOverlay();
    this.onFaces?.(0);
    if (!keepState) this.onState('idle');
  }

  async flip() {
    this.facing = this.facing === 'user' ? 'environment' : 'user';
    if (this.running || this.stream) {
      // stay on the grey "switching camera" screen instead of the start screen
      this.switching = true;
      this.stop({ keepState: true });
      try {
        await this.start();
      } finally {
        this.switching = false;
      }
    }
  }

  /** Freeze the current frame and report the faces; false if there is none. */
  freeze(index = 0) {
    if (!this.running || !this.tracks.length) return false;
    this.paused = true;
    this.video.pause();
    this.onState('frozen');
    const faces = this.tracks.map((tr) => ({ ...interpret(tr.raw, this.engine.cards), score: tr.score }));
    this.onInspect?.(faces, Math.min(index, faces.length - 1));
    return true;
  }

  resume() {
    if (!this.running || !this.paused) return;
    this.paused = false;
    this.video.play().catch(() => {});
    this.onState('running');
  }

  tap(e) {
    if (!this.running || this.paused) return;
    const r = this.overlay.getBoundingClientRect();
    const x = e.clientX - r.left;
    const y = e.clientY - r.top;
    const hit = this.hits.find((b) => x >= b.l - 12 && x <= b.r + 12 && y >= b.t - 12 && y <= b.b + 12);
    if (hit) this.freeze(hit.i);
  }

  async loop() {
    while (this.running) {
      if (!this.paused && this.video.readyState >= 2) {
        const vw = this.video.videoWidth;
        const vh = this.video.videoHeight;
        if (this.frame.width !== vw) this.frame.width = vw;
        if (this.frame.height !== vh) this.frame.height = vh;
        this.frame.getContext('2d').drawImage(this.video, 0, 0, vw, vh);
        const s = this.getSettings();
        try {
          const res = await analyze(this.engine, this.frame, vw, vh, this.canvases, {
            detectorMaxSide: s.detectorSize,
            scoreThreshold: s.scoreThreshold,
            maxFaces: s.maxFaces,
          });
          if (!this.running) break;
          if (!this.paused) {
            this.update(res, vw, vh, s);
            this.draw();
          }
          if (++this.frames % 150 === 0) {
            crumb('live:frames', { frames: this.frames, fps: Math.round(this.fps), faces: res.faces.length, engine: this.engine.label });
          }
        } catch (e) {
          if (this.running) {
            this.wanted = false;
            this.stop();
            this.onError(e);
          }
          break;
        }
      }
      await this.nextVideoFrame();
    }
    // free frame buffers only once no analysis is in flight (and no new loop runs)
    if (!this.running) {
      this.frame.width = this.frame.height = 0;
      freeCanvases(this.canvases);
    }
  }

  /** Wait for a new camera frame (never analyse the same frame twice). */
  nextVideoFrame() {
    const v = this.video;
    if (!this.running || this.paused || !('requestVideoFrameCallback' in v) || v.paused) return nextFrame();
    return new Promise((resolve) => {
      const timer = setTimeout(resolve, 250); // safety net if no frame arrives
      v.requestVideoFrameCallback(() => {
        clearTimeout(timer);
        resolve();
      });
    });
  }

  update(res, vw, vh, s) {
    const now = performance.now();
    if (this.lastDone) {
      const inst = 1000 / (now - this.lastDone);
      this.fps = this.fps ? this.fps * 0.9 + inst * 0.1 : inst;
    }
    this.lastDone = now;
    const t = res.timings;
    const k = 0.1;
    this.avg = this.avg
      ? Object.fromEntries(Object.entries(t).map(([key, v]) => [key, this.avg[key] * (1 - k) + v * k]))
      : { ...t };

    // Simple tracking: each detected face is matched to the previous face with the
    // largest box overlap (IoU above 0.3). With smoothing on, matched faces are
    // updated with an exponential moving average (new frame weighted 0.3) to
    // reduce flicker; unmatched faces start a new track.
    const alpha = s.smoothing ? 0.3 : 1;
    const next = [];
    for (const f of res.faces) {
      let best = null;
      let bestIou = 0.3;
      for (const tr of this.tracks) {
        const v = iou(tr.box, f.box);
        if (v > bestIou && !next.includes(tr)) {
          best = tr;
          bestIou = v;
        }
      }
      if (best) {
        best.box = f.box;
        best.landmarks = f.landmarks;
        best.score = f.score;
        best.raw.age += alpha * (f.raw.age - best.raw.age);
        best.raw.pFemale += alpha * (f.raw.pFemale - best.raw.pFemale);
        best.raw.probs = best.raw.probs.map((p, i) => p + alpha * (f.raw.probs[i] - p));
        next.push(best);
      } else {
        next.push({ box: f.box, landmarks: f.landmarks, score: f.score, raw: { ...f.raw, probs: Array.from(f.raw.probs) } });
      }
    }
    // stable numbering: left to right in the picture as the user sees it
    next.sort((a, b) => (this.facing === 'user' ? b.box.x - a.box.x : a.box.x - b.box.x));
    this.tracks = next;
    this.vw = vw;
    this.vh = vh;
    this.onFaces?.(next.length);
    this.renderTelemetry(res.faces.length);
  }

  // Fills the compact bar and the expanded panel (both carry data-t keys).
  renderTelemetry(nFaces) {
    const a = this.avg;
    const set = (key, v) => {
      for (const node of this.telemetry.querySelectorAll(`[data-t="${key}"]`)) node.textContent = v;
    };
    set('detect', ms(a.detect));
    set('ageGender', nFaces ? ms(a.ageGender / Math.max(1, nFaces)) : '–');
    set('expression', nFaces ? ms(a.expression / Math.max(1, nFaces)) : '–');
    set('other', ms(Math.max(0, a.total - a.detect - a.ageGender - a.expression)));
    set('total', ms(a.total));
    set('fps', this.fps ? this.fps.toFixed(0) : '–');
    set('faces', String(nFaces));
    const e = this.engine;
    const threads = runtimeInfo().wasmThreads;
    set('engine', e.backend === 'webgpu'
      ? `WebGPU · ${e.precision.toUpperCase()}`
      : `WASM · ${e.precision.toUpperCase()} · ${threads} thread${threads === 1 ? '' : 's'}`);
  }

  clearOverlay() {
    const ctx = this.overlay.getContext('2d');
    ctx.clearRect(0, 0, this.overlay.width, this.overlay.height);
  }

  draw() {
    const cv = this.overlay;
    const dpr = window.devicePixelRatio || 1;
    const cw = cv.clientWidth;
    const ch = cv.clientHeight;
    if (cv.width !== Math.round(cw * dpr)) cv.width = Math.round(cw * dpr);
    if (cv.height !== Math.round(ch * dpr)) cv.height = Math.round(ch * dpr);
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, ch);
    this.hits = [];
    if (!this.running || !this.vw) return;

    // same mapping as paintFrame (cover, or contain on a computer)
    const sc = fitScale(cw, ch, this.vw, this.vh);
    const ox = (cw - this.vw * sc) / 2;
    const oy = (ch - this.vh * sc) / 2;
    const mirror = this.facing === 'user';
    const mapX = (x) => (mirror ? cw - (x * sc + ox) : x * sc + ox);
    const mapY = (y) => y * sc + oy;
    const s = this.getSettings();
    const compact = this.tracks.length > 1;

    this.tracks.forEach((tr, i) => {
      const x1 = mapX(tr.box.x);
      const x2 = mapX(tr.box.x + tr.box.w);
      const l = Math.min(x1, x2);
      const r = Math.max(x1, x2);
      const t = mapY(tr.box.y);
      const b = mapY(tr.box.y + tr.box.h);
      this.drawBrackets(ctx, l, t, r, b);
      ctx.fillStyle = this.color;
      ctx.font = `500 10px ${MONO}`;
      ctx.textBaseline = 'bottom';
      ctx.fillText(String(i + 1).padStart(2, '0'), Math.max(6, l + 1), Math.max(16, t - 6));
      if (s.showLandmarks) {
        for (const [lx, ly] of tr.landmarks) {
          ctx.beginPath();
          ctx.arc(mapX(lx), mapY(ly), 2.6, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      const p = interpret(tr.raw, this.engine.cards);
      const tag = compact ? this.drawCompact(ctx, p, l, r, t, b, cw, ch) : this.drawReadout(ctx, p, l, r, t, b, cw, ch);
      this.hits.push({ i, l: Math.min(l, tag.x), r: Math.max(r, tag.x + tag.w), t: Math.min(t, tag.y), b: Math.max(b, tag.y + tag.h) });
    });
  }

  drawBrackets(ctx, l, t, r, b) {
    const len = Math.max(10, Math.min(r - l, b - t) * 0.2);
    ctx.save();
    ctx.lineWidth = 2.5;
    ctx.lineCap = 'round';
    ctx.strokeStyle = this.color;
    ctx.shadowColor = 'rgba(0,0,0,0.5)';
    ctx.shadowBlur = 4;
    ctx.beginPath();
    ctx.moveTo(l, t + len); ctx.lineTo(l, t); ctx.lineTo(l + len, t);
    ctx.moveTo(r - len, t); ctx.lineTo(r, t); ctx.lineTo(r, t + len);
    ctx.moveTo(r, b - len); ctx.lineTo(r, b); ctx.lineTo(r - len, b);
    ctx.moveTo(l + len, b); ctx.lineTo(l, b); ctx.lineTo(l, b - len);
    ctx.stroke();
    ctx.restore();
  }

  placeTag(w, hgt, l, r, t, b, cw, ch) {
    const x = Math.max(8, Math.min(cw - w - 8, (l + r) / 2 - w / 2));
    let y = b + 14;
    let below = true;
    if (y + hgt > ch - 8) {
      y = Math.max(8, t - hgt - 22);
      below = false;
    }
    return { x, y, w, h: hgt, below };
  }

  panel(ctx, g, l, r, t, b) {
    ctx.fillStyle = C.ink;
    ctx.strokeStyle = C.rule;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(g.x + 0.5, g.y + 0.5, g.w, g.h, 4);
    ctx.fill();
    ctx.stroke();
    // connector from the face frame to the panel
    const cx = Math.max(g.x + 12, Math.min(g.x + g.w - 12, (l + r) / 2));
    ctx.strokeStyle = this.color;
    ctx.beginPath();
    if (g.below) { ctx.moveTo(cx, b + 1); ctx.lineTo(cx, g.y); } else { ctx.moveTo(cx, g.y + g.h); ctx.lineTo(cx, t - 1); }
    ctx.stroke();
  }

  drawReadout(ctx, p, l, r, t, b, cw, ch) {
    const g = this.placeTag(204, 92, l, r, t, b, cw, ch);
    this.panel(ctx, g, l, r, t, b);
    const x = g.x + 12;
    const [lo, hi] = ageRange(p.age);
    ctx.textBaseline = 'alphabetic';
    ctx.fillStyle = C.muted;
    ctx.font = `400 10px ${MONO}`;
    ctx.fillText('AGE', x, g.y + 34);
    ctx.fillStyle = C.bone;
    ctx.font = `500 26px ${MONO}`;
    const ageText = String(Math.round(p.age));
    ctx.fillText(ageText, x + 36, g.y + 37);
    const aw = ctx.measureText(ageText).width;
    ctx.fillStyle = C.muted;
    ctx.font = `400 11px ${MONO}`;
    ctx.fillText(`${lo}–${hi}`, x + 36 + aw + 8, g.y + 36);
    ctx.fillStyle = C.track;
    ctx.fillRect(x, g.y + 50, g.w - 24, 1);

    const colW = (g.w - 24 - 12) / 2;
    const cols = [
      { label: p.gender.toUpperCase(), v: p.genderConf, color: C.bone },
      { label: p.expression.toUpperCase(), v: p.expressionConf, color: this.color },
    ];
    cols.forEach((c, i) => {
      const cx = x + i * (colW + 12);
      ctx.font = `400 10px ${MONO}`;
      ctx.fillStyle = C.bone;
      ctx.textAlign = 'left';
      ctx.fillText(c.label, cx, g.y + 68);
      ctx.fillStyle = C.muted;
      ctx.textAlign = 'right';
      ctx.fillText(String(Math.round(c.v * 100)), cx + colW, g.y + 68);
      ctx.textAlign = 'left';
      ctx.fillStyle = C.track;
      ctx.fillRect(cx, g.y + 75, colW, 3);
      ctx.fillStyle = c.color;
      ctx.fillRect(cx, g.y + 75, colW * c.v, 3);
    });
    return g;
  }

  drawCompact(ctx, p, l, r, t, b, cw, ch) {
    ctx.font = `400 10px ${MONO}`;
    const rest = `  ${p.gender === 'female' ? 'F' : 'M'}${Math.round(p.genderConf * 100)}  ${p.expression.toUpperCase()} ${Math.round(p.expressionConf * 100)}`;
    const restW = ctx.measureText(rest).width;
    ctx.font = `500 15px ${MONO}`;
    const ageText = String(Math.round(p.age));
    const ageW = ctx.measureText(ageText).width;
    const g = this.placeTag(ageW + restW + 20, 30, l, r, t, b, cw, ch);
    this.panel(ctx, g, l, r, t, b);
    ctx.textBaseline = 'middle';
    ctx.fillStyle = C.bone;
    ctx.fillText(ageText, g.x + 10, g.y + 16);
    ctx.font = `400 10px ${MONO}`;
    ctx.fillStyle = C.muted;
    ctx.fillText(rest, g.x + 10 + ageW, g.y + 16);
    return g;
  }
}
