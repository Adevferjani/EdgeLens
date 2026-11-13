// YuNet face detector (Wu et al., 2023): pre- and post-processing.
// The decoding mirrors OpenCV's cv::FaceDetectorYN (verified against it in
// training/yunet.py, max. difference 2e-5 px).

export const STRIDES = [8, 16, 32];

const ceil32 = (v) => Math.ceil(v / 32) * 32;

/**
 * Draw `source` scaled so that its long side is at most `maxSide`, pad to a
 * multiple of 32 and convert to the detector input: BGR, 0-255, NCHW.
 */
export function prepareDetectorInput(source, srcW, srcH, maxSide, canvas) {
  const scale = Math.min(1, maxSide / Math.max(srcW, srcH));
  const w = Math.max(1, Math.round(srcW * scale));
  const h = Math.max(1, Math.round(srcH * scale));
  const pw = ceil32(w);
  const ph = ceil32(h);
  if (canvas.width !== pw) canvas.width = pw;
  if (canvas.height !== ph) canvas.height = ph;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, pw, ph);
  ctx.drawImage(source, 0, 0, w, h);
  const rgba = ctx.getImageData(0, 0, pw, ph).data;
  const plane = pw * ph;
  // Reuse one buffer per canvas: avoids ~1 MB of garbage per camera frame.
  if (canvas._buf?.length !== 3 * plane) canvas._buf = new Float32Array(3 * plane);
  const data = canvas._buf;
  for (let i = 0, j = 0; i < plane; i++, j += 4) {
    data[i] = rgba[j + 2]; // B
    data[plane + i] = rgba[j + 1]; // G
    data[2 * plane + i] = rgba[j]; // R
  }
  return { data, width: pw, height: ph, scale };
}

function iou(a, b) {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w);
  const y2 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return inter / (a.w * a.h + b.w * b.h - inter);
}

/**
 * Decode the 12 YuNet output tensors into faces in source-image pixels.
 * Each face: {x, y, w, h, score, landmarks: [[x, y] x 5]}
 * Landmark order: right eye, left eye, nose tip, right and left mouth corner
 * (from the person's view, i.e. index 0 is on the left side of the image).
 */
export function decodeDetections(out, width, height, opts = {}) {
  const { scoreThreshold = 0.6, nmsThreshold = 0.3, topK = 100, scale = 1 } = opts;
  const cand = [];
  for (const s of STRIDES) {
    const cols = width / s;
    const cls = out[`cls_${s}`];
    const obj = out[`obj_${s}`];
    const bbox = out[`bbox_${s}`];
    const kps = out[`kps_${s}`];
    const n = cls.length;
    for (let i = 0; i < n; i++) {
      const score = Math.sqrt(Math.min(1, Math.max(0, cls[i])) * Math.min(1, Math.max(0, obj[i])));
      if (score < scoreThreshold) continue;
      const r = Math.floor(i / cols);
      const c = i - r * cols;
      const cx = (c + bbox[i * 4]) * s;
      const cy = (r + bbox[i * 4 + 1]) * s;
      const w = Math.exp(bbox[i * 4 + 2]) * s;
      const h = Math.exp(bbox[i * 4 + 3]) * s;
      const landmarks = [];
      for (let k = 0; k < 5; k++) {
        landmarks.push([(kps[i * 10 + 2 * k] + c) * s, (kps[i * 10 + 2 * k + 1] + r) * s]);
      }
      cand.push({ x: cx - w / 2, y: cy - h / 2, w, h, score, landmarks });
    }
  }
  cand.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const d of cand) {
    if (keep.length >= topK) break;
    if (keep.every((k) => iou(k, d) <= nmsThreshold)) keep.push(d);
  }
  const inv = 1 / scale;
  return keep.map((d) => ({
    x: d.x * inv,
    y: d.y * inv,
    w: d.w * inv,
    h: d.h * inv,
    score: d.score,
    landmarks: d.landmarks.map(([x, y]) => [x * inv, y * inv]),
  }));
}
