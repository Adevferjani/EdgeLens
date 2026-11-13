// Bit-exact port of Pillow's 8-bit bilinear resampling (libImaging/Resample.c).
// torchvision's Resize uses it for PIL images, so the test-set check feeds the models
// exactly the pixels they saw during evaluation on the PC (given the same decoded image).

// Fixed-point arithmetic as in Pillow: filter weights are integers scaled by
// 2^PRECISION_BITS, leaving 8 of 32 bits for the pixel value and 2 for overflow
// (Pillow's filters can have negative weights or weights that add up to slightly
// more than 1). HALF is the rounding offset.
const PRECISION_BITS = 32 - 8 - 2;
const ONE = 1 << PRECISION_BITS;
const HALF = 1 << (PRECISION_BITS - 1);
const cache = new Map();

function coefficients(inSize, outSize) {
  const key = `${inSize}>${outSize}`;
  if (cache.has(key)) return cache.get(key);
  const scale = inSize / outSize;
  const filterscale = Math.max(scale, 1);
  const support = 1 * filterscale; // bilinear filter support = 1
  const ksize = Math.ceil(support) * 2 + 1;
  const kk = new Int32Array(outSize * ksize);
  const bounds = new Int32Array(outSize * 2);
  const w = new Float64Array(ksize);
  const ss = 1 / filterscale;
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    let xmin = Math.trunc(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    let ww = 0;
    for (let x = 0; x < xmax; x++) {
      const t = Math.abs((x + xmin - center + 0.5) * ss);
      w[x] = t < 1 ? 1 - t : 0;
      ww += w[x];
    }
    for (let x = 0; x < xmax; x++) {
      const v = ww !== 0 ? w[x] / ww : w[x];
      kk[xx * ksize + x] = v < 0 ? Math.trunc(-0.5 + v * ONE) : Math.trunc(0.5 + v * ONE);
    }
    bounds[2 * xx] = xmin;
    bounds[2 * xx + 1] = xmax;
  }
  const c = { kk, bounds, ksize };
  cache.set(key, c);
  return c;
}

const clip8 = (v) => (v >= ONE << 8 ? 255 : v <= 0 ? 0 : v >> PRECISION_BITS);

/**
 * Resize an RGBA image (e.g. ImageData.data) to outW x outH like
 * PIL Image.resize(..., BILINEAR) on an RGB image. Returns RGB bytes (HWC).
 */
export function resizeRGBA(src, inW, inH, outW, outH) {
  // horizontal pass: inH rows x outW columns, 8-bit intermediate as in Pillow
  const H = coefficients(inW, outW);
  const tmp = new Uint8Array(outW * inH * 3);
  for (let y = 0; y < inH; y++) {
    const row = y * inW * 4;
    for (let xx = 0; xx < outW; xx++) {
      const xmin = H.bounds[2 * xx];
      const n = H.bounds[2 * xx + 1];
      const k = xx * H.ksize;
      let r = HALF;
      let g = HALF;
      let b = HALF;
      for (let x = 0; x < n; x++) {
        const p = row + (x + xmin) * 4;
        const c = H.kk[k + x];
        r += src[p] * c;
        g += src[p + 1] * c;
        b += src[p + 2] * c;
      }
      const o = (y * outW + xx) * 3;
      tmp[o] = clip8(r);
      tmp[o + 1] = clip8(g);
      tmp[o + 2] = clip8(b);
    }
  }
  // vertical pass
  const V = coefficients(inH, outH);
  const out = new Uint8Array(outW * outH * 3);
  for (let yy = 0; yy < outH; yy++) {
    const ymin = V.bounds[2 * yy];
    const n = V.bounds[2 * yy + 1];
    const k = yy * V.ksize;
    for (let xx = 0; xx < outW; xx++) {
      let r = HALF;
      let g = HALF;
      let b = HALF;
      for (let y = 0; y < n; y++) {
        const p = ((y + ymin) * outW + xx) * 3;
        const c = V.kk[k + y];
        r += tmp[p] * c;
        g += tmp[p + 1] * c;
        b += tmp[p + 2] * c;
      }
      const o = (yy * outW + xx) * 3;
      out[o] = clip8(r);
      out[o + 1] = clip8(g);
      out[o + 2] = clip8(b);
    }
  }
  return out;
}

/** RGB bytes (HWC) -> float32 CHW in [0, 1], like torchvision ToTensor. */
export function toCHW(rgb, w, h, out = new Float32Array(3 * w * h)) {
  const plane = w * h;
  for (let i = 0; i < plane; i++) {
    out[i] = rgb[3 * i] / 255;
    out[plane + i] = rgb[3 * i + 1] / 255;
    out[2 * plane + i] = rgb[3 * i + 2] / 255;
  }
  return out;
}
