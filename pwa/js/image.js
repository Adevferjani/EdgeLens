// Decoding photos for analysis (Photo mode and Lab → Evaluate).

async function loadImage(file) {
  const url = URL.createObjectURL(file);
  try {
    return await new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error(`${file.name} could not be read as an image.`));
      img.src = url;
    });
  } finally {
    URL.revokeObjectURL(url);
  }
}

export async function decode(file, canvas, maxSide = 1600) {
  let src = null;
  // Large files are camera photos: let the decoder shrink them (24 MP would
  // need ~100 MB). Browsers without resize support ignore the option.
  if (file.size > 1e6) {
    src = await createImageBitmap(file, { resizeWidth: maxSide, resizeQuality: 'high' }).catch(() => null);
  }
  src ??= await createImageBitmap(file).catch(() => null);
  src ??= await loadImage(file);
  const w0 = src.width || src.naturalWidth;
  const h0 = src.height || src.naturalHeight;
  const s = Math.min(1, maxSide / Math.max(w0, h0));
  canvas.width = Math.round(w0 * s);
  canvas.height = Math.round(h0 * s);
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(src, 0, 0, canvas.width, canvas.height);
  src.close?.();
  return canvas;
}
