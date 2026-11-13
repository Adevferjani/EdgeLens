// Face alignment and cropping.
//
// The face is rotated so that the eyes are horizontal, which removes in-plane
// rotation (head tilt) and brings the crop closer to the roughly upright faces
// of the training data. Then a square around the detector box is cut out:
//   side   = crop.scale * max(box width, box height)
//   centre = box centre + (crop.shift_x, crop.shift_y) * max(w, h), in the
//            rotated (face-aligned) frame
// crop.scale / shift come from training/calibrate_crop.py, which measures how
// the training images are framed relative to the same detector. The result is
// an RGB tensor in [0, 1], NCHW - the network normalises internally.

const FILL = 'rgb(115,115,115)'; // ~ImageNet mean, neutral after normalisation

export function faceGeometry(face, crop) {
  const [re, le] = face.landmarks;
  const angle = Math.atan2(le[1] - re[1], le[0] - re[0]);
  const base = Math.max(face.w, face.h);
  const side = base * crop.scale;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const sx = (crop.shift_x || 0) * base;
  const sy = (crop.shift_y || 0) * base;
  const cx = face.x + face.w / 2 + cos * sx - sin * sy;
  const cy = face.y + face.h / 2 + sin * sx + cos * sy;
  return { angle, side, cx, cy };
}

/** Draw the aligned crop of `face` into `canvas` (size x size) and return it as a tensor. */
export function cropFace(source, face, crop, size, canvas) {
  const { angle, side, cx, cy } = faceGeometry(face, crop);
  if (canvas.width !== size) canvas.width = size;
  if (canvas.height !== size) canvas.height = size;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = FILL;
  ctx.fillRect(0, 0, size, size);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.translate(size / 2, size / 2);
  ctx.scale(size / side, size / side);
  ctx.rotate(-angle);
  ctx.translate(-cx, -cy);
  ctx.drawImage(source, 0, 0);
  ctx.setTransform(1, 0, 0, 1, 0, 0);

  const rgba = ctx.getImageData(0, 0, size, size).data;
  const plane = size * size;
  if (canvas._buf?.length !== 3 * plane) canvas._buf = new Float32Array(3 * plane); // reused per canvas
  const t = canvas._buf;
  for (let i = 0, j = 0; i < plane; i++, j += 4) {
    t[i] = rgba[j] / 255;
    t[plane + i] = rgba[j + 1] / 255;
    t[2 * plane + i] = rgba[j + 2] / 255;
  }
  return t;
}
