# Architecture

EdgeLens consists of two parts. A training pipeline in PyTorch (`training/`) runs on a PC and produces
ONNX models. A progressive web app (`pwa/`) runs these models in the browser of the phone. The only
interface between the two parts are the files in `pwa/models/`: the ONNX models and one JSON model
card per model.

## 1. Training and deployment pipeline

```
PC                                                     iPhone (Safari)
UTKFace / FER2013                                      camera frame or photo
   │ train.py: fine-tune MobileNetV3-Small                │
   ▼                                                      ▼
best checkpoint (selected on the validation split)     YuNet face detector
   │ export_onnx.py: ONNX FP32 + weight-only INT8         │ rotate eyes level, crop
   ▼                                                      ▼
pwa/models/*.onnx + model cards  ── GitHub Pages ──►   age + gender model, expression model
   │ evaluate.py: test metrics, PC latency                │
   ▼                                                      ▼
runs/*/evaluation.json                                 age, age group, gender, expression
```

The two attribute models are trained separately (`train.py`), and the checkpoint with the best
validation score is kept. `export_onnx.py` converts it to ONNX in two precisions and writes the model
card. `evaluate.py` measures the test metrics of the PyTorch model and of both ONNX files, and the
latency on the PC. Every push to the `main` branch that changes `pwa/` publishes the app on GitHub
Pages through a GitHub Actions workflow.

## 2. Face detection and alignment

Faces are detected with YuNet (Wu et al., 2023), a pre-trained detector of 0.23 MB that returns a box,
a confidence score and five landmarks (eyes, nose, mouth corners) per face. YuNet was chosen over
MediaPipe because it runs in the same runtime as the other two models (ONNX Runtime Web), and its eye
landmarks allow the face to be aligned. The decoding of the YuNet outputs is implemented in Python
(`training/yunet.py`) and JavaScript and was checked against OpenCV's implementation (differences
below 10⁻⁴ px). The image is scaled so that its long side is at most 320 px
(adjustable from 256 to 640 px), and faces with a score below 0.6 are discarded. Live mode analyses up
to five faces, Photo mode up to ten (the largest ones).

Each face is rotated so that the eyes are level and cut out as a square crop. The training images of
UTKFace and FER2013 are already cropped faces, while the app receives a detector box, so the two can be
framed differently. `training/calibrate_crop.py` therefore runs YuNet on 1,000 training images per
dataset and measures the crop size and position relative to the detector box (median crop side for
UTKFace: 0.918 × the box side, centre shifted down by 5.9 % of the box side; FER2013: 0.965 × the box
side). These values are stored in the
model cards, and the app crops camera faces in the same way. Without the calibration the crops would
have covered a much larger area than the faces seen in training.

## 3. Models

Both attribute models use MobileNetV3-Small (Howard et al., 2019), a network designed for mobile CPUs,
initialised with ImageNet weights and fine-tuned (about 1.08 million parameters each).

| Model | Input | Output |
|---|---|---|
| YuNet | BGR image, long side ≤ 320 px, padded to a multiple of 32 | boxes, scores, 5 landmarks |
| Age and gender | RGB face crop, 160 × 160, values in [0, 1] | age in years (regression), P(female) |
| Facial expression | RGB face crop, 112 × 112, values in [0, 1] | probabilities of the 7 FER2013 classes |

The age and gender model has two output heads on a shared backbone. The expression model converts its
input to grayscale internally, because FER2013 contains only grayscale images. Both networks contain
the ImageNet normalisation as their first layer, so the app only has to crop, resize and scale the
pixels. The Hardswish and Hardsigmoid activations are written with basic operators so that every ONNX
Runtime Web backend can run the graph.

The app derives the final outputs from the model outputs: the age group (adult 18–59, elderly 60+),
the gender (female if P(female) ≥ 0.5), the most likely expression, and a forced choice between happy
and sad that compares only these two probabilities.

## 4. ONNX deployment in the browser

The models run with ONNX Runtime Web 1.31.1, which is included in the app, so no external server is
used for inference. On the iPhone the app uses the WebAssembly backend (CPU) with up to four threads.
Multi-threading requires a cross-origin isolated page; because GitHub Pages cannot set the necessary
HTTP headers, the service worker adds them to every response.

The default precision is FP32. The weight-only INT8 models (W8A32) store the weights as 8-bit integers,
which makes the files 72 % smaller; when a model is loaded, ONNX Runtime converts the weights back to
FP32 once, so the computation is the same as for the FP32 model.

The service worker stores all app files, the runtime and the models in a cache, so the app works
offline after the first visit. A list of file hashes, generated by `pwa/tools/build_sw.py`, lets an
update download only changed files, and a new version is activated only after the user confirms it.

## 5. Main design decisions

| Decision | Reason |
|---|---|
| Progressive web app instead of a native app | runs on the iPhone without an App Store build and can be tested in desktop browsers |
| MobileNetV3-Small for both attribute models | small and fast on mobile CPUs; ImageNet pre-training shortens training |
| YuNet instead of MediaPipe | one runtime for all models; landmarks for eye alignment; 0.23 MB |
| Crop calibration | aligns the framing of detected faces with the framing of the training images |
| Age-balanced sampling for the age and gender model | elderly people are rare in UTKFace; sampling them more often raised the recall of the 60+ group from 63.2 % to 69.5 % at the same age error |
| WebAssembly instead of WebGPU on the iPhone | in an on-device benchmark WebGPU was 3.5–6.8 times slower per stage, and sustained WebGPU use next to the camera crashed Safari |
| FP32 as default, weight-only INT8 as option | static INT8 quantisation destroyed the accuracy; weight-only INT8 keeps the accuracy but is not faster |

The measurements behind these decisions are in [results/evaluation-table.md](results/evaluation-table.md)
and [results/quantization.md](results/quantization.md).

## 6. Application structure

The app uses plain JavaScript modules without a framework or build step; `pwa/index.html` loads
`js/app.js`, which starts the other modules.

| Modules (`pwa/js/`) | Responsibility |
|---|---|
| `app.js` | entry point, mode switching, first-launch flow |
| `live.js`, `photo.js`, `evaluation.js`, `sheet.js` | Live, Photo and Lab → Evaluate views, face detail sheet |
| `pipeline.js`, `detector.js`, `facecrop.js`, `resize.js`, `image.js` | recognition pipeline and image handling |
| `inference.js`, `engine-manager.js` | ONNX Runtime sessions, backend and precision choice, model loading |
| `benchmark.js`, `testset.js`, `metrics.js`, `lab-*.js` | Lab tools: latency benchmark, test-set check, metrics |
| `settings.js`, `settings-panel.js`, `updates.js`, other modules | settings, service worker and updates, screens, messages |

The metrics in `metrics.js` use the same definitions as `training/metrics.py`, so results measured on
the phone and on the PC are comparable.

## References

Howard, A., Sandler, M., Chu, G., Chen, L.-C., Chen, B., Tan, M., Wang, W., Zhu, Y., Pang, R., Vasudevan, V.,
Le, Q. V., & Adam, H. (2019). Searching for MobileNetV3. In *Proceedings of the IEEE/CVF International
Conference on Computer Vision* (pp. 1314–1324). IEEE. https://doi.org/10.1109/ICCV.2019.00140

Wu, W., Peng, H., & Yu, S. (2023). YuNet: A tiny millisecond-level face detector. *Machine Intelligence
Research, 20*(5), 656–665. https://doi.org/10.1007/s11633-023-1423-y
