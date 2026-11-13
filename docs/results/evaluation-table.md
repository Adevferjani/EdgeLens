# Evaluation

This document describes how the deployed models were evaluated on the PC and on the iPhone 15, and
reports the results. The deployed models are the age and gender model from
`runs/age_gender_cuda_balanced` and the expression model from `runs/expression_cuda`, both trained on
the GPU. The exported measurement files are listed in section 3.

## 1. Methodology

### 1.1 Test data and metrics

Accuracy is measured on test partitions that were not used for training or model selection: 2,371
UTKFace images for age and gender (fixed stratified 10 % split) and the 3,589 images of the FER2013
PrivateTest partition for facial expression.

| Task | Metrics |
|---|---|
| Age | mean absolute error (MAE) in years, for all test images, for people aged 18+ and for people aged 60+ |
| Age group | adult (18–59) vs. elderly (60+) for people aged 18+: accuracy, macro-F1, recall per group, balanced accuracy |
| Gender | accuracy and macro-F1 |
| Facial expression | 7-class accuracy, macro-F1 and per-class F1; happy vs. sad on the images labelled happy or sad (n = 1,473), as a forced choice between the two probabilities and as the top class of all seven |

The main metric for the classification tasks is the macro-F1, because the classes are not balanced
(for example, only 269 of the 1,948 test images of people aged 18+ show people aged 60+).

### 1.2 Hardware and latency measurement

PC: Intel Core i5-12500H, NVIDIA GeForce RTX 4060 Laptop GPU (8 GB), ONNX Runtime 1.31.1. Edge device:
iPhone 15, Safari, ONNX Runtime Web 1.31.1 with the WebAssembly backend.

Latency is measured at batch size 1. On the PC, `training/evaluate.py` reports the mean of 200 runs
after 20 warm-up runs. On the iPhone, the app's Lab → Speed benchmark runs each model for at least two
seconds before the timed runs and reports the median. The first benchmark with four threads showed an
order effect: the configuration measured first was slower, because the phone raises its processor
clock only under sustained load. The final measurement was therefore repeated twice in opposite order
(FP32 first, INT8 first), and the mean of the two medians is reported. Timings on the phone also vary
between sessions (temperature, charging), so configurations are only compared within one session.

Two latencies are distinguished: the **inference time** of a single model, and the **full pipeline**
time for one camera frame with one face (face detection, alignment and cropping, both models).

### 1.3 On-device accuracy check

To show that the models give the same results on the phone as on the PC, the complete test partitions
were also evaluated on the iPhone. `training/tools/export_testset.py` packs the test images together
with the PC predictions of the same model files into one file (not published, because it contains
dataset images). The app's Lab → Test sets runs both models on every image and compares the metrics
and the individual predictions with the PC. The app resizes images with a port of the bilinear filter
used on the PC, so both receive the same pixels if the image is decoded identically.

### 1.4 Practical evaluation with 20 own photos

The task requires a practical evaluation with 20 images: 50 % adults and 50 % elderly people, and
within each age group 50 % male / female and 50 % happy / sad. Photos from digitised magazines were
considered first, but too few large, frontal sad faces with known ages could be found.

The 20 photos were therefore taken with phone cameras for this evaluation: some by the author (iPhone
15), the others by two contributors who photographed members of their families (iPhone 15 Pro and iPhone
17) and provided the true labels. Everyone shown gave consent. The labels are factual: the real age of
each person, the gender, and the expression shown (happy or sad). They were fixed before the photos were
analysed. All photos were evaluated in EdgeLens on the iPhone 15 (Lab → Evaluate, WebAssembly, FP32) and
deleted afterwards; only the anonymous export remains.

| | Male, happy | Male, sad | Female, happy | Female, sad | Total |
|---|---|---|---|---|---|
| Adult (18–59) | 3 | 2 | 2 | 3 | 10 |
| Elderly (60+) | 2 | 3 | 3 | 2 | 10 |

A group of ten cannot be split evenly by gender and expression at the same time, so the extra photo
alternates between the two age groups; every gender and expression combination has five photos.

## 2. Results

### 2.1 Overview: original papers, PC and iPhone (Table 1)

Main metric macro-F1, accuracy or MAE in brackets. Time = age and gender model / expression model in ms
(PC and iPhone with 4 threads as described in section 1.2; the iPhone one-thread values are means,
because that session recorded times in whole milliseconds).

| Model / configuration | Age group adult vs. 60+ (18+), F1 [accuracy] | Age MAE (years) | Gender, F1 [accuracy] | Expression, 7 classes, F1 [accuracy] | Time per inference (ms) |
|---|---|---|---|---|---|
| MobileNetV3-Small 1.0, original paper (Howard et al., 2019) | – (ImageNet top-1: 67.4 %) | – | – | – | 15.8 (Pixel 1, 224 × 224, one core) |
| FER2013 reference (Goodfellow et al., 2013) | – | – | – | [human ≈ 65 ± 5 %; best challenge entry 71.2 %] | – |
| YuNet, original paper (Wu et al., 2023) | WIDER Face AP easy / medium / hard: 0.892 / 0.883 / 0.811 (original size), 0.836 / 0.747 / 0.395 (320 × 320) | – | – | – | 16.3 (original size) / 2.2 (320 × 320), desktop CPU (Intel i7-12700K) |
| PC, PyTorch FP32 | 0.847 [93.1 %] | 5.08 | 0.940 [94.1 %] | 0.683 [68.4 %] | CPU 6.8 / 5.6; GPU 4.5 / 4.0 |
| PC, ONNX Runtime FP32 | 0.847 [93.1 %] | 5.08 | 0.940 [94.1 %] | 0.683 [68.4 %] | 1.9 / 1.8 (one thread, median: 1.52 / 1.16) |
| PC, ONNX Runtime, weight-only INT8 | 0.847 [92.9 %] | 5.15 | 0.942 [94.2 %] | 0.683 [68.3 %] | one thread, median: 1.48 / 1.18 |
| PC, model trained on the CPU, PyTorch FP32 | 0.841 [93.1 %] | 5.29 | 0.942 [94.2 %] | 0.671 [67.9 %] | same architecture as above |
| iPhone 15, ONNX Runtime Web, FP32 | 0.849 [93.1 %] | 5.08 | 0.940 [94.0 %] | 0.683 [68.4 %] | 6.3 / 4.1 (4 threads); 6.0 / 4.5 (1 thread) |
| iPhone 15, ONNX Runtime Web, weight-only INT8 | 0.847 [92.9 %] | 5.15 | 0.940 [94.1 %] | 0.683 [68.3 %] | 5.7 / 4.3 (4 threads); 6.4 / 4.3 (1 thread) |

The rows for the original papers describe the components in their original tasks: MobileNetV3 is
evaluated on ImageNet classification and YuNet on face detection, so their numbers cannot be compared
directly with age, gender or expression scores. FER2013 provides a direct reference for the
expression task. The UTKFace paper (Zhang et al., 2017) addresses age progression and reports no
age-estimation baseline. The MobileNetV3 time refers to a 224 × 224 input; the inputs of the own models
(160 × 160 and 112 × 112) require about 51 % and 25 % of those operations.

### 2.2 Accuracy details

| | PC, PyTorch FP32 | PC, ONNX FP32 | PC, ONNX INT8 | iPhone FP32 | iPhone INT8 |
|---|---|---|---|---|---|
| Age MAE, all / 18+ / 60+ (years) | 5.08 / 5.51 / 7.50 | 5.08 / 5.51 / 7.50 | 5.15 / 5.56 / 7.24 | 5.08 / 5.52 / 7.52 | 5.15 / 5.56 / 7.24 |
| Recall adult / 60+ (18+) | 96.8 % / 69.5 % | 96.8 % / 69.5 % | 96.4 % / 71.0 % | 96.8 % / 69.9 % | 96.3 % / 71.4 % |
| Balanced accuracy adult vs. 60+ | 83.2 % | 83.2 % | 83.7 % | 83.4 % | 83.8 % |
| Happy vs. sad, forced choice | 91.0 % | 91.0 % | 90.7 % | 91.0 % | 90.7 % |
| Happy vs. sad, top of 7 classes | 72.8 % | 72.8 % | 72.2 % | 72.8 % | 72.2 % |
| Model size (MB) | 4.43 (checkpoint) | 4.35 | 1.24 | 4.35 | 1.24 |

Per-class F1 of the expression model (PC and iPhone, FP32): angry 0.61, disgust 0.77, fear 0.53,
happy 0.87, sad 0.54, surprise 0.80, neutral 0.66. With INT8 only sad changes (0.53).

The age error grows with age (MAE 7.5 years for people aged 60+), and only 69.5 % of the elderly test
faces are assigned to the elderly group; the others are estimated younger than 60. Weight-only INT8
changes the overall age MAE by 0.07 years and the other metrics by at most 1.5 percentage points.

### 2.3 Agreement between iPhone and PC

| | iPhone FP32 vs. PC FP32 | iPhone INT8 vs. PC INT8 |
|---|---|---|
| FER2013 (lossless PNG, n = 3,589): same top class | 100 % | 100 % |
| FER2013: largest probability difference | 4 × 10⁻⁶ | 6 × 10⁻⁶ |
| UTKFace (JPEG, n = 2,371): mean / largest age difference (years) | 0.17 / 1.61 | 0.17 / 1.48 |
| UTKFace: same gender decision | 99.87 % (3 images differ) | 99.83 % (4 images differ) |
| UTKFace: largest P(female) difference | 0.098 | 0.116 |

For lossless PNG images the iPhone reproduces the PC results up to floating-point rounding, so ONNX
Runtime Web and ONNX Runtime compute the same function. The JPEG images differ slightly because Safari
uses Apple's JPEG decoder, while the PC uses libjpeg-turbo; in Chromium, which also uses libjpeg-turbo,
the decoded pixels and the predictions matched the PC. The test metrics hardly change, but single age
estimates move by up to 1.6 years, which shows that the age model is sensitive to small pixel changes.

### 2.4 Inference latency of the individual models

On the PC, ONNX Runtime is 3–3.5 times faster than PyTorch on the CPU (Table 1). On the iPhone
(median in ms, 4 threads, two runs in opposite order):

| Model | FP32, run 1 (FP32 first) | INT8, run 1 | INT8, run 2 (INT8 first) | FP32, run 2 |
|---|---|---|---|---|
| Age and gender | 6.44 | 6.20 | 5.22 | 6.10 |
| Expression | 3.94 | 4.06 | 4.44 | 4.34 |

The face detector always runs in FP32 and was measured once per run: 5.62 ms in run 1 and 6.76 ms in
run 2.

FP32 and INT8 differ by −1.0 to +0.4 ms, and the sign changes between the runs, so the difference lies
within the measurement spread. This matches the expectation, because the weight-only INT8 model
computes in FP32 after loading.

Two further comparisons were measured in an earlier session, before the trained weights were available,
with models of the same architecture (latency depends on the architecture, not on the weight values):

| Stage (mean, ms) | WebAssembly, 1 thread | WebAssembly, 4 threads | WebGPU |
|---|---|---|---|
| Face detection | 4.53 | 3.66 | 24.92 |
| Age and gender | 6.62 | 4.72 | 17.61 |
| Expression | 3.95 | 3.08 | 11.66 |
| Full pipeline (one face) | 22.74 | 19.49 | 68.69 |

Four threads were 1.2–1.4 times faster per model than one thread; the gain is small because the models
are small and pre-processing runs on one thread. WebGPU was 3.5–6.8 times slower than WebAssembly,
probably because for such small models at batch size 1 the overhead of the GPU calls outweighs the
computation. Static INT8 models were particularly slow on WebGPU (140 and 243 ms per model, mean). For these reasons the app
uses WebAssembly on the iPhone.

### 2.5 Full pipeline latency

| iPhone 15, 4 threads, one face (median, ms) | Run 1 | Run 2 |
|---|---|---|
| FP32 | 25.66 | 27.26 |
| Weight-only INT8 | 28.84 | 30.14 |

The full pipeline takes about 26–27 ms with FP32, which would allow 37–39 frames per second; in Live
mode the observed rate was about 31 frames per second, limited by the 30 fps camera. The pipeline was
about 3 ms slower with INT8 in both runs; with two runs this difference is not conclusive. In
Lab → Evaluate, where the detector uses a resolution of 640 instead of 320 px, the analysis of one
photo took 81.8 ms on average (section 2.7).

### 2.6 Training on GPU vs. CPU

Both models were also trained on the CPU with the same data, seed, hyperparameters and maximum number
of epochs.

| | Age and gender, GPU | Age and gender, CPU | Expression, GPU | Expression, CPU |
|---|---|---|---|---|
| Hardware | RTX 4060 Laptop, mixed precision | i5-12500H, 12 threads, FP32 | RTX 4060 Laptop, mixed precision | i5-12500H, 12 threads, FP32 |
| Epochs run (best epoch) | 30 (28) | 28 (20), early stopping | 30 (28) | 30 (30) |
| Mean time per epoch | 15.0 s | 105.5 s (7.0 ×) | 14.7 s | 92.2 s (6.3 ×) |
| Total training time | 7.5 min | 49.3 min | 7.4 min | 46.1 min |
| Test result | MAE 5.08, gender 94.1 %, 60+ recall 69.5 % | MAE 5.29, gender 94.2 %, 60+ recall 64.7 % | accuracy 68.4 %, F1 0.683 | accuracy 67.9 %, F1 0.671 |

The GPU trained 6–7 times faster per epoch. The accuracy differences come from one run per device;
mixed precision and non-deterministic GPU kernels lead to different training trajectories, so they
cannot be attributed to the device.

**Effect of age-balanced sampling** (GPU, PyTorch FP32, same test set):

| | Without (`runs/age_gender_cuda`) | With (`runs/age_gender_cuda_balanced`, deployed) |
|---|---|---|
| Age MAE, all / 60+ (years) | 5.09 / 8.82 | 5.08 / 7.50 |
| Recall adult / 60+ (18+) | 97.8 % / 63.2 % | 96.8 % / 69.5 % |
| Balanced accuracy adult vs. 60+ | 80.5 % | 83.2 % |
| Gender accuracy | 94.2 % | 94.1 % |

### 2.7 Practical evaluation with 20 own photos (Table 2)

iPhone 15, WebAssembly, FP32. Accuracy with counts, macro-F1 in brackets, 95 % Wilson confidence
interval for the whole set.

| Subset | n | Faces found | Age group, acc. [F1] | Age MAE (years) | Gender, acc. [F1] | Expression, 7 classes, acc. | Happy vs. sad, forced choice |
|---|---|---|---|---|---|---|---|
| All photos | 20 | 20/20 | 95 % (19/20) [0.95] | 4.1 (median 3.8) | 85 % (17/20) [0.85] | 65 % (13/20) | 95 % (19/20) [0.95] |
| 95 % CI (all) | – | – | 76–99 % | – | 64–95 % | 43–82 % | 76–99 % |
| Adult (18–59) | 10 | 10/10 | 100 % (10/10) | 3.9 | 100 % (10/10) [1.00] | 80 % (8/10) | 100 % (10/10) [1.00] |
| Elderly (60+) | 10 | 10/10 | 90 % (9/10) | 4.2 | 70 % (7/10) [0.70] | 50 % (5/10) | 90 % (9/10) [0.90] |
| Male | 10 | 10/10 | 100 % (10/10) | 4.3 | 90 % (9/10) | 50 % (5/10) | 100 % (10/10) |
| Female | 10 | 10/10 | 90 % (9/10) | 3.9 | 80 % (8/10) | 80 % (8/10) | 90 % (9/10) |
| Happy | 10 | 10/10 | 100 % (10/10) | 2.8 | 80 % (8/10) | 80 % (8/10) | 90 % (9/10) |
| Sad | 10 | 10/10 | 90 % (9/10) | 5.4 | 90 % (9/10) | 50 % (5/10) | 100 % (10/10) |

All 20 faces were detected (detector scores 0.72–0.94), and one photo took 78.0–87.2 ms to analyse
(mean 81.8 ms). The errors:

- **Age group (1):** a 64-year-old was estimated at 59.2 years and therefore counted as an adult.
- **Age:** the largest errors were +9.1 and +7.9 years (two elderly people). On average the estimates
  were 0.6 years too high; sad faces were estimated 2.0 years older, happy faces 0.8 years younger.
- **Gender (3):** all three errors concern elderly people.
- **Expression (7 of 20):** sad faces were classified as neutral (2), fear (2) or angry (1), and two
  happy faces as neutral. Six of these seven are still correct in the happy-vs-sad forced choice.

The 7-class macro-F1 of this set is low (0.31) by construction: only two classes occur in the true
labels, but the predictions fall into five classes, each of which counts in the average. The forced
choice between happy and sad matches the design of the task.

With n = 20, one photo changes a percentage by 5 points, and the confidence intervals are wide. The
photos come from a small private circle, show posed or clearly visible expressions and were taken with
three different phone cameras. The results confirm that the app works on new photos, but they do not
allow general conclusions about its accuracy. As on the test set, the elderly group is the weaker one,
especially for gender.

## 3. Evidence files

| File | Content |
|---|---|
| `runs/<run>/evaluation.json` | PC test metrics (PyTorch, ONNX FP32, ONNX INT8) and PC latency of each run |
| `runs/<run>/train_summary.json`, `history.csv` | training configuration, hardware, time per epoch, validation curve |
| `iphone15/testset-wasm-4threads-20261102-2116.csv` / `.json` | iPhone test-set metrics, latency and per-image predictions |
| `iphone15/benchmark-wasm-trained-4threads-fp32first-20261102-2320.csv`, `...-int8first-20261102-2323.csv` | final iPhone latency, both orders |
| `iphone15/benchmark-wasm-trained-w8a32-4threads-20261102-2308.csv` | first 4-thread benchmark that showed the order effect |
| `iphone15/benchmark-wasm-trained-w8a32-1thread-20261101-2150.csv` | iPhone latency with one thread |
| `iphone15/benchmark-wasm-1thread-20261101-1851.csv`, `...-4threads-20261101-1843.csv`, `benchmark-webgpu-20261101-1845.csv` | thread and backend comparison (same architecture) |
| `iphone15/own-photos-evaluation-20261105-0006.csv` | Table 2, one row per photo (anonymous IDs) |
| `pc/static-int8-experiment/` | static INT8 experiment (see [quantization.md](quantization.md)) |

## References

Goodfellow, I. J., Erhan, D., Carrier, P. L., Courville, A., Mirza, M., Hamner, B., Cukierski, W., Tang, Y.,
Thaler, D., Lee, D.-H., Zhou, Y., Ramaiah, C., Feng, F., Li, R., Wang, X., Athanasakis, D., Shawe-Taylor, J.,
Milakov, M., Park, J., … Bengio, Y. (2013). *Challenges in representation learning: A report on three machine
learning contests*. arXiv. https://doi.org/10.48550/arXiv.1307.0414

Howard, A., Sandler, M., Chu, G., Chen, L.-C., Chen, B., Tan, M., Wang, W., Zhu, Y., Pang, R., Vasudevan, V.,
Le, Q. V., & Adam, H. (2019). Searching for MobileNetV3. In *Proceedings of the IEEE/CVF International
Conference on Computer Vision* (pp. 1314–1324). IEEE. https://doi.org/10.1109/ICCV.2019.00140

Wu, W., Peng, H., & Yu, S. (2023). YuNet: A tiny millisecond-level face detector. *Machine Intelligence
Research, 20*(5), 656–665. https://doi.org/10.1007/s11633-023-1423-y

Zhang, Z., Song, Y., & Qi, H. (2017). Age progression/regression by conditional adversarial autoencoder.
In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 5810–5818). IEEE.
https://doi.org/10.1109/CVPR.2017.463
