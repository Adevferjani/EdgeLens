# Reproducibility

This document describes the environment, data, training configuration and evaluation procedure, and
lists the commands needed to repeat the experiments on a Windows PC. All commands are run in
PowerShell from the repository root.

## 1. Environment

| | Training and evaluation PC | Edge device |
|---|---|---|
| Device | Gigabyte G5 KF laptop | iPhone 15 |
| Processor | Intel Core i5-12500H (12 threads) | Apple A16 Bionic |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB | not used (WebAssembly backend) |
| Software | Windows 11, Python 3.12.7, PyTorch 2.15.0 with CUDA 13.2, torchvision 0.29.2, ONNX Runtime 1.31.1 | iOS 27.2 (beta) with Safari 27.2, ONNX Runtime Web 1.31.1 |

The hardware and software versions of each training run are also recorded in
`runs/*/train_summary.json`. `training/requirements-lock.txt` contains the exact versions of all
Python packages of the training environment; `training/requirements.txt` lists only the direct
dependencies without version numbers, for other systems. The web app has no build step; all
third-party code it uses is included in `pwa/vendor/` with a fixed version.

## 2. Set-up on Windows

Python 3.12 (64-bit) from python.org and, for GPU training, an NVIDIA driver that supports CUDA 13.2
(`nvidia-smi` shows the highest supported version) are required.

```powershell
git clone https://github.com/Adevferjani/EdgeLens.git
cd EdgeLens
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass   # allows activation in this window only
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r training/requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu132
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # must print True
```

Without an NVIDIA GPU, install `torch` and `torchvision` from PyPI, then
`pip install -r training/requirements.txt`, and use `--device cpu` for training. In every new
PowerShell window the two activation lines have to be run again.

## 3. Data

| Dataset | Source | Location | Partitions (train / validation / test) |
|---|---|---|---|
| UTKFace, 23,708 face images with age and gender labels | https://www.kaggle.com/datasets/jangedoo/utkface-new | `data/UTKFace/` (only the folder named `UTKFace` from the archive) | 18,966 / 2,371 / 2,371 |
| FER2013, 35,887 grayscale images, 7 expression classes | https://www.kaggle.com/datasets/deadskull7/fer2013 | `data/fer2013/` (CSV version) | 28,709 / 3,589 / 3,589 |

The UTKFace split is fixed in `splits/utkface_split.csv` (80 / 10 / 10 %, stratified by gender and
age band 0–17, 18–39, 40–59, 60+, seed 42); all scripts read this file, so every run uses the same
images. FER2013 is split by the dataset's own `Usage` column (Training, PublicTest as validation,
PrivateTest as test). The test partitions are used only for the final evaluation, never for training
or model selection. The `data/` folder is excluded from version control.

## 4. Training configuration

Both models are trained with the same settings (`training/train.py` defaults):

| Setting | Value |
|---|---|
| Network | MobileNetV3-Small, ImageNet pre-trained, all layers fine-tuned |
| Input size | 160 × 160 (age and gender), 112 × 112 (expression) |
| Optimiser | AdamW, learning rate 0.001, weight decay 0.01, batch size 64 |
| Schedule | 30 epochs, one warm-up epoch, cosine decay; early stopping after 8 epochs without improvement |
| Losses | age: L1 on age / 100; gender: binary cross-entropy; expression: cross-entropy with inverse square-root class weights; label smoothing 0.1 |
| Augmentation | random resized crop, horizontal flip, colour jitter; additionally random grayscale (age and gender) and ±10° rotation (expression) |
| Age-balanced sampling | age and gender model only: images are drawn with weight (1 / number of images in the age decade)^0.5 |
| Precision | mixed precision on the GPU, FP32 on the CPU |

**Random seeds.** Training uses seed 42 for Python, NumPy and PyTorch; crop calibration and test-set
subsets use a fixed NumPy seed. CPU runs are repeatable on the same machine. GPU runs are not
bit-exact because of mixed precision and non-deterministic cuDNN kernels, so a repeated GPU run gives
slightly different results. The checkpoints are not part of the repository; the deployed models are
the files in `pwa/models/`.

**Model selection.** After each epoch the model is evaluated on the validation split, and the epoch
with the best score is kept. For the expression model the score is the macro-F1. For the age and
gender model the score is −(age MAE / MAE of a constant median-age prediction + gender error / 0.5),
so both tasks count equally. This criterion replaced the validation loss, which was dominated by the
gender cross-entropy and selected an early epoch although the age error was still decreasing.

## 5. Export and quantisation

`training/calibrate_crop.py` measures the face framing of the training images (see
[architecture.md](architecture.md), section 2) and stores it in `runs/crop_calibration.json`.
`training/export_onnx.py` exports the selected checkpoint to ONNX (opset 17), checks that the ONNX
outputs match PyTorch (largest difference of the deployed models: 6.5 × 10⁻⁵ years for the age and
below 10⁻⁶ for the probabilities), and writes two variants: FP32 and weight-only INT8 (W8A32, symmetric per-channel
INT8 weights, FP32 activations). The model card stores the file sizes, inputs, outputs, labels, the
crop calibration and the PyTorch-to-ONNX difference. Static INT8 quantisation is available as an option
(`--int8-mode static`) but is not used (see [results/quantization.md](results/quantization.md)).

## 6. Evaluation

`training/evaluate.py` computes the test metrics of the PyTorch checkpoint and of both ONNX files, and
measures the latency on the PC at batch size 1 (20 warm-up runs, 200 timed runs). The metrics are the
age MAE, the accuracy of the age group adult vs. 60+ for people aged 18 or older, gender accuracy and
macro-F1, expression accuracy, macro-F1 and per-class F1, and the happy-vs-sad accuracy.

On the iPhone, the same models are evaluated in the app's Lab: the test sets packed by
`training/tools/export_testset.py`, a latency benchmark, and the 20 own test photos. The measurement
procedure and all results are described in [results/evaluation-table.md](results/evaluation-table.md).

## 7. Commands

| Step | Command | Time on the PC above |
|---|---|---|
| Quick check with generated data | `python training/tools/make_dummy_data.py` and `python training/train.py --task age_gender --data data_dummy/UTKFace --epochs 1 --no-pretrained --workers 0 --out runs/smoke --split-dir runs/smoke` | about 2 min |
| Train age and gender (GPU, deployed) | `python training/train.py --task age_gender --data data/UTKFace --device cuda --balance-ages --out runs/age_gender_cuda_balanced` | 7.5 min |
| Train expression (GPU, deployed) | `python training/train.py --task expression --data data/fer2013 --device cuda` | 7.4 min |
| Train age and gender (CPU) | `python training/train.py --task age_gender --data data/UTKFace --device cpu --balance-ages` | 49.3 min |
| Train expression (CPU) | `python training/train.py --task expression --data data/fer2013 --device cpu` | 46.1 min |
| Calibrate, export, evaluate | `python training/finalize.py --utkface data/UTKFace --fer data/fer2013 --age-gender-run runs/age_gender_cuda_balanced` | not recorded |
| Test-set package for the iPhone | `python training/tools/export_testset.py` | not recorded |

`finalize.py` runs the crop calibration (only if `runs/crop_calibration.json` does not exist yet),
both exports, the evaluation of all four runs and `pwa/tools/build_sw.py`, which updates the offline
file list of the app. The individual steps are:

```powershell
python training/calibrate_crop.py --utkface data/UTKFace --fer data/fer2013
python training/export_onnx.py --run runs/age_gender_cuda_balanced
python training/export_onnx.py --run runs/expression_cuda
python training/evaluate.py --run runs/age_gender_cuda_balanced --data data/UTKFace --onnx
python training/evaluate.py --run runs/expression_cuda --data data/fer2013 --onnx
python training/evaluate.py --run runs/age_gender_cpu --data data/UTKFace
python training/evaluate.py --run runs/expression_cpu --data data/fer2013
python pwa/tools/build_sw.py
```

`runs/age_gender_cuda` contains an earlier GPU run without age-balanced sampling, kept for comparison
(same command without `--balance-ages` and `--out`).

**Expected results** (test partitions; full tables in
[results/evaluation-table.md](results/evaluation-table.md)):

| Age and gender run | Age MAE (years) | Gender accuracy | 60+ recall (age 18+) |
|---|---|---|---|
| `age_gender_cuda_balanced` (deployed) | 5.08 | 94.1 % | 69.5 % |
| `age_gender_cpu` | 5.29 | 94.2 % | 64.7 % |

| Expression run | Accuracy | Macro-F1 |
|---|---|---|
| `expression_cuda` (deployed) | 68.4 % | 0.683 |
| `expression_cpu` | 67.9 % | 0.671 |

## 8. Running, testing and deploying the app

The app is served locally with `python -m http.server 8000 -d pwa` and opened at
http://localhost:8000 in Chrome or Edge. Automated checks:

```powershell
pip install pytest ruff
python -m pytest tests/python -q      # training code and deployed models, no datasets needed
ruff check .                          # Python code style
cd tests/web; npm install; npx playwright install chromium; node smoke.mjs   # app in headless Chromium
```

The browser test starts the app with a simulated camera and checks start-up, model loading, settings,
the service worker and offline use. The iPhone was tested by hand.

The app is deployed with GitHub Pages: in the repository settings, Pages → Source is set to GitHub
Actions, and every push to `main` that changes `pwa/` publishes the folder
(`.github/workflows/pages.yml`).

## 9. Troubleshooting on Windows

| Problem | Solution |
|---|---|
| `Activate.ps1 cannot be loaded because running scripts is disabled` | run the `Set-ExecutionPolicy -Scope Process ...` line in the same window, then activate again |
| `torch.cuda.is_available()` prints `False` | update the NVIDIA driver, then reinstall PyTorch with `pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu132` |
| Training stops or hangs at the start of an epoch | add `--workers 0` (data loading in the main process, slower) |
| Error that the split does not match the images | `--data` points to the wrong folder; use the folder named `UTKFace` |
