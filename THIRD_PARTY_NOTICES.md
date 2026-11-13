# Third-party components

| Component | Used for | Licence | Source |
|---|---|---|---|
| ONNX Runtime Web 1.31.1 | running the models in the browser (WebGPU / WebAssembly) | MIT – `pwa/vendor/ort/LICENSE` | https://github.com/microsoft/onnxruntime |
| fflate 0.8.3 (`pwa/vendor/fflate.mjs`) | reading the test-set ZIP file in the app | MIT – `pwa/vendor/LICENSE-fflate.txt` | https://github.com/101arrowz/fflate |
| YuNet face detector (`face_detection_yunet_2026may.onnx`) | face detection + 5 landmarks | MIT – `pwa/models/LICENSE-yunet.txt` | https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet |
| Instrument Sans (variable, Latin subset) | interface typeface | SIL Open Font License 1.1 – `pwa/vendor/fonts/OFL-instrument-sans.txt` | https://github.com/Instrument/instrument-sans (via Fontsource) |
| Martian Mono (variable, Latin subset) | numbers and labels | SIL Open Font License 1.1 – `pwa/vendor/fonts/OFL-martian-mono.txt` | https://github.com/evilmartians/mono (via Fontsource) |
| MobileNetV3-Small ImageNet weights (torchvision) | starting point for fine-tuning | BSD-3-Clause (torchvision) | https://github.com/pytorch/vision |
| PyTorch, torchvision, ONNX, onnxruntime, scikit-learn, NumPy, Pillow, OpenCV | training and export | BSD / MIT / Apache-2.0 | see the respective projects |

## Datasets (not included in this repository)

| Dataset | Terms |
|---|---|
| UTKFace (Zhang et al., 2017) | non-commercial research and educational use only |
| FER2013 (Goodfellow et al., 2013) | released for the ICML 2013 challenge; research use |

## References

Goodfellow, I. J., Erhan, D., Carrier, P. L., Courville, A., Mirza, M., Hamner, B., Cukierski, W., Tang, Y.,
Thaler, D., Lee, D.-H., Zhou, Y., Ramaiah, C., Feng, F., Li, R., Wang, X., Athanasakis, D., Shawe-Taylor, J.,
Milakov, M., Park, J., … Bengio, Y. (2013). *Challenges in representation learning: A report on three machine
learning contests*. arXiv. https://doi.org/10.48550/arXiv.1307.0414

Zhang, Z., Song, Y., & Qi, H. (2017). Age progression/regression by conditional adversarial autoencoder.
In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 5810–5818). IEEE.
https://doi.org/10.1109/CVPR.2017.463
