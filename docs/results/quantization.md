# Quantisation

The goal of quantisation was to reduce the download size of the models for the phone without losing
accuracy. Two approaches were compared: static INT8 quantisation of weights and activations (W8A8) and
weight-only INT8 quantisation (W8A32).

## 1. Static INT8 quantisation

ONNX Runtime static quantisation was applied with per-channel INT8 weights, UINT8 activations and MinMax
calibration on 300 validation images. The experiment used the first age and gender model
(`runs/age_gender_cuda`, before age-balanced sampling) and the expression model; results on the full
test sets (`pc/static-int8-experiment/`):

| | FP32 | Static INT8 |
|---|---|---|
| Age MAE (years) | 5.09 | 16.40 |
| Gender accuracy | 94.2 % | 61.7 % |
| Expression accuracy, 7 classes | 68.4 % | 28.9 % |
| Happy vs. sad, forced choice | 91.0 % | 63.5 % |
| File size per model | 4.35 MB | 1.47 MB |

Static INT8 made both models unusable. Eight further quantiser settings were tried on a subset of 100
images (among them percentile calibration, per-tensor weights, signed activations, dynamic quantisation
and excluding the first layer and the squeeze-and-excitation blocks); none of them restored the
accuracy (age MAE 14–18 years instead of 4.8 on this subset).

## 2. Cause of the accuracy loss

`training/tools/quant_sensitivity.py` simulates the quantisation in PyTorch on 200 validation images
per model (`runs/<task>_cuda/quant_sensitivity.json`):

| Simulation | Age MAE (years) | Gender accuracy | Expression accuracy |
|---|---|---|---|
| FP32 | 5.66 | 94.5 % | 59.5 % |
| Weights INT8 per channel, activations FP32 | 5.69 | 94.5 % | 60.0 % |
| Weights INT8 per tensor, activations FP32 | 15.85 | 54.5 % | 27.5 % |
| All activations UINT8, weights FP32 | 30.71 | 47.0 % | 1.0 % |
| Only the output of the first layer UINT8 | 20.24 | – | 15.5 % |
| All activations UINT8 except the 10 most sensitive | 10.26 | 90.5 % | 20.5 % |

The weights can be quantised almost without loss if each output channel has its own scale. The
activations are the problem: the output of the first convolution reaches values up to 297 in the age
and gender model (120 in the expression model), while 99.99 % of the values lie between −0.4 and 56
(expression: 15). With one 8-bit scale per tensor, the step size is about 1.2, so the many small values
are rounded away. Clipping the rare large values also destroys the model, so they carry information.
Keeping the most sensitive activations in FP32 recovers only part of the accuracy, because large
values occur in many layers. The usual remedies for MobileNet-type networks are cross-layer
equalisation (Nagel et al., 2019) and quantisation-aware training (Krishnamoorthi, 2018); both would
require changes to the training and were not pursued.

## 3. Weight-only INT8

In the weight-only variant (`training/quantize_weights.py`), the weights of every convolution and
fully connected layer are stored as INT8 with one scale per output channel, and the activations stay
in FP32. ONNX Runtime converts the weights back to FP32 once when the model is loaded, so the
computation is the same as for the FP32 model. Results of the deployed models on the full test sets
(`runs/age_gender_cuda_balanced/evaluation.json`, `runs/expression_cuda/evaluation.json`):

| | FP32 | Weight-only INT8 |
|---|---|---|
| File size, age and gender / expression | 4.35 / 4.36 MB | 1.24 / 1.24 MB (−72 %) |
| Age MAE (years) | 5.08 | 5.15 |
| Gender accuracy | 94.1 % | 94.2 % |
| Age group adult vs. 60+ (18+), accuracy | 93.1 % | 92.9 % |
| Expression accuracy, 7 classes | 68.4 % | 68.3 % |
| Happy vs. sad, forced choice | 91.0 % | 90.7 % |

The latency is the same as for FP32. On the PC with one thread (`runs/int8_folding_check.json`, median)
the age and gender model took 1.48 ms with INT8 and 1.52 ms with FP32, the expression model 1.18 and
1.16 ms. Without the one-time conversion at load time, the weights would be converted in every run, and
the INT8 model would be 2–3 times slower (3.33 and 3.16 ms). On the iPhone the differences between
FP32 and INT8 were within the measurement spread (see [evaluation-table.md](evaluation-table.md),
section 2.4).

## 4. Conclusion

INT8 only makes inference faster if the runtime has fast INT8 kernels and the activations can be
quantised without large errors. Neither applies here: static INT8 was not faster than FP32 on the
iPhone with WebAssembly and was much slower with WebGPU, and its accuracy collapsed. For a web app the
main benefit of quantisation is the smaller download for the offline cache, and weight-only INT8
achieves this without a loss in accuracy or speed. The app therefore uses FP32 by default and offers
weight-only INT8 as an option.

## References

Krishnamoorthi, R. (2018). *Quantizing deep convolutional networks for efficient inference: A whitepaper*.
arXiv. https://doi.org/10.48550/arXiv.1806.08342

Nagel, M., van Baalen, M., Blankevoort, T., & Welling, M. (2019). Data-free quantization through weight
equalization and bias correction. In *Proceedings of the IEEE/CVF International Conference on Computer
Vision* (pp. 1325–1334). IEEE. https://doi.org/10.1109/ICCV.2019.00141
