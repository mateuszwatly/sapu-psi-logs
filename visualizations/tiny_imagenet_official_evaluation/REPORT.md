# Tiny ImageNet Official-Split Evaluation

Both checkpoints were evaluated with identical deterministic preprocessing:
`Resize(73) -> CenterCrop(64) -> ImageNet normalization`.

| Checkpoint | Parameters | Epoch | Training-split best validation |
|---|---:|---:|---:|
| 702K | 701,640 | 65 | 41.45% |
| 2048-unit | 1,650,568 | 61 | 39.39% |

## Validation (4,909 images)

| Model | Loss | Top-1 | Top-5 | Top-10 | MRR | Mean rank | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 702K | 2.6393 | 41.62% | 67.65% | 77.06% | 0.537 | 10.89 | 0.122 | 0.738 |
| 2048-unit | 2.6921 | 39.21% | 65.35% | 75.58% | 0.516 | 11.73 | 0.099 | 0.751 |
| ensemble | 2.4317 | 43.86% | 68.89% | 78.55% | 0.556 | 10.01 | 0.037 | 0.698 |

- prediction agreement: **47.81%**
- correct only for 702K: **10.59%**
- correct only for 2048-unit: **8.19%**
- oracle accuracy if either model is correct: **49.81%**

## Test (4,923 images)

| Model | Loss | Top-1 | Top-5 | Top-10 | MRR | Mean rank | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 702K | 2.6799 | 41.05% | 66.46% | 76.01% | 0.530 | 10.83 | 0.124 | 0.746 |
| 2048-unit | 2.7860 | 38.11% | 63.29% | 74.06% | 0.501 | 12.05 | 0.112 | 0.771 |
| ensemble | 2.4934 | 42.72% | 67.97% | 77.39% | 0.544 | 10.19 | 0.042 | 0.710 |

- prediction agreement: **46.70%**
- correct only for 702K: **10.95%**
- correct only for 2048-unit: **8.00%**
- oracle accuracy if either model is correct: **49.06%**

## Combined Unseen Splits (9,832 images)

| Model | Loss | Top-1 | Top-5 | Top-10 | Macro Top-1 | MRR | Mean rank | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 702K | 2.6597 | 41.33% | 67.06% | 76.54% | 41.24% | 0.533 | 10.86 | 0.123 |
| 2048-unit | 2.7391 | 38.66% | 64.32% | 74.82% | 38.57% | 0.509 | 11.89 | 0.105 |
| ensemble | 2.4626 | 43.29% | 68.43% | 77.97% | 43.18% | 0.550 | 10.10 | 0.039 |

Combined prediction agreement is **47.25%**. The models' complementary correct predictions produce an oracle top-1 ceiling of **49.43%**.

The ensemble is the arithmetic mean of the two softmax probability vectors; it is not trained or calibrated on these splits.
