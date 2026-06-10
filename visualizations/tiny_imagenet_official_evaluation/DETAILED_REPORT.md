# Detailed Tiny ImageNet Evaluation

This report uses all **9,832** labeled held-out images from the clean dataset's
validation and test splits. Both checkpoints were evaluated with deterministic
`Resize(73) -> CenterCrop(64) -> ImageNet normalization` preprocessing.

## Main Results

Accuracy cells show the estimate followed by a 95% Wilson confidence interval.

| Model | Parameters | Top-1 | Top-5 | Top-10 | NLL [95% CI] |
|---|---:|---:|---:|---:|---:|
| 702K | 701,640 | 41.33% [40.36%, 42.31%] | 67.06% [66.12%, 67.98%] | 76.54% [75.69%, 77.36%] | 2.6597 [2.6083, 2.7110] |
| 2048-unit | 1,650,568 | 38.66% [37.70%, 39.63%] | 64.32% [63.37%, 65.26%] | 74.82% [73.95%, 75.67%] | 2.7391 [2.6905, 2.7877] |
| Ensemble | 2,352,208 total | 43.29% [42.31%, 44.27%] | 68.43% [67.50%, 69.34%] | 77.97% [77.14%, 78.78%] | 2.4626 [2.4175, 2.5076] |

The 702K model leads the 2048-unit model by
**2.67 percentage points** in top-1
(95% paired CI 1.82 to
3.53, exact McNemar
`p=1.11e-09`). The simple untrained probability
ensemble adds **1.95 points**
over the 702K model (95% paired CI
1.36 to
2.54,
`p=8.8e-11`).

## Complementarity

- Same top-1 prediction: **47.25%**
- Correct only for 702K: **10.77%**
- Correct only for 2048-unit: **8.10%**
- Either model correct (oracle): **49.43%**
- Simple ensemble top-1: **43.29%**
- Remaining oracle-to-ensemble gap: **6.14 points**

The larger model is weaker alone but still uniquely solves about 8.1% of all
images. That disagreement is why averaging the probability vectors improves
accuracy, NLL, Brier score, and calibration.

## Ranking And Calibration

| Model | Top-2 | Top-3 | Top-20 | Top-50 | Mean rank | Rank p90 | MRR | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 702K | 53.02% | 59.21% | 85.77% | 94.48% | 10.86 | 30 | 0.533 | 0.123 | 0.742 |
| 2048-unit | 50.51% | 56.67% | 84.22% | 93.60% | 11.89 | 35 | 0.509 | 0.105 | 0.761 |
| Ensemble | 54.53% | 60.99% | 86.90% | 95.04% | 10.10 | 27 | 0.550 | 0.039 | 0.704 |

## Confidence Filtering

Accuracy after retaining only the most confident predictions:

| Coverage | 702K | 2048-unit | Ensemble |
|---:|---:|---:|---:|
| 100% | 41.33% | 38.66% | 43.29% |
| 75% | 50.57% | 47.65% | 53.25% |
| 50% | 63.22% | 59.58% | 66.82% |
| 25% | 82.59% | 77.95% | 84.99% |
| 10% | 94.51% | 94.51% | 96.54% |

## Classes Favoring The 702K Model

| Class | 702K | 2048-unit | Ensemble | Difference |
|---|---:|---:|---:|---:|
| sunglasses, dark glasses, shades | 34.8% | 8.7% | 30.4% | +26.1 pp |
| orangutan, orang, orangutang, Pongo pygmaeus | 60.4% | 35.4% | 52.1% | +25.0 pp |
| tarantula | 48.0% | 26.0% | 44.0% | +22.0 pp |
| coral reef | 62.0% | 42.0% | 52.0% | +20.0 pp |
| golden retriever | 44.0% | 26.0% | 36.0% | +18.0 pp |
| tractor | 56.2% | 39.6% | 52.1% | +16.7 pp |
| black stork, Ciconia nigra | 61.2% | 44.9% | 61.2% | +16.3 pp |
| tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui | 40.0% | 24.0% | 32.0% | +16.0 pp |
| snorkel | 46.0% | 30.0% | 40.0% | +16.0 pp |
| butcher shop, meat market | 58.0% | 42.0% | 56.0% | +16.0 pp |

## Classes Favoring The 2048-Unit Model

| Class | 702K | 2048-unit | Ensemble | Difference |
|---|---:|---:|---:|---:|
| brown bear, bruin, Ursus arctos | 46.9% | 61.2% | 67.3% | -14.3 pp |
| backpack, back pack, knapsack, packsack, rucksack, haversack | 34.0% | 46.0% | 42.0% | -12.0 pp |
| lion, king of beasts, Panthera leo | 50.0% | 60.0% | 56.0% | -10.0 pp |
| cockroach, roach | 26.0% | 36.0% | 38.0% | -10.0 pp |
| projectile, missile | 18.8% | 27.1% | 25.0% | -8.3 pp |
| fly | 42.9% | 51.0% | 53.1% | -8.2 pp |
| plunger, plumber's helper | 0.0% | 8.2% | 4.1% | -8.2 pp |
| brain coral | 44.0% | 52.0% | 48.0% | -8.0 pp |
| sea cucumber, holothurian | 46.0% | 54.0% | 54.0% | -8.0 pp |
| viaduct | 50.0% | 58.0% | 58.0% | -8.0 pp |

## Largest Ensemble Gains Over 702K

| Class | 702K | 2048-unit | Ensemble | Difference |
|---|---:|---:|---:|---:|
| brown bear, bruin, Ursus arctos | 46.9% | 61.2% | 67.3% | +20.4 pp |
| cockroach, roach | 26.0% | 36.0% | 38.0% | +12.0 pp |
| altar | 50.0% | 56.0% | 62.0% | +12.0 pp |
| baboon | 24.0% | 30.0% | 36.0% | +12.0 pp |
| fly | 42.9% | 51.0% | 53.1% | +10.2 pp |
| torch | 46.0% | 50.0% | 56.0% | +10.0 pp |
| beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon | 26.0% | 30.0% | 36.0% | +10.0 pp |
| European fire salamander, Salamandra salamandra | 72.0% | 78.0% | 82.0% | +10.0 pp |
| triumphal arch | 68.8% | 72.9% | 77.1% | +8.3 pp |
| crane | 29.2% | 33.3% | 37.5% | +8.3 pp |

## Runtime On This CPU

- 702K: **110.4 images/s**
- 2048-unit: **12.4 images/s**
- The 702K checkpoint was **8.9x faster** in this run.

GPU throughput was not measured because CUDA was unavailable.

## Files

- `accuracy_confidence_intervals.csv`: top-1/5/10 counts and Wilson intervals
- `paired_significance.csv`: paired accuracy differences and exact McNemar tests
- `loss_confidence_intervals.csv`: NLL intervals and paired loss tests
- `per_class_differences.csv`: all per-class model metrics and deltas
- `combined_selective_accuracy.csv`: confidence/coverage tradeoffs
- `statistical_summary.png`: compact visual comparison
- split directories: logits, every prediction, errors, confusion matrices,
  calibration bins, selective accuracy, and per-class metrics
