# Tiny ImageNet Checkpoint Comparison

| Property | 702K sparse model | 1.65M cross-reservoir model |
|---|---:|---:|
| Best validation accuracy | 41.45% | 39.39% |
| Checkpoint epoch | 65 | 61 |
| Parameters | 701,640 | 1,650,568 |
| Effective tau-neuron state | 512 | 2,048 |
| Shared recurrent matrix | 64 x 64 | 256 x 256 |
| Shared recurrent sparsity | 95.0% | 0.0% |
| Recurrent/cross parameters | 4,096 | 73,792 |
| Dense state-to-state equivalent | 262,144 | 4,194,304 |

The smaller model is the completed/pruned checkpoint. The larger model was still training when its epoch-61 `best.pt` was analyzed, so accuracy is not a final architecture comparison.

## Shared Matrix

| Metric | 702K sparse | 1.65M dense |
|---|---:|---:|
| Stable rank | 2.33 | 2.02 |
| Stable rank / width | 3.64% | 0.79% |
| Effective rank | 5.80 | 17.37 |
| Effective rank / width | 9.06% | 6.78% |
| First-mode energy | 43.0% | 49.4% |
| First-five energy | 89.1% | 64.2% |
| First-16 energy | 99.7% | 76.9% |
| Non-normality | 0.712 | 0.694 |
| Row-strength variation | 1.665 | 0.305 |
| Strong-edge reciprocity | 0.060 | 0.093 |

Both models learn a strong dominant direction and similarly high non-normality. Their residual structure differs:

- The sparse model puts **89.1%** of energy in five modes. Its 95% pruning leaves a very uneven, hub-like row-strength distribution (CV **1.66**).
- The dense model puts more energy in its first mode (**49.4%**) but retains a broader absolute tail: effective rank **17.4** versus **5.8**.
- Relative to width, the dense model is more compressed: effective rank is **6.8%** of width versus **9.1%**.

## Complete Tau-Neuron Operator

The smaller model repeats the same matrix independently across eight taus. The larger model additionally couples the taus through its learned cross-reservoir path.

| Tau-scaled metric | 702K: 512 units | 1.65M: 2048 units |
|---|---:|---:|
| Stable rank | 3.27 | 3.07 |
| Effective rank | 13.10 | 36.73 |
| Effective rank / state width | 2.56% | 1.79% |
| First-mode energy | 30.6% | 32.5% |
| First-five energy | 69.2% | 57.5% |

After tau scaling, both systems have almost the same stable rank and leading energy. The larger system nevertheless retains substantially more effective dimensions (**36.7** versus **13.1**), while using four times as many tau-neuron units.

## Cross-Reservoir Difference

The large model's rank-16 cross-neuron path has effective rank **1.08**, with **98.8%** of its energy in one mode. This creates a global communication channel absent from the smaller model.

## Interpretation

The 702K model obtains its compact dynamics through explicit magnitude pruning: a tiny number of recurrent edges and a handful of dominant modes. The 1.65M model remains fully dense but self-organizes into a dominant shared mode plus an almost rank-one cross-tau channel.

The larger architecture is therefore not simply a wider version of the smaller one. It trades explicit sparsity for dense low-dimensional coordination and global communication across timescales. Whether that improves generalization must be judged after its training schedule and official validation evaluation are complete.
