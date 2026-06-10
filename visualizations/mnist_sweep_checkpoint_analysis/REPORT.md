# MNIST Sweep Checkpoint-Only Analysis

- Models: **42**
- Shared evaluation set: **500 images**
- No training or initialization checkpoints were used.
- Weight null: each recurrent matrix was repeatedly shuffled while preserving its exact entries.

## Main Findings

Across encoders with the same decoder, unbiased spike CKA is **0.466**, membrane CKA is **0.573**, and normalized temporal CKA is **0.258**.
After randomly breaking image correspondence, the respective baselines are **-0.000**, **-0.000**, and **0.000**.

Prediction agreement across encoders with the same decoder is **0.934**, while exact recurrent-weight correlation is **0.001**.

Changing the decoder while keeping the encoder produces more similar representations than changing the encoder while keeping the decoder: spike CKA is **0.659** versus **0.466**, and membrane CKA is **0.767** versus **0.573**.

Sorted firing-rate profiles remain highly similar across encoders (**0.973**) even though sample-wise spike CKA is only **0.466**. Similar activity histograms therefore do not imply that the same images recruit the population similarly.

Raw singular-spectrum cosine averages **0.945** across model pairs. The shuffled-weight control averages **0.995**. This control is required because sorted, nonnegative spectra have a high cosine even without shared topology.

Activation-derived neuron matching tests whether permutation symmetry explains the weight mismatch. Averaged across cross-encoder comparisons:

- matched activation correlation: **0.583**, versus **0.304** for random assignments
- recurrent correlation: **-0.001** before, **0.038** after alignment
- top-edge Jaccard: **0.051** before, **0.057** after alignment, and **0.053** for random assignments
- top-5 singular-subspace overlap: **0.112** before, **0.153** after alignment (random-subspace expectation: **0.078**)

## Recurrent Structure Versus Shuffled Weights

Shuffling preserves every recurrent weight value but destroys which neurons those values connect.

| Metric | Observed mean | Shuffled mean | Models above +1.96 shuffle SD | Models below -1.96 shuffle SD |
|---|---:|---:|---:|---:|
| Stable rank | 4.142 | 14.409 | 0.0% | 100.0% |
| Effective rank | 15.004 | 36.817 | 0.0% | 100.0% |
| Leading-mode energy | 0.290 | 0.080 | 100.0% | 0.0% |
| Spectral radius | 1.233 | 0.994 | 73.8% | 4.8% |
| Non-normality | 0.458 | 0.183 | 100.0% | 0.0% |
| Weight reciprocity | 0.036 | -0.000 | 31.0% | 0.0% |
| Row-strength variation | 0.199 | 0.135 | 64.3% | 2.4% |
| Strong-edge reciprocity | 0.098 | 0.099 | 11.9% | 9.5% |

The consistent low effective/stable rank, high leading-mode energy, and high non-normality are genuine organization beyond the weight histogram. Strong-edge reciprocity is not: its observed mean is approximately the shuffle expectation.

## Functional Readout

A nearest-class-centroid classifier was evaluated on half of each digit using centroids formed from the other half. Across checkpoints:

- final accuracy versus spike-centroid accuracy: Spearman **0.858**
- final accuracy versus membrane-centroid accuracy: Spearman **0.750**
- final accuracy versus digit-selective neuron fraction: Spearman **0.552**
- final accuracy versus effective recurrent rank: Spearman **0.479**
- final accuracy versus recurrent non-normality: Spearman **-0.403**

`spike_mlp` is the clearest decoder outlier: mean spike rate **0.215**, effective rank **8.30**, and accuracy **87.86%**. The other decoders average **0.014** spikes per unit-step.

## Topology Versus Function

Spearman correlations between exact recurrent-weight correlation and functional similarity:

- spike cka: **0.088**
- membrane cka: **0.071**
- temporal cka: **0.065**
- prediction agreement: **0.038**

## Encoder/Decoder Variance

| Metric | Encoder | Decoder | Residual/interaction | Largest component |
|---|---:|---:|---:|---|
| Accuracy | 41.0% | 24.7% | 34.3% | encoder |
| Spike rate | 1.0% | 94.9% | 4.1% | decoder |
| Spike participation | 4.4% | 90.0% | 5.6% | decoder |
| Digit-selective fraction | 57.8% | 19.9% | 22.4% | encoder |
| Spike centroid accuracy | 63.6% | 12.5% | 23.9% | encoder |
| Membrane centroid accuracy | 59.6% | 21.1% | 19.3% | encoder |
| Stable rank | 40.8% | 27.0% | 32.2% | encoder |
| Effective rank | 41.1% | 38.1% | 20.8% | encoder |
| Spectral radius | 2.6% | 64.3% | 33.2% | decoder |
| Non-normality | 49.1% | 15.9% | 35.0% | encoder |
| Weight reciprocity | 8.4% | 62.6% | 29.0% | decoder |
| Strong-edge reciprocity | 14.2% | 19.3% | 66.5% | residual |

## Largest Departures From Shuffled Topology

Positive z means the observed matrix metric is larger than entry-shuffled versions of the same matrix; negative means smaller.

| Model | Metric | z-score |
|---|---|---:|
| rows__spike_mlp | top1 energy | 119.91 |
| lif_2x2__spike_mlp | top1 energy | 115.90 |
| lif_2x2__spike_mlp | non normality | 115.30 |
| rows__spike_mlp | non normality | 114.06 |
| linear_patch__linear | non normality | 107.65 |
| linear_patch__linear | top1 energy | 99.97 |
| lif_2x2__both_mlp | top1 energy | 99.45 |
| cnn3__both_mlp | non normality | 93.45 |
| mlp_patch__spike_mlp | top1 energy | 93.13 |
| rows__both_mlp | non normality | 92.69 |
