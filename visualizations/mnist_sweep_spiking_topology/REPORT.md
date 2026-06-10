# MNIST Sweep Spiking and Topology Analysis

- Checkpoints analyzed: **42**
- Evaluation set: **500 balanced MNIST test images** (50 per digit)
- Shared reservoir: **64 neuron positions × 3 taus = 192 spiking units**
- Strong-edge topology: top **10%** absolute off-diagonal recurrent weights

## Main Answer

**No: the models do not converge to the same recurrent neuron-to-neuron topology. They do converge to a similar statistical firing regime.**

Across different encoders with the same decoder, exact neuron firing-profile correlation averages **0.665**. After sorting neurons within each tau (permutation-insensitive), it is **0.973**.

Exact recurrent-weight correlation across encoders averages **0.001**. Strong-edge Jaccard overlap averages **0.051**, versus a random-overlap baseline of about **0.053**.

These metrics separate three questions: exact neuron identity, firing-rate distribution regardless of neuron permutation, and learned recurrent wiring.

The decoder creates the largest firing-regime split. `spike_mlp` averages **0.215** spikes per unit-step, versus **0.014** for all other decoders. Its per-tau rates are **0.422**, **0.180**, and **0.043** for tau 1.1, 8, and 64.

## Encoder Summary

| Encoder | Balanced accuracy | Mean spike rate | Activity similarity across decoders | Recurrent similarity across decoders |
|---|---:|---:|---:|---:|
| linear_patch | 96.10% | 0.057 | 0.764 | 0.017 |
| mlp_patch | 95.30% | 0.058 | 0.723 | 0.008 |
| lif_2x2 | 84.97% | 0.050 | 0.557 | -0.004 |
| cnn2 | 98.17% | 0.034 | 0.660 | 0.005 |
| cnn3 | 99.10% | 0.043 | 0.709 | 0.008 |
| res_cnn | 98.93% | 0.050 | 0.723 | 0.000 |
| rows | 97.07% | 0.042 | 0.650 | 0.007 |

## Decoder Summary

| Decoder | Balanced accuracy | Mean spike rate | Tau 1.1 | Tau 8 | Tau 64 | Exact activity across encoders | Sorted activity across encoders | Recurrent correlation | Top-edge Jaccard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| linear | 96.80% | 0.013 | 0.039 | 0.001 | 0.000 | 0.678 | 0.989 | -0.005 | 0.048 |
| membrane_mlp | 97.66% | 0.016 | 0.047 | 0.001 | 0.000 | 0.701 | 0.992 | 0.000 | 0.046 |
| spike_mlp | 87.86% | 0.215 | 0.422 | 0.180 | 0.043 | 0.723 | 0.964 | 0.003 | 0.055 |
| both_mlp | 97.57% | 0.017 | 0.049 | 0.001 | 0.000 | 0.666 | 0.988 | 0.005 | 0.049 |
| all_state_mlp | 97.86% | 0.011 | 0.034 | 0.000 | 0.000 | 0.657 | 0.988 | -0.001 | 0.051 |
| lif_count | 96.23% | 0.014 | 0.041 | 0.000 | 0.000 | 0.565 | 0.918 | 0.005 | 0.055 |

## Pairwise Group Means

| Metric | Same encoder | Same decoder | Different encoder and decoder |
|---|---:|---:|---:|
| activity correlation | 0.684 | 0.665 | 0.653 |
| digit activity correlation | 0.535 | 0.503 | 0.489 |
| sorted activity correlation | 0.943 | 0.973 | 0.937 |
| recurrent correlation | 0.006 | 0.001 | 0.001 |
| recurrent spectrum cosine | 0.962 | 0.953 | 0.940 |
| top edge jaccard | 0.056 | 0.051 | 0.053 |

## Interpretation Notes

- Same-encoder comparisons vary the decoder while preserving encoder structure.
- Same-decoder comparisons vary the encoder and directly address the question.
- Exact neuron correlations assume neuron index 17 in one model corresponds to neuron index 17 in another.
- Sorted activity and recurrent singular-value comparisons are insensitive to neuron permutations, but they do not prove identical wiring.
- Models with the same encoder were initialized with the same backbone seed before decoder construction. Different encoders consume different random numbers before backbone initialization, so cross-encoder weight similarity reflects both initialization and training.
