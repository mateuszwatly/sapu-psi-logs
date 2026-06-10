# best.pt Recurrent Structure

- Checkpoint: `/home/mike/sapu-psi-logs/best.pt`
- Epoch: **61**
- Best validation accuracy: **39.39%**
- Architecture: `res_cnn -> tpsapu_cross_reservoir -> both_transformer`
- Reservoirs: **8 taus x 256 neurons = 2048 tau-neuron units**

## Shared Recurrent Matrix

The learned `256 x 256` shared matrix has stable rank **2.02** and effective rank **17.37**. Its leading singular mode holds **49.4%** of all shared recurrent energy.

Entry-shuffled controls average stable rank **62.55**, effective rank **154.69**, and leading-mode energy **1.6%**.

The trained matrix is strongly non-normal: **0.694** versus **0.089** after shuffling. Its spectral radius is **7.86**, versus **3.58**.

Strong-edge reciprocity is **0.093**, close to the shuffled expectation **0.100**.

## Cross-Reservoir Path

The cross-neuron map is architecturally limited to rank 16, so rank <=16 alone is not a learned finding. However, training collapses nearly all permitted capacity into one mode:

- stable rank: **1.01**
- effective rank: **1.08**
- leading-mode energy: **98.8%**
- factor-shuffled leading-mode energy: **21.4%**

After the configured cross gain, this branch has Frobenius strength equal to **61.3%** of the repeated within-tau shared path.

The strongest receiving tau is **128** with absolute incoming mixing strength **5.618**. The strongest source tau is **1.1** with outgoing strength **2.380**.

The used tau-mixing matrix has effective rank **1.71**, versus **4.36** after shuffling its off-diagonal entries.

## Complete Recurrent-Current Operator

This combines the shared within-tau matrix and every cross-tau block into one `2048 x 2048` linear map from previous spikes to recurrent current.

- raw stable rank: **3.35** (structured null **160.40**)
- raw effective rank: **60.14** (structured null **1212.64**)
- raw leading-mode energy: **29.9%** (structured null **0.6%**)

After applying each target reservoir's `1/tau` update scaling:

- stable rank: **3.07** (structured null **78.13**)
- effective rank: **36.73** (structured null **354.30**)
- leading-mode energy: **32.5%** (structured null **1.3%**)

## Interpretation

The checkpoint has not learned a broadly distributed recurrent transform. It has organized both the shared topology and the cross-tau path around a few collective neuron directions. The cross path is especially close to rank one.

This does not mean only one neuron is active. A singular mode is a distributed combination of many neurons. It also does not establish dynamical instability: LIF leakage, thresholding, reset, detached spikes, and input drive all modify the nonlinear state evolution.
