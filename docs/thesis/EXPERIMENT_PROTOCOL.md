# Thesis Experiment Protocol Entry

- Chapter 3: fixed demonstration-count efficiency; nested 10/25/50/100 trajectory subsets, initial 25/50 and three training seeds.
- Chapter 4: fixed interaction-budget held-out-scenario performance; separate RL scenarios, validation, and frozen final tests.
- A: gated, detached self-conditioned action denoising with optional bounded feedback perturbation.
- B: real-action-conditioned future EMA-visual and normalized-proprioceptive prediction; future observations are training targets only.
- C: frozen-base, normalized residual SAC with aligned execution/prediction error and grouped bounded residuals.
- D: fixed-pool curriculum from success, progress, and first-unmet-stage failure frequency; it is not transition replay priority.
- Excluded primary directions: learnable variance, consistency/few-step acceleration, wavelets, Mamba, bidirectional cross-modal attention, multiscale/depthwise denoising, temporal-correlated noise.
- S0 baseline check limit: one real-data check of at most 200 gradient steps or 20 minutes; it is not a performance claim.
