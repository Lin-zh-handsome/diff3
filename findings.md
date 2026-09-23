# Findings

## Phase 0

- Server-side checkout: `/home/hanjinwei/p1/project/diffusion_policy`.
- Current HEAD: `ecc892b1fbeb590c1e415f4face4fadc1ca30f62` (2026-09-19, `experiment: publish baseline PEC comparison results`).
- The working tree is already extensively dirty with staged historical deletions and modifications, including `diffusion_policy/policy/diffusion_unet_lowdim_policy.py`. These changes predate this workpoint and must not be reset, removed, or included in a new commit.
- Existing `origin` points to `git@github.com:Lin-zh-handsome/diff.git`; the requested publication target is the separate `Lin-zh-handsome/diff2` repository. A separate remote will be used only when this task has a minimal, isolated commit ready to publish.
- The low-dimensional UNet policy and `ConditionalUnet1D` are present. Detailed interface audit and baseline execution remain pending.

## Phase 0 interface audit

- The selected policy is `DiffusionUnetLowdimPolicy`; its sampling loop is `conditional_sample`, and its training entry is `compute_loss`.
- External low-dimensional trajectory semantics are `[B,H,D]`. With the current global-conditioning configuration, the model input contains actions only: `[B,16,2]`; the observation condition is `[B,40]` from two `[20]` observation steps. `ConditionalUnet1D` transposes internally to channel-first `[B,D,H]` and returns `[B,H,D]`.
- Inference initializes Gaussian trajectory noise, calls the model once at each scheduler timestep, then calls the unchanged `DDPMScheduler.step(...).prev_sample`. It does not currently construct or pass a previous clean-action estimate.
- Training samples an independent timestep per batch item, calls `DDPMScheduler.add_noise`, predicts epsilon, and optimizes masked MSE against the sampled noise. This is epsilon prediction (epsilon-prediction, 噪声预测), not another parameterization.
- The workspace creates EMA (Exponential Moving Average, 指数移动平均) as a deep copy of the complete policy, trains `model`, and evaluates `ema_model` when enabled. Checkpoint loading ultimately calls PyTorch `load_state_dict` without `strict=False`, so it is strict by default.
- The repository default low-dimensional task is Push-T (`obs_dim=20`, `action_dim=2`), but no Push-T environment or task experiment will be used as this workpoint's core task.
- The staged historical diff removes an earlier, unrelated Self-Exposure implementation and restores the normal diffusion-loss path in the current file. It must remain isolated from this workpoint commit.

## Authorized baseline cleanup

- Per user authorization, historical base-policy modifications and experiment outputs may be removed from this server-side checkout.
- Restored only unrelated baseline source files that had been staged for deletion/modification: `.gitignore`, `diffusion_policy/common/replay_buffer.py`, `diffusion_policy/env/pusht/pusht_env.py`, and `diffusion_policy/dataset/mujoco_image_dataset.py`.
- Kept the intended removal of the old Self-Exposure code path: its dedicated configuration is deleted and `diffusion_unet_lowdim_policy.py` is reduced to the original diffusion-loss behavior. The remaining 1,388 staged deletions are authorized historical outputs/diagnostics.

## Phase 0 execution evidence

- Runtime: `robodiff` Conda environment on `CUDA_VISIBLE_DEVICES=0` (`cuda:0`).
- A real batch was loaded from the existing low-dimensional dataset and passed to the current policy's `compute_loss`; a two-step call to the actual `conditional_sample` loop was also run. No environment rollout was created.
- Result: parameter count `65,783,430`; loss `1.3082119226455688` (finite); input action `[1,16,2]`; input observation `[1,16,20]`; predicted action chunk `[1,16,2]`; executed action slice `[1,8,2]`; predicted values finite.
- This is a Phase 0 interface/baseline check only, not training evidence or a task-performance result.

## Phase 1 decision required

- Concatenating feedback directly into `ConditionalUnet1D.input_dim` would change the first network layer and conflicts with strict checkpoint loading. The two viable implementations are: (1) extend the input dimension and intentionally make existing checkpoints incompatible, or (2) add a separate feedback projection and fuse it after the existing input path, retaining strict loading for the original model weights. The user must select before Phase 1.

## Phase 1 selected design and benchmark

- User selected direct channel concatenation. The SC model will consume `[B,H,2D] = concat([A_k,F_k])` and emit `[B,H,D]`; old checkpoints are intentionally incompatible with the SC variant.
- The first non-Push-T benchmark is RoboMimic `Can` low-dimensional, configured as `task=can_lowdim` with 23-D observations and 7-D actions. It permits the same low-dimensional UNet policy path used by this workpoint.
- No non-Push-T dataset is currently present under `/home/hanjinwei/p1`; the configured Can dataset path is `data/robomimic/datasets/can/ph/low_dim.hdf5`. Code/interface validation can proceed, but closed-loop Can results require that dataset before training.

## Phase 1 implementation and evidence

- `ConditionalUnet1D` now separates `input_dim` from `output_dim`; without SC it preserves the existing behavior by defaulting `output_dim` to `input_dim`.
- With SC enabled, `DiffusionUnetLowdimPolicy` concatenates the noisy trajectory and feedback at the action channel dimension, so the model receives `[B,H,2D]` and emits epsilon prediction `[B,H,D]`.
- Training uses a no-gradient zero-feedback pre-forward, reconstructs the epsilon-based clean estimate, detaches its action channels into feedback, then performs the loss-bearing second forward. Inference initializes feedback to zero and forwards the reconstructed estimate to the next diffusion step without changing `scheduler.step`.
- The default low-dimensional training configuration now targets `can_lowdim`; with SC disabled it resolves to input/output `7/7`, and with SC enabled it resolves to `14/7`.
- Direct SC check on GPU with Can-shaped synthetic tensors: finite loss; feedback `[1,16,7]`; `feedback_requires_grad=False`; finite predicted chunk `[1,16,7]`; executed slice `[1,8,7]`. This verifies the Phase 1 data path, not benchmark effectiveness.

## Analog Bits training update

- SC training now uses one random timestep per batch and samples a 50% batch-level branch: either a no-gradient zero-feedback pre-forward followed by detached clean-action feedback, or zero feedback directly. Both branches perform the same final gradient-bearing epsilon-prediction loss.
- Inference self-conditioning, scheduler behavior, clean-estimate recovery, and feedback construction are unchanged.

## Next benchmark experiment

- First effectiveness comparison: RoboMimic Can low-dimensional (`ph` demonstrations), `original_dp` versus `sc_only`, using identical seed, data split, horizons, scheduler, optimizer, and training budget. Report closed-loop success rate and reward from the existing RoboMimic runner.
- Confirmation benchmark after Can: RoboMimic Square low-dimensional with the same protocol. The Can and Square HDF5 files are not presently available, so no effectiveness result should be claimed until the data is supplied or download authorization is given.
