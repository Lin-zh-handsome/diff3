# Findings & Decisions

## Requirements
- Official RoboTwin 2.0 CNN DP in XPolicyLab, Vanilla Self-Conditioning only.
- 50% detached same-timestep self-conditioning prepass; 50% zero feedback; all batches use the original epsilon MSE.
- Preserve scheduler.step and official DDPM inference loop; detached clean action estimate becomes next step feedback.
- Keep original official training/evaluation config and scripts, use Aloha-AgileX, demo_clean, 50 demos, seed, 100 episodes.
- First task beat_block_hammer; next tasks only after first completes. Following the user's clarification, fetch only this task's official archive through a domestic HF mirror if absent; do not switch tasks or data.

## Research Findings
- Official RoboTwin source is `https://github.com/RoboTwin-Platform/RoboTwin`, current main checkout `ea8b211`; it pins XPolicyLab at `fa431ecd893ee706883e64fe5fe1464ec8cd928d`.
- Server checkout path: `/home/hanjinwei/p1/project/RoboTwin`.
- Server has an existing Clash HTTP proxy on localhost port 7890 and SOCKS listener on 7891. Use HTTP proxy for GitHub code transfers; use a domestic HF mirror for large task data.
- Official data layout documented by RoboTwin: `data/demo_clean/<task>/aloha_agilex/data/`; the DP processing stage converts this into its Zarr input.
- Server has two available RTX 3090 GPUs (indices 0 and 1 at discovery time).
- The official downloader `scripts/download_xpolicylab_data.sh` accepts selected task names and fetches `dataset/<task>/demo_clean.zip` from `TianxingChen/RoboTwin2.0`, then normalizes it into the official XPolicyLab layout.
- The pinned DP source, config, and official processing/training/evaluation launchers are present under `XPolicyLab/policy/DP/`.
- Official mirror download of `dataset/beat_block_hammer/demo_clean.zip` completed (225 MB) and extracted through the official script to `data/demo_clean/beat_block_hammer/aloha_agilex/`; its `data/` subfolder contains exactly 50 HDF5 episodes.
- The official config already matches requested training dimensions and hyperparameters: horizon 8, 3 obs, 6 action steps, 100 diffusion timesteps, cosine-squared schedule, fixed-small variance, epsilon prediction, `[256,512,1024]` U-Net widths, kernel 5, 8 groups, scale prediction, batch 128, AdamW 1e-4, EMA, 600 epochs, seed 42.
- Official Hydra resolution adds `agent_pos.shape=[14]` with the `+` override because the pinned default YAML does not declare this observation key; its encoder defaults the added entry to low-dimensional input. The resolved SC policy has U-Net input/output dimensions 28/14, 85,646,222 U-Net parameters and 96,822,734 total policy parameters.
- `robodiff` has Torch 1.12.1 while the official DP installer requests Torch 2.4.1; created a separate environment at `/home/hanjinwei/p1/envs/robotwin-sc` by cloning it and will install/update dependencies there.
- The DP processor expects `data/<bench>/<task>/<env_cfg>/data`; the archive is at `data/demo_clean/<task>/aloha_agilex/data`, so create a symlink under `data/RoboTwin/<task>/aloha_agilex` for the official processor while retaining the canonical source.
- Official processing completed for all 50 episodes and created a 1.1 GB Zarr dataset. The separate Python 3.10 environment is `/home/hanjinwei/p1/envs/robotwin-sc`; `h5py 3.10`, the DP installer-pinned `numpy 1.23.5`, and `opencv-python-headless 4.9.0.80` are required for the downloaded HDF5 image format and Torch array interoperability here.
- The training worker is `/home/hanjinwei/p1/project/RoboTwin/XPolicyLab/policy/DP/train.py`, launched with the resolved config saved as `artifacts/robotwin_self_condition/beat_block_hammer/config/resolved_robot_dp.yaml`. Workspace initialization is commented out in the pinned `robotworkspace.py`, so despite `logging.mode=online`, no WandB run is created by this training code.

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Append feedback on the final action channel dimension | Existing DP trajectory representation is batch/time/action_dim and must preserve D-dimensional epsilon output. |
| Add `output_dim=None` to ConditionalUnet1D, defaulting to input_dim | Existing checkpoints/configs keep the original output shape when self-conditioning is disabled. |
| Use one Bernoulli choice per training batch, matching existing DP policy structure | User specified 50% self-conditioning; both branches keep one normal denoising loss. |
| Record external logs/checkpoints/eval under `artifacts/robotwin_self_condition/<task>/` | Matches requested result contract without modifying official training data locations. |
| Download only `beat_block_hammer/demo_clean.zip` via domestic HF mirror if needed | User clarified large data should use a domestic source; official downloader normalizes it into the expected path. |
| Keep existing `robodiff` intact and install DP requirements in a new p1-local conda prefix | Existing Torch version is older than the RoboTwin DP install spec. |
| Preserve the upstream YAML, using a Hydra `+` override for `agent_pos` | The official launcher refers to `agent_pos` but the pinned default YAML omits that key; adding its shape retains the official model/data contract without broad configuration edits. |
| Pin pandas 1.5.3 in the isolated env | The newer pandas wheel required NumPy 2 internals while the official DP setup pins NumPy 1.23.5; pandas 1.5.3 restores official workspace imports without changing the source project. |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| RoboTwin checkout absent in local project listing and `/home/hanjinwei/p1/project` | Prepared official main checkout under the requested p1 project folder. |
| Direct GitHub clone was slow and left the XPolicyLab submodule unfinished | Restarted pinned commit fetch through already running local Clash HTTP proxy. |
| Server does not have ripgrep (`rg`) installed | Used `find` and `sed` for remote source inspection. |
| System Python did not include pip | Re-ran the official data downloader with the existing environment Python on PATH and the HF mirror endpoint. |
| A PowerShell command expanded a remote shell variable and failed to resolve the intended pip path | Re-ran with explicit absolute paths. |

## Resources
- RoboTwin official DP instructions: https://robotwin-platform.github.io/doc/usage/DP.html
- RoboTwin official repo: https://github.com/RoboTwin-Platform/RoboTwin
- XPolicyLab repo: https://github.com/XPolicyLab/XPolicyLab
