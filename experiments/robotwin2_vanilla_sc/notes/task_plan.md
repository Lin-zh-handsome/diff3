# Task Plan: RoboTwin 2.0 DP Vanilla Self-Conditioning

## Goal
Modify the official XPolicyLab DP minimally, train and evaluate `beat_block_hammer` on the official RoboTwin protocol, then run the remaining requested tasks after the first task is complete.

## Next Step
Download and process `demo_clean` for `handover_block`, `stack_bowls_three`, and `pick_dual_bottles`, then train/evaluate each through the official XPolicyLab entry points with seed 42 and the same SC configuration.

## Current Phase
Phase 4

## Phases

### Phase 1: Requirements and repository discovery
- [x] Locate official RoboTwin 2.0 checkout under the requested p1 workspace.
- [x] Inspect pinned XPolicyLab DP source, official launch scripts, and expected data path.
- [x] Record current GPU state and official data download route.
- **Status:** complete

### Phase 2: Minimal Self-Conditioning implementation
- [x] Separate ConditionalUnet1D input_dim and output_dim.
- [x] Add the config toggle with vanilla 50/50 same-timestep training.
- [x] Preserve official scheduler.step inference and feed detached x0 estimates between denoising steps.
- **Status:** complete

### Phase 3: Official beat_block_hammer processing and training
- [x] Use only demo_clean / Aloha-AgileX / 50 demonstrations / seed 42.
- [x] Prepare separate p1-local Conda environment with official DP dependencies and fit raw data to the official DP processor path.
- [x] Run official processing entry point.
- [x] Launch official training entry point on GPU 0 (Python PID 1855894).
- [x] Record process, logs, resolved config, parameter count, and final checkpoint.
- **Status:** complete

### Phase 4: Official evaluation and result record
- [x] Launch official demo_clean evaluation for 100 episodes on GPU 0 (policy) and GPU 1 (simulation).
- [x] Save task, seed, demos, epoch, success count/rate, best/final checkpoint, parameter count under artifacts/robotwin_self_condition/beat_block_hammer/.
- [x] Archive/link all 100 official evaluation episode videos and the result summary.
- **Status:** complete

### Phase 5: Remaining requested tasks
- [ ] Repeat the official processing, training, and 100-episode evaluation for handover_block, stack_bowls_three, and pick_dual_bottles.
- [ ] Record each task's result.json and artifacts.
- **Status:** in_progress; official data download started (PID 2235502).

### Phase 6: Delivery
- [ ] Summarize changed files, method locations, commands, GPU/PIDs, outputs, config, and parameter count.
- **Status:** pending

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Use RoboTwin-Platform/RoboTwin main and its pinned XPolicyLab submodule | No checkout of the named repository was present in the local workspace or server p1/project; official source is the only supported checkout identified. |
| Route repository code fetches through the server's existing Clash HTTP proxy | User explicitly directed proxy use for code; large benchmark data will not be routed through it. |
| Fetch only beat_block_hammer/demo_clean from the official HF dataset through a domestic mirror if needed | User clarified large task data may use a domestic source; the official repository's downloader writes the exact XPolicyLab path. |
| Use a p1-local cloned environment rather than mutate robodiff | Existing environment has Torch 1.12.1; isolated prefix receives official DP dependencies. |

## Errors Encountered
| Error | Resolution |
|-------|------------|
| GitHub clone without proxy was slow and left submodule initialization incomplete | Fetched the pinned XPolicyLab commit through existing proxy at 127.0.0.1:7890; verified the DP source file materialized. |
| Server lacks `rg` | Used `find` and `sed` for source and launcher inspection. |
| System Python lacked pip when running the official data downloader | Used existing `robodiff` Python via PATH; official domestic-mirror download succeeded. |
| `h5py 3.16` could not map the archive's HDF5 float type | Installed `h5py 3.10` in the new environment; it reads the official episode without changing source data. |
| OpenCV and Torch could not consume NumPy arrays after dependency reconciliation | Matched the official DP install's `numpy 1.23.5` and used `opencv-python-headless 4.9.0.80`; OpenCV decoding and Torch NumPy interop now work. |
| Hydra rejected the official `agent_pos` shape override because the YAML omits that key | Used Hydra's `+task.shape_meta.obs.agent_pos.shape=[14]`, preserving the intended values while adding the dataset's 14-D observation entry. |
| RoboTwin evaluation lacked official scene assets | Ran official `scripts/_download_assets.sh` with `HF_ENDPOINT=https://hf-mirror.com` and proxy variables unset; downloaded and extracted assets without altering evaluation code. |
| Evaluation's `toppra 0.6.10` extension was built for NumPy 2 | Built and installed `toppra 0.6.3` against the DP-compatible NumPy 1.23.5 in the isolated environment. |
