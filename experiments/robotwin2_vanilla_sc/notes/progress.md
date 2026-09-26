# Progress Log

## Session: 2026-09-26

### Current Status
- **Phase:** 3 - Official beat_block_hammer processing and training
- **Started:** 2026-09-26

### Actions Taken
- Read the requested implementation scope and official RoboTwin instructions.
- Located no existing RoboTwin checkout in the local project workspace or `/home/hanjinwei/p1/project`; the only existing server repo there is the separate Diffusion Policy project.
- Cloned official RoboTwin main into `/home/hanjinwei/p1/project/RoboTwin` (commit `ea8b211`).
- Completed XPolicyLab submodule retrieval through the pre-existing server HTTP proxy at 127.0.0.1:7890; official DP policy, U-Net, YAML, and processing/training/evaluation scripts are present.
- Confirmed task data was absent; following the user's latest direction, use the official downloader with a domestic HF mirror for only `beat_block_hammer/demo_clean`.
- Confirmed official data path, launcher protocol, requested config values, and two currently idle RTX 3090s.
- Downloaded the official 225 MB `demo_clean.zip` through `https://hf-mirror.com` and extracted it to the official path; verified exactly 50 HDF5 episodes.
- Implemented Vanilla Self-Conditioning in the policy, U-Net output projection, and policy YAML only; syntax check passed.
- Created `/home/hanjinwei/p1/envs/robotwin-sc` as a separate Python 3.10 environment; installed the official DP stack there without modifying `robodiff`.
- Resolved environment compatibility using `h5py 3.10`, installer-pinned `numpy 1.23.5`, and `opencv-python-headless 4.9.0.80`; the original HDF5 archive and source data were left untouched.
- Ran the official `process_data.sh RoboTwin beat_block_hammer aloha_agilex joint 50`; all 50 episodes processed into `data/RoboTwin-beat_block_hammer-aloha_agilex-joint.zarr` (1.1 GB).
- Parsed the actual official Hydra config with seed 42 and SC enabled. Policy instantiation reports action D=14, U-Net input/output=28/14, 85,646,222 U-Net params, and 96,822,734 total policy params. The current GPU check shows two RTX 3090 cards available, with 0.1 GiB used on GPU 0 and effectively idle GPU 1.
- Launched the official training entry point with the requested seed/config on GPU 0. Python PID 1855894 has completed epoch 0 and is training epoch 1; the log shows 43 batches per epoch, and GPU 0 is using about 23.9 GiB. The process is confirmed active; no model-forward smoke test was added.
- Saved the actual resolved Hydra config to `artifacts/robotwin_self_condition/beat_block_hammer/config/resolved_robot_dp.yaml`; training logs are under `artifacts/robotwin_self_condition/beat_block_hammer/logs/beat_block_hammer-seed42-20260926/`, and its `checkpoints` symlink targets the requested artifact checkpoint directory.
- Pinned workspace comments out WandB initialization; `logging.mode=online` remains in the resolved config but no WandB run is created by this code.
- Created this isolated planning folder, preserving the prior root planning files for the other project.

### Test Results
| Check | Expected | Actual | Status |
|------|----------|--------|--------|
| Official DP source path | Present under pinned XPolicyLab | Present | PASS |
| Task data | `data/demo_clean/beat_block_hammer/aloha_agilex/data/` contains 50 demos | Exactly 50 HDF5 episodes | PASS |
| Self-Conditioning source syntax | Modified Python files parse | `py_compile` passed | PASS |
| Official data processing | 50 demos convert to the DP Zarr input | Processed all 50; output is 1.1 GB | PASS |
| Official Hydra config and dimensions | Resolved seed-42 SC config, input 2D, output D | action D=14; U-Net 28 -> 14 | PASS |
| Model parameters | Instantiate policy and count parameters | U-Net 85,646,222; full policy 96,822,734 | PASS |
| GPU availability | A free 3090 for training | GPU 0: 114 MiB used; GPU 1: 1 MiB used | PASS |
| Training started | Official train worker remains active with a training step completed | Python PID 1855894, epoch 1 running on GPU 0 | PASS |

### Errors
| Error | Resolution |
|-------|------------|
| GitHub clone without proxy slow; interrupted submodule helper could not resolve checkout | Fetch pinned submodule commit directly with `git -c http.proxy=http://127.0.0.1:7890` and verify root gitlink alignment. |
| Server does not have `rg` installed | Used `find`/`sed` and repository-provided scripts instead. |
| System Python has no pip; first downloader invocation exited before accessing HF | Re-ran with existing environment Python on PATH and domestic HF endpoint; 225 MB archive extracted successfully. |
| PowerShell expanded a remote variable while checking pip configuration | Re-ran with explicit absolute paths. |
| h5py 3.16 failed to convert the official HDF5 float type | Installed h5py 3.10 in the isolated new environment; episode data reads correctly. |
| OpenCV array binding and Torch `from_numpy` failed with a mixed NumPy stack | Reinstalled the OpenCV headless wheel and matched `numpy==1.23.5` from the official DP install script. |
| Hydra rejected agent_pos override because the key is absent from pinned default YAML | Added the key via the `+task.shape_meta.obs.agent_pos.shape=[14]` CLI override. |
| First launch stopped during workspace import because the requirements install left pandas built for NumPy 2 | Installed pandas 1.5.3 compatible with the DP installer-pinned NumPy 1.23.5; confirmed the official workspace imports and restarted. |


### Publication snapshot update
- Captured while PID 1855894 was still training. The included `artifacts/logs.json.txt` ends at epoch 48, global step 2104; official evaluation is still pending. No checkpoint or task dataset is included.
