# Progress Log

## Session: 2026-09-26

### Current Status
- **Phase:** 5 - Remaining official task runs
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
- Completed the official 600-epoch training run on GPU 0 (43 batches/epoch, 25,800 optimizer updates; last logged zero-indexed global_step 25,799). Final checkpoint: `XPolicyLab/policy/DP/checkpoints/RoboTwin-beat_block_hammer-aloha_agilex-joint-42/600.ckpt` (about 1.5 GB), linked at `artifacts/robotwin_self_condition/beat_block_hammer/checkpoints/600.ckpt`.
- Saved the actual resolved Hydra config to `artifacts/robotwin_self_condition/beat_block_hammer/config/resolved_robot_dp.yaml`; training logs are under `artifacts/robotwin_self_condition/beat_block_hammer/logs/beat_block_hammer-seed42-20260926/`, and its `checkpoints` symlink targets the requested artifact checkpoint directory.
- Pinned workspace comments out WandB initialization; `logging.mode=online` remains in the resolved config but no WandB run is created by this code.
- Prepared official RoboTwin scene resources using `scripts/_download_assets.sh` through `https://hf-mirror.com` with proxy variables unset; assets were extracted and the embodiment paths configured.
- Confirmed the training log reaches epoch 599/600 and produced the final 600.ckpt (1.55 GB); the official training run is complete.
- First evaluation launch stopped before any episode because CuRobo was absent. Installed official CuRobo v0.7.8 in the new environment; compiled its CUDA extensions using CUDA 12.8 (the server is Ubuntu 24.04, where its CUDA 12.1 compiler failed on glibc headers), installed the official Warp 1.12.0 version, and applied RoboTwin's documented SAPIEN/MPLib environment patches.
- The next evaluation launch reached the episode runner but stopped at video recording because the `ffmpeg` executable was missing; linked the already-installed `imageio-ffmpeg` binary into the new environment's `bin/ffmpeg`.
- Restarted the official `eval.sh` flow with `demo_clean`, seed 42, 100 target episodes, policy GPU 0 and simulation GPU 1. Current evaluation wrapper PID 1956737, policy server PID 1956744, and eval client PID 1956913; 67 episode videos are complete and the active rollout has reached step 154/400. Final summary/result JSON is not present yet.
- Rechecked the evaluation after the user's update: the run finished all 100 episodes, produced 100 episode videos, and `_result.txt` reports 19/100 successes (19.0%), with 33 skipped seeds.
- Saved `artifacts/robotwin_self_condition/beat_block_hammer/result.json`; linked the official timestamped video/result directory into the task's `artifacts/.../eval/` directory. No best checkpoint was tracked by the official workspace; the result records this explicitly and points to final `600.ckpt`.
- Downloaded and extracted all three remaining official `demo_clean` archives through `https://hf-mirror.com` with proxy variables unset; each contains 50 HDF5 demonstrations. Official processing completed and produced Zarr datasets: handover 2.5 GB, stack bowls 4.8 GB, pick dual bottles 1.5 GB.
- The official `train.sh` wrapper could not add the dataset's missing Hydra `agent_pos` key, so it exited before training. Launched the same official `train.py` Hydra entry point with only the previously required `+task.shape_meta.obs.agent_pos.shape=[14]` override; Handover training PID 2238700 is active on GPU 0 and Stack Bowls PID 2239568 on GPU 1. Both configs resolve to seed 42, 600 epochs, Vanilla SC true, and action D=14; both resolved configs are saved in task artifacts. Handover has 107 batches/epoch and Stack Bowls 179 batches/epoch for their different official processed datasets.
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
| GPU availability | A free 3090 for training | GPU 0: 114 MiB used before run; GPU 1: 1 MiB | PASS |
| Training completion | Official run completes 600 epochs and writes final checkpoint | 600 epochs complete; official 600.ckpt exists, linked into artifacts | PASS |
| Evaluation | Official 100-episode demo_clean evaluation | 100 episodes, 19 successes, 19.0%; official `_result.txt` saved and linked in artifacts | PASS |
| Evaluation prerequisites | CuRobo and video encoder available to official eval | CuRobo extensions import; ffmpeg 7.0.2-static available in the isolated env | PASS |
| Result record | Required task metadata and metrics persisted | `result.json` parsed; final checkpoint and missing best-checkpoint status recorded | PASS |
| Remaining task data | Official demo_clean archives and processed datasets available | 50 demos/task; official Zarr datasets at 2.5 GB, 4.8 GB, and 1.5 GB | PASS |
| Handover training startup | Official SC training starts on GPU 0 with seed 42 and 600 epochs | PID 2238700 active; resolved config confirms `use_self_condition=true`, seed 42 | IN_PROGRESS |
| Stack Bowls training startup | Official SC training starts on GPU 1 with seed 42 and 600 epochs | PID 2239568 active; resolved config confirms `use_self_condition=true`, seed 42 | IN_PROGRESS |

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
| First evaluation attempt lacked `assets/objects/objaverse/list.json` | Downloaded/extracted all official assets with the RoboTwin helper and domestic HF mirror. |
| Evaluation client then failed importing `toppra` against NumPy 1.23.5 | Rebuilt `toppra 0.6.3` locally with Cython 0.29.37 against the pinned NumPy; official eval client starts. |
| First episode attempt failed because CuRobo was not installed | Installed pinned official CuRobo v0.7.8 plus its required CUDA/Warp environment components; applied official SAPIEN/MPLib compatibility patches. |
| Next episode attempt could not start video recording because `ffmpeg` was not on PATH | Added the existing `imageio-ffmpeg` binary to the isolated environment's `bin/ffmpeg`; restarted official evaluation. |
| User's first completion note disagreed with the live server state | Rechecked the task after the evaluation finished; official output confirms 100 episodes and 19/100 successes. |
