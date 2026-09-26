# RoboTwin 2.0: Vanilla Self-Conditioning for Diffusion Policy

This package records the current RoboTwin 2.0 experiment based on the official image-based DP policy.

## Pinned upstream source

- RoboTwin: `RoboTwin-Platform/RoboTwin`, commit `ea8b211`.
- XPolicyLab submodule: commit `fa431ecd893ee706883e64fe5fe1464ec8cd928d`.
- The full modified files are under `source_snapshot/`; `patches/xpolicylab_dp_vanilla_sc.patch` can be applied from the pinned XPolicyLab checkout root with `git apply`.

## Method and configuration

Only Vanilla Self-Conditioning was added to the official CNN-based image DP policy. Training follows the 50% same-timestep detached self-condition / 50% zero-feedback branch; inference carries the detached clean-action estimate through the existing scheduler loop. The model retains a 14-D action output and uses a 28-D U-Net input when self-conditioning is enabled. The official scheduler and training loss are unchanged.

The resolved run uses `beat_block_hammer`, `demo_clean`, `aloha_agilex`, 50 demonstrations, seed 42, and the official 600-epoch training configuration. Exact source and resolved configs plus Hydra run snapshots are included under `config/`.

## Run status at publication

Training was still running when this snapshot was captured. The exact last logged epoch and global step are in `artifacts/logs.json.txt`; this is an interim training snapshot, not a completed training result. Official 100-episode evaluation has not yet been run, so no success rate is claimed. Logs in `artifacts/` stop at the captured snapshot and will not update automatically.

## Included and excluded

Included: the three modified policy/config source files, an applyable patch, resolved/Hydra configs, training logs, and implementation/progress notes. Dataset archives, processed datasets, environment files, and model checkpoints are not included.
