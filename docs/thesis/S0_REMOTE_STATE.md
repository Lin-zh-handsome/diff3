# S0 Remote Baseline State

- Date: 2026-09-21 Asia/Shanghai
- Repository: `/home/hanjinwei/p1/project/diffusion_policy`
- Upstream baseline boundary: Git commit `548a52b`.
- Current worktree: file-level cleanup staged but not committed; no history rewrite or push performed.
- Removed under explicit user authorization: post-`548a52b` PEC/Push-T/MuJoCo diagnostics, self-exposure implementation, and historical artifacts/checkpoints.
- Preserved: `data/pusht/pusht_cchi_v7_replay.zarr`, official source tree, and existing Conda environments.
- Hardware evidence: two NVIDIA RTX 3090 GPUs, 24 GB each; no training process was running at inspection.
- Candidate runtime: Conda env `robodiff`, Python with PyTorch 1.12.1 and CUDA visible on two devices.
- Evidence boundary: no thesis baseline performance result has been produced in this S0 session.
