# TM-DP Chapter 3: CNN Diffusion Policy

This branch integrates the supplied TM-DP starter into the server's CNN-DP checkout for Push-T Image. It adds four controlled modes: `vanilla`, `tsm`, `me`, and `full` (TM-DP). The original action U-Net, epsilon prediction objective, horizon 16, two observed frames, eight executed actions, and 100 denoising steps remain unchanged.

## Integration

- The installed Robomimic `ResNet18Conv` contains torchvision ResNet-18 `BasicBlock` modules. TSM and ME wrap the 128- and 256-channel residual blocks. The installer skips existing wrappers; the starter otherwise wrapped the same blocks twice in this checkout.
- The base policy flattens observations in batch-major order from `[B, To, ...]` to `[B*To, ...]`. Temporal operations reshape back to `[B, To, ...]`, so no channel shift or motion difference crosses sample boundaries.
- ME modifies the visual residual branch. A validity mask makes the first observation an exact ME identity because it has no preceding frame. MotionConditionFusion independently adds a learned motion residual to encoded features. The base policy flattens those features into `global_cond` for the unchanged ConditionalUnet1D FiLM conditioning path.
- All four modes set `policy.crop_shape=null`. The original per-frame random crop is disabled for a matched spatial input.

The real-data interface check is `check_tm_dp_integration.py`. It captures the input of an actual selected ResNet block and its first convolution to verify TSM executes there, verifies `[B, 132]` global conditioning, and checks ME `expand` / fusion `projection` gradients on the first backward pass followed by `reduce`, `spatial`, and `gate` gradients after one in-memory parameter update. No checkpoint or training result is written. The unit tests additionally force a nonzero learned `expand.bias` and verify the first frame remains unchanged.

## Training

Run from this repository root, with data at `data/pusht/pusht_cchi_v7_replay.zarr` and the `robodiff` conda environment at `/home/hanjinwei/miniconda3/envs/robodiff`:

```bash
bash run_tm_dp.sh vanilla 0
bash run_tm_dp.sh tsm 0
bash run_tm_dp.sh me 0
bash run_tm_dp.sh full 0
# Optional isolation ablations, only when explicitly requested:
# bash run_tm_dp.sh me_only 0
# bash run_tm_dp.sh fusion_only 0
# bash run_tm_dp.sh me_fusion 0
```

The existing `me` mode still enables both ME and motion fusion, preserving its checkpoint layout and meaning. The new `me_only`, `fusion_only`, and `me_fusion` modes expose separate switches; `use_motion_condition_fusion` defaults to the old ME setting when absent from older configs. The four core commands share the base CNN-DP configuration: seed 42, batch size 64, 3050 epochs, learning rate 1e-4, EMA, 90 demonstration episodes, and Push-T Image. W&B logs stay offline. Each mode writes its Hydra config, training log, `latest.ckpt`, and rollout-score-ranked checkpoints under `data/outputs/tm_dp/<mode>/`. The launcher log is `data/outputs/tm_dp/<mode>_train.log`.

Training is currently stopped at the user's request. For any future resume, use only one physical GPU and run every training and evaluation job sequentially. `bash run_tm_dp_queue.sh 0` uses physical GPU 0 for all four modes; replace `0` with `1` only if that card is selected. The launcher masks all other GPUs from each process. Existing `latest.ckpt` files are used to resume partial runs, and the workspace stops at the common 3050-epoch budget. Do not launch this queue until the user explicitly asks to resume.

## Independent evaluation

`evaluate_tm_dp.py` reloads a saved checkpoint and runs 50 closed-loop Push-T Image episodes with held-out seeds 200000 through 200049. The training rollout seeds are 100000 through 100049. Success means the maximum episode reward equals 1.0, corresponding to at least 95% target coverage. The evaluator also records mean episode score, per-seed scores, single-plan latency (batch size 1), visual encoding time, diffusion denoising time, parameter count, and separate peak CUDA memory for the batched closed-loop rollout and batch-size-1 planning. Peak statistics are reset before each phase and again after planning warmup. It writes `metrics.json` under `data/outputs/tm_dp/eval_<mode>/`.

```bash
/home/hanjinwei/miniconda3/envs/robodiff/bin/python evaluate_tm_dp.py \
  --checkpoint data/outputs/tm_dp/full/checkpoints/epoch=XXXX-test_mean_score=Y.YYY.ckpt \
  --output-dir data/outputs/tm_dp/eval_full --device cuda:0
```

The queue selects the checkpoint with the highest training rollout score for each mode before the held-out evaluation. Validation loss is never used as a task-success metric.

## Revision notes

The ME first-frame mask changes learned-model behavior only when the old `expand` output for that invalid frame had become nonzero, for example through a trained bias. Later frames and the zero-initialized starting behavior are unchanged. Existing checkpoints remain on disk and their parameter key layout is unchanged, but their previous training logs and rollout scores were produced under the earlier code; they are not post-fix results. Formal training and evaluation remain stopped.

Before a future queue run, `run_tm_dp_queue.sh` creates its log directory. A mode counts as complete only when the final epoch is logged and `latest.ckpt` exists, contains model state/config, and can be deserialized. A final log with a missing or corrupt checkpoint stops the queue instead of silently restarting training.
