# TM-DP Chapter 3: CNN Diffusion Policy

This branch integrates the supplied TM-DP starter into the server's CNN-DP checkout for Push-T Image. It adds four controlled modes: `vanilla`, `tsm`, `me`, and `full` (TM-DP). The original action U-Net, epsilon prediction objective, horizon 16, two observed frames, eight executed actions, and 100 denoising steps remain unchanged.

## Integration

- The installed Robomimic `ResNet18Conv` contains torchvision ResNet-18 `BasicBlock` modules. TSM and ME wrap the 128- and 256-channel residual blocks. The installer skips existing wrappers; the starter otherwise wrapped the same blocks twice in this checkout.
- The base policy flattens observations in batch-major order from `[B, To, ...]` to `[B*To, ...]`. Temporal operations reshape back to `[B, To, ...]`, so no channel shift or motion difference crosses sample boundaries.
- ME modifies the visual residual branch. MotionConditionFusion then adds a learned motion residual to each frame's encoded features. The base policy flattens those features into `global_cond` for the unchanged ConditionalUnet1D FiLM conditioning path.
- All four modes set `policy.crop_shape=null`. The original per-frame random crop is disabled for a matched spatial input.

The real-data interface check is `check_tm_dp_integration.py`. It confirms the selected blocks, per-sample shift, nonzero ME and fusion gradients, and `[B, 132]` global conditioning on a two-sample Push-T Image batch. This is an interface check, not task success evidence.

## Training

Run from this repository root, with data at `data/pusht/pusht_cchi_v7_replay.zarr` and the `robodiff` conda environment at `/home/hanjinwei/miniconda3/envs/robodiff`:

```bash
bash run_tm_dp.sh vanilla 0
bash run_tm_dp.sh tsm 0
bash run_tm_dp.sh me 0
bash run_tm_dp.sh full 0
```

The four commands share the base CNN-DP configuration: seed 42, batch size 64, 3050 epochs, learning rate 1e-4, EMA, 90 demonstration episodes, and Push-T Image. W&B logs stay offline. Each mode writes its Hydra config, training log, `latest.ckpt`, and rollout-score-ranked checkpoints under `data/outputs/tm_dp/<mode>/`. The launcher log is `data/outputs/tm_dp/<mode>_train.log`.

The server's GPU 0 is used for this run. GPU 1 was occupied by another workload at launch, so `run_tm_dp_queue.sh <vanilla-main-pid>` queues the other three modes and then the independent evaluation on GPU 0. The queue stops if any training run fails.

## Independent evaluation

`evaluate_tm_dp.py` reloads a saved checkpoint and runs 50 closed-loop Push-T Image episodes with held-out seeds 200000 through 200049. The training rollout seeds are 100000 through 100049. Success means the maximum episode reward equals 1.0, corresponding to at least 95% target coverage. The evaluator also records mean episode score, per-seed scores, single-plan latency (batch size 1), visual encoding time, diffusion denoising time, parameter count, and peak CUDA memory. It writes `metrics.json` under `data/outputs/tm_dp/eval_<mode>/`.

```bash
/home/hanjinwei/miniconda3/envs/robodiff/bin/python evaluate_tm_dp.py \
  --checkpoint data/outputs/tm_dp/full/checkpoints/epoch=XXXX-test_mean_score=Y.YYY.ckpt \
  --output-dir data/outputs/tm_dp/eval_full --device cuda:0
```

The queue selects the checkpoint with the highest training rollout score for each mode before the held-out evaluation. Validation loss is never used as a task-success metric.
