# RoboMimic Can low-dimensional: Workpoint 1 results

Completed paired seed-42 training runs for Original Diffusion Policy and Vanilla Self-Conditioning.

| Run | Self-conditioning | Seed | Epochs | Optimizer updates | Final train loss | Final validation loss | Best retained rollout score |
|---|---:|---:|---:|---:|---:|---:|---:|
| `seed42_original_dp` | off | 42 | 5,000 | 420,000 | 0.000098 | 0.490013 | 1.000 |
| `seed42_sc_only` | on | 42 | 5,000 | 420,000 | 0.000102 | 0.278128 | 1.000 |

Each run folder includes full per-step metrics, console and training logs, resolved Hydra configuration, command-line overrides, and every retained training checkpoint. This is a single paired seed and does not establish a multi-seed performance difference.

The dataset used was RoboMimic Can PH low-dimensional data at `data/robomimic/datasets/can/ph/low_dim.hdf5`; the input dataset is not included in this repository.
