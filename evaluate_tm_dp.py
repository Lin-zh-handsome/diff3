"""Independent Push-T Image rollout and single-plan latency from a saved checkpoint."""
import argparse
import copy
import json
import pathlib
import time

import dill
import hydra
import numpy as np
import torch
from diffusion_policy.workspace import train_diffusion_unet_hybrid_workspace  # registers eval resolver
from diffusion_policy.common.pytorch_util import dict_apply


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--timing-plans', type=int, default=20)
    args = parser.parse_args()
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.checkpoint, map_location='cpu', pickle_module=dill)
    cfg = copy.deepcopy(payload['cfg'])
    cfg.task.env_runner.n_train = 0
    cfg.task.env_runner.n_train_vis = 0
    cfg.task.env_runner.n_test = 50
    cfg.task.env_runner.n_test_vis = 0
    cfg.task.env_runner.n_envs = 10
    cfg.task.env_runner.test_start_seed = 200000
    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace = workspace_cls(cfg, output_dir=str(out))
    workspace.load_payload(payload, include_keys=[])
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(args.device).eval()

    runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=str(out))
    try:
        rollout = runner.run(policy)
    finally:
        runner.env.close()
    per_seed = {str(seed): float(rollout[f'test/sim_max_reward_{seed}'])
                for seed in range(200000, 200050)}
    successes = {seed: reward >= 1.0 for seed, reward in per_seed.items()}

    dataset = hydra.utils.instantiate(cfg.task.dataset)
    example = dataset[0]['obs']
    obs = dict_apply(example, lambda x: x[:policy.n_obs_steps].unsqueeze(0).to(args.device))
    visual_times = []
    denoise_times = []
    total_times = []
    synchronize = lambda: torch.cuda.synchronize(args.device) if torch.cuda.is_available() else None
    original_encoder = policy.obs_encoder.forward
    original_sampler = policy.conditional_sample

    def timed_encoder(*a, **kw):
        synchronize()
        start = time.perf_counter()
        result = original_encoder(*a, **kw)
        synchronize()
        visual_times.append((time.perf_counter() - start) * 1000)
        return result

    def timed_sampler(*a, **kw):
        synchronize()
        start = time.perf_counter()
        result = original_sampler(*a, **kw)
        synchronize()
        denoise_times.append((time.perf_counter() - start) * 1000)
        return result

    policy.obs_encoder.forward = timed_encoder
    policy.conditional_sample = timed_sampler
    with torch.no_grad():
        for i in range(args.timing_plans + 3):
            synchronize()
            start = time.perf_counter()
            policy.predict_action(obs)
            synchronize()
            if i >= 3:
                total_times.append((time.perf_counter() - start) * 1000)
    policy.obs_encoder.forward = original_encoder
    policy.conditional_sample = original_sampler
    result = {
        'checkpoint': str(pathlib.Path(args.checkpoint).resolve()),
        'task': 'pusht_image',
        'test_seeds': list(range(200000, 200050)),
        'success_definition': 'max episode reward == 1.0 (PushTEnv coverage reaches 0.95)',
        'success_count': sum(successes.values()),
        'success_rate': float(np.mean(list(successes.values()))),
        'mean_max_reward': float(np.mean(list(per_seed.values()))),
        'per_seed_max_reward': per_seed,
        'planning_latency_ms_mean': float(np.mean(total_times)),
        'visual_encoding_ms_mean': float(np.mean(visual_times[3:])),
        'diffusion_denoising_ms_mean': float(np.mean(denoise_times[3:])),
        'parameter_count': sum(p.numel() for p in policy.parameters()),
        'gpu_peak_allocated_mib': torch.cuda.max_memory_allocated(args.device) / 2**20,
        'gpu_peak_reserved_mib': torch.cuda.max_memory_reserved(args.device) / 2**20,
        'timing_batch_size': 1,
        'timing_plans': args.timing_plans,
        'device': str(args.device),
        'num_inference_steps': policy.num_inference_steps,
        'n_obs_steps': policy.n_obs_steps,
        'horizon': policy.horizon,
        'n_action_steps': policy.n_action_steps,
    }
    (out / 'metrics.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'per_seed_max_reward'}, indent=2))


if __name__ == '__main__':
    main()
