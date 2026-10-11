import torch
import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate
from diffusion_policy.workspace import train_diffusion_unet_hybrid_workspace
from diffusion_policy.model.vision.temporal_motion_encoder import temporal_shift, MotionExcitation2d

with initialize_config_dir(config_dir='/home/hanjinwei/p1/project/thesis-ch3-tm-dp/diffusion_policy/config', version_base=None):
    cfg = compose(config_name='train_diffusion_unet_hybrid_workspace', overrides=[
        'task=pusht_image',
        'policy._target_=diffusion_policy.policy.tm_diffusion_unet_hybrid_image_policy.TMDiffusionUnetHybridImagePolicy',
        '+policy.use_tsm=true', '+policy.use_motion_excitation=true',
        'policy.crop_shape=null', 'n_obs_steps=2'])
OmegaConf.resolve(cfg)
model = hydra.utils.instantiate(cfg.policy)
dataset = hydra.utils.instantiate(cfg.task.dataset)
model.set_normalizer(dataset.get_normalizer())
model.cuda()
batch = default_collate([dataset[0], dataset[1]])
batch = {key: {k: v.cuda() for k, v in value.items()} if isinstance(value, dict) else value.cuda() for key, value in batch.items()}
seen = {}
original_forward = model.model.forward
def capture_forward(sample, timestep, local_cond=None, global_cond=None):
    seen['global_cond'] = tuple(global_cond.shape) if global_cond is not None else None
    return original_forward(sample, timestep, local_cond=local_cond, global_cond=global_cond)
model.model.forward = capture_forward
print('wrappers=', [(n, type(m.block).__name__) for n, m in model.obs_encoder.named_modules() if type(m).__name__ == 'TemporalBasicBlock'])
loss = model.compute_loss(batch)
loss.backward()
me_grads = [m.expand.weight.grad for m in model.modules() if isinstance(m, MotionExcitation2d)]
fusion_grad = model.obs_encoder.motion_condition.projection.weight.grad
print('loss=', float(loss.detach().cpu()))
print('global_cond=', seen['global_cond'], 'expected=', (2, model.n_obs_steps * model.obs_feature_dim))
print('temporal_blocks=', model.obs_encoder.installed_blocks)
print('ME_grad_nonzero=', any(g is not None and torch.count_nonzero(g).item() > 0 for g in me_grads))
print('fusion_grad_nonzero=', fusion_grad is not None and torch.count_nonzero(fusion_grad).item() > 0)
x = torch.arange(4., device='cuda').view(4, 1, 1, 1).expand(4, 8, 1, 1)
y = temporal_shift(x, 2)
print('TSM_batch_isolation=', y[:, 0, 0, 0].tolist() == [1., 0., 3., 0.])
assert seen['global_cond'] == (2, model.n_obs_steps * model.obs_feature_dim)
assert any(g is not None and torch.count_nonzero(g).item() > 0 for g in me_grads)
assert fusion_grad is not None and torch.count_nonzero(fusion_grad).item() > 0
assert y[:, 0, 0, 0].tolist() == [1., 0., 3., 0.]
