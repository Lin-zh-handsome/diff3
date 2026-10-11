"""One real Push-T Image batch checks TM-DP wiring and gradient flow; no training run."""
import torch
import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate
from diffusion_policy.workspace import train_diffusion_unet_hybrid_workspace
from diffusion_policy.model.vision.temporal_motion_encoder import (
    temporal_shift, MotionExcitation2d, TemporalBasicBlock)


def has_gradient(parameter):
    return parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0


with initialize_config_dir(config_dir='/home/hanjinwei/p1/project/thesis-ch3-tm-dp/diffusion_policy/config', version_base=None):
    cfg = compose(config_name='train_diffusion_unet_hybrid_workspace', overrides=[
        'task=pusht_image',
        'policy._target_=diffusion_policy.policy.tm_diffusion_unet_hybrid_image_policy.TMDiffusionUnetHybridImagePolicy',
        '+policy.use_tsm=true', '+policy.use_motion_excitation=true',
        '+policy.use_motion_condition_fusion=true',
        'policy.crop_shape=null', 'n_obs_steps=2'])
OmegaConf.resolve(cfg)
model = hydra.utils.instantiate(cfg.policy)
dataset = hydra.utils.instantiate(cfg.task.dataset)
model.set_normalizer(dataset.get_normalizer())
model.cuda()
batch = default_collate([dataset[0], dataset[1]])
batch = {key: {k: v.cuda() for k, v in value.items()} if isinstance(value, dict) else value.cuda() for key, value in batch.items()}

motion_blocks = [m for m in model.modules() if isinstance(m, MotionExcitation2d)]
fusion = model.obs_encoder.motion_condition
assert motion_blocks and fusion is not None
selected = [p for module in motion_blocks for p in module.parameters()] + list(fusion.parameters())
optimizer = torch.optim.SGD(selected, lr=0.1)
seen = {}
original_forward = model.model.forward
def capture_forward(sample, timestep, local_cond=None, global_cond=None):
    seen['global_cond'] = tuple(global_cond.shape) if global_cond is not None else None
    return original_forward(sample, timestep, local_cond=local_cond, global_cond=global_cond)
model.model.forward = capture_forward
selected_tsm_block = next(m for m in model.obs_encoder.modules()
                          if isinstance(m, TemporalBasicBlock) and m.use_tsm)
def block_input(module, args):
    seen['block_input'] = args[0].detach().clone()
def conv_input(module, args):
    seen['conv1_input'] = args[0].detach().clone()
h1 = selected_tsm_block.register_forward_pre_hook(block_input)
h2 = selected_tsm_block.block.conv1.register_forward_pre_hook(conv_input)

loss = model.compute_loss(batch)
loss.backward()
assert seen['global_cond'] == (2, model.n_obs_steps * model.obs_feature_dim)
assert torch.equal(seen['conv1_input'], temporal_shift(
    seen['block_input'], model.n_obs_steps, selected_tsm_block.fold_div))
assert not torch.equal(seen['conv1_input'], seen['block_input'])
assert all(has_gradient(m.expand.weight) for m in motion_blocks)
assert has_gradient(fusion.projection.weight)
assert all(not has_gradient(m.reduce.weight) and not has_gradient(m.spatial.weight)
           for m in motion_blocks)
assert not has_gradient(fusion.gate[0].weight)
optimizer.step()
model.zero_grad(set_to_none=True)
second_loss = model.compute_loss(batch)
second_loss.backward()
assert all(has_gradient(m.reduce.weight) and has_gradient(m.spatial.weight)
           for m in motion_blocks)
assert has_gradient(fusion.gate[0].weight)
assert has_gradient(fusion.gate[0].bias)
h1.remove()
h2.remove()
print('actual_resnet_tsm_blocks=', model.obs_encoder.installed_blocks)
print('tsm_inside_selected_block=', True)
print('global_cond=', seen['global_cond'])
print('ME_expand_step1=', True, 'ME_reduce_spatial_step2=', True)
print('fusion_projection_step1=', True, 'fusion_gate_step2=', True)
print('losses=', float(loss.detach().cpu()), float(second_loss.detach().cpu()))
