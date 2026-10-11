"""Lightweight TSM + Motion Excitation for Robomimic ResNet-18 vision encoders.

The original CNN-DP flattens [batch, time] before calling its observation encoder.
This adapter reconstructs the temporal grouping INSIDE eligible ResNet blocks.
It assumes the flattened observations preserve batch-major, time-minor order,
which is the layout used by DiffusionUnetHybridImagePolicy.

Adapted from the mechanisms in:
- Lin et al., TSM, ICCV 2019 (shift the residual branch input)
- Li et al., TEA, CVPR 2020 (motion-difference channel excitation)
This is NOT an implementation of the full TEA action recognition network.
"""
from typing import Iterable

import torch
from torch import nn
from torchvision.models.resnet import BasicBlock


def temporal_shift(x: torch.Tensor, n_obs_steps: int, fold_div: int = 8) -> torch.Tensor:
    """Temporal channel shift on flattened BTCHW; never mix different episodes."""
    if x.ndim != 4 or n_obs_steps < 2:
        raise ValueError("TSM requires [B*To,C,H,W] and To >= 2")
    bt, c, h, w = x.shape
    if bt % n_obs_steps:
        raise ValueError("BT must be divisible by n_obs_steps")
    fold = c // fold_div
    if fold == 0:
        return x
    frames = x.reshape(bt // n_obs_steps, n_obs_steps, c, h, w)
    shifted = torch.zeros_like(frames)
    shifted[:, :-1, :fold] = frames[:, 1:, :fold]
    shifted[:, 1:, fold:2 * fold] = frames[:, :-1, fold:2 * fold]
    shifted[:, :, 2 * fold:] = frames[:, :, 2 * fold:]
    return shifted.reshape_as(x)


class MotionExcitation2d(nn.Module):
    """Two/few-frame TEA-ME-style excitation with a bounded residual gate.

    A previous-frame depthwise spatial transform is compared against the current
    reduced feature. The first frame has a zero motion descriptor. The expand
    projection is zero initialized, so the block starts as an identity. The
    first frame stays an identity even after the projection bias is trained.
    """
    def __init__(self, channels: int, n_obs_steps: int, reduction: int = 16):
        super().__init__()
        if n_obs_steps < 2:
            raise ValueError("Motion excitation requires at least two observations")
        hidden = max(8, channels // reduction)
        self.n_obs_steps = n_obs_steps
        self.reduce = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.spatial = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1,
                                 groups=hidden, bias=False)
        self.expand = nn.Conv2d(hidden, channels, kernel_size=1)
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bt, channels, h, w = x.shape
        t = self.n_obs_steps
        if bt % t:
            raise ValueError("Temporal batch grouping mismatch in MotionExcitation2d")
        z = self.reduce(x)
        zt = z.reshape(bt // t, t, z.shape[1], h, w)
        diff = torch.zeros_like(zt)
        prev = self.spatial(zt[:, :-1].reshape(-1, z.shape[1], h, w))
        diff[:, 1:] = zt[:, 1:] - prev.reshape(bt // t, t - 1, z.shape[1], h, w)
        pooled = diff.reshape(bt, z.shape[1], h, w).mean(dim=(-2, -1), keepdim=True)
        excitation = 2.0 * torch.sigmoid(self.expand(pooled)) - 1.0
        # The first frame has no preceding observation. Mask after expand so a
        # learned expand.bias cannot excite that invalid motion descriptor.
        valid = torch.ones((bt // t, t, 1, 1, 1), device=x.device, dtype=x.dtype)
        valid[:, 0] = 0
        excitation = excitation.reshape(bt // t, t, channels, 1, 1) * valid
        return x * (1.0 + excitation.reshape(bt, channels, 1, 1))


class TemporalBasicBlock(nn.Module):
    """Wrap an original torchvision ResNet BasicBlock without changing its skip.

    TSM shifts ONLY the residual path, consistent with residual TSM insertion.
    ME modulates the residual branch before it meets the identity shortcut.
    """
    def __init__(self, block: BasicBlock, n_obs_steps: int,
                 use_tsm: bool, use_motion_excitation: bool, fold_div: int = 8):
        super().__init__()
        self.block = block
        self.n_obs_steps = n_obs_steps
        self.use_tsm = use_tsm
        self.fold_div = fold_div
        self.motion = (MotionExcitation2d(block.conv2.out_channels, n_obs_steps)
                       if use_motion_excitation else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        residual = temporal_shift(x, self.n_obs_steps, self.fold_div) if self.use_tsm else x
        out = self.block.conv1(residual)
        out = self.block.bn1(out)
        out = self.block.relu(out)
        out = self.block.conv2(out)
        out = self.block.bn2(out)
        out = self.motion(out)
        if self.block.downsample is not None:
            identity = self.block.downsample(identity)
        out = out + identity
        out = self.block.relu(out)
        return out


def install_temporal_blocks(encoder: nn.Module, n_obs_steps: int,
                            use_tsm: bool, use_motion_excitation: bool,
                            tsm_channels: Iterable[int] = (128, 256),
                            me_channels: Iterable[int] = (256,),
                            fold_div: int = 8) -> int:
    """Replace selected ResNet BasicBlocks in-place, across all image cameras."""
    tsm_channels, me_channels = set(tsm_channels), set(me_channels)
    count = 0

    def visit(parent: nn.Module):
        nonlocal count
        for name, child in list(parent.named_children()):
            if isinstance(child, TemporalBasicBlock):
                continue
            if isinstance(child, BasicBlock):
                channels = child.conv2.out_channels
                tsm = use_tsm and channels in tsm_channels
                me = use_motion_excitation and channels in me_channels
                if tsm or me:
                    setattr(parent, name, TemporalBasicBlock(
                        child, n_obs_steps, tsm, me, fold_div))
                    count += 1
            else:
                visit(child)

    visit(encoder)
    if (use_tsm or use_motion_excitation) and count == 0:
        raise RuntimeError("No ResNet BasicBlock selected; inspect Robomimic visual backbone")
    return count


class MotionConditionFusion(nn.Module):
    """Adapt motion features to per-observation CNN-DP FiLM global conditions."""
    def __init__(self, feature_dim: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(2 * feature_dim, feature_dim), nn.Sigmoid())
        self.projection = nn.Linear(feature_dim, feature_dim)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        # [B,To,D]. First observation has no preceding in-window motion.
        motion = torch.cat((torch.zeros_like(feature[:, :1]),
                            feature[:, 1:] - feature[:, :-1]), dim=1)
        gate = self.gate(torch.cat((feature, motion), dim=-1))
        return feature + gate * self.projection(motion)


class TemporalMotionObsEncoder(nn.Module):
    """Compatible observation encoder wrapper returning flattened [B*To,D]."""
    def __init__(self, obs_encoder: nn.Module, n_obs_steps: int, feature_dim: int,
                 use_tsm: bool, use_motion_excitation: bool,
                 tsm_channels=(128, 256), me_channels=(256,), fold_div=8,
                 use_motion_condition_fusion=None):
        super().__init__()
        if n_obs_steps < 2:
            raise ValueError("TM-DP requires n_obs_steps >= 2")
        self.encoder = obs_encoder
        self.n_obs_steps = n_obs_steps
        if use_motion_condition_fusion is None:
            use_motion_condition_fusion = use_motion_excitation
        self.installed_blocks = install_temporal_blocks(
            self.encoder, n_obs_steps, use_tsm, use_motion_excitation,
            tsm_channels, me_channels, fold_div)
        self.motion_condition = (MotionConditionFusion(feature_dim)
                                 if use_motion_condition_fusion else None)

    def forward(self, obs_dict):
        feat = self.encoder(obs_dict)
        if feat.ndim != 2 or feat.shape[0] % self.n_obs_steps:
            raise ValueError("TM-DP observation encoder expected flattened [B*To,D]")
        if self.motion_condition is None:
            return feat
        feat = feat.reshape(-1, self.n_obs_steps, feat.shape[-1])
        return self.motion_condition(feat).reshape(-1, feat.shape[-1])
