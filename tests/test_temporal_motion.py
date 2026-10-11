import unittest
import torch
from torch import nn
from torchvision.models.resnet import BasicBlock
from diffusion_policy.model.vision.temporal_motion_encoder import (
    temporal_shift, MotionExcitation2d, TemporalMotionObsEncoder)


class FakeObsEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.entry = nn.Conv2d(3, 128, 3, padding=1)
        self.residual = BasicBlock(128, 128)
        self.out = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(128, 32)

    def forward(self, obs):
        x = self.entry(obs['image'])
        return self.proj(self.out(self.residual(x)).flatten(1))


class TMTest(unittest.TestCase):
    def test_shift_no_episode_mixing(self):
        x = torch.arange(4.).reshape(4, 1, 1, 1).repeat(1, 8, 1, 1)
        out = temporal_shift(x, 2, fold_div=8)
        self.assertEqual(out[0, 0, 0, 0].item(), 1)
        self.assertEqual(out[1, 0, 0, 0].item(), 0)
        self.assertEqual(out[2, 0, 0, 0].item(), 3)
        self.assertEqual(out[3, 0, 0, 0].item(), 0)
        self.assertEqual(out[2, 1, 0, 0].item(), 0)
        self.assertEqual(out[3, 1, 0, 0].item(), 2)

    def test_modes_backward(self):
        for tsm, me in [(True, False), (False, True), (True, True)]:
            with self.subTest(tsm=tsm, me=me):
                torch.manual_seed(0)
                net = TemporalMotionObsEncoder(
                    FakeObsEncoder(), 2, 32, tsm, me,
                    tsm_channels=(128,), me_channels=(128,))
                x = torch.randn(4, 3, 12, 12, requires_grad=True)
                y = net({'image': x})
                self.assertEqual(y.shape, (4, 32))
                y.square().mean().backward()
                self.assertIsNotNone(x.grad)
                self.assertTrue(torch.isfinite(x.grad).all())

    def test_excitation_identity_at_init(self):
        x = torch.randn(4, 64, 8, 8)
        me = MotionExcitation2d(64, 2)
        self.assertTrue(torch.allclose(x, me(x), atol=1e-6))


if __name__ == '__main__':
    unittest.main()
