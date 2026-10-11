import unittest
import torch
from torch import nn
from torchvision.models.resnet import BasicBlock
from diffusion_policy.model.vision.temporal_motion_encoder import (
    temporal_shift, MotionExcitation2d, MotionConditionFusion,
    TemporalBasicBlock, TemporalMotionObsEncoder)


def has_gradient(parameter):
    return parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0


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
        self.assertEqual(out[:, 0, 0, 0].tolist(), [1., 0., 3., 0.])
        self.assertEqual(out[:, 1, 0, 0].tolist(), [0., 0., 0., 2.])

    def test_selected_basicblock_receives_shifted_input(self):
        torch.manual_seed(0)
        net = TemporalMotionObsEncoder(FakeObsEncoder(), 2, 32, True, False,
                                       tsm_channels=(128,), me_channels=(128,))
        block = net.encoder.residual
        self.assertIsInstance(block, TemporalBasicBlock)
        self.assertEqual(net.installed_blocks, 1)
        seen = {}
        def block_input(module, args):
            seen['before_shift'] = args[0].detach().clone()
        def conv_input(module, args):
            seen['conv1_input'] = args[0].detach().clone()
        h1 = block.register_forward_pre_hook(block_input)
        h2 = block.block.conv1.register_forward_pre_hook(conv_input)
        try:
            net({'image': torch.randn(4, 3, 12, 12)})
        finally:
            h1.remove()
            h2.remove()
        expected = temporal_shift(seen['before_shift'], 2)
        self.assertTrue(torch.equal(seen['conv1_input'], expected))
        self.assertFalse(torch.equal(seen['conv1_input'], seen['before_shift']))

    def test_me_first_frame_stays_identity_after_bias_changes(self):
        torch.manual_seed(1)
        module = MotionExcitation2d(16, 2)
        with torch.no_grad():
            module.expand.weight.fill_(0.1)
            module.expand.bias.fill_(1.0)
        x = torch.randn(4, 16, 8, 8)
        y = module(x)
        self.assertTrue(torch.equal(y[0], x[0]))
        self.assertTrue(torch.equal(y[2], x[2]))
        self.assertFalse(torch.equal(y[1], x[1]))

    def test_me_gradients_reach_reduce_and_spatial_after_update(self):
        torch.manual_seed(2)
        module = MotionExcitation2d(16, 2)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        x = torch.randn(4, 16, 8, 8)
        module(x).square().mean().backward()
        self.assertTrue(has_gradient(module.expand.weight))
        self.assertFalse(has_gradient(module.reduce.weight))
        self.assertFalse(has_gradient(module.spatial.weight))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        module(x).square().mean().backward()
        self.assertTrue(has_gradient(module.reduce.weight))
        self.assertTrue(has_gradient(module.spatial.weight))

    def test_fusion_gate_gradient_after_projection_update(self):
        torch.manual_seed(3)
        module = MotionConditionFusion(32)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        feature = torch.randn(2, 2, 32)
        module(feature).square().mean().backward()
        self.assertTrue(has_gradient(module.projection.weight))
        self.assertFalse(has_gradient(module.gate[0].weight))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        module(feature).square().mean().backward()
        self.assertTrue(has_gradient(module.gate[0].weight))
        self.assertTrue(has_gradient(module.gate[0].bias))

    def test_independent_me_and_fusion_switches(self):
        for me, fusion, block_count in [(True, False, 1), (False, True, 0), (True, True, 1)]:
            with self.subTest(me=me, fusion=fusion):
                net = TemporalMotionObsEncoder(
                    FakeObsEncoder(), 2, 32, False, me,
                    tsm_channels=(128,), me_channels=(128,),
                    use_motion_condition_fusion=fusion)
                self.assertEqual(net.installed_blocks, block_count)
                self.assertEqual(net.motion_condition is not None, fusion)
                self.assertEqual(any(isinstance(m, MotionExcitation2d) for m in net.modules()), me)
                self.assertEqual(net({'image': torch.randn(4, 3, 12, 12)}).shape, (4, 32))


    def test_legacy_me_flag_still_enables_fusion(self):
        net = TemporalMotionObsEncoder(
            FakeObsEncoder(), 2, 32, False, True,
            tsm_channels=(128,), me_channels=(128,))
        self.assertIsNotNone(net.motion_condition)
        self.assertTrue(any(isinstance(m, MotionExcitation2d) for m in net.modules()))


if __name__ == '__main__':
    unittest.main()
