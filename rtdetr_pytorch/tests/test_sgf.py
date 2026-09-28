import sys
from pathlib import Path
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.zoo.rtdetr.hybrid_encoder import HybridEncoder, SpatialGatedFusion


class SGFTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)

    def test_lossless_rearrangement_and_staged_gradient_flow(self):
        branch = SpatialGatedFusion(4, 8)
        s2, f3 = torch.randn(2, 4, 16, 16), torch.randn(2, 8, 8, 8)
        self.assertTrue(torch.equal(torch.nn.functional.pixel_shuffle(branch.space_to_depth(s2), 2), s2))
        output = branch(s2, f3)
        self.assertTrue(torch.equal(output, f3))
        (output-(f3+1)).square().mean().backward()
        self.assertGreater(branch.residual_scale.grad.abs().item(), 0)
        self.assertEqual(branch.gate.weight.grad.count_nonzero().item(), 0)
        with torch.no_grad():
            branch.residual_scale -= .1 * branch.residual_scale.grad
        branch.zero_grad(set_to_none=True)
        (branch(s2, f3)-(f3+1)).square().mean().backward()
        self.assertGreater(branch.gate.weight.grad.abs().sum().item(), 0)
        self.assertGreater(branch.projection.conv.weight.grad.abs().sum().item(), 0)

    def test_zero_lambda_preserves_all_encoder_outputs(self):
        options = dict(in_channels=[8, 16, 32], hidden_dim=8, nhead=2, dim_feedforward=16, depth_mult=0.34)
        baseline = HybridEncoder(**options).eval()
        sgf = HybridEncoder(**options, sgf_s2_channels=4).eval()
        missing, unexpected = sgf.load_state_dict(baseline.state_dict(), strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(all(k.startswith('sgf.') for k in missing))
        features = [torch.randn(1, c, size, size) for c, size in [(4, 32), (8, 16), (16, 8), (32, 4)]]
        with torch.inference_mode():
            for a, b in zip(baseline(features[1:]), sgf(features)):
                self.assertTrue(torch.equal(a, b))


if __name__ == '__main__':
    unittest.main()
