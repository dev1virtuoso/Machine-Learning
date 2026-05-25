import unittest
import torch
import numpy as np
from module import LepauteConfig, DepthAwareSE3Warping, TransformerModel, DenseSE3Tracker

class TestLepauteModule(unittest.TestCase):
    def setUp(self):
        # Override for strict CI/CD environments and force deterministic hardware graphs
        self.config = LepauteConfig(device="cpu")
        self.batch_size = 2
        self.dummy_input = torch.randn(self.batch_size, 3, self.config.orig_h, self.config.orig_w)
        self.dummy_xi = torch.zeros(self.batch_size, 6) 
        self.dummy_depth = torch.ones(self.batch_size, self.config.orig_h, self.config.orig_w)

    def test_equivariant_block_identity(self):
        block = DepthAwareSE3Warping(self.config)
        feat = torch.randn(self.batch_size, 16, 14, 14)
        depth_resized = torch.ones(self.batch_size, 14, 14)
        
        output = block(feat, depth_resized, self.dummy_xi)
        self.assertEqual(output.shape, feat.shape)
        self.assertTrue(torch.allclose(output, feat, atol=1e-4))

    def test_warping_nan_safety(self):
        block = DepthAwareSE3Warping(self.config)
        feat = torch.randn(self.batch_size, 16, 14, 14)
        zero_depth = torch.zeros(self.batch_size, 14, 14) 
        xi_motion = torch.ones(self.batch_size, 6) * 1.5
        
        output = block(feat, zero_depth, xi_motion)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_multitask_backprop_flow(self):
        model = TransformerModel(self.config)
        xi = torch.randn(1, 6, requires_grad=True)
        img = torch.randn(1, 3, self.config.orig_h, self.config.orig_w, requires_grad=True)
        depth = torch.ones(1, self.config.orig_h, self.config.orig_w, requires_grad=True)
        
        # Test the newly integrated pose decoupling outputs
        embed, pose_pred = model(img, depth, xi)
        
        loss = embed.sum() + pose_pred.sum()
        loss.backward()
        
        self.assertEqual(pose_pred.shape, (1, 6))
        self.assertIsNotNone(img.grad)
        self.assertIsNotNone(xi.grad)

    def test_tracker_initialization_and_degradation(self):
        tracker = DenseSE3Tracker(self.config)
        frame1 = np.zeros((self.config.orig_h, self.config.orig_w, 3), dtype=np.uint8)
        frame2 = np.ones((self.config.orig_h, self.config.orig_w, 3), dtype=np.uint8) * 255
        
        xi, rmse, depth = tracker.estimate_pose(frame1, frame2, run_depth=True)
        self.assertEqual(len(xi), 6)
        self.assertTrue(rmse > 0)
        self.assertIsNotNone(tracker.depth_ema)

if __name__ == "__main__":
    unittest.main()