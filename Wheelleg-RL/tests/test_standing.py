"""CPU-only numerical tests: uv run python tests/test_standing.py."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch

path = Path(__file__).resolve().parents[1] / "src/wheelleg/mdp/standing.py"
spec = importlib.util.spec_from_file_location("standing", path)
standing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(standing)


class StandingTests(unittest.TestCase):
    def test_local_clearance_and_threshold(self):
        root = torch.tensor([[0., 0., .129], [0., 0., .13], [0., 0., 1.15]])
        sensor = SimpleNamespace(data=SimpleNamespace(
            hit_pos_w=torch.tensor([[[0., 0., 0.]], [[0., 0., 0.]], [[0., 0., 1.]]]),
            distances=torch.tensor([[.129], [.13], [.15]])))
        env = SimpleNamespace(num_envs=3, scene={
            "wheelleg": SimpleNamespace(data=SimpleNamespace(root_link_pos_w=root)),
            "base_clearance": sensor})
        self.assertEqual(standing.below_standing_height(env).tolist(), [True, False, False])
        sensor.data.distances[2, 0] = -1
        self.assertTrue(standing.below_standing_height(env)[2])

    def test_pose_and_knee_contact(self):
        q = torch.tensor([[0., 1.02, -1.57, 0., 1.02, -1.57]])
        asset = SimpleNamespace(data=SimpleNamespace(joint_pos=q),
                                find_joints=lambda names, preserve_order: (list(range(6)), names))
        env = SimpleNamespace(num_envs=1, scene={"wheelleg": asset,
            "knee_ground_contact": SimpleNamespace(data=SimpleNamespace(found=torch.tensor([[0, 1]])))})
        self.assertEqual(standing.standard_pose_error(env).item(), 0.)
        self.assertTrue(standing.knee_ground_contact(env).item())
        asset.data.joint_pos = q + .2
        self.assertGreater(standing.standard_pose_error(env).item(), 0.)


if __name__ == "__main__":
    unittest.main()
