"""Torch-only numerical tests for the standing constraints.

Run on a machine with the training runtime installed: uv run python tests/test_standing.py
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

root = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, root / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


standing = _load("wheelleg_standing", "src/wheelleg/mdp/standing.py")
stance = _load("wheelleg_stance", "src/wheelleg/stance.py")


def _env(clearances):
    """Fake env whose clearance ray hits `clearances` metres below base_link."""
    n = len(clearances)
    root_z = torch.tensor([[0.0, 0.0, float(c)] for c in clearances])
    hits = torch.zeros(n, 1, 1, 3)
    distances = torch.tensor([[float(c)] for c in clearances])
    sensor = SimpleNamespace(data=SimpleNamespace(hit_pos_w=hits, distances=distances))
    return SimpleNamespace(
        num_envs=n,
        scene={
            "wheelleg": SimpleNamespace(data=SimpleNamespace(root_link_pos_w=root_z)),
            "base_clearance": sensor,
        },
    )


class ClearanceTests(unittest.TestCase):
    def test_clearance_is_measured_not_assumed(self):
        env = _env([0.145, 0.12])
        self.assertAlmostEqual(standing.base_clearance(env)[0].item(), 0.145, places=6)

    def test_missed_ray_is_never_treated_as_low(self):
        env = _env([0.145])
        env.scene["base_clearance"].data.distances[0, 0] = -1.0
        self.assertLess(standing.base_clearance(env)[0].item(), 0)
        self.assertFalse(standing.collapsed(env)[0].item())
        self.assertEqual(standing.low_height_barrier(env)[0].item(), 0.0)

    def test_collapse_threshold_sits_below_the_required_height(self):
        env = _env([0.145, 0.13, 0.125, 0.109])
        collapsed = standing.collapsed(env).tolist()
        self.assertEqual(collapsed, [False, False, False, True])

    def test_barrier_is_negligible_at_the_stance_and_severe_when_crawling(self):
        env = _env([0.145, 0.129, 0.115])
        barrier = standing.low_height_barrier(env)
        self.assertEqual(barrier[0].item(), 0.0)
        self.assertLess(barrier[1].item(), 0.01)
        self.assertGreater(barrier[2].item() * 100.0, 1.0)

    def test_height_error_zero_at_the_trained_stance(self):
        env = _env([stance.STANDING_CLEARANCE])
        self.assertAlmostEqual(standing.standing_height_error(env)[0].item(), 0.0, places=9)


class PoseTests(unittest.TestCase):
    def test_pose_error_zero_at_the_nominal_stance(self):
        q = torch.tensor([[stance.NOMINAL_STANCE * 2]])
        asset = SimpleNamespace(
            data=SimpleNamespace(joint_pos=q),
            find_joints=lambda names, preserve_order: (list(range(6)), names),
        )
        env = SimpleNamespace(num_envs=1, scene={"wheelleg": asset})
        self.assertAlmostEqual(standing.standing_pose_error(env)[0].item(), 0.0, places=9)
        asset.data.joint_pos = q + 0.2
        self.assertGreater(standing.standing_pose_error(env)[0].item(), 0.0)

    def test_knee_contact_terminates(self):
        env = SimpleNamespace(num_envs=2, scene={
            "knee_ground_contact": SimpleNamespace(data=SimpleNamespace(
                found=torch.tensor([[0, 0], [0, 1]])))})
        self.assertEqual(standing.knee_ground_contact(env).tolist(), [False, True])


if __name__ == "__main__":
    unittest.main()
