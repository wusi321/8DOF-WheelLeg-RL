"""Run with uv run python tests/test_wheel_tracking.py (Torch required)."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

# Load the actual reward without importing task registration or GPU simulation.
source = Path(__file__).resolve().parents[1] / "src/wheelleg/mdp/rewards.py"
tree = ast.parse(source.read_text(encoding="utf-8"))
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "wheel_roll_tracking")
module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
namespace = {"torch": torch}
exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
reward = namespace["wheel_roll_tracking"]


class WheelTrackingTests(unittest.TestCase):
    def test_straight_reverse_turn_and_stop(self):
        commands = torch.tensor([[1., 0., 0.], [-1., 0., 0.],
                                 [0., 0., 1.], [0., 0., -1.], [0., 0., 0.]])
        targets = torch.tensor([[10., 10.], [-10., -10.],
                                [-1.6, 1.6], [1.6, -1.6], [0., 0.]])
        for ids in ([0, 1], [1, 0]):
            velocities = torch.zeros(5, 2)
            velocities[:, ids] = targets

            def find_joints(names, preserve_order=False):
                self.assertEqual(names, ("left_wheel_joint", "right_wheel_joint"))
                self.assertTrue(preserve_order)
                return ids, list(names)

            asset = SimpleNamespace(data=SimpleNamespace(joint_vel=velocities), find_joints=find_joints)
            env = SimpleNamespace(scene={"wheelleg": asset},
                                  command_manager=SimpleNamespace(get_command=lambda _: commands))
            cfg = SimpleNamespace(name="wheelleg")
            result = reward(env, "twist", 0.1, 0.32, 3., cfg)
            self.assertEqual(tuple(result.shape), (5,))
            torch.testing.assert_close(result, torch.ones(5))
            asset.data.joint_vel = velocities + 1.
            self.assertTrue(torch.all(reward(env, "twist", 0.1, 0.32, 3., cfg) < 1.))


if __name__ == "__main__":
    unittest.main()
