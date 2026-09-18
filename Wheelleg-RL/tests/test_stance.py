"""Torch-free geometry tests for the stance contract and the robot config.

Run: uv run python tests/test_stance.py
"""
import importlib.util
import re
import unittest
from pathlib import Path

root = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, root / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stance = _load("wheelleg_stance", "src/wheelleg/stance.py")


class ReferenceStanceTests(unittest.TestCase):
    def test_supplied_reference_stand_is_below_the_locomotion_requirement(self):
        """Documents why the trained stance is not the supplied one."""
        hip, thigh, knee = stance.REFERENCE_STANCE
        clearance = stance.clearance(thigh, knee, hip)
        self.assertAlmostEqual(clearance, 0.125454, places=5)
        self.assertLess(clearance, stance.MIN_CLEARANCE)
        self.assertAlmostEqual(clearance, 0.125454, delta=0.002)

    def test_nominal_stance_clears_the_requirement_with_margin(self):
        hip, thigh, knee = stance.NOMINAL_STANCE
        clearance = stance.clearance(thigh, knee, hip)
        self.assertAlmostEqual(clearance, stance.STANDING_CLEARANCE, places=6)
        self.assertGreater(clearance - stance.MIN_CLEARANCE, 0.01)

    def test_nominal_stance_centres_the_wheel_under_the_base(self):
        """The support line must sit under the body, not 2 cm in front of it."""
        hip, thigh, knee = stance.NOMINAL_STANCE
        pose = stance.leg_pose(thigh, knee, hip)
        self.assertAlmostEqual(pose.axle_x, 0.0, places=6)
        # The reference pose is also nearly centred, so this is a small change.
        self.assertLess(abs(stance.leg_pose(*stance.REFERENCE_STANCE[1:]).axle_x), 0.01)

    def test_nominal_stance_puts_the_com_over_the_contact_line(self):
        hip, thigh, knee = stance.NOMINAL_STANCE
        self.assertLess(abs(stance.com_x(thigh, knee, hip)), 0.002)

    def test_axle_offset_is_monotonic_in_shank_tilt(self):
        offsets = [
            stance.leg_pose(t, shank - t).axle_x
            for shank in (-0.8, -0.6, -0.4, -0.2)
            for t in (stance._thigh_for_clearance(stance.STANDING_CLEARANCE, shank),)
            if t is not None
        ]
        self.assertEqual(offsets, sorted(offsets, reverse=True))

    def test_nominal_stance_respects_joint_limits(self):
        hip, thigh, knee = stance.NOMINAL_STANCE
        self.assertEqual(hip, 0.0)
        self.assertTrue(stance.THIGH_LIMIT[0] < thigh < stance.THIGH_LIMIT[1])
        self.assertTrue(stance.KNEE_LIMIT[0] < knee < stance.KNEE_LIMIT[1])

    def test_clearance_is_monotonic_in_leg_extension(self):
        heights = [stance.clearance(t, -0.55 - t) for t in (1.02, 0.9, 0.75, 0.5, 0.25)]
        self.assertEqual(heights, sorted(heights))
        self.assertGreater(heights[-1], heights[0])

    def test_thresholds_are_ordered_and_reachable(self):
        self.assertLess(stance.COLLAPSE_CLEARANCE, stance.MIN_CLEARANCE)
        self.assertLess(stance.MIN_CLEARANCE, stance.STANDING_CLEARANCE)
        self.assertLess(stance.STANDING_CLEARANCE, stance.clearance(0.0, 0.0))
        # The requirement must be reachable at all, with room to spare.
        self.assertGreater(stance.clearance(0.0, 0.0) - stance.MIN_CLEARANCE, 0.03)

    def test_wheel_geometry_is_consistent_with_the_reward(self):
        self.assertAlmostEqual(stance.WHEEL_RADIUS, 0.03, places=6)
        self.assertTrue(0.15 < stance.WHEEL_TRACK < 0.30)


class RobotConfigTests(unittest.TestCase):
    def test_robot_config_spawns_at_the_nominal_stance(self):
        source = (root / "src/wheelleg/robot_cfg.py").read_text(encoding="utf-8")
        self.assertIn("from .stance import", source)
        self.assertIn("NOMINAL_STANCE", source)
        self.assertIn("STANDING_CLEARANCE", source)
        # No stale hard-coded spawn height may reappear.
        self.assertNotIn("0.36", source)
        self.assertNotRegex(source, re.compile(r"pos=\(0\.0, 0\.0, 0\.\d+\)"))

    def test_collisions_are_limited_to_collision_geoms(self):
        source = (root / "src/wheelleg/robot_cfg.py").read_text(encoding="utf-8")
        self.assertIn('geom_names_expr=(".*_collision",)', source)


class EnvironmentConfigTests(unittest.TestCase):
    def test_locomotion_tasks_enforce_the_standing_contract(self):
        source = (root / "src/wheelleg/config/env_cfgs.py").read_text(encoding="utf-8")
        for term in ("standing.collapsed", "standing.low_height_barrier",
                     "standing.knee_ground_contact", "standing.standing_height_error"):
            self.assertIn(term, source)
        self.assertIn("COLLAPSE_CLEARANCE", source)
        self.assertIn("MIN_CLEARANCE", source)

    def test_recovery_does_not_terminate_on_low_height(self):
        """A recovery task must be allowed to start from the ground."""
        source = (root / "src/wheelleg/config/env_cfgs.py").read_text(encoding="utf-8")
        self.assertIn("enforce_standing=False", source)

    def test_reward_clamp_is_gone(self):
        """The global >=0 reward clamp would neutralise the anti-crawl barrier."""
        base = (root / "src/wheelleg/config/base_env_cfg.py").read_text(encoding="utf-8")
        self.assertNotIn("only_positive_rewards", base)
        self.assertFalse((root / "src/wheelleg/mdp/only_positive_rewards.py").exists())


if __name__ == "__main__":
    unittest.main()
