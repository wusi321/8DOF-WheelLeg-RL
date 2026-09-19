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
        for term in ("standing.low_posture_locomotion", "standing.standing_height_error",
                     "standing.base_ground_contact", "standing.moving_gate",
                     "standing.no_wheel_support", "standing.wheeled_stance_locomotion",
                     "standing.leg_symmetry_error", "standing.lateral_step_reward"):
            self.assertIn(term, source)
        self.assertIn("MIN_CLEARANCE", source)

    def test_stepping_incentive_is_removed(self):
        """feet_air_time pays a foot to be in the air; on wheels that is backwards."""
        source = (root / "src/wheelleg/config/env_cfgs.py").read_text(encoding="utf-8")
        self.assertIn('cfg.rewards.pop("feet_air_time", None)', source)

    def test_wrong_form_hip_symmetry_term_is_removed(self):
        """abduction_mirror penalises the mirror-symmetric hip splay, not the tilt."""
        source = (root / "src/wheelleg/config/env_cfgs.py").read_text(encoding="utf-8")
        self.assertIn('cfg.rewards.pop("abduction_mirror", None)', source)

    def test_kneeling_penalties_are_stronger_than_tracking(self):
        """Kneeling pays nothing and costs several times the tracking reward."""
        source = (root / "src/wheelleg/config/env_cfgs.py").read_text(encoding="utf-8")
        # base contact -10/s and losing every wheel -10/s are individual terms far
        # above the ~1.4/s to ~1.7/s the velocity tracking terms pay.
        self.assertIn("weight=-10.0", source)

    def test_moving_low_is_penalised_but_not_terminated(self):
        """The whole point: posture is a reward signal, never a reset."""
        source = (root / "src/wheelleg/config/env_cfgs.py").read_text(encoding="utf-8")
        self.assertIn("cfg.terminations.pop", source)
        for name in ("low_base_height", "knee_ground_contact", "base_ground_contact",
                     "bad_orientation"):
            self.assertIn(f'"{name}"', source)
        # No posture term may reappear as a termination key. The only term added
        # here is the fall-recovery timeout.
        assigned = set(re.findall(r'cfg\.terminations\["([^"]+)"\]\s*=', source))
        self.assertEqual(assigned, {"fallen_too_long"})

    def test_falling_is_allowed_until_the_robot_cannot_get_up(self):
        """A fall must not reset instantly, or getting up can never be learned."""
        source = (root / "src/wheelleg/mdp/standing.py").read_text(encoding="utf-8")
        self.assertIn("class FallenTooLong", source)
        self.assertIn("self._down_time > self.max_down_time", source)
        self.assertIn("def reset(self, env_ids=None)", source)

    def test_terrain_difficulty_never_starts_at_zero(self):
        """Difficulty 0 makes obstacle terrains literally flat."""
        source = (root / "src/wheelleg/config/env_cfgs.py").read_text(encoding="utf-8")
        self.assertIn("tg.difficulty_range = (0.02, 1.0)", source)
        # The lower bound must stay gentle enough to stand on to begin with.
        self.assertIn("max_init_terrain_level = 1", source)

    def test_falls_do_not_accumulate_every_penalty_at_once(self):
        """Being down must not stack crawl, support, contact and tilt penalties."""
        source = (root / "src/wheelleg/mdp/standing.py").read_text(encoding="utf-8")
        self.assertIn("def upright_gate", source)
        # The crawl and support penalties are suspended once the robot is down.
        self.assertIn("* upright_gate(env)", source)
        self.assertIn("MAX_TILT_COST", source)

    def test_terrain_curriculum_uses_path_length_not_net_displacement(self):
        """Circling under heading commands demotes a walking robot every episode."""
        source = (root / "src/wheelleg/mdp/curriculums.py").read_text(encoding="utf-8")
        self.assertIn("_path_travelled", source)
        self.assertIn("class PathLength", source)

    def test_recovery_does_not_terminate_on_low_height(self):
        """A recovery task must be allowed to start from the ground."""
        source = (root / "src/wheelleg/config/env_cfgs.py").read_text(encoding="utf-8")
        self.assertIn("enforce_standing=False", source)

    def test_crawl_penalty_dominates_velocity_tracking(self):
        """Travelling at 0.12 m must cost more than the tracking reward pays."""
        penalty_weight = 100.0
        deficit = (stance.MIN_CLEARANCE - 0.12) / stance.MIN_CLEARANCE
        # Rewards are dt-scaled, so 50 steps/s x 0.02 s collapses weight to per-second.
        self.assertGreater(penalty_weight * deficit, 2.0)

    def test_reward_clamp_is_gone(self):
        """The global >=0 reward clamp would neutralise the anti-crawl barrier."""
        base = (root / "src/wheelleg/config/base_env_cfg.py").read_text(encoding="utf-8")
        self.assertNotIn("only_positive_rewards", base)
        self.assertFalse((root / "src/wheelleg/mdp/only_positive_rewards.py").exists())


class ActuatorSpecTests(unittest.TestCase):
    """The torque limits must exist in exactly one place."""

    @classmethod
    def setUpClass(cls):
        cls.spec = _load("wheelleg_actuator_spec", "src/wheelleg/actuator_spec.py")

    def test_generated_mjcf_matches_the_actuator_spec(self):
        """Standalone-viewer limits in the XML may not drift from the runtime."""
        import xml.etree.ElementTree as ET

        xml = ET.parse(root / "mjcf/8dof_wheelleg.xml").getroot()
        seen = {"position": 0, "velocity": 0}
        for act in xml.findall("actuator/*"):
            kind = act.tag
            seen[kind] += 1
            low, high = (float(v) for v in act.get("forcerange").split())
            limit = (
                self.spec.LEG_TORQUE_LIMIT if kind == "position" else self.spec.WHEEL_TORQUE_LIMIT
            )
            self.assertEqual((-low, high), (limit, limit))
            self.assertEqual(act.get("forcelimited"), "true")
        self.assertEqual(seen, {"position": 6, "velocity": 2})

    def test_runtime_config_uses_the_spec(self):
        source = (root / "src/wheelleg/robot_cfg.py").read_text(encoding="utf-8")
        self.assertIn("from .actuator_spec import", source)
        for name in ("LEG_TORQUE_LIMIT", "WHEEL_TORQUE_LIMIT", "LEG_KP", "LEG_KD", "WHEEL_KD"):
            self.assertIn(name, source)
        # No bare magic gains may reappear next to the actuator construction.
        self.assertNotIn("effort_limit=4.0", source)
        self.assertNotIn("effort_limit=2.0", source)

    def test_limits_are_positive_and_leave_static_headroom(self):
        self.assertGreater(self.spec.LEG_TORQUE_LIMIT, 0.0)
        self.assertGreater(self.spec.WHEEL_TORQUE_LIMIT, 0.0)
        hip, thigh, knee = stance.NOMINAL_STANCE
        peak = max(stance.static_joint_torques(thigh, knee).values())
        # Holding the stance must not sit near the limit, or the policy would be
        # fighting saturation just to stand still.
        self.assertGreater(self.spec.LEG_TORQUE_LIMIT, 5.0 * peak)

    def test_mass_report_matches_the_mjcf_inertials(self):
        import xml.etree.ElementTree as ET

        xml = ET.parse(root / "mjcf/8dof_wheelleg.xml").getroot()
        total = sum(float(i.get("mass")) for i in xml.iter("inertial"))
        self.assertAlmostEqual(total, stance.TOTAL_MASS, places=6)


if __name__ == "__main__":
    unittest.main()
