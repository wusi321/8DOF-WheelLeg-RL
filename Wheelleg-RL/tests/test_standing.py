"""Torch-only numerical tests for the posture constraints.

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


def _env(clearances, speeds=None, yaws=None, base_contact=None, air=None,
         wheel_contact=None, command=None):
    """Fake env whose clearance ray hits `clearances` metres below base_link."""
    n = len(clearances)
    root_z = torch.tensor([[0.0, 0.0, float(c)] for c in clearances])
    hits = torch.zeros(n, 1, 1, 3)
    distances = torch.tensor([[float(c)] for c in clearances])
    speeds = [0.0] * n if speeds is None else speeds
    yaws = [0.0] * n if yaws is None else yaws
    contact = [0] * n if base_contact is None else base_contact
    air = [0.0] * n if air is None else air
    wheel_contact = [[1, 1]] * n if wheel_contact is None else wheel_contact
    command = [[1.0, 0.0, 0.0]] * n if command is None else command
    return SimpleNamespace(
        num_envs=n,
        command_manager=SimpleNamespace(
            get_command=lambda name: torch.tensor([[float(v) for v in c] for c in command])),
        scene={
            "wheelleg": SimpleNamespace(data=SimpleNamespace(
                root_link_pos_w=root_z,
                root_link_lin_vel_b=torch.tensor([[float(s), 0.0, 0.0] for s in speeds]),
                root_link_ang_vel_b=torch.tensor([[0.0, 0.0, float(y)] for y in yaws]),
            )),
            "base_clearance": SimpleNamespace(
                data=SimpleNamespace(hit_pos_w=hits, distances=distances)),
            "base_ground_contact": SimpleNamespace(
                data=SimpleNamespace(found=torch.tensor([[c] for c in contact]))),
            "feet_ground_contact": SimpleNamespace(data=SimpleNamespace(
                current_air_time=torch.tensor([[float(a)] for a in air]),
                found=torch.tensor([[float(v) for v in row] for row in wheel_contact]))),
        },
    )


class ClearanceTests(unittest.TestCase):
    def test_clearance_is_measured_not_assumed(self):
        env = _env([0.145, 0.12])
        self.assertAlmostEqual(standing.base_clearance(env)[0].item(), 0.145, places=6)

    def test_missed_ray_is_never_treated_as_crawling(self):
        env = _env([0.145], speeds=[1.0])
        env.scene["base_clearance"].data.distances[0, 0] = -1.0
        self.assertLess(standing.base_clearance(env)[0].item(), 0)
        self.assertEqual(standing.low_posture_locomotion(env)[0].item(), 0.0)


class MovingGateTests(unittest.TestCase):
    def test_parked_robot_has_no_gate(self):
        env = _env([0.10, 0.10], speeds=[0.0, 0.01])
        self.assertEqual(standing.moving_gate(env).tolist(), [0.0, 0.0])

    def test_translation_and_rotation_both_count_as_moving(self):
        env = _env([0.10] * 4,
                   speeds=[standing.LINEAR_SPEED_MOVING, 0.0, 0.0, 10.0],
                   yaws=[0.0, standing.YAW_RATE_MOVING, 0.0, 0.0])
        gate = standing.moving_gate(env)
        self.assertEqual(gate[0].item(), 1.0)
        self.assertEqual(gate[1].item(), 1.0)
        self.assertEqual(gate[2].item(), 0.0)
        self.assertEqual(gate[3].item(), 1.0)

    def test_gate_ramps_between_parked_and_moving(self):
        env = _env([0.10], speeds=[standing.LINEAR_SPEED_MOVING / 2])
        self.assertAlmostEqual(standing.moving_gate(env)[0].item(), 0.5, places=6)


class LowPostureLocomotionTests(unittest.TestCase):
    def test_parked_robot_may_crouch_or_lie_anywhere(self):
        """Low posture while stopped is legal, including flat on the ground."""
        env = _env([0.145, 0.10, 0.02, 0.0], speeds=[0.0] * 4)
        self.assertEqual(standing.low_posture_locomotion(env).tolist(), [0.0] * 4)

    def test_moving_at_the_required_height_is_free(self):
        env = _env([0.145, stance.MIN_CLEARANCE, 0.1317], speeds=[1.0] * 3)
        self.assertEqual(standing.low_posture_locomotion(env).tolist(), [0.0] * 3)

    def test_moving_below_the_requirement_is_penalised_progressively(self):
        env = _env([0.128, 0.12, 0.10], speeds=[1.0] * 3)
        penalty = standing.low_posture_locomotion(env)
        self.assertGreater(penalty[0].item(), 0.0)
        self.assertGreater(penalty[1].item(), penalty[0].item())
        self.assertGreater(penalty[2].item(), penalty[1].item())

    def test_crawl_penalty_beats_the_tracking_reward(self):
        """A 0.12 m crawl must be unprofitable, or the policy will choose it."""
        env = _env([0.12], speeds=[1.0])
        per_second = 100.0 * standing.low_posture_locomotion(env)[0].item()
        self.assertGreater(per_second, 5.0)

    def test_spinning_in_place_while_low_is_penalised(self):
        env = _env([0.11], speeds=[0.0], yaws=[1.0])
        self.assertGreater(standing.low_posture_locomotion(env)[0].item(), 0.0)


class HeightAndContactTests(unittest.TestCase):
    def test_height_error_zero_at_the_trained_stance(self):
        env = _env([stance.STANDING_CLEARANCE])
        self.assertAlmostEqual(standing.standing_height_error(env)[0].item(), 0.0, places=9)

    def test_lying_on_the_ground_is_expensive_but_not_fatal(self):
        env = _env([0.0])
        # Costs reward, and nothing in this module can end an episode.
        self.assertGreater(standing.standing_height_error(env)[0].item(), 0.5)
        self.assertFalse(hasattr(standing, "collapsed"))

    def test_base_contact_is_detected_not_terminated(self):
        env = _env([0.02, 0.02], base_contact=[1, 0])
        self.assertEqual(standing.base_ground_contact(env).tolist(), [True, False])


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


def _sym_env(hip_l, thigh_l, knee_l, hip_r, thigh_r, knee_r):
    q = torch.tensor([[hip_l, thigh_l, knee_l, hip_r, thigh_r, knee_r]])
    return SimpleNamespace(num_envs=1, scene={"wheelleg": SimpleNamespace(
        data=SimpleNamespace(joint_pos=q),
        find_joints=lambda names, preserve_order: (list(range(6)), names))})


class SymmetryTests(unittest.TestCase):
    def _mirrored(self):
        # The hip axes are not mirrored in the model, so a level pair is L = -R.
        return _sym_env(0.1, 0.85, -1.23, -0.1, 0.85, -1.23)

    def test_mirrored_legs_have_no_error(self):
        self.assertAlmostEqual(standing.leg_symmetry_error(self._mirrored())[0].item(), 0.0,
                               places=9)

    def test_same_sign_hips_are_treated_as_a_tilt(self):
        tilted = _sym_env(0.2, 0.85, -1.23, 0.2, 0.85, -1.23)
        self.assertGreater(standing.leg_symmetry_error(tilted)[0].item(), 0.0)

    def test_thigh_and_knee_mismatch_is_penalised_more_than_hips(self):
        knee = _sym_env(0.0, 0.85, -1.23, 0.0, 0.85, -0.93)
        hip = _sym_env(0.3, 0.85, -1.23, -0.3, 0.85, -1.23)
        self.assertGreater(standing.leg_symmetry_error(knee)[0].item(),
                           standing.leg_symmetry_error(hip)[0].item())

    def test_step_mismatch_scales_with_the_difference(self):
        small = _sym_env(0.0, 0.85, -1.23, 0.0, 0.65, -1.23)
        large = _sym_env(0.0, 0.85, -1.23, 0.0, 0.35, -1.23)
        self.assertGreater(standing.leg_symmetry_error(large)[0].item(),
                           standing.leg_symmetry_error(small)[0].item())


class WheelGroundTests(unittest.TestCase):
    def test_brief_lift_over_a_bump_is_free(self):
        env = _env([0.14] * 2, air=[0.0, standing.WHEEL_AIR_ALLOWANCE])
        self.assertEqual(standing.wheel_off_ground(env).tolist(), [0.0, 0.0])

    def test_sustained_lift_saturates(self):
        full = standing.WHEEL_AIR_ALLOWANCE + standing.WHEEL_AIR_HORIZON
        env = _env([0.14] * 3, air=[full, 0.4, 0.35])
        penalty = standing.wheel_off_ground(env)
        self.assertEqual(penalty[0].item(), 1.0)
        self.assertGreater(penalty[1].item(), penalty[2].item())
        self.assertGreater(penalty[2].item(), 0.0)

    def test_air_time_is_the_worst_wheel(self):
        env = _env([0.14])
        env.scene["feet_ground_contact"].data.current_air_time = torch.tensor([[0.1, 0.6]])
        self.assertAlmostEqual(standing.wheel_air_time(env)[0].item(), 0.6, places=6)

    def test_contact_fraction_counts_both_wheels(self):
        env = _env([0.14] * 3, wheel_contact=[[1, 1], [1, 0], [0, 0]])
        self.assertEqual(standing.wheel_contact_fraction(env).tolist(), [1.0, 0.5, 0.0])


class WheeledStanceLocomotionTests(unittest.TestCase):
    def test_kneeling_with_wheels_up_earns_nothing(self):
        """The exact failure seen in play: high reward for kneeling is impossible."""
        env = _env([0.03], speeds=[0.5], wheel_contact=[[0, 0]], command=[[0.5, 0.0, 0.0]])
        self.assertEqual(standing.wheeled_stance_locomotion(env)[0].item(), 0.0)

    def test_wheels_down_but_body_low_earns_nothing(self):
        env = _env([0.12], speeds=[0.5], command=[[0.5, 0.0, 0.0]])
        self.assertEqual(standing.wheeled_stance_locomotion(env)[0].item(), 0.0)

    def test_standing_still_earns_nothing(self):
        env = _env([0.145], speeds=[0.0], command=[[0.5, 0.0, 0.0]])
        self.assertEqual(standing.wheeled_stance_locomotion(env)[0].item(), 0.0)

    def test_no_command_earns_nothing(self):
        env = _env([0.145], speeds=[0.5], command=[[0.0, 0.0, 0.0]])
        self.assertEqual(standing.wheeled_stance_locomotion(env)[0].item(), 0.0)

    def test_reward_grows_towards_the_stance_height(self):
        env = _env([stance.MIN_CLEARANCE, 0.1375, stance.STANDING_CLEARANCE],
                   speeds=[0.5] * 3, command=[[0.5, 0.0, 0.0]] * 3)
        reward = standing.wheeled_stance_locomotion(env)
        self.assertEqual(reward[0].item(), 0.0)
        self.assertAlmostEqual(reward[1].item(), 0.5, places=6)
        self.assertAlmostEqual(reward[2].item(), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
