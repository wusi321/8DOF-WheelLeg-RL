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
         wheel_contact=None, command=None, gravity=None, step_dt=0.02, lateral=None):
    """Fake env whose clearance ray hits `clearances` metres below base_link.

    ``speeds`` is the body-frame forward speed and ``lateral`` the sideways one.
    """
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
    # Default to a perfectly upright body: projected gravity is (0, 0, -1).
    gravity = [[0.0, 0.0, -1.0]] * n if gravity is None else gravity
    lateral = [0.0] * n if lateral is None else lateral
    return SimpleNamespace(
        num_envs=n,
        step_dt=step_dt,
        # Not a fresh episode, so per-episode state persists across calls.
        episode_length_buf=torch.full((n,), 5, dtype=torch.long),
        command_manager=SimpleNamespace(
            get_command=lambda name: torch.tensor([[float(v) for v in c] for c in command])),
        scene={
            "wheelleg": SimpleNamespace(
                data=SimpleNamespace(
                    root_link_pos_w=root_z,
                    root_link_lin_vel_b=torch.tensor(
                        [[float(s), float(y), 0.0] for s, y in zip(speeds, lateral)]),
                    root_link_ang_vel_b=torch.tensor([[0.0, 0.0, float(y)] for y in yaws]),
                    projected_gravity_b=torch.tensor([[float(v) for v in g] for g in gravity]),
                    # Legs at zero are nowhere near the folded pose.
                    joint_pos=torch.zeros(n, 6),
                ),
                find_joints=lambda names, preserve_order: (list(range(6)), names)),
            "base_clearance": SimpleNamespace(
                data=SimpleNamespace(hit_pos_w=hits, distances=distances)),
            "base_ground_contact": SimpleNamespace(
                data=SimpleNamespace(found=torch.tensor([[c] for c in contact]))),
            "feet_ground_contact": SimpleNamespace(data=SimpleNamespace(
                current_air_time=torch.tensor([[float(a)] for a in air]),
                found=torch.tensor([[float(v) for v in row] for row in wheel_contact]))),
        },
    )


def _set_clearance(env, value, index=0):
    """Move one environment to a new clearance without rebuilding it."""
    env.scene["base_clearance"].data.distances[index, 0] = value
    env.scene["wheelleg"].data.root_link_pos_w[index, 2] = value


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


def _sym_env(hip_l, thigh_l, knee_l, hip_r, thigh_r, knee_r, command=(1.0, 0.0, 0.0),
             wheel_z=(0.0, 0.0)):
    q = torch.tensor([[hip_l, thigh_l, knee_l, hip_r, thigh_r, knee_r]])
    cmd = torch.tensor([[float(v) for v in command]])
    z = torch.zeros(1, 2, 3)
    z[0, :, 2] = torch.tensor([float(v) for v in wheel_z])
    return SimpleNamespace(
        num_envs=1,
        command_manager=SimpleNamespace(get_command=lambda name: cmd),
        scene={
            "wheelleg": SimpleNamespace(
                data=SimpleNamespace(
                    joint_pos=q,
                    body_link_pos_w=z,
                    root_link_lin_vel_b=torch.zeros(1, 3),
                ),
                find_joints=lambda names, preserve_order: (list(range(6)), names)),
        })


class SymmetryTests(unittest.TestCase):
    def _asset(self):
        # The SceneEntityCfg the manager would have resolved for the two wheels.
        return SimpleNamespace(name="wheelleg", body_ids=[0, 1])

    def _mirrored(self, **kwargs):
        # The hip axes are not mirrored in the model, so a level pair is L = -R.
        return _sym_env(0.1, 0.85, -1.23, -0.1, 0.85, -1.23, **kwargs)

    def test_mirrored_legs_have_no_error(self):
        env = self._mirrored()
        self.assertAlmostEqual(
            standing.leg_symmetry_error(env, asset_cfg=self._asset())[0].item(), 0.0, places=9)

    def test_same_sign_hips_are_treated_as_a_tilt(self):
        env = _sym_env(0.2, 0.85, -1.23, 0.2, 0.85, -1.23)
        self.assertGreater(
            standing.leg_symmetry_error(env, asset_cfg=self._asset())[0].item(), 0.0)

    def test_thigh_and_knee_mismatch_is_penalised_more_than_hips(self):
        asset = self._asset()
        knee = _sym_env(0.0, 0.85, -1.23, 0.0, 0.85, -0.93)
        hip = _sym_env(0.3, 0.85, -1.23, -0.3, 0.85, -1.23)
        self.assertGreater(
            standing.leg_symmetry_error(knee, asset_cfg=asset)[0].item(),
            standing.leg_symmetry_error(hip, asset_cfg=asset)[0].item())

    def test_step_mismatch_scales_with_the_difference(self):
        asset = self._asset()
        small = _sym_env(0.0, 0.85, -1.23, 0.0, 0.65, -1.23)
        large = _sym_env(0.0, 0.85, -1.23, 0.0, 0.35, -1.23)
        self.assertGreater(
            standing.leg_symmetry_error(large, asset_cfg=asset)[0].item(),
            standing.leg_symmetry_error(small, asset_cfg=asset)[0].item())

    def test_forward_commands_on_level_ground_demand_symmetry(self):
        env = _sym_env(0.0, 0.85, -1.23, 0.0, 0.45, -1.23, command=(1.0, 0.0, 0.0))
        self.assertGreater(
            standing.leg_symmetry_error(env, asset_cfg=self._asset())[0].item(), 0.0)

    def test_sideways_commands_release_symmetry_for_stepping(self):
        """Sideways travel has to step, so the mismatch is legitimate there."""
        env = _sym_env(0.0, 0.85, -1.23, 0.0, 0.45, -1.23,
                       command=(0.0, standing.LATERAL_COMMAND_REF, 0.0))
        self.assertEqual(
            standing.leg_symmetry_error(env, asset_cfg=self._asset())[0].item(), 0.0)

    def test_uneven_ground_releases_symmetry_so_the_body_can_stay_level(self):
        """Wheels at different heights: the legs must be allowed to differ."""
        env = _sym_env(0.0, 0.85, -1.23, 0.0, 0.45, -1.23,
                       wheel_z=(standing.UNEVEN_HEIGHT_REFERENCE, 0.0))
        self.assertEqual(
            standing.leg_symmetry_error(env, asset_cfg=self._asset())[0].item(), 0.0)

    def test_a_gentle_bank_only_partly_releases_symmetry(self):
        env = _sym_env(0.0, 0.85, -1.23, 0.0, 0.45, -1.23,
                       wheel_z=(standing.UNEVEN_HEIGHT_REFERENCE / 2, 0.0))
        level = _sym_env(0.0, 0.85, -1.23, 0.0, 0.45, -1.23)
        asset = self._asset()
        self.assertAlmostEqual(
            standing.leg_symmetry_error(env, asset_cfg=asset)[0].item(),
            0.5 * standing.leg_symmetry_error(level, asset_cfg=asset)[0].item(),
            places=6)

    def test_wheel_height_difference_ignores_body_attitude(self):
        """It measures the ground, so tilting the body does not change it."""
        env = _sym_env(0.0, 0.85, -1.23, 0.0, 0.85, -1.23, wheel_z=(0.02, -0.02))
        self.assertAlmostEqual(
            standing.wheel_height_difference(env, self._asset())[0].item(), 0.04, places=6)


class WheelSupportTests(unittest.TestCase):
    def test_one_leg_lifted_still_has_support(self):
        """A sideways step needs one wheel up; that must not count as a fault."""
        env = _env([0.14])
        env.scene["feet_ground_contact"].data.current_air_time = torch.tensor([[0.9, 0.0]])
        self.assertEqual(standing.wheel_support_time(env)[0].item(), 0.0)
        self.assertEqual(standing.no_wheel_support(env)[0].item(), 0.0)

    def test_losing_every_wheel_is_a_fault(self):
        env = _env([0.14])
        env.scene["feet_ground_contact"].data.current_air_time = torch.tensor([[0.9, 1.2]])
        self.assertAlmostEqual(standing.wheel_support_time(env)[0].item(), 0.9, places=6)
        self.assertGreater(standing.no_wheel_support(env)[0].item(), 0.0)

    def test_brief_hop_is_free(self):
        env = _env([0.14] * 2)
        env.scene["feet_ground_contact"].data.current_air_time = torch.tensor(
            [[standing.WHEEL_AIR_ALLOWANCE] * 2, [0.03, 0.03]])
        self.assertEqual(standing.no_wheel_support(env).tolist(), [0.0, 0.0])

    def test_no_support_saturates(self):
        full = standing.WHEEL_AIR_ALLOWANCE + standing.WHEEL_AIR_HORIZON
        env = _env([0.14] * 2)
        env.scene["feet_ground_contact"].data.current_air_time = torch.tensor(
            [[full, full], [0.4, 0.4]])
        penalty = standing.no_wheel_support(env)
        self.assertEqual(penalty[0].item(), 1.0)
        self.assertGreater(penalty[1].item(), 0.0)

    def test_air_time_is_the_worst_wheel(self):
        env = _env([0.14])
        env.scene["feet_ground_contact"].data.current_air_time = torch.tensor([[0.1, 0.6]])
        self.assertAlmostEqual(standing.wheel_air_time(env)[0].item(), 0.6, places=6)

    def test_contact_fraction_counts_both_wheels(self):
        env = _env([0.14] * 3, wheel_contact=[[1, 1], [1, 0], [0, 0]])
        self.assertEqual(standing.wheel_contact_fraction(env).tolist(), [1.0, 0.5, 0.0])


class LateralStepTests(unittest.TestCase):
    def test_lateral_demand_scales_with_the_command(self):
        env = _env([0.14], command=[[0.0, standing.LATERAL_COMMAND_REF / 2, 0.0]])
        self.assertAlmostEqual(standing.lateral_command_demand(env)[0].item(), 0.5, places=6)
        env = _env([0.14], command=[[0.0, standing.LATERAL_COMMAND_REF * 4, 0.0]])
        self.assertEqual(standing.lateral_command_demand(env)[0].item(), 1.0)
        env = _env([0.14], command=[[0.0, 0.0, 0.0]])
        self.assertEqual(standing.lateral_command_demand(env)[0].item(), 0.0)

    def test_step_reward_needs_a_sideways_command(self):
        env = _env([0.14], wheel_contact=[[1, 0]], command=[[1.0, 0.0, 0.0]])
        env.scene["feet_ground_contact"].data.current_air_time = torch.tensor([[0.3, 0.0]])
        self.assertEqual(standing.lateral_step_reward(env)[0].item(), 0.0)

    def test_step_reward_pays_for_a_sideways_step(self):
        env = _env([0.14], lateral=[standing.LATERAL_SPEED_MOVING], wheel_contact=[[1, 0]],
                   command=[[0.0, standing.LATERAL_COMMAND_REF, 0.0]])
        env.scene["feet_ground_contact"].data.current_air_time = torch.tensor([[0.3, 0.0]])
        self.assertEqual(standing.lateral_step_reward(env)[0].item(), 1.0)

    def test_step_reward_does_not_pay_when_every_wheel_is_up(self):
        env = _env([0.14], wheel_contact=[[0, 0]],
                   command=[[0.0, standing.LATERAL_COMMAND_REF, 0.0]])
        env.scene["feet_ground_contact"].data.current_air_time = torch.tensor([[0.3, 0.3]])
        self.assertEqual(standing.lateral_step_reward(env)[0].item(), 0.0)

    def test_step_reward_needs_real_sideways_motion(self):
        """Chattering a wheel up and down is not travelling sideways."""
        env = _env([0.14], wheel_contact=[[1, 0]],
                   command=[[0.0, standing.LATERAL_COMMAND_REF, 0.0]])
        env.scene["feet_ground_contact"].data.current_air_time = torch.tensor([[0.3, 0.0]])
        self.assertEqual(standing.lateral_step_reward(env)[0].item(), 0.0)
        env.scene["wheelleg"].data.root_link_lin_vel_b = torch.tensor(
            [[0.0, standing.LATERAL_SPEED_MOVING, 0.0]])
        self.assertEqual(standing.lateral_step_reward(env)[0].item(), 1.0)


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


class FallenMaskTests(unittest.TestCase):
    def test_upright_and_high_is_not_fallen(self):
        env = _env([0.145])
        self.assertEqual(standing.fallen_mask(env)[0].item(), 0.0)
        self.assertEqual(standing.recovered_mask(env)[0].item(), 1.0)

    def test_tilt_alone_counts_as_fallen(self):
        """The box chassis can wedge on a side without the body sensor firing."""
        import math

        env = _env([0.145], gravity=[(math.sin(1.0), 0.0, -math.cos(1.0))])
        self.assertEqual(standing.fallen_mask(env)[0].item(), 1.0)

    def test_low_alone_counts_as_fallen(self):
        env = _env([0.03])
        self.assertEqual(standing.fallen_mask(env)[0].item(), 1.0)

    def test_a_crouch_below_the_gate_is_not_recovered(self):
        """A crouch just under the arming gate must not read as standing."""
        env = _env([0.09])
        self.assertEqual(standing.fallen_mask(env)[0].item(), 0.0)
        self.assertEqual(standing.recovered_mask(env)[0].item(), 0.0)


class PotentialShapingTests(unittest.TestCase):
    """Shaping pays for change, never for holding a pose, so it cannot be farmed."""

    def test_rising_pays_and_holding_pays_zero(self):
        low = _env([0.02])
        high = _env([0.145])
        up = standing.height_progress(high)
        self.assertGreater(up[0].item(), 0.0)
        # Holding the same height pays exactly nothing on every further step.
        for _ in range(5):
            self.assertAlmostEqual(standing.height_progress(high)[0].item(), 0.0, places=9)

    def test_falling_is_charged(self):
        high = _env([0.145])
        low = _env([0.02])
        standing.height_progress(high)
        self.assertLess(standing.height_progress(low)[0].item(), 0.0)

    def test_upright_progress_pays_for_standing_up(self):
        import math

        down = _env([0.02], gravity=[(math.sin(1.4), 0.0, -math.cos(1.4))])
        up = _env([0.145])
        standing.upright_progress(down)
        self.assertGreater(standing.upright_progress(up)[0].item(), 0.0)

    def test_a_fresh_episode_does_not_pay_for_the_reset(self):
        """Otherwise every reset is a free bounty."""
        env = _env([0.02])
        env.episode_length_buf = torch.tensor([0])
        standing.height_progress(env)
        # The reset teleports the robot up; the first step must not pay for that.
        _set_clearance(env, 0.145)
        env.episode_length_buf = torch.tensor([1])
        self.assertAlmostEqual(standing.height_progress(env)[0].item(), 0.0, places=9)


class FallenTaxTests(unittest.TestCase):
    def test_an_upright_robot_is_never_taxed(self):
        env = _env([0.145])
        for _ in range(5):
            self.assertEqual(standing.fallen_tax(env)[0].item(), 0.0)

    def test_a_fall_arms_the_tax(self):
        env = _env([0.03])
        self.assertEqual(standing.fallen_tax(env)[0].item(), 1.0)

    def test_the_tax_has_hysteresis_until_genuinely_up(self):
        """Otherwise a crouch under the gate becomes a free rest state."""
        env = _env([0.02])
        self.assertEqual(standing.fallen_tax(env)[0].item(), 1.0)
        _set_clearance(env, 0.09)  # crouch: above the arming gate, below recovered
        self.assertEqual(standing.fallen_mask(env)[0].item(), 0.0)
        self.assertEqual(standing.fallen_tax(env)[0].item(), 1.0)
        _set_clearance(env, 0.145)  # genuinely up
        self.assertEqual(standing.fallen_tax(env)[0].item(), 0.0)

    def test_a_fresh_episode_clears_the_tax(self):
        env = _env([0.02])
        standing.fallen_tax(env)
        env.episode_length_buf = torch.tensor([0])
        _set_clearance(env, 0.145)
        self.assertEqual(standing.fallen_tax(env)[0].item(), 0.0)


class RecoverySuccessTests(unittest.TestCase):
    def test_a_completed_recovery_pays_once(self):
        import math

        down = _env([0.02], gravity=[(math.sin(1.4), 0.0, -math.cos(1.4))])
        up = _env([0.145])
        term = standing.recovery_success
        for _ in range(40):  # 0.8 s fallen, past min_fallen_s
            term(down, up_clearance=0.11)
        self.assertEqual(term(up, up_clearance=0.11)[0].item(), 1.0)
        # Re-arming needs another fall, so oscillating around the gate pays zero.
        self.assertEqual(term(up, up_clearance=0.11)[0].item(), 0.0)

    def test_being_merely_high_is_not_a_recovery(self):
        up = _env([0.145])
        self.assertEqual(standing.recovery_success(up, up_clearance=0.11)[0].item(), 0.0)


class FoldedPoseTests(unittest.TestCase):
    def _folded_env(self, joints):
        q = torch.tensor([list(joints)])
        return SimpleNamespace(num_envs=1, scene={"wheelleg": SimpleNamespace(
            data=SimpleNamespace(joint_pos=q),
            find_joints=lambda names, preserve_order: (list(range(6)), names))})

    def test_the_folded_pose_reports_complete(self):
        hip, thigh, knee = standing.FOLDED_STANCE
        env = self._folded_env([hip, thigh, knee] * 2)
        self.assertTrue(standing.fold_complete(env)[0].item())
        self.assertAlmostEqual(standing.folded_pose_error(env)[0].item(), 0.0, places=9)

    def test_a_sprawled_pose_is_not_folded(self):
        env = self._folded_env([0.0, 0.85, -1.23] * 2)
        self.assertFalse(standing.fold_complete(env)[0].item())
        self.assertGreater(standing.folded_pose_error(env)[0].item(), 0.0)

    def test_one_leg_folded_is_not_enough(self):
        hip, thigh, knee = standing.FOLDED_STANCE
        env = self._folded_env([hip, thigh, knee, 0.0, 0.85, -1.23])
        self.assertFalse(standing.fold_complete(env)[0].item())

    def test_the_folded_pose_is_inside_the_joint_limits(self):
        """The supplied angles are clamped: hip 0.91 and thigh 1.31 are outside."""
        hip, thigh, knee = standing.FOLDED_STANCE
        self.assertLessEqual(hip, stance.HIP_LIMIT[1])
        self.assertLessEqual(thigh, stance.THIGH_LIMIT[1])
        self.assertGreaterEqual(knee, stance.KNEE_LIMIT[0])
        self.assertLess(knee, 0.0)
        # The reported values really were outside, which is why clamping matters.
        self.assertGreater(0.91, stance.HIP_LIMIT[1])
        self.assertGreater(1.31, stance.THIGH_LIMIT[1])


class FallenTooLongTests(unittest.TestCase):
    def test_an_upright_robot_never_terminates(self):
        env = _env([0.145])
        for _ in range(400):
            self.assertFalse(standing.fallen_too_long(env, max_down_time=1.0).any().item())

    def test_a_fall_gets_a_window_before_the_backstop(self):
        env = _env([0.02])
        seen = [standing.fallen_too_long(env, max_down_time=1.0).item() for _ in range(60)]
        self.assertFalse(any(seen[:40]))  # 0.8 s down is still allowed
        self.assertTrue(seen[-1])         # 1.2 s down: recycled

    def test_folding_starts_the_stand_up_deadline(self):
        """Once the legs are folded the robot gets 2 s to be upright again."""
        hip, thigh, knee = standing.FOLDED_STANCE
        env = _env([0.02])
        env.scene["wheelleg"].data.joint_pos = torch.tensor([[hip, thigh, knee] * 2])
        seen = [standing.fallen_too_long(env, max_down_time=99.0).item() for _ in range(140)]
        # 2 s at 20 ms is 100 steps; nothing before that, then it fires.
        self.assertFalse(any(seen[:95]))
        self.assertTrue(seen[-1])

    def test_a_jammed_fall_that_never_folds_hits_the_backstop(self):
        """Legs never reach the folded pose, so only the down timer applies."""
        env = _env([0.02])  # joint_pos stays at zero: nowhere near folded
        seen = [standing.fallen_too_long(env, max_down_time=1.0).item() for _ in range(60)]
        self.assertFalse(any(seen[:40]))
        self.assertTrue(seen[-1])

    def test_standing_up_clears_both_clocks(self):
        down = _env([0.02])
        standing.fallen_too_long(down, max_down_time=1.0)
        up = _env([0.145])
        self.assertFalse(standing.fallen_too_long(up, max_down_time=1.0).item())


class AttemptScaleTests(unittest.TestCase):
    def test_attempt_taxes_are_reduced_while_down(self):
        self.assertAlmostEqual(standing.attempt_scale(_env([0.03]))[0].item(),
                               standing.FALLEN_ATTEMPT_SCALE, places=9)

    def test_attempt_taxes_are_full_while_up(self):
        self.assertAlmostEqual(standing.attempt_scale(_env([0.145]))[0].item(), 1.0, places=9)


class BodyLevelTests(unittest.TestCase):
    """The body's z axis must stay vertical rather than following the slope."""

    @staticmethod
    def _gravity(pitch):
        """Projected gravity for a body pitch, positive meaning leaning backwards.

        ``body_tilt`` recovers the pitch as ``asin(g_x)``, so this is its inverse.
        """
        import math

        return [(math.sin(pitch), 0.0, -math.cos(pitch))]

    def test_upright_body_costs_nothing(self):
        env = _env([0.145], gravity=self._gravity(0.0))
        self.assertAlmostEqual(standing.body_level_error(env)[0].item(), 0.0, places=9)

    def test_forward_lean_is_free_within_the_allowance(self):
        env = _env([0.145], gravity=self._gravity(-0.9 * standing.FORWARD_LEAN_ALLOWANCE))
        self.assertAlmostEqual(standing.body_level_error(env)[0].item(), 0.0, places=9)

    def test_excessive_forward_lean_is_charged(self):
        env = _env([0.145], gravity=self._gravity(-2 * standing.FORWARD_LEAN_ALLOWANCE))
        self.assertGreater(standing.body_level_error(env)[0].item(), 0.0)

    def test_backward_lean_costs_more_than_the_same_forward_lean(self):
        angle = standing.FORWARD_LEAN_ALLOWANCE  # at the edge of the free band
        back = _env([0.145], gravity=self._gravity(angle))
        forward = _env([0.145], gravity=self._gravity(-angle))
        self.assertGreater(standing.body_level_error(back)[0].item(),
                           standing.body_level_error(forward)[0].item())

    def test_roll_on_a_bank_is_charged(self):
        """Lying along a lateral slope must not be free."""
        import math

        roll = 0.2
        env = _env([0.145], gravity=[(0.0, math.sin(roll), -math.cos(roll))])
        self.assertAlmostEqual(standing.body_level_error(env)[0].item(), roll, places=6)

    def test_tilt_cost_is_linear_not_squared(self):
        """A squared cost is flat near upright and cannot pull back a steady bias."""
        small = standing.body_level_error(_env([0.145], gravity=self._gravity(0.1)))[0].item()
        large = standing.body_level_error(_env([0.145], gravity=self._gravity(0.2)))[0].item()
        self.assertAlmostEqual(large / small, 2.0, places=5)


class CrouchAtSpeedTests(unittest.TestCase):
    def test_parked_robot_keeps_the_full_requirement(self):
        env = _env([0.13], speeds=[0.0])
        self.assertAlmostEqual(standing.effective_min_clearance(env)[0].item(),
                               stance.MIN_CLEARANCE, places=9)

    def test_high_speed_relaxes_both_thresholds_by_the_same_drop(self):
        env = _env([0.145], speeds=[standing.CROUCH_SPEED_REFERENCE * 2])
        self.assertAlmostEqual(standing.effective_min_clearance(env)[0].item(),
                               stance.MIN_CLEARANCE - standing.CROUCH_CLEARANCE_DROP, places=9)
        self.assertAlmostEqual(standing.effective_target_height(env)[0].item(),
                               stance.STANDING_CLEARANCE - standing.CROUCH_CLEARANCE_DROP,
                               places=9)

    def test_reversing_gets_the_same_allowance(self):
        forward = _env([0.145], speeds=[standing.CROUCH_SPEED_REFERENCE])
        reverse = _env([0.145], speeds=[-standing.CROUCH_SPEED_REFERENCE])
        self.assertAlmostEqual(standing.crouch_gate(forward)[0].item(),
                               standing.crouch_gate(reverse)[0].item(), places=9)

    def test_a_fast_crouch_is_not_charged_as_a_crawl(self):
        height = stance.MIN_CLEARANCE - standing.CROUCH_CLEARANCE_DROP
        env = _env([height], speeds=[standing.CROUCH_SPEED_REFERENCE * 2])
        self.assertEqual(standing.low_posture_locomotion(env)[0].item(), 0.0)

    def test_the_same_crouch_is_a_crawl_when_parked_and_moving_slowly(self):
        height = stance.MIN_CLEARANCE - standing.CROUCH_CLEARANCE_DROP
        env = _env([height], speeds=[standing.LINEAR_SPEED_MOVING])
        self.assertGreater(standing.low_posture_locomotion(env)[0].item(), 0.0)

    def test_the_requirement_never_falls_below_the_documented_floor(self):
        env = _env([0.145], speeds=[10.0])
        floor = stance.MIN_CLEARANCE - standing.CROUCH_CLEARANCE_DROP
        self.assertAlmostEqual(standing.effective_min_clearance(env)[0].item(), floor, places=9)
        self.assertGreater(floor, 0.11)


class FallenTooLongTests(unittest.TestCase):
    def _term(self, max_down_time=1.0, tilt_limit=0.9):
        cfg = SimpleNamespace(params={"max_down_time": max_down_time, "tilt_limit": tilt_limit})
        return cfg

    def test_an_upright_robot_never_terminates(self):
        env = _env([0.145])
        term = standing.FallenTooLong(self._term(0.1), env)
        for _ in range(20):
            self.assertFalse(term(env, 0.1, 0.9).any().item())

    def test_a_fall_is_tolerated_for_a_while_then_ends_the_episode(self):
        """The delay is what gives the policy time to practise standing up."""
        import math

        env = _env([0.02], base_contact=[1],
                   gravity=[(math.sin(1.4), 0.0, -math.cos(1.4))])
        term = standing.FallenTooLong(self._term(max_down_time=0.5), env)
        seen = [term(env, 0.5, 0.9).item() for _ in range(40)]
        self.assertFalse(any(seen[:20]))  # 0.4 s down: still allowed
        self.assertTrue(seen[-1])         # 0.8 s down: give up

    def test_recovering_resets_the_timer(self):
        import math

        down = _env([0.02], gravity=[(math.sin(1.4), 0.0, -math.cos(1.4))])
        up = _env([0.145])
        term = standing.FallenTooLong(self._term(max_down_time=0.5), down)
        for _ in range(20):
            term(down, 0.5, 0.9)
        self.assertFalse(term(up, 0.5, 0.9).item())
        for _ in range(20):
            self.assertFalse(term(up, 0.5, 0.9).any().item())

    def test_reset_clears_the_accumulated_time(self):
        import math

        env = _env([0.02], gravity=[(math.sin(1.4), 0.0, -math.cos(1.4))])
        term = standing.FallenTooLong(self._term(max_down_time=0.5), env)
        for _ in range(40):
            term(env, 0.5, 0.9)
        term.reset()
        self.assertFalse(term(env, 0.5, 0.9).item())


class DownStateTests(unittest.TestCase):
    """A fall must not stack every posture penalty on top of the next."""

    @staticmethod
    def _down_env(**kwargs):
        import math

        return _env([0.03], gravity=[(math.sin(1.4), 0.0, -math.cos(1.4))],
                    base_contact=[1], **kwargs)

    def test_a_fallen_robot_is_detected(self):
        self.assertTrue(standing.is_down(self._down_env()).all().item())
        self.assertFalse(standing.is_down(_env([0.145])).any().item())

    def test_a_fallen_robot_is_not_charged_the_crawl_penalty(self):
        """Wriggling on the ground is not choosing a low gait."""
        env = self._down_env(speeds=[0.5], wheel_contact=[[0, 0]])
        self.assertEqual(standing.low_posture_locomotion(env)[0].item(), 0.0)

    def test_a_fallen_robot_is_not_charged_for_unsupported_wheels(self):
        env = self._down_env()
        env.scene["feet_ground_contact"].data.current_air_time = torch.tensor([[3.0, 3.0]])
        self.assertEqual(standing.no_wheel_support(env)[0].item(), 0.0)

    def test_a_fallen_robot_is_not_charged_body_contact_twice(self):
        self.assertEqual(standing.base_ground_contact_cost(self._down_env())[0].item(), 0.0)

    def test_an_upright_robot_keeps_the_crawl_and_support_penalties(self):
        env = _env([0.02], speeds=[0.5], base_contact=[0], wheel_contact=[[0, 0]])
        env.scene["feet_ground_contact"].data.current_air_time = torch.tensor([[3.0, 3.0]])
        self.assertGreater(standing.low_posture_locomotion(env)[0].item(), 0.0)
        self.assertGreater(standing.no_wheel_support(env)[0].item(), 0.0)

    def test_an_upright_robot_keeps_the_body_contact_penalty(self):
        """The contact gate looks only at tilt, so it cannot switch itself off."""
        env = _env([0.02], base_contact=[1])
        self.assertTrue(standing.is_down(env)[0].item())  # contact alone means down
        self.assertEqual(standing.base_ground_contact_cost(env)[0].item(), 1.0)

    def test_tilt_cost_saturates(self):
        """Lying down must not produce an unbounded penalty."""
        import math

        flat = _env([0.145], gravity=[(math.sin(math.pi / 2), 0.0, -math.cos(math.pi / 2))])
        self.assertAlmostEqual(standing.body_level_error(flat)[0].item(),
                               standing.MAX_TILT_COST, places=6)


if __name__ == "__main__":
    unittest.main()
