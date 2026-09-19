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
        command_manager=SimpleNamespace(
            get_command=lambda name: torch.tensor([[float(v) for v in c] for c in command])),
        scene={
            "wheelleg": SimpleNamespace(data=SimpleNamespace(
                root_link_pos_w=root_z,
                root_link_lin_vel_b=torch.tensor(
                    [[float(s), float(y), 0.0] for s, y in zip(speeds, lateral)]),
                root_link_ang_vel_b=torch.tensor([[0.0, 0.0, float(y)] for y in yaws]),
                projected_gravity_b=torch.tensor([[float(v) for v in g] for g in gravity]),
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


if __name__ == "__main__":
    unittest.main()
