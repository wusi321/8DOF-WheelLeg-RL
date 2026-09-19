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
                     "standing.base_contact_penalty_relieved", "standing.moving_gate",
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
        self.assertIn("def fallen_too_long", source)
        self.assertIn("env._fold_seconds >= fold_stand_deadline", source)
        self.assertIn("env._down_seconds >= max_down_time", source)

    def test_recovery_shaping_is_potential_based(self):
        """A per-step bonus for being folded would be farmed by parking there."""
        source = (root / "src/wheelleg/mdp/standing.py").read_text(encoding="utf-8")
        self.assertIn("def upright_progress", source)
        self.assertIn("def height_progress", source)
        self.assertIn("def recovery_success", source)
        self.assertIn("def fallen_tax", source)
        # Progress must be a delta against the previous step, never a state value.
        self.assertIn("delta = value - previous", source)

    def test_attempt_taxes_are_scaled_while_fallen(self):
        source = (root / "src/wheelleg/mdp/standing.py").read_text(encoding="utf-8")
        self.assertIn("def attempt_scale", source)
        for name in ("action_rate_motion_relieved", "joint_acc_motion_relieved",
                     "joint_pos_limits_fallen_scaled"):
            self.assertIn(f"def {name}", source)
        cfg = (root / "src/wheelleg/config/env_cfgs.py").read_text(encoding="utf-8")
        for name in ("standing.action_rate_motion_relieved",
                     "standing.joint_acc_motion_relieved",
                     "standing.joint_pos_limits_fallen_scaled",
                     "standing.fallen_tax", "standing.recovery_success",
                     "standing.upright_progress", "standing.height_progress"):
            self.assertIn(name, cfg)

    def test_leg_motion_taxes_stand_down_when_a_wheel_has_to_lift(self):
        """Charged in full at a step, the cheapest policy is never to lift a wheel.

        Play showed the legs not moving at all when the robot was blocked. The
        relief must key on the commanded-but-unachieved speed, which is what being
        blocked looks like, and it must never reach zero -- reduced, not removed,
        or thrashing works on the flat too.
        """
        source = (root / "src/wheelleg/mdp/standing.py").read_text(encoding="utf-8")
        self.assertIn("def leg_motion_scale", source)
        self.assertIn("blocked = (asked > blocked_speed).float() * shortfall", source)
        self.assertIn("relax = torch.maximum(", source)
        self.assertIn("return 1.0 - (1.0 - scale) * relax", source)
        # Reduced, never removed.
        self.assertIn("LEG_MOTION_RELIEF = 0.2", source)

    def test_swing_clearance_is_measured_against_the_terrain_under_the_robot(self):
        """No world z and no terrain origin: every term is a difference.

        The scanner reports the base origin's height above the terrain beneath each
        ray, so a wheel's clearance is that distance plus how far the wheel hangs
        relative to the base. That is what makes the term survive a curriculum whose
        rows sit at different heights.
        """
        source = (root / "src/wheelleg/mdp/standing.py").read_text(encoding="utf-8")
        self.assertIn("def wheel_swing_clearance", source)
        self.assertIn("heights = height_scan(env, sensor_name, offset=0.0)", source)
        self.assertIn("ground = torch.mean(heights, dim=1)", source)
        self.assertIn(
            "clearance = ground.unsqueeze(-1) + (z - asset.data.root_link_pos_w[:, 2:3])",
            source,
        )
        # Rolling wheels owe nothing; only a wheel that has left the ground is shaped.
        self.assertIn("off_ground = (clearance > rolling).float()", source)
        self.assertIn("* speed * off_ground", source)
        # Silent without a command, and on a task with no scanner.
        self.assertIn("return cost * (asked > command_threshold).float()", source)
        self.assertIn("except KeyError:", source)
        cfg = (root / "src/wheelleg/config/env_cfgs.py").read_text(encoding="utf-8")
        self.assertIn("standing.wheel_swing_clearance", cfg)
        # The module stays mjlab-free: it must not build a SceneEntityCfg itself.
        self.assertNotIn("SceneEntityCfg(", source)

    def test_terrain_traversal_pays_where_tracking_has_gone_to_zero(self):
        """Gated on the command, not on movement: the funded seconds are the stuck ones."""
        source = (root / "src/wheelleg/mdp/standing.py").read_text(encoding="utf-8")
        self.assertIn("def terrain_level_bonus", source)
        self.assertIn('levels = getattr(terrain, "terrain_levels", None)', source)
        self.assertIn("active = (asked > active_threshold).float()", source)
        self.assertIn(
            "return torch.clamp(levels.float() / reference, 0.0, 1.0) * active", source
        )
        cfg = (root / "src/wheelleg/config/env_cfgs.py").read_text(encoding="utf-8")
        self.assertIn("standing.terrain_level_bonus", cfg)

    def test_body_level_is_muted_only_for_a_failed_fall(self):
        """A robot told to stand is the one that gets the allowance, not a folded one.

        A folded-commanded robot must still lie *flat* on the ground, so its level
        requirement keeps full weight; only the failed fall -- down while standing
        is commanded -- gets the attempt scale.
        """
        source = (root / "src/wheelleg/mdp/standing.py").read_text(encoding="utf-8")
        self.assertIn("def recovery_scale", source)
        self.assertIn("fallen_mask(env).bool() & (standing_command(env) > 0.5)", source)
        self.assertIn("def body_level_error_recovery_scaled", source)
        cfg = (root / "src/wheelleg/config/env_cfgs.py").read_text(encoding="utf-8")
        self.assertIn("func=standing.body_level_error_recovery_scaled", cfg)
        # The logged metric must stay the unscaled tilt, or the muting hides itself.
        self.assertIn(
            'cfg.metrics["body_level_error"] = MetricsTermCfg(func=standing.body_level_error)',
            cfg,
        )

    def test_down_window_is_short(self):
        """Every second down is a second of gradient saying 'hold still'."""
        import re

        source = (root / "src/wheelleg/mdp/standing.py").read_text(encoding="utf-8")
        for name in ("MAX_DOWN_TIME", "FOLD_STAND_DEADLINE"):
            match = re.search(rf"^{name} = ([\d.]+)", source, re.MULTILINE)
            self.assertIsNotNone(match, f"{name} is missing")
            self.assertLessEqual(float(match.group(1)), 3.0)

    def test_leg_joint_positions_mirrors_the_hip(self):
        """The hip axes are not mirrored in the model, so the right hip is negated.

        Both ``*_hip_joint`` rotate about +X, so one shared angle tucks the left
        wheel up and drives the right one down: the body is skewed rather than
        folded, and it cannot lie flat. ``leg_symmetry_error`` already tests the
        hip as a *sum* for the same reason.
        """
        pose = stance.FOLDED_STANCE
        left_hip, left_thigh, left_knee, right_hip, right_thigh, right_knee = (
            stance.leg_joint_positions(pose)
        )
        self.assertAlmostEqual(left_hip, pose[0])
        self.assertAlmostEqual(right_hip, -pose[0])
        # The sum the symmetry reward measures must vanish for a folded robot.
        self.assertAlmostEqual(left_hip + right_hip, 0.0)
        # Thigh and knee axes are +Y on both sides, where a shared angle is
        # already mirror-symmetric, so those keep their sign.
        self.assertAlmostEqual(left_thigh, right_thigh)
        self.assertAlmostEqual(left_knee, right_knee)

    def test_mirrored_fold_lands_both_wheels_at_equal_height(self):
        """Why the negated hip is the right convention, shown on the rotation.

        A hip angle rotates a point (y, z) of the leg about +X. Mirroring the
        whole leg means y -> -y *and* the angle -> -angle, and that pair leaves z
        untouched while flipping y. So the negated hip puts both wheels at the
        same height and opposite lateral offset -- a fold -- whereas the shared
        sign puts one wheel up and the other down.
        """
        hip = stance.FOLDED_STANCE[0]
        point = (0.0, 0.05, -0.06)  # a wheel-ish point in the left hip frame
        left = stance._rot_x(point, hip)
        mirrored = stance._rot_x((0.0, -point[1], point[2]), -hip)
        self.assertAlmostEqual(left[2], mirrored[2], places=12)

        # The bug: the same angle on the right leg does not do this.
        same_sign = stance._rot_x((0.0, -point[1], point[2]), hip)
        self.assertNotAlmostEqual(left[2], same_sign[2], places=3)

    def test_no_stance_is_expanded_by_duplication(self):
        """``pose * 2`` is the bug this convention exists to prevent."""
        for rel in (
            "src/wheelleg/mdp/standing.py",
            "src/wheelleg/mdp/posture.py",
            "src/wheelleg/mdp/lowpass_actions.py",
            "src/wheelleg/config/env_cfgs.py",
        ):
            source = (root / rel).read_text(encoding="utf-8")
            self.assertNotIn("STANCE[0], NOMINAL_STANCE[1]", source)
            self.assertNotIn("[folded[0], folded[1], folded[2]] * 2", source)
            self.assertNotIn("FOLDED_STANCE * 2", source)

    def test_every_spawn_dictates_where_its_posture_starts_and_goes(self):
        """The event manager runs before the command manager, so they must couple.

        Without it a standing spawn is handed a folded command and the action
        offset drags its legs out on the first step. And it takes *two* values: a
        spawn folded and told to stand must start folded and ramp up, because that
        transition is the get-up. Starting alpha at the commanded value made the
        legs jump instead.
        """
        source = (root / "src/wheelleg/mdp/standing.py").read_text(encoding="utf-8")
        self.assertIn("env._spawn_alpha[env_ids] = torch.where(", source)
        self.assertIn("env._spawn_command[env_ids] = torch.where(", source)
        posture = (root / "src/wheelleg/mdp/posture.py").read_text(encoding="utf-8")
        self.assertIn('pose = getattr(self._env, "_spawn_alpha", None)', posture)
        self.assertIn('want = getattr(self._env, "_spawn_command", None)', posture)
        # Both consumed, so a mid-episode resample is a free draw again.
        self.assertIn('pose[env_ids] = float("nan")', posture)
        self.assertIn('want[env_ids] = float("nan")', posture)
        # The start and the goal must be separate assignments, or there is no ramp.
        self.assertIn("self._target[env_ids] = command", posture)
        self.assertIn("reset = ~torch.isnan(start)", posture)
        # And the posture must travel rather than jump.
        self.assertIn("def _update_command", posture)
        self.assertIn("torch.clamp(self._target - self.alpha, -step, step)", posture)

    def test_a_timer_resample_may_not_move_the_posture_directly(self):
        """Only a reset may set alpha; a mid-episode tick may only set the target.

        Assigning alpha unconditionally snapped a robot holding the folded pose
        upright in a single step when its command was resampled, which is the jolt
        the ramp exists to prevent, and it was visible in play as a sudden jump
        from lying to standing partway through an episode.
        """
        posture = (root / "src/wheelleg/mdp/posture.py").read_text(encoding="utf-8")
        self.assertIn(
            "self.alpha[env_ids] = torch.where(reset, start, self.alpha[env_ids])",
            posture,
        )
        self.assertNotIn("self.alpha[env_ids] = alpha", posture)

    def test_the_crouch_sits_where_the_interpolation_passes(self):
        """The crouch is the midpoint, so a crouch spawn starts at alpha 0.5."""
        mid = tuple(
            0.5 * (folded + standing)
            for folded, standing in zip(stance.FOLDED_STANCE, stance.NOMINAL_STANCE)
        )
        for computed, stored in zip(mid, stance.CROUCH_STANCE):
            self.assertAlmostEqual(computed, stored, places=3)
        self.assertAlmostEqual(stance.CROUCH_ALPHA, 0.5)

    def test_leg_action_scale_stays_in_a_usable_band(self):
        """Small enough not to open the policy in a falling regime, large enough to act.

        At 0.125/0.25 rad the knee's whole one-sigma action was under 0.1 rad. At a
        quarter of the travel the initial exploration amplitude threw the machine
        over, the policy learned that large actions fall, and the standard deviation
        collapsed from 0.37 to 0.15. Both ends are now guarded: the scale is exactly
        a fraction of the travel, and that fraction stays between 8% and 20% of it.
        """
        for kind, limits in (
            ("hip", stance.HIP_LIMIT),
            ("thigh", stance.THIGH_LIMIT),
            ("knee", stance.KNEE_LIMIT),
        ):
            span = limits[1] - limits[0]
            self.assertAlmostEqual(
                stance.LEG_ACTION_SCALE[kind],
                stance.ACTION_RANGE_FRACTION * span,
                places=12,
            )
            self.assertGreater(stance.LEG_ACTION_SCALE[kind], 0.08 * span)
            self.assertLessEqual(stance.LEG_ACTION_SCALE[kind], 0.20 * span)

    def test_the_half_crouch_is_not_a_posture_but_is_still_a_spawn(self):
        """Holding a half-crouch and rolling in it tipped the robot onto its back.

        That was the arithmetic midpoint of the two stances, never solved for axle
        position, so its wheel sat forward of the body. It is no longer reachable
        as a command -- alpha is binary and standing is the only posture a running
        episode is given -- but starting halfway up is still a valid mid-recovery
        spawn, so the probability stays.
        """
        cfg = (root / "src/wheelleg/config/env_cfgs.py").read_text(encoding="utf-8")
        self.assertIn('"crouch_probability": 0.15,', cfg)
        posture = (root / "src/wheelleg/mdp/posture.py").read_text(encoding="utf-8")
        self.assertNotIn("folded_fraction", posture)
        # Standing is the only command a running episode issues; the folded pose
        # can only arrive from a spawn.
        self.assertIn("command = torch.ones(len(env_ids), device=self.device)", posture)

    def test_tracking_is_gated_on_clearance_not_on_tilt(self):
        """Gating on tilt withdrew the walking payment precisely while walking.

        A robot rolling with the body leaning forty degrees is still travelling.
        The run that used the tilt-inclusive gate averaged sixty degrees of lean,
        so the payment was off most of the time, walking reward fell away exactly
        when the robot walked, and the standard deviation collapsed.
        """
        source = (root / "src/wheelleg/mdp/standing.py").read_text(encoding="utf-8")
        self.assertIn("def grounded_gate", source)
        self.assertIn("return (base_clearance(env) > clearance_gate).float()", source)
        for fn in (
            "track_linear_velocity_x",
            "track_linear_velocity_y",
            "track_angular_velocity_z",
        ):
            self.assertIn(
                f"return {fn}(env, std, command_name) * grounded_gate(env)", source
            )
        # And the tilt-inclusive gate must not be what pays the tracking reward.
        self.assertNotIn("env, std, command_name) * gait_gate(env)", source)

    def test_solve_stance_refuses_a_target_it_cannot_reach(self):
        """It used to return the bracket bound: 0.02 and 0.12 gave the same pose.

        The thigh bracket tops out at REFERENCE_STANCE[1], so below about 0.128 m
        there is no solution at all. That floor is also the reason a prone get-up
        is a dynamic manoeuvre rather than a pose change: no stance puts the axle
        under the base origin that low, so the first ten centimetres of the lift
        are necessarily taken with the contact ahead of the centre of mass.
        """
        hip, thigh, knee = stance.solve_stance(stance.STANDING_CLEARANCE, 0.0)
        self.assertAlmostEqual(
            stance.leg_pose(thigh, knee, hip).base_clearance,
            stance.STANDING_CLEARANCE,
            places=4,
        )
        # And the stance the policy is initialised to must stay exactly what it was.
        self.assertAlmostEqual(knee, stance.NOMINAL_STANCE[2], places=9)
        for target in (0.02, 0.09, 0.12):
            with self.assertRaises(ValueError):
                stance.solve_stance(target, 0.0)

    def test_the_recovery_bounty_uses_the_shared_fallen_definition(self):
        """A folded robot is level, so a tilt-only test never armed the bounty.

        It was taxed as fallen the whole time -- ``fallen_tax`` uses tilt *or*
        clearance -- and could not be paid for getting up, which is the one
        arrangement the recovery economy exists to avoid.
        """
        source = (root / "src/wheelleg/mdp/standing.py").read_text(encoding="utf-8")
        self.assertIn("fallen = fallen_mask(env, tilt_gate=fallen_tilt).bool()", source)
        self.assertNotIn("fallen = total_tilt(env) > fallen_tilt", source)

    def test_terrain_difficulty_is_not_diluted(self):
        """Ten rows spread one range so thin the robot never met a step.

        Difficulty is level/(num_rows-1), and the robot reached row 4 while sitting
        at level 0.2-0.6 per terrain type -- 0.8 cm steps. Five rows put full
        difficulty at row 4, where it already is, and everything is capped at the
        10 cm the geometry allows a two-wheel machine to step over from a rest.
        """
        source = (root / "src/wheelleg/config/base_env_cfg.py").read_text(encoding="utf-8")
        self.assertIn("num_rows=5, num_cols=20, curriculum=True", source)
        self.assertIn(
            "PyramidStairsTerrainCfg(proportion=0.15, step_height_range=(0.0, 0.10)", source
        )
        self.assertIn(
            "InvertedPyramidStairsTerrainCfg(proportion=0.25, step_height_range=(0.0, 0.10)",
            source,
        )
        self.assertIn("grid_height_range=(0.0, 0.10)", source)
        self.assertIn("wall_height_range=(0.04, 0.10)", source)
        # Ascending stairs are the task; they must not be outnumbered seven to one.
        self.assertNotIn("(0.0, 0.12)", source)
        # And the *effective* values live in env_cfgs, which overwrites the base
        # recipe's ranges. A cap set only in base_env_cfg does nothing for this task,
        # which is exactly how an earlier attempt at this was silently reverted.
        cfg = (root / "src/wheelleg/config/env_cfgs.py").read_text(encoding="utf-8")
        self.assertIn("tg.sub_terrains[name].step_height_range = (0.0, 0.10)", cfg)
        self.assertIn('tg.sub_terrains["random_grid"].grid_height_range = (0.0, 0.10)', cfg)
        self.assertIn('tg.sub_terrains["rc_wall"].wall_height_range = (0.04, 0.10)', cfg)
        self.assertNotIn("(0.0, 0.12)", cfg)

    def test_body_contact_is_free_for_a_robot_that_is_working(self):
        """A two-wheel machine has to rest its chassis on a step to swing a leg up.

        Four wheels lifting one leg still leave three; two leave one. Gating on
        "not fallen" alone would charge that strategy, because the ray under a
        chassis resting on a step reads a centimetre or two and the robot therefore
        counts as fallen while it climbs.
        """
        source = (root / "src/wheelleg/mdp/standing.py").read_text(encoding="utf-8")
        self.assertIn("def base_contact_relief", source)
        self.assertIn("up = 1.0 - fallen_mask(env)", source)
        self.assertIn(
            "relaxed = torch.maximum(up, (asked > command_threshold).float())", source
        )
        self.assertIn("return 1.0 - (1.0 - scale) * relaxed", source)
        cfg = (root / "src/wheelleg/config/env_cfgs.py").read_text(encoding="utf-8")
        self.assertIn("standing.base_contact_penalty_relieved", cfg)
        self.assertIn('"scale": standing.BASE_CONTACT_RELIEF', cfg)

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
        self.assertIn("def gait_gate", source)
        # The crawl and support penalties are suspended once the robot is down.
        self.assertIn("* gait_gate(env)", source)
        self.assertIn("MAX_TILT_COST", source)

    def test_one_fallen_definition_only(self):
        """Two definitions let a low robot keep paying the crawl penalty."""
        source = (root / "src/wheelleg/mdp/standing.py").read_text(encoding="utf-8")
        self.assertIn("FALLEN_TILT", source)
        self.assertIn("FALLEN_CLEARANCE", source)
        # The old second definition must be gone.
        self.assertNotIn("DOWN_TILT_LIMIT", source)
        # is_down must delegate to fallen_mask rather than re-deriving a gate.
        self.assertIn("return fallen_mask(env, tilt_limit, clearance_gate).bool()", source)

    def test_posture_command_drives_the_action_offset(self):
        """Option A: zero action must mean the commanded posture, not the stance."""
        cfg = (root / "src/wheelleg/config/env_cfgs.py").read_text(encoding="utf-8")
        self.assertIn("PostureCommandCfg", cfg)
        self.assertIn("PostureOffsetPositionActionCfg", cfg)
        self.assertIn("posture_command_name=POSTURE_COMMAND_NAME", cfg)
        self.assertIn("def _posture_command", cfg)
        # Both tasks must install it.
        self.assertIn("_posture_command(_posture_contract", cfg)
        # And the command must be observed, or the policy cannot know its mapping.
        self.assertIn('for group in ("actor", "critic")', cfg)

    def test_gait_rules_are_gated_by_the_commanded_posture(self):
        """A robot told to lie down is not falling and is not crawling."""
        source = (root / "src/wheelleg/mdp/standing.py").read_text(encoding="utf-8")
        self.assertIn("def standing_command", source)
        self.assertIn("standing_command(env) > 0.5", source)
        # The fall tax and the body-contact cost fold the command in too.
        self.assertIn("* standing_command(env)", source)

    def test_terrain_curriculum_uses_path_length_not_net_displacement(self):
        """Circling under heading commands demotes a walking robot every episode."""
        source = (root / "src/wheelleg/mdp/curriculums.py").read_text(encoding="utf-8")
        self.assertIn("_path_travelled", source)
        self.assertIn("class PathLength", source)

    def test_recovery_does_not_terminate_on_low_height(self):
        """A recovery task must be allowed to start from the ground."""
        source = (root / "src/wheelleg/config/env_cfgs.py").read_text(encoding="utf-8")
        self.assertIn("enforce_standing=False", source)

    def test_reverse_curriculum_spawns_are_configured(self):
        """Prone starts are what let the stand-up branch ever be discovered."""
        cfg = (root / "src/wheelleg/config/env_cfgs.py").read_text(encoding="utf-8")
        self.assertIn("standing.spawn_fallen_state", cfg)
        self.assertIn('cfg.events["spawn_fallen"]', cfg)
        self.assertIn("folded_probability", cfg)
        self.assertIn("crouch_probability", cfg)

    def test_spawn_heights_are_between_the_ground_and_standing(self):
        """The spawn poses must be reachable poses, not arbitrary numbers."""
        # The folded body rests on its base mesh, so the base origin is near zero.
        self.assertLess(stance.FOLDED_REST_Z, 0.01)
        # The mid-recovery crouch is a real crouch: above the ground, below stance.
        self.assertGreater(stance.CROUCH_SPAWN_Z, 0.02)
        self.assertLess(stance.CROUCH_SPAWN_Z, stance.STANDING_CLEARANCE)
        self.assertGreater(stance.CROUCH_SPAWN_Z, stance.MIN_CLEARANCE / 2)
        # The folded pose really is the pose from the supplied data file, with the
        # two over-limit joints pinned to the hard limits rather than literals.
        hip, thigh, knee = stance.FOLDED_STANCE
        self.assertEqual(hip, stance.HIP_LIMIT[1])
        self.assertEqual(thigh, stance.THIGH_LIMIT[1])
        self.assertAlmostEqual(knee, -2.62, places=6)
        self.assertGreater(knee, stance.KNEE_LIMIT[0])
        # Folded is a genuine crouch: far lower than the nominal stance.
        self.assertLess(stance.clearance(thigh, knee, hip),
                        stance.clearance(*stance.NOMINAL_STANCE[1:]) - 0.10)

    def test_height_error_is_not_normalised_by_a_target_that_can_go_to_zero(self):
        """The posture target falls to the folded rest height of 2 mm.

        Dividing the discrepancy by that target squared to over five thousand and
        produced a -7000 per episode penalty with a matching value loss.
        """
        source = (root / "src/wheelleg/mdp/standing.py").read_text(encoding="utf-8")
        self.assertIn("(clearance - target) / STANDING_CLEARANCE", source)
        self.assertNotIn("(clearance - target) / target", source)
        # The worst case must stay around 1, whatever the commanded posture.
        worst = max(((0.145 - t) / 0.145) ** 2 for t in (0.002, 0.02, 0.073, 0.145))
        self.assertLess(worst, 1.0)

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
