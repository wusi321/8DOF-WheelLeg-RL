"""8DOF wheel-leg task recipes built on the copied MJLab manager recipe."""
from .base_env_cfg import *
from .base_env_cfg import flat_env_cfg as _base_flat_env_cfg, rough_env_cfg as _base_rough_env_cfg
from ..robot_cfg import get_robot_cfg
from ..stance import MIN_CLEARANCE, STANDING_CLEARANCE
from ..mdp import standing
from ..mdp.curriculums import PathLength


def _posture_contract(cfg, enforce_standing=True):
    """Locomotion posture contract: moving must be standing, crawling is not a gait.

    Height is never a termination. Resetting on low posture would only ever
    rescue the robot before it can fall, so instead the body may sit or lie low
    while parked, while *travelling* below ``MIN_CLEARANCE`` is penalised.

    ``enforce_standing=False`` (Recovery) only loosens the terminations, so a
    robot that starts on the ground is not reset immediately.
    """
    cfg.scene.sensors = (*cfg.scene.sensors,
        RayCastSensorCfg(
            name="base_clearance",
            frame=ObjRef(type="body", name="base_link", entity="wheelleg"),
            pattern=GridPatternCfg(size=(0.0, 0.0), resolution=0.1),
            ray_alignment="yaw", max_distance=5.0,
            exclude_parent_body=True, include_geom_groups=(0,),
        ),
    )
    # Posture and ground contact must not end the episode. A fall is ended by its
    # own outcome instead: the body may lie there until it has failed to get up
    # for a while, which is the only way standing back up can be learned.
    for name in ("low_base_height", "knee_ground_contact", "base_ground_contact"):
        cfg.terminations.pop(name, None)
    cfg.terminations.pop("bad_orientation", None)
    cfg.terminations["fallen_too_long"] = TerminationTermCfg(
        func=standing.FallenTooLong,
        params={
            "max_down_time": standing.MAX_DOWN_TIME,
            "tilt_limit": standing.DOWN_TILT_LIMIT,
        },
    )

    if enforce_standing:
        cfg.rewards["base_height_l2"] = RewardTermCfg(
            func=standing.standing_height_error,
            weight=-4.0,
            params={"target_height": STANDING_CLEARANCE},
        )
        # Rewards are scaled by dt, so a weight reads as "per second" (50 steps/s
        # x 0.02 s). At 0.12 m the crawl penalty is about 8/s, against roughly
        # 1.4/s of velocity tracking, so travelling low is never worth it.
        cfg.rewards["low_posture_locomotion"] = RewardTermCfg(
            func=standing.low_posture_locomotion,
            weight=-100.0,
            params={
                "min_clearance": MIN_CLEARANCE,
                "linear_speed": standing.LINEAR_SPEED_MOVING,
                "yaw_rate": standing.YAW_RATE_MOVING,
            },
        )
        # A wheeled robot must not travel on its body, and it must not lose every
        # wheel. Both are penalties, not resets, so a fall is still allowed to
        # happen; they only make kneeling unprofitable.
        cfg.rewards["base_contact_penalty"] = RewardTermCfg(
            func=standing.base_ground_contact, weight=-10.0)
        cfg.rewards["no_wheel_support"] = RewardTermCfg(
            func=standing.no_wheel_support,
            weight=-10.0,
            params={
                "sensor_name": "feet_ground_contact",
                "allowance": standing.WHEEL_AIR_ALLOWANCE,
                "horizon": standing.WHEEL_AIR_HORIZON,
            },
        )
        # Sideways travel cannot be rolled, only stepped, so the step itself is
        # paid for and the symmetry requirement is lifted while going sideways.
        cfg.rewards["lateral_step"] = RewardTermCfg(
            func=standing.lateral_step_reward,
            weight=1.0,
            params={
                "command_name": "twist",
                "sensor_name": "feet_ground_contact",
            },
        )
        cfg.rewards["wheeled_stance_locomotion"] = RewardTermCfg(
            func=standing.wheeled_stance_locomotion,
            weight=3.0,
            params={
                "command_name": "twist",
                "min_clearance": MIN_CLEARANCE,
                "target_clearance": STANDING_CLEARANCE,
                "sensor_name": "feet_ground_contact",
            },
        )
        # Keep the two legs posed alike *where that is correct*. The weight is
        # small and the term is switched off entirely once the two wheels sit at
        # different heights, because levelling the body on a bank or across a step
        # requires the high-side leg to retract and the low-side leg to extend.
        cfg.rewards["leg_symmetry"] = RewardTermCfg(
            func=standing.leg_symmetry_error,
            weight=-1.0,
            params={
                "command_name": "twist",
                "asset_cfg": SceneEntityCfg("wheelleg", body_names=standing._WHEEL_BODIES),
                "uneven_reference": standing.UNEVEN_HEIGHT_REFERENCE,
            },
        )
        # Keep the body's z axis vertical instead of letting it follow the slope.
        # Ramped linearly in radians so it can actually pull back the steady tilt
        # that rolling along a bank produces; a squared cost is flat near upright.
        cfg.rewards["body_level"] = RewardTermCfg(
            func=standing.body_level_error,
            weight=-12.0,
            params={
                "forward_allowance": standing.FORWARD_LEAN_ALLOWANCE,
                "backward_scale": standing.BACKWARD_LEAN_SCALE,
            },
        )
        cfg.rewards["standing_pose"] = RewardTermCfg(
            func=standing.standing_pose_error, weight=-0.5)
        # `feet_air_time` rewards a foot for being *in the air* for 0.1-0.5 s,
        # which is a stepping incentive for legged robots. On a wheeled machine it
        # pays the policy to lift its wheels off the ground, so it is removed
        # rather than retuned.
        cfg.rewards.pop("feet_air_time", None)
        # `joint_mirror` mirrors the thigh/knee pairs, which is the same quantity
        # leg_symmetry now measures correctly and gates on terrain, so it would
        # fight the level-adaptation this task needs.
        cfg.rewards.pop("joint_mirror", None)
        # The rough recipe's abduction mirror takes the difference of the two hip
        # angles. With unmirrored hip axes that rewards the sideways tilt and
        # penalises the mirror-symmetric splay, so it is dropped in favour of the
        # correctly measured leg_symmetry.
        cfg.rewards.pop("abduction_mirror", None)

    cfg.metrics["base_clearance_m"] = MetricsTermCfg(func=standing.base_clearance)
    cfg.metrics["moving_gate"] = MetricsTermCfg(func=standing.moving_gate)
    cfg.metrics["wheel_air_time_s"] = MetricsTermCfg(func=standing.wheel_air_time)
    cfg.metrics["wheel_support_time_s"] = MetricsTermCfg(func=standing.wheel_support_time)
    cfg.metrics["wheel_contact_fraction"] = MetricsTermCfg(func=standing.wheel_contact_fraction)
    cfg.metrics["lateral_command"] = MetricsTermCfg(func=standing.lateral_command_demand)
    cfg.metrics["body_level_error"] = MetricsTermCfg(func=standing.body_level_error)
    cfg.metrics["wheel_height_diff_m"] = MetricsTermCfg(
        func=standing.wheel_height_difference,
        params={"asset_cfg": SceneEntityCfg("wheelleg", body_names=standing._WHEEL_BODIES)},
    )
    return cfg


def _adapt(cfg, play=False):
    cfg.scene.entities = {"wheelleg": get_robot_cfg()}
    cfg.scene.num_envs = 2048
    # Reserve per-world constraints for floating-base ground contact.
    cfg.sim.njmax = 512
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}
    return cfg


def flat_env_cfg(play=False):
    return _posture_contract(_adapt(_base_flat_env_cfg(play=False), play))


def rough_env_cfg(play=False, enforce_standing=True):
    cfg = _base_rough_env_cfg(play=False)
    tg = cfg.scene.terrain.terrain_generator
    for name in ("pyramid_stairs", "pyramid_stairs_inv"):
        if name in tg.sub_terrains:
            tg.sub_terrains[name].step_height_range = (0.0, 0.12)
    if "random_grid" in tg.sub_terrains:
        tg.sub_terrains["random_grid"].grid_height_range = (0.0, 0.12)
    if "rc_wall" in tg.sub_terrains:
        tg.sub_terrains["rc_wall"].wall_height_range = (0.04, 0.12)
    # Obstacle height is interpolated by difficulty, and difficulty is
    # level/(num_rows-1). With the default range (0.0, 1.0) the easiest level
    # generates *zero-height* obstacles: a pyramid staircase with step_height 0
    # is flat ground, and so is a random grid with grid_height 0. The measured
    # terrain curriculum then averaged level 0.556, i.e. roughly 7 mm of
    # obstacle, so the policy spent its whole run on effectively flat ground and
    # never learned to lift a wheel over a step. Starting the range above zero
    # guarantees a real obstacle at every level.
    tg.difficulty_range = (0.3, 1.0)
    # PathLength feeds the terrain curriculum, which only advances when the robot
    # actually travels; see the note in terrain_levels_vel_strict.
    cfg.metrics["path_length_m"] = MetricsTermCfg(func=PathLength)
    return _posture_contract(_adapt(cfg, play), enforce_standing)


def recovery_env_cfg(play=False):
    cfg = rough_env_cfg(play, enforce_standing=False)
    cfg.episode_length_s = 12.0
    return _adapt(cfg, play)
