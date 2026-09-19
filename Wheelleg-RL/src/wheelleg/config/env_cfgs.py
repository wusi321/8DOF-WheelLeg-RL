"""8DOF wheel-leg task recipes built on the copied MJLab manager recipe."""
from .base_env_cfg import *
from .base_env_cfg import flat_env_cfg as _base_flat_env_cfg, rough_env_cfg as _base_rough_env_cfg
from ..robot_cfg import get_robot_cfg
from ..stance import MIN_CLEARANCE, STANDING_CLEARANCE
from ..mdp import standing
from ..mdp.curriculums import PathLength
from ..mdp.lowpass_actions import PostureOffsetPositionActionCfg
from ..mdp.posture import POSTURE_COMMAND_NAME, PostureCommandCfg, posture_pose_error


def _posture_command(cfg):
    """Add the commanded posture and make the leg action follow it.

    The leg action is a delta from the standing stance, with a scale of 0.125 rad
    for the hip and 0.25 for the thigh and knee, so commanding the folded pose
    from there needs raw actions of +7.26 / -5.55 against an initial action std
    of 0.80. The policy therefore cannot choose to fold, and a folded spawn only
    lasts until the actuators drag the legs back to the stance. With the offset
    following the command, zero action means "hold the commanded posture" and
    both lying down and standing up become small-correction problems.
    """
    cfg.commands[POSTURE_COMMAND_NAME] = PostureCommandCfg(
        resampling_time_range=(5.0, 10.0),
        folded_fraction=0.35,
        standing_fraction=0.35,
    )
    # The actor must see the command: it cannot observe its own height, so a
    # posture that depended on state alone would be invisible to the policy.
    for group in ("actor", "critic"):
        cfg.observations[group].terms[POSTURE_COMMAND_NAME] = ObservationTermCfg(
            func=velocity_mdp.generated_commands,
            params={"command_name": POSTURE_COMMAND_NAME},
        )
    cfg.actions["leg_joint_pos"] = PostureOffsetPositionActionCfg(
        entity_name="wheelleg",
        actuator_names=("(left|right)_hip_joint", "(left|right)_thigh_joint",
                        "(left|right)_knee_joint"),
        scale={".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25},
        use_default_offset=True,
        control_frequency=50.0, cut_off_frequency=5.0,
        min_delay=0, max_delay=2,
        posture_command_name=POSTURE_COMMAND_NAME,
    )
    return cfg


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
        func=standing.fallen_too_long,
        params={
            "max_down_time": standing.MAX_DOWN_TIME,
            "fold_stand_deadline": standing.FOLD_STAND_DEADLINE,
        },
    )

    # Reverse-curriculum spawns. Without them the policy only ever starts
    # standing, falls at once, and never observes the successful stand-up branch,
    # so the progress potentials have no positive side to discover. This is the
    # MicroDuck VelStand fix for "learns the start, never the last mile".
    cfg.events["spawn_fallen"] = EventTermCfg(
        func=standing.spawn_fallen_state,
        mode="reset",
        params={
            "folded_probability": 0.5,
            "crouch_probability": 0.25,
            "asset_cfg": SceneEntityCfg("wheelleg"),
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
            func=standing.base_ground_contact_cost, weight=-10.0)
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
        # Pose tracking towards the *commanded* posture, not a fixed stance: a
        # folded-commanded robot has to be allowed, and required, to stay folded.
        cfg.rewards["posture_pose"] = RewardTermCfg(
            func=posture_pose_error, weight=-2.0)
        # Keep the body's z axis vertical instead of letting it follow the slope.
        # Ramped linearly in radians so it can actually pull back the steady tilt
        # that rolling along a bank produces; a squared cost is flat near upright.
        cfg.rewards["body_level"] = RewardTermCfg(
            func=standing.body_level_error,
            weight=-10.0,
            params={
                "forward_allowance": standing.FORWARD_LEAN_ALLOWANCE,
                "backward_scale": standing.BACKWARD_LEAN_SCALE,
                "max_cost": standing.MAX_TILT_COST,
            },
        )
        cfg.rewards["standing_pose"] = RewardTermCfg(
            func=standing.standing_pose_error, weight=-0.5)
        # `joint_pos_limits` matters here: the folded pose sits on the hip and
        # thigh hard limits, so it is scaled while down (above).

        # -- Fall recovery, following the VelStand task in the MicroDuck
        # reference. Rising pays and holding pays zero, so none of it can be
        # farmed by parking in a convenient pose.
        cfg.rewards["upright_progress"] = RewardTermCfg(
            func=standing.upright_progress, weight=5.0)
        cfg.rewards["height_progress"] = RewardTermCfg(
            func=standing.height_progress,
            weight=30.0,
            params={"ceiling": standing.HEIGHT_CEILING},
        )
        # A flat tax on staying down. Without it, lying still is cheap while
        # attempting a recovery pays the attempt taxes below.
        cfg.rewards["fallen_tax"] = RewardTermCfg(
            func=standing.fallen_tax, weight=-0.5)
        # One-shot bounty for finishing the stand, with a reachable definition
        # (25 degrees and 0.11 m, not the full 0.145 m stance).
        cfg.rewards["recovery_success"] = RewardTermCfg(
            func=standing.recovery_success,
            weight=10.0,
            params={
                "up_tilt": standing.RECOVERED_TILT,
                "up_clearance": standing.RECOVERED_CLEARANCE,
            },
        )
        # Attempt taxes: reduced, not removed, while the robot is down. The folded
        # pose sits on the hip and thigh hard limits, so an unscaled joint-limit
        # penalty would charge the robot for adopting the pose it must stand from.
        scale = standing.FALLEN_ATTEMPT_SCALE
        cfg.rewards["action_rate"] = RewardTermCfg(
            func=standing.action_rate_fallen_scaled,
            weight=cfg.rewards["action_rate"].weight if "action_rate" in cfg.rewards else -0.01,
            params={"scale": scale})
        leg_names = ("(left|right)_hip_joint", "(left|right)_thigh_joint",
                     "(left|right)_knee_joint")
        # Flat carries one joint-acceleration term over every joint; rough pops it
        # for split leg and wheel terms. Scale whichever this task has.
        if "joint_acc" in cfg.rewards:
            cfg.rewards["joint_acc"] = RewardTermCfg(
                func=standing.joint_acc_fallen_scaled,
                weight=cfg.rewards["joint_acc"].weight,
                params={"asset_cfg": SceneEntityCfg("wheelleg"), "scale": scale})
        if "leg_joint_acc_l2" in cfg.rewards:
            cfg.rewards["leg_joint_acc_l2"] = RewardTermCfg(
                func=standing.joint_acc_fallen_scaled,
                weight=cfg.rewards["leg_joint_acc_l2"].weight,
                params={"asset_cfg": SceneEntityCfg("wheelleg", joint_names=leg_names),
                        "scale": scale})
        if "joint_pos_limits" in cfg.rewards:
            cfg.rewards["joint_pos_limits"] = RewardTermCfg(
                func=standing.joint_pos_limits_fallen_scaled,
                weight=cfg.rewards["joint_pos_limits"].weight,
                params={"asset_cfg": SceneEntityCfg("wheelleg"), "scale": scale})
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
    cfg.metrics["fallen"] = MetricsTermCfg(func=standing.fallen_mask)
    cfg.metrics["recovered"] = MetricsTermCfg(func=standing.recovered_mask)
    cfg.metrics["fold_complete"] = MetricsTermCfg(func=standing.fold_complete)
    cfg.metrics["folded_pose_error"] = MetricsTermCfg(func=standing.folded_pose_error)
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
    return _posture_command(_posture_contract(_adapt(_base_flat_env_cfg(play=False), play)))


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
    # is flat ground, and so is a random grid with grid_height 0. That let the
    # policy spend a whole run on effectively flat ground and never learn to lift
    # a wheel over a step.
    #
    # The lower bound must still be small enough to start from: an earlier
    # attempt at 0.3 put every environment on 3.6-7.8 cm obstacles from the first
    # episode, which this robot cannot stand on, and the policy collapsed. 0.02
    # is a 2.4 mm step, i.e. effectively flat but non-degenerate, and the terrain
    # curriculum -- now driven by distance actually travelled -- raises the
    # difficulty as the robot proves it can get around.
    tg.difficulty_range = (0.02, 1.0)
    # Start every environment at the two easiest levels for the same reason;
    # max_init_terrain_level=5 previously dropped half of them onto 6-8 cm steps.
    cfg.scene.terrain.max_init_terrain_level = 1
    # PathLength feeds the terrain curriculum, which only advances when the robot
    # actually travels; see the note in terrain_levels_vel_strict.
    cfg.metrics["path_length_m"] = MetricsTermCfg(func=PathLength)
    return _posture_command(_posture_contract(_adapt(cfg, play), enforce_standing))


def recovery_env_cfg(play=False):
    cfg = rough_env_cfg(play, enforce_standing=False)
    cfg.episode_length_s = 12.0
    return _adapt(cfg, play)
