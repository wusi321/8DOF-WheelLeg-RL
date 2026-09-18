"""8DOF wheel-leg task recipes built on the copied MJLab manager recipe."""
from .base_env_cfg import *
from .base_env_cfg import flat_env_cfg as _base_flat_env_cfg, rough_env_cfg as _base_rough_env_cfg
from ..robot_cfg import get_robot_cfg
from ..stance import MIN_CLEARANCE, STANDING_CLEARANCE
from ..mdp import standing


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
    # Posture and ground contact must not end the episode; a completed fall still
    # does, through bad_orientation, and physics blow-ups through nan_detection.
    for name in ("low_base_height", "knee_ground_contact", "base_ground_contact"):
        cfg.terminations.pop(name, None)

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
        # A wheeled robot must not travel on its body, and a wheel held in the air
        # is a lost wheel. Both are penalties, not resets, so a fall is still
        # allowed to happen; they only make kneeling unprofitable.
        cfg.rewards["base_contact_penalty"] = RewardTermCfg(
            func=standing.base_ground_contact, weight=-10.0)
        cfg.rewards["wheel_off_ground"] = RewardTermCfg(
            func=standing.wheel_off_ground,
            weight=-10.0,
            params={
                "sensor_name": "feet_ground_contact",
                "allowance": standing.WHEEL_AIR_ALLOWANCE,
                "horizon": standing.WHEEL_AIR_HORIZON,
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
        # Keep the two legs posed alike; an asymmetric gait is not a wheeled gait.
        cfg.rewards["leg_symmetry"] = RewardTermCfg(
            func=standing.leg_symmetry_error, weight=-10.0)
        cfg.rewards["standing_pose"] = RewardTermCfg(
            func=standing.standing_pose_error, weight=-0.5)
        # `feet_air_time` rewards a foot for being *in the air* for 0.1-0.5 s,
        # which is a stepping incentive for legged robots. On a wheeled machine it
        # pays the policy to lift its wheels off the ground, so it is removed
        # rather than retuned.
        cfg.rewards.pop("feet_air_time", None)

    cfg.metrics["base_clearance_m"] = MetricsTermCfg(func=standing.base_clearance)
    cfg.metrics["moving_gate"] = MetricsTermCfg(func=standing.moving_gate)
    cfg.metrics["wheel_air_time_s"] = MetricsTermCfg(func=standing.wheel_air_time)
    cfg.metrics["wheel_contact_fraction"] = MetricsTermCfg(func=standing.wheel_contact_fraction)
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
    return _posture_contract(_adapt(cfg, play), enforce_standing)


def recovery_env_cfg(play=False):
    cfg = rough_env_cfg(play, enforce_standing=False)
    cfg.episode_length_s = 12.0
    return _adapt(cfg, play)
