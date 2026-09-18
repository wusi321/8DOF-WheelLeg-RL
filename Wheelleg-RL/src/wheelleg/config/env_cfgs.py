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
        cfg.rewards["base_contact_penalty"] = RewardTermCfg(
            func=standing.base_ground_contact, weight=-2.0)
        cfg.rewards["standing_pose"] = RewardTermCfg(
            func=standing.standing_pose_error, weight=-0.5)

    cfg.metrics["base_clearance_m"] = MetricsTermCfg(func=standing.base_clearance)
    cfg.metrics["moving_gate"] = MetricsTermCfg(func=standing.moving_gate)
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
