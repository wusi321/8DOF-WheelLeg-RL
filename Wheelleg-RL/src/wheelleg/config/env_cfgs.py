"""8DOF wheel-leg task recipes built on the copied MJLab manager recipe."""
from .base_env_cfg import *
from .base_env_cfg import flat_env_cfg as _base_flat_env_cfg, rough_env_cfg as _base_rough_env_cfg
from ..robot_cfg import get_robot_cfg
from ..mdp import standing


def _standing_constraints(cfg):
    cfg.scene.sensors = (*cfg.scene.sensors,
        RayCastSensorCfg(
            name="base_clearance",
            frame=ObjRef(type="body", name="base_link", entity="wheelleg"),
            pattern=GridPatternCfg(size=(0.0, 0.0), resolution=0.1),
            ray_alignment="yaw", max_distance=5.0,
            exclude_parent_body=True, include_geom_groups=(0,),
        ),
        ContactSensorCfg(
            name="knee_ground_contact",
            primary=ContactMatch(mode="body", pattern=("left_shank_link", "right_shank_link"), entity="wheelleg"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found",), reduce="none", num_slots=1,
        ),
    )
    cfg.terminations["low_base_height"] = TerminationTermCfg(
        func=standing.below_standing_height, params={"minimum_height": 0.13})
    cfg.terminations["knee_ground_contact"] = TerminationTermCfg(func=standing.knee_ground_contact)
    cfg.rewards["base_height_l2"] = RewardTermCfg(func=standing.standing_height_error, weight=-2.0)
    cfg.rewards["standard_standing_pose"] = RewardTermCfg(func=standing.standard_pose_error, weight=-2.0)
    if "wheel_roll_tracking" in cfg.rewards:
        cfg.rewards["wheel_roll_tracking"].params.update(wheel_radius=0.03, wheel_track=0.216)
    cfg.metrics["base_clearance_m"] = MetricsTermCfg(func=standing.base_clearance)
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
    return _standing_constraints(_adapt(_base_flat_env_cfg(play=False), play))

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
    cfg = _adapt(cfg, play)
    return _standing_constraints(cfg) if enforce_standing else cfg

def recovery_env_cfg(play=False):
    cfg = rough_env_cfg(play, enforce_standing=False)
    cfg.episode_length_s = 12.0
    return _adapt(cfg, play)
