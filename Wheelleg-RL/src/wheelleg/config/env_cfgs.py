"""8DOF wheel-leg task recipes built on the copied MJLab manager recipe."""
from ._reference_env_cfgs import *
from ._reference_env_cfgs import flat_env_cfg as _reference_flat_env_cfg, rough_env_cfg as _reference_rough_env_cfg
from ..robot_cfg import get_robot_cfg

def _adapt(cfg, play=False):
    cfg.scene.entities = {"robot": get_robot_cfg()}
    cfg.scene.num_envs = 2048
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}
    return cfg

def flat_env_cfg(play=False):
    return _adapt(_reference_flat_env_cfg(play=False), play)

def rough_env_cfg(play=False):
    cfg = _reference_rough_env_cfg(play=False)
    tg = cfg.scene.terrain.terrain_generator
    for name in ("pyramid_stairs", "pyramid_stairs_inv"):
        if name in tg.sub_terrains:
            tg.sub_terrains[name].step_height_range = (0.0, 0.12)
    if "random_grid" in tg.sub_terrains:
        tg.sub_terrains["random_grid"].grid_height_range = (0.0, 0.12)
    if "rc_wall" in tg.sub_terrains:
        tg.sub_terrains["rc_wall"].wall_height_range = (0.04, 0.12)
    return _adapt(cfg, play)

def recovery_env_cfg(play=False):
    cfg = rough_env_cfg(play)
    cfg.episode_length_s = 12.0
    return _adapt(cfg, play)

def crawl_env_cfg(play=False):
    return recovery_env_cfg(play)
