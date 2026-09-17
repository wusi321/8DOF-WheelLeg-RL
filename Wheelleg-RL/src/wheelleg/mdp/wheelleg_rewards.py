"""Small 8DOF-specific reward terms kept separate from the reference library."""
from __future__ import annotations
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

def joint_symmetry(env, asset_cfg: SceneEntityCfg, pairs: tuple[tuple[str, str], ...]) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ids = {name: i for i, name in enumerate(asset.joint_names)}
    result = torch.zeros(env.num_envs, device=env.device)
    for left, right in pairs:
        if left in ids and right in ids:
            result += torch.square(asset.data.joint_pos[:, ids[left]] - asset.data.joint_pos[:, ids[right]])
    return result

def speed_limit_penalty(env, max_speed: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ids = asset.find_joints(asset_cfg.joint_names)[0]
    return torch.sum(torch.relu(torch.abs(asset.data.joint_vel[:, ids]) - max_speed) ** 2, dim=1)

def no_motion_penalty(env, command_name: str, threshold: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    commanded = torch.linalg.vector_norm(command[:, :3], dim=1)
    motion = torch.linalg.vector_norm(asset.data.root_link_lin_vel_b[:, :2], dim=1) + torch.abs(asset.data.root_link_ang_vel_b[:, 2])
    return torch.where(commanded < threshold, torch.relu(0.03 - motion), torch.zeros_like(motion))

def recovery_progress(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Potential shaping for standing after a fall; holding a fallen pose cannot farm reward."""
    asset = env.scene[asset_cfg.name]
    upright = torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 1.0)
    return upright
