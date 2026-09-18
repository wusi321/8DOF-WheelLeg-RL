"""Locomotion clearance and posture constraints (not recovery objectives)."""
import torch


def base_clearance(env):
    """Vertical distance from base_link origin to terrain directly underneath."""
    sensor = env.scene["base_clearance"]
    data = sensor.data
    root_z = env.scene["wheelleg"].data.root_link_pos_w[:, 2]
    hit_z = data.hit_pos_w[..., 2].reshape(env.num_envs, -1)[:, 0]
    distance = data.distances.reshape(env.num_envs, -1)[:, 0]
    # A missing ground hit is not evidence of sufficient clearance.
    valid = (distance >= 0) & torch.isfinite(hit_z)
    return torch.where(valid, root_z - hit_z, torch.zeros_like(root_z))


def below_standing_height(env, minimum_height=0.13):
    return base_clearance(env) < minimum_height


def standing_height_error(env, target_height=0.15):
    return torch.square((base_clearance(env) - target_height) / target_height)


def standard_pose_error(env):
    asset = env.scene["wheelleg"]
    ids, _ = asset.find_joints(
        ("left_hip_joint", "left_thigh_joint", "left_knee_joint",
         "right_hip_joint", "right_thigh_joint", "right_knee_joint"),
        preserve_order=True,
    )
    q = asset.data.joint_pos[:, ids]
    target = q.new_tensor([0.0, 1.02, -1.57, 0.0, 1.02, -1.57])
    return torch.mean(torch.square(q - target), dim=1)


def knee_ground_contact(env):
    found = env.scene["knee_ground_contact"].data.found
    return (found.reshape(env.num_envs, -1) > 0).any(dim=1)
