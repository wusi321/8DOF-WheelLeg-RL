"""Locomotion clearance and posture constraints (not recovery objectives).

All heights here are the ``base_link`` frame origin above the ground directly
beneath it, measured by a single downward ray. See ``wheelleg.stance`` for the
geometry those numbers come from.
"""
import torch

from ..stance import COLLAPSE_CLEARANCE, MIN_CLEARANCE, NOMINAL_STANCE, STANDING_CLEARANCE

# Sentinel returned when the ray finds no ground. It is deliberately negative
# so that "unknown" can never be mistaken for a measurable clearance, and every
# consumer below treats it as "not measurable" rather than "too low".
INVALID_CLEARANCE = -1.0

_LEG_JOINTS = (
    "left_hip_joint", "left_thigh_joint", "left_knee_joint",
    "right_hip_joint", "right_thigh_joint", "right_knee_joint",
)


def base_clearance(env):
    """Vertical distance from the base_link origin to the terrain below."""
    data = env.scene["base_clearance"].data
    root_z = env.scene["wheelleg"].data.root_link_pos_w[:, 2]
    hit_z = data.hit_pos_w[..., 2].reshape(env.num_envs, -1)[:, 0]
    distance = data.distances.reshape(env.num_envs, -1)[:, 0]
    valid = (distance >= 0) & torch.isfinite(hit_z)
    return torch.where(valid, root_z - hit_z, torch.full_like(root_z, INVALID_CLEARANCE))


def _measurable(clearance):
    return clearance > INVALID_CLEARANCE


def collapsed(env, min_clearance=COLLAPSE_CLEARANCE):
    """True once the robot is on the ground rather than standing on its wheels."""
    clearance = base_clearance(env)
    return _measurable(clearance) & (clearance < min_clearance)


def low_height_barrier(env, min_clearance=MIN_CLEARANCE):
    """Squared barrier that grows as the body sinks below the required height.

    Zero at or above ``min_clearance`` and small just below it, so normal
    squatting, buffering and leg lifting during a stride cost nothing while a
    sustained low crawl becomes strongly unprofitable.
    """
    clearance = base_clearance(env)
    deficit = torch.clamp((min_clearance - clearance) / min_clearance, 0.0, 1.0)
    return torch.where(_measurable(clearance), deficit * deficit, torch.zeros_like(deficit))


def standing_height_error(env, target_height=STANDING_CLEARANCE):
    """Normalised squared deviation from the working stance height."""
    clearance = torch.clamp(base_clearance(env), min=0.0)
    return torch.square((clearance - target_height) / target_height)


def standing_pose_error(env):
    """Mean squared deviation of the six leg joints from the nominal stance."""
    asset = env.scene["wheelleg"]
    ids, _ = asset.find_joints(_LEG_JOINTS, preserve_order=True)
    joints = asset.data.joint_pos[:, ids]
    target = joints.new_tensor([NOMINAL_STANCE[0], NOMINAL_STANCE[1], NOMINAL_STANCE[2]] * 2)
    return torch.mean(torch.square(joints - target), dim=1)


def knee_ground_contact(env):
    """True when a shank (knee) collides with the terrain — crawling or kneeling."""
    found = env.scene["knee_ground_contact"].data.found
    return (found.reshape(env.num_envs, -1) > 0).any(dim=1)
