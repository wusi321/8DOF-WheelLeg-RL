"""Locomotion posture constraints: stand while moving, but do not reset on posture.

Design
------
Height is deliberately **not** a termination condition. Resetting the moment the
body dips below the target means the robot never experiences the fall it is
supposed to learn to avoid, so it can only ever be "rescued" by the reset.

Instead:

- standing still at a low posture is allowed and costs almost nothing;
- lying on the ground is allowed (nothing resets) but is made unprofitable by the
  ungated height reward and a base-contact penalty;
- *travelling* while low is penalised hard, so a crawl is never a valid gait.

The moving penalty is linear in the height deficit rather than squared, because
while the robot is moving the required height is a requirement, not a soft
preference: a squared barrier would be nearly free just below the threshold and
would let the policy settle into a crawl a centimetre under the target.

All heights are the ``base_link`` frame origin above the ground directly beneath
it, measured by a single downward ray. See ``wheelleg.stance`` for the geometry.
"""
import torch

from ..stance import MIN_CLEARANCE, NOMINAL_STANCE, STANDING_CLEARANCE

# Sentinel returned when the ray finds no ground. It is deliberately negative so
# that "unknown" can never be mistaken for a measurable clearance, and every
# consumer below treats it as "not measurable" rather than "too low".
INVALID_CLEARANCE = -1.0

# A robot counts as travelling above either of these. Rotation is included so
# that spinning in place while crouched is not a way around the moving penalty.
LINEAR_SPEED_MOVING = 0.15  # m/s
YAW_RATE_MOVING = 0.3  # rad/s

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


def moving_gate(env, linear_speed=LINEAR_SPEED_MOVING, yaw_rate=YAW_RATE_MOVING):
    """How strongly the robot is travelling, 0 (parked) to 1 (clearly moving)."""
    data = env.scene["wheelleg"].data
    linear = torch.nan_to_num(data.root_link_lin_vel_b[:, :2])
    yaw = torch.nan_to_num(data.root_link_ang_vel_b[:, 2])
    speed = torch.linalg.norm(linear, dim=1)
    gate = torch.maximum(speed / linear_speed, torch.abs(yaw) / yaw_rate)
    return torch.clamp(gate, 0.0, 1.0)


def low_posture_locomotion(
    env,
    min_clearance=MIN_CLEARANCE,
    linear_speed=LINEAR_SPEED_MOVING,
    yaw_rate=YAW_RATE_MOVING,
):
    """Height deficit while travelling: the crawl penalty.

    Zero for a parked robot at any height, so low posture and lying on the ground
    are legal; it rises linearly as a *moving* robot sinks below the requirement.
    """
    clearance = base_clearance(env)
    deficit = torch.clamp((min_clearance - clearance) / min_clearance, 0.0, 1.0)
    deficit = torch.where(_measurable(clearance), deficit, torch.zeros_like(deficit))
    return deficit * moving_gate(env, linear_speed, yaw_rate)


def standing_height_error(env, target_height=STANDING_CLEARANCE):
    """Normalised squared deviation from the working stance height.

    Always active, so a low posture is mildly discouraged and lying on the ground
    is expensive, without ever ending the episode.
    """
    clearance = torch.clamp(base_clearance(env), min=0.0)
    return torch.square((clearance - target_height) / target_height)


def standing_pose_error(env):
    """Mean squared deviation of the six leg joints from the nominal stance."""
    asset = env.scene["wheelleg"]
    ids, _ = asset.find_joints(_LEG_JOINTS, preserve_order=True)
    joints = asset.data.joint_pos[:, ids]
    target = joints.new_tensor([NOMINAL_STANCE[0], NOMINAL_STANCE[1], NOMINAL_STANCE[2]] * 2)
    return torch.mean(torch.square(joints - target), dim=1)


def base_ground_contact(env):
    """True when base_link touches the terrain: the robot is down, not on its wheels."""
    found = env.scene["base_ground_contact"].data.found
    return (found.reshape(env.num_envs, -1) > 0).any(dim=1)
