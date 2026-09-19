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

# Sideways command magnitude that counts as a full lateral request. The task
# samples |lin_vel_y| up to 0.5 m/s, so half of that is treated as "clearly
# going sideways" for the symmetry relaxation and the step reward.
LATERAL_COMMAND_REF = 0.25  # m/s

# A wheel may leave the ground to roll over a bump or take a deliberate step, so
# a lift is free up to the allowance; beyond it the penalty ramps to full over
# the horizon. This is the knob to loosen if rough terrain needs longer lifts.
WHEEL_AIR_ALLOWANCE = 0.25  # s
WHEEL_AIR_HORIZON = 0.75  # s

# Thigh and knee mismatch is penalised in full; hip mismatch counts for less so
# that lateral balance and turning can still use abduction.
_SYMMETRY_WEIGHTS = (0.5, 1.0, 1.0)  # hip, thigh, knee

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


def wheel_air_time(env, sensor_name="feet_ground_contact"):
    """Longest current air time across the wheels, in seconds."""
    air = env.scene[sensor_name].data.current_air_time
    if air is None:
        raise ValueError(f"sensor {sensor_name!r} must set track_air_time=True")
    return air.amax(dim=1)


def wheel_support_time(env, sensor_name="feet_ground_contact"):
    """How long *no* wheel has touched the ground, in seconds.

    The minimum of the two air times: it only starts counting once both wheels
    have left the ground. A single lifted wheel is a step, not a loss of support.
    """
    air = env.scene[sensor_name].data.current_air_time
    if air is None:
        raise ValueError(f"sensor {sensor_name!r} must set track_air_time=True")
    return air.amin(dim=1)


def no_wheel_support(
    env,
    sensor_name="feet_ground_contact",
    allowance=WHEEL_AIR_ALLOWANCE,
    horizon=WHEEL_AIR_HORIZON,
):
    """Penalty in [0, 1] that ramps up while the robot has no wheel on the ground.

    Lifting one leg to step sideways is free for as long as it takes, because the
    other wheel still carries the robot. Only losing *every* wheel is a fault, and
    that is what the kneeling gait did.
    """
    excess = torch.clamp(wheel_support_time(env, sensor_name) - allowance, min=0.0)
    return torch.clamp(excess / horizon, 0.0, 1.0)


def wheel_contact_fraction(env, sensor_name="feet_ground_contact"):
    """Fraction of wheels currently touching the terrain, 0 to 1."""
    found = env.scene[sensor_name].data.found
    return (found.reshape(env.num_envs, -1) > 0).float().mean(dim=1)


def lateral_command_demand(env, command_name="twist", reference=LATERAL_COMMAND_REF):
    """Sideways command strength, 0 (none) to 1 (at or above the reference)."""
    command = env.command_manager.get_command(command_name)
    return torch.clamp(torch.abs(command[:, 1]) / reference, 0.0, 1.0)


def leg_symmetry_error(env, command_name="twist", lateral_reference=LATERAL_COMMAND_REF):
    """How differently the two legs are posed, relaxed for sideways travel.

    The hip axes are *not* mirrored in the model (both are ``+X``), so a
    mirror-symmetric pair needs ``left_hip == -right_hip`` and the hip term is the
    sum. The thigh and knee axes are both ``+Y``, where rotation does not involve
    the lateral offset, so equal angles are already mirror-symmetric and their
    term is the difference.

    A two-wheel differential robot cannot roll sideways, so sideways travel has to
    be *stepped*: one leg lifts while the other supports. That asymmetry is
    correct, so the requirement fades out as the sideways command grows and stays
    in force for forward, backward and turning commands.
    """
    asset = env.scene["wheelleg"]
    ids, _ = asset.find_joints(_LEG_JOINTS, preserve_order=True)
    q = asset.data.joint_pos[:, ids]
    left, right = q[:, :3], q[:, 3:]
    mismatch = torch.stack(
        (left[:, 0] + right[:, 0], left[:, 1] - right[:, 1], left[:, 2] - right[:, 2]),
        dim=1,
    )
    weights = mismatch.new_tensor(_SYMMETRY_WEIGHTS)
    error = torch.mean(torch.square(mismatch) * weights, dim=1)
    return error * (1.0 - lateral_command_demand(env, command_name, lateral_reference))


def lateral_step_reward(
    env,
    command_name="twist",
    reference=LATERAL_COMMAND_REF,
    sensor_name="feet_ground_contact",
):
    """Pay for stepping a leg while a sideways command is active.

    Wheels cannot roll sideways, so a sideways command is only achievable by
    lifting a leg, translating it and planting it again. Velocity tracking alone
    cannot prefer that over simply slipping, so the step itself is rewarded. The
    other wheel must still be carrying the robot, which rules out a fall.
    """
    lifting = (wheel_air_time(env, sensor_name) > 0.0).float()
    supporting = (wheel_support_time(env, sensor_name) <= 0.0).float()
    return lateral_command_demand(env, command_name, reference) * lifting * supporting


def wheeled_stance_locomotion(
    env,
    command_name="twist",
    command_threshold=0.1,
    min_clearance=MIN_CLEARANCE,
    target_clearance=STANDING_CLEARANCE,
    sensor_name="feet_ground_contact",
):
    """Reward for travelling in a wheeled standing posture.

    Paid only when a locomotion command is active, the robot is actually
    travelling and the body is clear of the ground, and it scales with how close
    the body is to the working stance height. Kneeling, lying down and standing
    still all earn nothing, so it cannot be farmed by refusing to move.

    Wheel support is scored as ``0.5 + 0.5 * contact fraction`` rather than the
    fraction itself: a stepping gait rests on one wheel between plants and must
    not be taxed for it, while a robot with no wheel down loses the reward.
    """
    command = env.command_manager.get_command(command_name)
    commanded = (
        torch.linalg.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
        > command_threshold
    ).float()
    clearance = base_clearance(env)
    span = max(target_clearance - min_clearance, 1e-6)
    height = torch.clamp((clearance - min_clearance) / span, 0.0, 1.0)
    height = torch.where(_measurable(clearance), height, torch.zeros_like(height))
    support = 0.5 + 0.5 * wheel_contact_fraction(env, sensor_name)
    return commanded * moving_gate(env) * height * support
