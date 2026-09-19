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

# At speed the body is allowed to drop a little to resist pitching over
# backwards, and to lean forward rather than being held bolt upright. The drop is
# capped so the clearance requirement never falls below MIN_CLEARANCE - drop.
CROUCH_SPEED_REFERENCE = 0.6  # m/s of forward speed for the full allowance
CROUCH_CLEARANCE_DROP = 0.015  # m allowed off both height thresholds

# Body attitude: roll is always costly, a forward lean is free up to the
# allowance, and leaning backwards costs extra because it is how the robot
# wheelies over under a large forward command. The cost saturates so that being
# on the ground cannot produce an unbounded penalty.
FORWARD_LEAN_ALLOWANCE = 0.30  # rad, about 17 degrees
BACKWARD_LEAN_SCALE = 1.5
MAX_TILT_COST = 0.6  # rad, about 34 degrees

# How far the body may tip, and for how long, before the episode is ended. A
# get-up takes a second or two, so three seconds is enough time to practise while
# halving how long a failed episode spends accumulating penalties.
DOWN_TILT_LIMIT = 0.9  # rad, about 52 degrees
MAX_DOWN_TIME = 3.0  # s

# Sideways speed that counts as genuinely translating sideways rather than
# merely rotating, used by the lateral step reward.
LATERAL_SPEED_MOVING = 0.10  # m/s

# Height difference between the two wheel axles at which the terrain counts as
# uneven enough that the legs *must* differ in length to keep the body level.
UNEVEN_HEIGHT_REFERENCE = 0.03  # m

_WHEEL_BODIES = ("left_wheel_link", "right_wheel_link")

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


def forward_speed(env):
    """Body-frame forward speed, m/s. Negative means reversing."""
    return torch.nan_to_num(env.scene["wheelleg"].data.root_link_lin_vel_b[:, 0])


def total_tilt(env):
    """Angle between the body's z axis and world up, in radians."""
    gravity_z = torch.nan_to_num(env.scene["wheelleg"].data.projected_gravity_b[:, 2])
    return torch.acos(torch.clamp(-gravity_z, -1.0, 1.0))


def is_down(env, tilt_limit=DOWN_TILT_LIMIT):
    """True when the robot has fallen: badly tilted, or the body on the ground."""
    return (total_tilt(env) > tilt_limit) | base_ground_contact(env)


def upright_gate(env, tilt_limit=DOWN_TILT_LIMIT):
    """1 while the robot is up, 0 once it is down.

    The posture rewards exist to shape a *gait*: they must stop applying once the
    robot has fallen, because being on the ground is not a gait choice. Without
    this the fall state accumulated roughly -20/s from half a dozen terms at once,
    episode returns ran to -390, the value loss blew up and the policy collapsed
    to a near-deterministic "stay down" behaviour it could not explore out of.
    """
    return (~is_down(env, tilt_limit)).float()


def crouch_gate(env, reference=CROUCH_SPEED_REFERENCE):
    """How much of the high-speed crouch allowance is available, 0 to 1.

    A robot driving fast must resist pitching over backwards, and dropping the
    body is the cheap way to do it, so both height thresholds relax with forward
    speed. Reversing gets the same allowance because it tips the other way.
    """
    return torch.clamp(torch.abs(forward_speed(env)) / reference, 0.0, 1.0)


def effective_min_clearance(env, min_clearance=MIN_CLEARANCE):
    """Clearance required while moving, relaxed at speed by the crouch allowance."""
    return min_clearance - CROUCH_CLEARANCE_DROP * crouch_gate(env)


def effective_target_height(env, target_height=STANDING_CLEARANCE):
    """Stance height to track, relaxed at speed by the crouch allowance."""
    return target_height - CROUCH_CLEARANCE_DROP * crouch_gate(env)


def low_posture_locomotion(
    env,
    min_clearance=MIN_CLEARANCE,
    linear_speed=LINEAR_SPEED_MOVING,
    yaw_rate=YAW_RATE_MOVING,
):
    """Height deficit while travelling: the crawl penalty.

    Zero for a parked robot at any height, so low posture and lying on the ground
    are legal; it rises linearly as a *moving* robot sinks below the requirement,
    which itself drops a little at speed to allow a forward-leaning crouch.

    Also zero once the robot is down: a fallen robot wriggling is not choosing a
    crawling gait, and charging it the full crawl rate for the seconds it is down
    is what turned falls into -100 episodes.
    """
    floor = effective_min_clearance(env, min_clearance)
    clearance = base_clearance(env)
    deficit = torch.clamp((floor - clearance) / floor, 0.0, 1.0)
    deficit = torch.where(_measurable(clearance), deficit, torch.zeros_like(deficit))
    return deficit * moving_gate(env, linear_speed, yaw_rate) * upright_gate(env)


def standing_height_error(env, target_height=STANDING_CLEARANCE):
    """Normalised squared deviation from the working stance height.

    Always active, so a low posture is mildly discouraged and lying on the ground
    is expensive, without ever ending the episode. The target drops at speed so a
    high-speed crouch is not charged as a posture error.
    """
    target = effective_target_height(env, target_height)
    clearance = torch.clamp(base_clearance(env), min=0.0)
    return torch.square((clearance - target) / target)


def body_tilt(env):
    """Body roll and pitch in radians.

    ``pitch`` is positive when the body leans *backwards* and negative when it
    leans forwards, because the projected gravity x component is ``-sin(pitch)``.
    """
    gravity = torch.nan_to_num(env.scene["wheelleg"].data.projected_gravity_b)
    pitch = torch.asin(torch.clamp(gravity[:, 0], -1.0, 1.0))
    roll = torch.asin(torch.clamp(gravity[:, 1], -1.0, 1.0))
    return roll, pitch


def body_level_error(
    env,
    forward_allowance=FORWARD_LEAN_ALLOWANCE,
    backward_scale=BACKWARD_LEAN_SCALE,
    max_cost=MAX_TILT_COST,
):
    """Cost of tilting the body away from world vertical, in radians.

    The body's z axis must stay upright rather than following the slope: tilt in
    roll is always charged, so the machine cannot simply lie along a bank, and
    pitch is charged as well once past the forward allowance. The allowance
    itself scales with speed, so a parked robot must be level while a moving one
    may lean forward to resist pitching over backwards.

    Ramped linearly rather than squared: a squared tilt cost is flat near upright
    and cannot pull back the steady bias that rolling on a slope produces. The
    cost saturates at ``max_cost`` so that lying on the ground cannot turn into a
    club large enough to swamp the value function.
    """
    roll, pitch = body_tilt(env)
    allowed = forward_allowance * crouch_gate(env)
    forward_excess = torch.clamp(-pitch - allowed, min=0.0)
    cost = torch.abs(roll) + backward_scale * torch.clamp(pitch, min=0.0) + forward_excess
    return torch.clamp(cost, max=max_cost)


def wheel_height_difference(env, asset_cfg=None):
    """Height difference between the two wheel axles, in metres.

    This measures how unevenly the ground supports the robot rather than how the
    body is oriented: a robot tilted along a bank and a robot held level on the
    same bank see the same axle difference. That makes it the right signal for
    "the legs must now differ in length".
    """
    if asset_cfg is None:
        raise ValueError(
            "wheel_height_difference needs asset_cfg with body_names="
            f"{_WHEEL_BODIES} so the body ids are resolved by the manager"
        )
    asset = env.scene[asset_cfg.name]
    z = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
    return torch.abs(z[:, 0] - z[:, 1])


def terrain_unevenness_gate(env, asset_cfg=None, reference=UNEVEN_HEIGHT_REFERENCE):
    """0 on level ground, 1 once the wheels differ by ``reference`` metres.

    Used to switch *off* the symmetry requirement where asymmetry is required.
    """
    difference = wheel_height_difference(env, asset_cfg)
    return torch.clamp(difference / reference, 0.0, 1.0)


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


def base_ground_contact_cost(env):
    """Body-on-ground cost, suspended while the robot is already down.

    It is here to discourage travelling on the belly, not to add a second charge
    for being on the ground; the height error and the tilt cost already cover that.
    """
    return base_ground_contact(env).float() * _not_down_ignoring_base_contact(env)


def _not_down_ignoring_base_contact(env, tilt_limit=DOWN_TILT_LIMIT):
    """Upright gate that only looks at tilt.

    ``is_down`` ORs in body contact, so it cannot be reused inside the body
    contact term itself or the term would always switch itself off.
    """
    return (total_tilt(env) <= tilt_limit).float()


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
    that is what the kneeling gait did. Suspended once the robot is down, where
    wheels in the air are the expected consequence of the fall rather than a
    posture choice.
    """
    excess = torch.clamp(wheel_support_time(env, sensor_name) - allowance, min=0.0)
    return torch.clamp(excess / horizon, 0.0, 1.0) * upright_gate(env)


def wheel_contact_fraction(env, sensor_name="feet_ground_contact"):
    """Fraction of wheels currently touching the terrain, 0 to 1."""
    found = env.scene[sensor_name].data.found
    return (found.reshape(env.num_envs, -1) > 0).float().mean(dim=1)


def lateral_command_demand(env, command_name="twist", reference=LATERAL_COMMAND_REF):
    """Sideways command strength, 0 (none) to 1 (at or above the reference)."""
    command = env.command_manager.get_command(command_name)
    return torch.clamp(torch.abs(command[:, 1]) / reference, 0.0, 1.0)


def leg_symmetry_error(
    env,
    command_name="twist",
    lateral_reference=LATERAL_COMMAND_REF,
    asset_cfg=None,
    uneven_reference=UNEVEN_HEIGHT_REFERENCE,
):
    """How differently the two legs are posed, where that difference is wrong.

    The hip axes are *not* mirrored in the model (both are ``+X``), so a
    mirror-symmetric pair needs ``left_hip == -right_hip`` and the hip term is the
    sum. The thigh and knee axes are both ``+Y``, where rotation does not involve
    the lateral offset, so equal angles are already mirror-symmetric and their
    term is the difference.

    Equal leg length is only correct on level ground. When the two wheels rest at
    different heights -- a lateral bank, or the two faces of a step -- the body
    can only stay level if the leg on the high side retracts and the leg on the
    low side extends, which is exactly a thigh/knee difference. The requirement
    therefore fades out with the *measured* axle height difference, and
    separately with a sideways command, since sideways travel has to be stepped.
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
    level_ground = 1.0 - terrain_unevenness_gate(env, asset_cfg, uneven_reference)
    return error * level_ground * (1.0 - lateral_command_demand(env, command_name, lateral_reference))


def lateral_step_reward(
    env,
    command_name="twist",
    reference=LATERAL_COMMAND_REF,
    motion_reference=LATERAL_SPEED_MOVING,
    sensor_name="feet_ground_contact",
):
    """Pay for stepping a leg while actually travelling sideways.

    Wheels cannot roll sideways, so a sideways command is only achievable by
    lifting a leg, translating it and planting it again. Velocity tracking alone
    cannot prefer that over simply slipping, so the step itself is rewarded. The
    other wheel must still be carrying the robot, and the body must genuinely be
    moving sideways relative to its own heading -- otherwise a robot that merely
    chatters its wheels up and down collects the reward for doing nothing.
    """
    data = env.scene["wheelleg"].data
    lateral_speed = torch.abs(torch.nan_to_num(data.root_link_lin_vel_b[:, 1]))
    translating = torch.clamp(lateral_speed / motion_reference, 0.0, 1.0)
    lifting = (wheel_air_time(env, sensor_name) > 0.0).float()
    supporting = (wheel_support_time(env, sensor_name) <= 0.0).float()
    return lateral_command_demand(env, command_name, reference) * translating * lifting * supporting


class FallenTooLong:
    """End an episode only once the robot has been down for a while.

    Resetting the instant the body tips teaches the policy that falling is a way
    to end the episode, and never gives it the seconds it needs to practise
    standing back up. Here the episode continues while the robot is down and ends
    only after ``max_down_time`` seconds without recovering, so getting up is
    worth learning.
    """

    def __init__(self, cfg, env):
        self.max_down_time = float(cfg.params["max_down_time"])
        self.tilt_limit = float(cfg.params["tilt_limit"])
        self._down_time = torch.zeros(env.num_envs, device=env.device)
        self._dt = env.step_dt

    def _is_down(self, env):
        gravity_z = torch.nan_to_num(env.scene["wheelleg"].data.projected_gravity_b[:, 2])
        tilted = torch.acos(torch.clamp(-gravity_z, -1.0, 1.0)) > self.tilt_limit
        return tilted | base_ground_contact(env)

    def __call__(self, env, max_down_time, tilt_limit):
        del max_down_time, tilt_limit  # Read once in __init__.
        down = self._is_down(env)
        self._down_time = torch.where(
            down, self._down_time + self._dt, torch.zeros_like(self._down_time)
        )
        return self._down_time > self.max_down_time

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._down_time[env_ids] = 0.0


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
