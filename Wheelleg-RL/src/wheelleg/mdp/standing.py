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

from ..stance import (
    CROUCH_ALPHA,
    CROUCH_SPAWN_Z,
    CROUCH_STANCE,
    FOLDED_REST_Z,
    FOLDED_STANCE,
    MIN_CLEARANCE,
    NOMINAL_STANCE,
    SPAWN_MARGIN,
    STANDING_CLEARANCE,
    leg_joint_positions,
)

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

# Fallen gates: tilt OR height, not body contact, because the box chassis can
# wedge on a side without the body sensor firing. There is exactly one such
# definition in this module: it decides what counts as fallen for the recovery
# rewards, for the standing-up deadline, and for suspending the gait penalties.
# Using two different definitions was a real bug -- a robot that was fallen by
# height but not by tilt kept paying the full crawl penalty while down.
FALLEN_TILT = 0.70  # rad, about 40 degrees
FALLEN_CLEARANCE = 0.08  # m

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


def is_down(env, tilt_limit=FALLEN_TILT, clearance_gate=FALLEN_CLEARANCE):
    """True when the robot has fallen.

    The single definition, shared by the gait-penalty suspension and the recovery
    terms: badly tilted, or too low.
    """
    return fallen_mask(env, tilt_limit, clearance_gate).bool()


def upright_gate(env, tilt_limit=FALLEN_TILT, clearance_gate=FALLEN_CLEARANCE):
    """1 while the robot is up, 0 once it is down.

    The posture rewards exist to shape a *gait*: they must stop applying once the
    robot has fallen, because being on the ground is not a gait choice. Without
    this the fall state accumulated roughly -20/s from half a dozen terms at once,
    episode returns ran to -390, the value loss blew up and the policy collapsed
    to a near-deterministic "stay down" behaviour it could not explore out of.
    """
    return (~is_down(env, tilt_limit, clearance_gate)).float()


def crouch_gate(env, reference=CROUCH_SPEED_REFERENCE):
    """How much of the high-speed crouch allowance is available, 0 to 1.

    A robot driving fast must resist pitching over backwards, and dropping the
    body is the cheap way to do it, so both height thresholds relax with forward
    speed. Reversing gets the same allowance because it tips the other way.
    """
    return torch.clamp(torch.abs(forward_speed(env)) / reference, 0.0, 1.0)


def standing_command(env, command_name="posture"):
    """1 when standing is commanded, 0 when lying folded is commanded.

    The gait penalties and the fall deadlines belong to the *locomotion* posture.
    A robot that was told to lie down is not falling: charging it a crawl penalty
    or recycling it for being on the ground would make the commanded posture
    impossible to hold.
    """
    from .posture import posture_alpha  # Local: posture pulls in mjlab.

    return posture_alpha(env, command_name)


def gait_gate(env, command_name="posture", **fallen_kwargs):
    """1 only while the robot is both standing-commanded and genuinely up."""
    return upright_gate(env, **fallen_kwargs) * standing_command(env, command_name)


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
    return deficit * moving_gate(env, linear_speed, yaw_rate) * gait_gate(env)


def standing_height_error(env, target_height=STANDING_CLEARANCE):
    """Squared deviation from the height the *commanded* posture wants.

    Always active, so a low posture is mildly discouraged and lying on the ground
    is expensive, without ever ending the episode. The standing target drops at
    speed so a high-speed crouch is not charged as a posture error, and the whole
    target blends down to the folded body's rest height when lying down is
    commanded -- otherwise a folded robot would be punished for being low, which
    is exactly what it was told to be.

    The error is normalised by the *standing* clearance, a constant, not by the
    target. Normalising by the target is what a single fixed height target could
    get away with, but the target now falls to the folded rest height of 2 mm:
    dividing a 0.14 m discrepancy by 0.002 squares to over five thousand and the
    penalty reached -7000 per episode, taking the value loss with it. A constant
    denominator bounds the term at about 1 whatever the commanded posture.
    """
    from .posture import posture_height_target  # Local: posture pulls in mjlab.

    target = posture_height_target(env, effective_target_height(env, target_height))
    clearance = torch.clamp(base_clearance(env), min=0.0)
    return torch.square((clearance - target) / STANDING_CLEARANCE)


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
    target = joints.new_tensor(leg_joint_positions(NOMINAL_STANCE))
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
    tilted = total_tilt(env) > FALLEN_TILT
    return base_ground_contact(env).float() * (~tilted).float() * standing_command(env)


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
    return torch.clamp(excess / horizon, 0.0, 1.0) * gait_gate(env)


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


# ---------------------------------------------------------------------------
# Fall recovery
#
# The structure below follows the VelStand task in the MicroDuck reference
# project (`microduck_rl/src/mjlab_microduck/tasks/`), whose comments record
# several failed runs spent on exactly this problem. Its rules, which this
# module follows deliberately:
#
# * Never gate a positive reward on being in a bad state. A per-step bonus for
#   being folded or lying pays the policy to park there; paying the *change* of
#   a potential (Ng et al. shaping) pays for rising, charges for falling and
#   pays exactly zero for holding, so it cannot be farmed. Penalties on bad
#   states are safe.
# * A hard skill must not be taxed while it is being discovered: scale the
#   smoothness and limit penalties down while the robot is down, or "do nothing"
#   wins.
# * Judge recovery complete with a *reachable* threshold. Their first bounty
#   demanded a trunk height above anything the policy achieved and never fired.
# * A fall needs a bounded recovery window, or a failed recovery farms the
#   episode and starves walking of data.
# ---------------------------------------------------------------------------

# Fallen gates, in the spirit of the reference: tilt OR height, not body contact,
# because the box chassis can wedge on a side without the body sensor firing.
# The single definition lives with the other thresholds at the top of the module.
# "Recovered": the reference used 25 degrees and a height it knew was reachable.
RECOVERED_TILT = 0.44  # rad, about 25 degrees
RECOVERED_CLEARANCE = 0.11  # m, 76% of the standing clearance
# Height potential saturates just below full stand so hopping pays nothing extra.
HEIGHT_CEILING = 0.14  # m
# Start of the standing-up move: the pose from 完全趴下落地.txt, clamped to the
# model's hard joint limits (the reported hip 0.91 and thigh 1.31 sit 0.002 rad
# outside them).
FOLDED_STANCE = (0.90757, 1.30900, -2.62)
FOLD_TOLERANCE = 0.20  # rad per joint
# Once the legs are folded, the robot has this long to be upright again before
# the environment is recycled; MAX_DOWN_TIME is the backstop for a fall that
# never reaches the folded pose at all.
#
# The backstop is deliberately short. A chassis wedged on a corner cannot act its
# way out of the pose it is in, so every second it spends down is a second of
# gradient that says "hold still": at 6 s, ``body_level`` alone accumulated about
# -38 of a -61 episode and the policy annealed to a frozen heap instead of
# standing up. A short window also multiplies the number of recovery attempts per
# iteration, and shrinks each failed attempt's accumulated penalty relative to
# the one-shot stand-up bounty -- the quantity that has to win for a recovery to
# ever be discovered.
FOLD_STAND_DEADLINE = 2.0  # s
MAX_DOWN_TIME = 2.5  # s
# Attempt taxes are reduced, never removed, while the robot is down.
FALLEN_ATTEMPT_SCALE = 0.1


def _fresh(env):
    """True on the first step after a reset, where per-episode state restarts."""
    return env.episode_length_buf <= 1


def fallen_mask(env, tilt_gate=FALLEN_TILT, clearance_gate=FALLEN_CLEARANCE):
    """1.0 where the robot counts as fallen: too tilted, or too low."""
    tilted = total_tilt(env) > tilt_gate
    low = base_clearance(env) < clearance_gate
    return (tilted | low).float()


def recovered_mask(env, tilt_gate=RECOVERED_TILT, clearance=RECOVERED_CLEARANCE):
    """1.0 where the robot counts as genuinely standing again."""
    upright = total_tilt(env) < tilt_gate
    high = base_clearance(env) > clearance
    return (upright & high).float()


def _potential_delta(env, key, value):
    """Per-step change of a potential, restarting cleanly after a reset."""
    previous = getattr(env, key, None)
    if previous is None:
        previous = value.clone()
        setattr(env, key, previous)
    fresh = _fresh(env)
    previous[fresh] = value[fresh]
    delta = value - previous
    setattr(env, key, value.clone())
    return delta


def upright_progress(env):
    """Potential-based shaping on cos(tilt): rising pays, holding pays zero."""
    gravity_z = torch.nan_to_num(
        env.scene["wheelleg"].data.projected_gravity_b[:, 2], nan=-1.0
    )
    return _potential_delta(env, "_upright_potential", -gravity_z)


def height_progress(env, ceiling=HEIGHT_CEILING):
    """Potential-based shaping on clearance: the last mile is mostly height.

    Extending the knees out of a deep crouch changes height far more than tilt,
    which is exactly where the tilt term is flat.
    """
    clearance = torch.clamp(base_clearance(env), min=0.0, max=ceiling)
    return _potential_delta(env, "_height_potential", clearance)


def fallen_tax(
    env,
    tilt_gate=FALLEN_TILT,
    clearance_gate=FALLEN_CLEARANCE,
    release_tilt=RECOVERED_TILT,
    release_clearance=RECOVERED_CLEARANCE,
):
    """Flat 1.0 per step while down, with hysteresis on the way out.

    Without it, lying still is cheap while attempting a recovery pays the
    attempt taxes, so waiting to be recycled is the rational policy. The release
    needs the *recovered* thresholds, not the arming gate, or a crouch just
    under the gate becomes a free rest state that recoveries park in.
    """
    fallen = fallen_mask(env, tilt_gate, clearance_gate).bool()
    up = recovered_mask(env, release_tilt, release_clearance).bool()
    armed = getattr(env, "_fallen_tax_armed", None)
    if armed is None:
        armed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._fallen_tax_armed = armed
    armed[_fresh(env)] = False
    armed |= fallen
    armed &= ~up
    # Being on the ground is not a fault when lying down was commanded.
    return armed.float() * standing_command(env)


def recovery_success(
    env,
    min_fallen_s=0.5,
    fallen_tilt=FALLEN_TILT,
    up_tilt=RECOVERED_TILT,
    up_clearance=RECOVERED_CLEARANCE,
):
    """One-shot bounty for a completed recovery.

    Fires on the frame an environment that has been fallen for at least
    ``min_fallen_s`` becomes upright and high enough again. Re-arming requires
    being fallen again, so oscillating around the gate pays nothing.

    "Fallen" here is the module's single definition -- tilt *or* clearance -- and
    not tilt alone. A robot lying folded with the body flat on the ground is
    level, so a tilt-only test never armed the bounty for exactly the posture a
    recovery starts from, while ``fallen_tax`` charged it every step of the way:
    the robot was taxed for being down and could not be paid for getting up. That
    is the one arrangement the whole recovery economy exists to avoid.
    """
    fallen = fallen_mask(env, tilt_gate=fallen_tilt).bool()
    up = recovered_mask(env, up_tilt, up_clearance).bool()
    seconds = getattr(env, "_recovery_fallen_s", None)
    if seconds is None:
        seconds = torch.zeros(env.num_envs, device=env.device)
        env._recovery_fallen_s = seconds
        env._recovery_armed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    fresh = _fresh(env)
    seconds[fresh] = 0.0
    env._recovery_armed[fresh] = False
    env._recovery_fallen_s = torch.where(
        fallen, seconds + env.step_dt, torch.zeros_like(seconds)
    )
    env._recovery_armed |= env._recovery_fallen_s >= min_fallen_s
    fired = env._recovery_armed & up
    env._recovery_armed &= ~fired
    return fired.float()


def folded_pose_error(env, folded=FOLDED_STANCE):
    """Mean squared deviation of the six leg joints from the folded pose."""
    asset = env.scene["wheelleg"]
    ids, _ = asset.find_joints(_LEG_JOINTS, preserve_order=True)
    joints = asset.data.joint_pos[:, ids]
    target = joints.new_tensor(leg_joint_positions(folded))
    return torch.mean(torch.square(joints - target), dim=1)


def fold_complete(env, folded=FOLDED_STANCE, tolerance=FOLD_TOLERANCE):
    """True once every leg joint is within ``tolerance`` of the folded pose.

    This is the pose the robot can stand up from, so it is the point at which a
    recovery window starts being timed.
    """
    asset = env.scene["wheelleg"]
    ids, _ = asset.find_joints(_LEG_JOINTS, preserve_order=True)
    joints = asset.data.joint_pos[:, ids]
    target = joints.new_tensor(leg_joint_positions(folded))
    return ((joints - target).abs().amax(dim=1) < tolerance)


def fallen_too_long(
    env,
    max_down_time=MAX_DOWN_TIME,
    fold_stand_deadline=FOLD_STAND_DEADLINE,
    tilt_gate=FALLEN_TILT,
    clearance_gate=FALLEN_CLEARANCE,
):
    """End an episode once the robot is clearly not getting back up.

    Two clocks. The first is a plain backstop on continuous time spent down,
    without which a failed recovery keeps the episode alive and starves walking
    of data. The second starts once the legs have folded -- the pose a stand-up
    begins from -- and allows ``fold_stand_deadline`` seconds to be upright
    again, so a chassis wedged on a corner is recycled instead of lying there.
    """
    down = fallen_mask(env, tilt_gate, clearance_gate).bool()
    folded = fold_complete(env).bool()
    # A robot told to lie down is not failing to get up.
    commanded_down = down & (standing_command(env) > 0.5)
    down_s = getattr(env, "_down_seconds", None)
    fold_s = getattr(env, "_fold_seconds", None)
    if down_s is None:
        down_s = torch.zeros(env.num_envs, device=env.device)
        fold_s = torch.zeros(env.num_envs, device=env.device)
        env._down_seconds = down_s
        env._fold_seconds = fold_s
    fresh = _fresh(env)
    down_s[fresh] = 0.0
    fold_s[fresh] = 0.0
    step = env.step_dt
    env._down_seconds = torch.where(
        commanded_down, down_s + step, torch.zeros_like(down_s)
    )
    env._fold_seconds = torch.where(
        commanded_down & folded, fold_s + step, torch.zeros_like(fold_s)
    )
    return (env._down_seconds >= max_down_time) | (env._fold_seconds >= fold_stand_deadline)


def attempt_scale(env, scale=FALLEN_ATTEMPT_SCALE):
    """Multiplier for attempt taxes: ``scale`` while down, 1 while up.

    Smoothness and joint-limit penalties exist to keep a gait clean. Charging
    them in full while the robot is trying to stand up prices the attempt above
    doing nothing, which is how a recovery never gets discovered.
    """
    fallen = fallen_mask(env).bool()
    return torch.where(fallen, torch.full_like(fallen, scale, dtype=torch.float),
                       torch.ones_like(fallen, dtype=torch.float))


def recovery_scale(env, scale=FALLEN_ATTEMPT_SCALE):
    """Multiplier for gait terms charged to a *failed* fall.

    ``attempt_scale`` fires for any fallen environment, which is right for the
    smoothness taxes: flailing on the ground is not a gait choice whatever the
    posture command says. This one is narrower. It fires only while the robot is
    down *and* has been told to stand, which is the state it is actively trying
    to leave and the only state in which these terms cannot be acted on.

    The distinction matters for orientation terms. A folded-commanded robot is
    *supposed* to be on the ground and still has to lie flat on it, so its level
    requirement keeps full weight. A robot told to stand that is already down gets
    no usable gradient from them, and charging the full tilt cost made
    ``body_level`` the largest term in the entire economy -- roughly -38 of a -61
    episode -- so the cheapest way to improve was to stop moving.
    """
    fall = fallen_mask(env).bool() & (standing_command(env) > 0.5)
    return torch.where(fall, torch.full_like(fall, scale, dtype=torch.float),
                       torch.ones_like(fall, dtype=torch.float))


# ---------------------------------------------------------------------------
# The leg-motion taxes, and when they must stand down.
#
# action_rate and joint_acc exist to keep a gait smooth, and left ungated they buy
# something worse than a smooth gait: a robot working its way over an obstacle has
# to move its legs fast and far, and charging the full tax for exactly that motion
# makes "keep the wheels down and push" cheaper than "stop and lift a wheel". A
# machine that never lifts a wheel never climbs, which is what play showed -- the
# legs did not move at all when it was blocked by a step.
#
# The reference 16DOF project gates its equivalent term on measured tilt and wheel
# contact. Tilt is kept, and the third term is the one that matters here: the
# *shortfall* between the commanded speed and the achieved speed, which is what
# pushing against a step looks like on a wheeled machine. Being fallen is included
# for the same reason as recovery_scale -- a robot standing up is working hard too.
# ---------------------------------------------------------------------------
LEG_MOTION_RELIEF = 0.2
"""How far the leg-motion taxes relax when the legs are being worked hard."""
RELIEF_TILT_START = 0.08  # rad, below this the robot is not really leaning
RELIEF_TILT_END = 0.30  # rad, at or above this the relief is full
RELIEF_BLOCKED_SPEED = 0.10  # m/s of command below which "blocked" means nothing


def leg_motion_scale(
    env,
    scale=LEG_MOTION_RELIEF,
    command_name="twist",
    tilt_start=RELIEF_TILT_START,
    tilt_end=RELIEF_TILT_END,
    blocked_speed=RELIEF_BLOCKED_SPEED,
):
    """1 normally, ``scale`` where the robot is being asked to work its legs.

    Never 0: the taxes are reduced, not removed, or a robot could thrash its way
    up an obstacle and keep the habit on the flat.
    """
    tilt_relax = torch.clamp(
        (total_tilt(env) - tilt_start) / max(tilt_end - tilt_start, 1e-6), 0.0, 1.0
    )
    asset = env.scene["wheelleg"]
    command = env.command_manager.get_command(command_name)
    speed = torch.norm(asset.data.root_link_lin_vel_b[:, :2], dim=1)
    asked = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    shortfall = torch.clamp(
        (asked - speed) / torch.clamp(asked, min=1e-6), 0.0, 1.0
    )
    blocked = (asked > blocked_speed).float() * shortfall
    relax = torch.maximum(
        torch.maximum(tilt_relax, blocked), fallen_mask(env)
    )
    return 1.0 - (1.0 - scale) * relax


def action_rate_motion_relieved(env, scale=LEG_MOTION_RELIEF):
    """mjlab action_rate_l2, relaxed while the legs are being worked hard."""
    from mjlab.envs import mdp as envs_mdp  # Lazy: keeps this module mjlab-free.

    return envs_mdp.action_rate_l2(env) * leg_motion_scale(env, scale)


def joint_acc_motion_relieved(env, asset_cfg=None, scale=LEG_MOTION_RELIEF):
    """mjlab joint_acc_l2, relaxed while the legs are being worked hard."""
    from mjlab.envs import mdp as envs_mdp

    return envs_mdp.joint_acc_l2(env, asset_cfg=asset_cfg) * leg_motion_scale(env, scale)


MAX_TERRAIN_LEVEL = 8.0
"""Levels in the curriculum grid, used to normalise the traversal bonus."""


def terrain_level_bonus(env, command_name="twist", reference=MAX_TERRAIN_LEVEL,
                        active_threshold=0.1):
    """Pay a little for being *asked* to travel a hard terrain row.

    Gated on the command, not on movement, and that distinction is the whole point:
    the seconds worth funding are the ones spent working at an obstacle, where the
    robot cannot make the commanded speed and the tracking reward has already gone
    to zero. Gating on movement would exclude exactly those seconds. A robot with
    no command is not attempting the row and is not paid, and parking is still not
    a way to farm this because the tracking reward requires real speed.
    """
    terrain = getattr(env.scene, "terrain", None)
    levels = getattr(terrain, "terrain_levels", None)
    if levels is None:
        return torch.zeros(env.num_envs, device=env.device)
    command = env.command_manager.get_command(command_name)
    asked = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    active = (asked > active_threshold).float()
    return torch.clamp(levels.float() / reference, 0.0, 1.0) * active


def body_level_error_recovery_scaled(env, scale=FALLEN_ATTEMPT_SCALE, **kwargs):
    """``body_level_error`` with the failed-fall allowance applied.

    The unscaled form stays the logged metric, so the tilt the robot is actually
    holding is still visible while this reward term is muted.
    """
    return body_level_error(env, **kwargs) * recovery_scale(env, scale)


def grounded_gate(env, clearance_gate=FALLEN_CLEARANCE):
    """1 while the body is clear of the ground, 0 while it rests on it.

    Deliberately *not* the fallen mask. That one also fires on tilt, and a robot
    rolling along on its wheels with the body leaning forty degrees is still
    travelling. Gating the tracking payment on tilt as well switched the reward
    off for most of a run whose mean body lean was sixty degrees: it withdrew the
    walking payment precisely while the robot was walking, so the policy walked
    less, leaned more, and the standard deviation collapsed from 0.37 to 0.15.
    This gate is about the belly being on the ground, which is the one thing the
    payment must not reward.
    """
    return (base_clearance(env) > clearance_gate).float()


# ---------------------------------------------------------------------------
# Velocity tracking, withdrawn while the body is on the ground.
#
# A machine dragging itself along on its belly collected the tracking reward,
# because every travelling *penalty* is suspended once it is down, so tracking
# was the one thing still paying there. Gating the payment rather than adding a
# crawl penalty is what leaves a robot that is wriggling without translating --
# which is what standing up looks like from the outside -- charged nothing. The
# robot's only income while down is then the recovery bounty, and the moment it
# is up on its wheels the command pays again, so getting up is followed by
# walking rather than by lying back down.
# ---------------------------------------------------------------------------
def track_linear_velocity_x_standing(env, std, command_name="twist"):
    """``track_linear_velocity_x``, withdrawn while the body is on the ground."""
    from .rewards import track_linear_velocity_x

    return track_linear_velocity_x(env, std, command_name) * grounded_gate(env)


def track_linear_velocity_y_standing(env, std, command_name="twist"):
    """``track_linear_velocity_y``, withdrawn while the body is on the ground."""
    from .rewards import track_linear_velocity_y

    return track_linear_velocity_y(env, std, command_name) * grounded_gate(env)


def track_angular_velocity_z_standing(env, std, command_name="twist"):
    """``track_angular_velocity_z``, withdrawn while the body is on the ground."""
    from .rewards import track_angular_velocity_z

    return track_angular_velocity_z(env, std, command_name) * grounded_gate(env)


# Reverse-curriculum spawn states. The poses and their rest heights are geometry,
# so they live in ``wheelleg.stance`` next to the nominal stance; they are
# imported above and re-exported here for the task config.
#   FOLDED_STANCE, FOLDED_REST_Z, CROUCH_STANCE, CROUCH_SPAWN_Z, SPAWN_MARGIN


def spawn_fallen_state(
    env,
    env_ids=None,
    folded_probability=0.35,
    crouch_probability=0.0,
    folded_hold_probability=0.5,
    asset_cfg=None,
):
    """Reset a slice of episodes already on the ground: reverse-curriculum spawns.

    With only a standing start, the policy falls immediately and never observes
    the successful standing-up branch, so the progress potentials have nothing
    positive to discover and the run settles into a flat "lying still" optimum --
    every episode almost identical, no advantage signal, no learning. MicroDuck's
    VelStand notes reach the same conclusion, calling prone and mid-recovery
    spawns "the reliable fix for learns-the-start-never-the-last-mile".

    The remaining share is a plain standing start, and it is the largest one.
    Balance and travel are what the machine is for: a spawn mix that begins most
    episodes on the ground starves the walking data every other reward term is
    written for, which is how one tilt penalty came to dominate the economy.

    ``folded_probability`` starts episodes in the fully folded pose from
    完全趴下落地.txt; ``crouch_probability`` starts them halfway up, which is
    where the last mile of a stand-up lives and where a policy trained only from
    prone rarely finds itself.

    Each spawn also *chooses the posture command* it is about to be given,
    through ``env._spawn_alpha`` (where the posture starts, matching the pose just
    written) and ``env._spawn_command`` (where it is told to go). That coupling is
    not cosmetic: the event manager runs before the command manager inside
    ``_reset_idx``, so without it the two are drawn independently and a standing
    spawn is handed a folded command most of the time -- at which point the action
    offset drags its legs out from under it and it face-plants on the first step.
    Two values rather than one because a spawn folded and told to stand has to
    start folded and ramp up: that transition is the get-up itself. A folded spawn
    is told to stand with probability ``1 - folded_hold_probability`` and to stay
    folded otherwise, which is what trains the flat, wheels-stowed pose itself.
    """
    from mjlab.envs.mdp.events import resolve_env_ids  # Lazy: keeps this pure.

    env_ids = resolve_env_ids(env, env_ids)
    if asset_cfg is None:
        raise ValueError("spawn_fallen_state needs asset_cfg for the wheelleg entity")
    asset = env.scene[asset_cfg.name]

    draw = torch.rand(len(env_ids), device=env.device)
    folded = draw < folded_probability
    crouch = (draw >= folded_probability) & (draw < folded_probability + crouch_probability)
    selected = folded | crouch

    # Tell each spawn where its posture starts and where it is told to go. Both
    # matter: a spawn folded and told to stand must start folded and ramp up,
    # because that transition *is* the get-up, and starting alpha at the commanded
    # value instead snapped the offset to the standing stance in one step and made
    # the legs jump. A folded spawn held flat is told to stay. Everything else
    # starts standing; the crouch starts halfway, which is where it sits on the
    # interpolation.
    if not hasattr(env, "_spawn_alpha"):
        env._spawn_alpha = torch.full(
            (env.num_envs,), float("nan"), device=env.device
        )
        env._spawn_command = torch.full(
            (env.num_envs,), float("nan"), device=env.device
        )
    hold = torch.rand(len(env_ids), device=env.device) < folded_hold_probability
    env._spawn_alpha[env_ids] = torch.where(
        folded,
        torch.zeros_like(draw),
        torch.where(
            crouch, torch.full_like(draw, CROUCH_ALPHA), torch.ones_like(draw)
        ),
    )
    env._spawn_command[env_ids] = torch.where(
        folded & hold, torch.zeros_like(draw), torch.ones_like(draw)
    )

    if not bool(selected.any()):
        return
    chosen = env_ids[selected]
    is_folded = folded[selected]

    leg_ids, _ = asset.find_joints(_LEG_JOINTS, preserve_order=True)
    leg_ids = torch.tensor(leg_ids, device=env.device, dtype=torch.long)
    folded_pose = torch.tensor(leg_joint_positions(FOLDED_STANCE), device=env.device)
    crouch_pose = torch.tensor(leg_joint_positions(CROUCH_STANCE), device=env.device)
    joint_pos = torch.where(is_folded.unsqueeze(1), folded_pose, crouch_pose)
    asset.write_joint_state_to_sim(
        joint_pos, torch.zeros_like(joint_pos), env_ids=chosen, joint_ids=leg_ids
    )

    root = asset.data.default_root_state[chosen].clone()
    pose = root[:, 0:3].clone()
    height = torch.where(
        is_folded,
        torch.full_like(is_folded, FOLDED_REST_Z, dtype=torch.float),
        torch.full_like(is_folded, CROUCH_SPAWN_Z + SPAWN_MARGIN, dtype=torch.float),
    )
    pose[:, 2] = height + env.scene.env_origins[chosen, 2]
    # The default orientation is upright: the folded body lies belly-down with its
    # z axis still vertical, it does not tip over.
    asset.write_root_link_pose_to_sim(
        torch.cat([pose, root[:, 3:7]], dim=-1), env_ids=chosen
    )
    asset.write_root_link_velocity_to_sim(
        torch.zeros_like(root[:, 7:13]), env_ids=chosen
    )


def action_rate_fallen_scaled(env, scale=FALLEN_ATTEMPT_SCALE):
    """mjlab action_rate_l2, reduced while the robot is down."""
    from mjlab.envs import mdp as envs_mdp  # Lazy: keeps this module mjlab-free.

    return envs_mdp.action_rate_l2(env) * attempt_scale(env, scale)


def joint_acc_fallen_scaled(env, asset_cfg=None, scale=FALLEN_ATTEMPT_SCALE):
    """mjlab joint_acc_l2, reduced while the robot is down."""
    from mjlab.envs import mdp as envs_mdp

    return envs_mdp.joint_acc_l2(env, asset_cfg=asset_cfg) * attempt_scale(env, scale)


def joint_pos_limits_fallen_scaled(env, asset_cfg=None, scale=FALLEN_ATTEMPT_SCALE):
    """mjlab joint_pos_limits, reduced while the robot is down.

    The folded pose sits on the hard hip and thigh limits, so leaving this at
    full weight would charge the robot for adopting the very pose it must stand
    up from.
    """
    from mjlab.envs import mdp as envs_mdp

    return envs_mdp.joint_pos_limits(env, asset_cfg=asset_cfg) * attempt_scale(env, scale)


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
