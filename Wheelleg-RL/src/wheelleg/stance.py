"""Nominal wheel-leg stance geometry — the single source of truth.

Why this module exists
----------------------
The pose in ``robot_description/标准站立.txt`` (hip 0, thigh 1.02, knee -1.57 rad)
is the mechanical reference stand, but it is *not* reachable at the required
locomotion clearance: with the wheels resting on flat ground that pose puts the
``base_link`` origin only ~0.1255 m above the floor. Flat-ground locomotion has
to keep the body at or above ``MIN_CLEARANCE``, so the trained stance is a
slightly extended version of the reference pose, solved here from the MJCF body
offsets rather than copied, so a geometry change cannot silently invalidate it.

The extension is chosen so the wheel axles end up under the base origin, which
puts the support line under the centre of mass instead of 2 cm in front of it.
All leg joints below the hip rotate about the body-frame Y axis, so wheel axle
position and level-ground clearance are closed-form.
"""
from __future__ import annotations

import math
from typing import NamedTuple

# Fixed parent -> child body offsets, metres, from mjcf/8dof_wheelleg.xml.
HIP_BODY = (-0.04625, 0.0787267604552256, 0.03)
THIGH_BODY = (0.0774767604552631, 0.0167732395447309, 0.0)
SHANK_BODY = (0.0, -0.00300000000000789, -0.0900000000000087)
WHEEL_BODY = (0.00127323954474146, 0.0235000000000081, -0.0926849666847982)

# Body masses (kg) and their inertial offsets in their own frames, used to check
# that the support line sits under the centre of mass and to size the actuators.
BODY_MASSES = {
    "base": (0.5, (0.0, 0.0, 0.03)),
    "hip": (0.2, (0.05, 0.0, 0.0)),
    "thigh": (0.1, (0.0, 0.0, -0.04)),
    "shank": (0.15, (0.0, 0.01, -0.08)),
    "wheel": (0.1, (0.0, 0.01, 0.0)),
}

GRAVITY = 9.81

TOTAL_MASS = BODY_MASSES["base"][0] + 2.0 * sum(
    BODY_MASSES[name][0] for name in ("hip", "thigh", "shank", "wheel")
)

# Outer rim radius of the wheel meshes, which are centred on the axle after the
# converter's documented rim correction. Ground contact is one radius below it.
WHEEL_RADIUS = 0.03

# Nominal tread-centre track at hip = 0, used by the wheel roll reward.
WHEEL_TRACK = 0.216

# Joint limits from the MJCF.
THIGH_LIMIT = (-0.8726646259971648, 1.3089969389957472)
KNEE_LIMIT = (-2.792526803190927, 0.0)
HIP_LIMIT = (-0.9075712110370514, 0.9075712110370514)

# The mechanically supplied reference stand.
REFERENCE_STANCE = (0.0, 1.02, -1.57)

# Clearance the locomotion task must hold *while moving*, in metres.
MIN_CLEARANCE = 0.13
# Working stance: above MIN_CLEARANCE so a stride can squat and lift legs freely.
STANDING_CLEARANCE = 0.145


class LegPose(NamedTuple):
    axle: tuple[float, float, float]
    base_clearance: float
    axle_x: float
    shank_angle: float


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _rot_y(v, angle):
    c, s = math.cos(angle), math.sin(angle)
    x, y, z = v
    return (c * x + s * z, y, -s * x + c * z)


def _rot_x(v, angle):
    c, s = math.cos(angle), math.sin(angle)
    x, y, z = v
    return (x, c * y - s * z, s * y + c * z)


def leg_pose(thigh: float, knee: float, hip: float = 0.0) -> LegPose:
    """Axle position and base-origin clearance for a left-leg pose.

    ``shank_angle`` is ``thigh + knee``: the absolute tilt of the shank. Wheel
    spin does not move the axle, so it does not affect clearance.

    ``HIP_BODY`` is a base-frame offset, while every offset below the hip joint
    rotates with the leg, so only ``THIGH_BODY`` and below are composed here.
    """
    below_hip = _add(THIGH_BODY, _rot_y(_add(SHANK_BODY, _rot_y(WHEEL_BODY, knee)), thigh))
    axle = _add(HIP_BODY, _rot_x(below_hip, hip))
    return LegPose(
        axle=axle,
        base_clearance=WHEEL_RADIUS - axle[2],
        axle_x=axle[0],
        shank_angle=thigh + knee,
    )


def clearance(thigh: float, knee: float, hip: float = 0.0) -> float:
    return leg_pose(thigh, knee, hip).base_clearance


def joint_x(thigh: float, knee: float) -> dict[str, float]:
    """Base-frame x of the hip/thigh/knee axes and of the wheel contact point.

    Rotation about the hip's X axis leaves every x coordinate unchanged, so
    abduction does not appear here.
    """
    thigh_x = HIP_BODY[0] + THIGH_BODY[0]
    return {
        "hip": HIP_BODY[0],
        "thigh": thigh_x,
        "knee": thigh_x + _rot_y(SHANK_BODY, thigh)[0],
        "contact": leg_pose(thigh, knee).axle_x,
    }


def static_joint_torques(thigh: float, knee: float) -> dict[str, float]:
    """Vertical-load torque (N*m) each leg joint holds at a level stance.

    A vertical force ``F`` at the wheel contact creates a torque ``F * dx`` about
    the body-frame Y axis, where ``dx`` is the horizontal arm to that joint.
    """
    load = TOTAL_MASS * GRAVITY / 2.0  # one leg carries half the robot
    x = joint_x(thigh, knee)
    return {name: load * abs(x["contact"] - x[name]) for name in ("hip", "thigh", "knee")}


def com_x(thigh: float, knee: float, hip: float = 0.0) -> float:
    """Base-frame x of the whole-robot centre of mass at a symmetric stance.

    Only x matters for the support line, so the mirrored right leg contributes
    the same x as the left and is counted twice by mass.
    """
    shank_angle = thigh + knee
    hip_frame = {
        "hip": ((0.0, 0.0, 0.0), 0.0),
        "thigh": (THIGH_BODY, thigh),
        "shank": (_add(THIGH_BODY, _rot_y(SHANK_BODY, thigh)), shank_angle),
        "wheel": (
            _add(_add(THIGH_BODY, _rot_y(SHANK_BODY, thigh)), _rot_y(WHEEL_BODY, shank_angle)),
            shank_angle,
        ),
    }

    def x_of(origin, y_angle, offset):
        point = _add(origin, _rot_y(offset, y_angle))
        return _add(HIP_BODY, _rot_x(point, hip))[0]

    weighted = 0.0
    total = 0.0
    for name, (origin, y_angle) in hip_frame.items():
        mass, offset = BODY_MASSES[name]
        weighted += mass * x_of(origin, y_angle, offset) * 2
        total += mass * 2
    base_mass, base_offset = BODY_MASSES["base"]
    weighted += base_mass * base_offset[0]
    total += base_mass
    return weighted / total


def _thigh_for_clearance(target: float, shank_angle: float):
    """Bisect the thigh angle at a fixed shank tilt to reach ``target``.

    Returns ``None`` when the target is unreachable at that shank tilt.
    Clearance is strictly increasing as the thigh angle falls towards 0.
    """
    lo, hi = REFERENCE_STANCE[1], 0.0
    if clearance(hi, shank_angle - hi) < target:
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if clearance(mid, shank_angle - mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def solve_stance(target: float = STANDING_CLEARANCE, axle_x: float = 0.0):
    """Solve ``(hip, thigh, knee)`` for a level stance.

    Finds the shank tilt at which the axle sits at ``axle_x`` (0 = under the base
    origin) while the base origin clears the ground by ``target``.
    """
    # Bracket the shank tilt: axle_x decreases monotonically as the shank tilt
    # rises towards 0, so scanning downwards finds the pair that straddles it.
    lo = hi = None
    step = 0.005
    shank = -0.05
    previous = None
    while shank > -1.4:
        thigh = _thigh_for_clearance(target, shank)
        if thigh is not None:
            value = leg_pose(thigh, shank - thigh).axle_x
            if previous is not None and previous[1] <= axle_x < value:
                # lo holds axle_x > request, hi holds axle_x <= request.
                lo, hi = shank, previous[0]
                break
            previous = (shank, value)
        shank -= step
    if lo is None:
        raise ValueError(f"no stance reaches {target} m clearance at axle_x={axle_x}")

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        thigh = _thigh_for_clearance(target, mid)
        assert thigh is not None
        if leg_pose(thigh, mid - thigh).axle_x > axle_x:
            lo = mid
        else:
            hi = mid

    shank = 0.5 * (lo + hi)
    thigh = _thigh_for_clearance(target, shank)
    assert thigh is not None
    knee = shank - thigh
    if not THIGH_LIMIT[0] < thigh < THIGH_LIMIT[1]:
        raise ValueError(f"solved thigh {thigh} outside joint limits")
    if not KNEE_LIMIT[0] < knee < KNEE_LIMIT[1]:
        raise ValueError(f"solved knee {knee} outside joint limits")
    return (REFERENCE_STANCE[0], thigh, knee)


# Solved once: the pose the policy is initialised to and asked to hold.
NOMINAL_STANCE = solve_stance()

# The fully folded pose from robot_description/完全趴下落地.txt. The reported hip
# 0.91 and thigh 1.31 round just past the hard joint limits, so they are pinned
# to the limits here rather than carried as literals; the knee value is inside its
# limit as given. Both pinned joints are also well outside the 0.95 soft limits,
# which is why the joint-limit penalty is scaled down while the robot is down.
# With the legs folded and the body resting on its base mesh the wheel rims clear
# the ground by 1.1 mm, so the wheels really are stowed and the body carries the
# robot. Verified by scripts/folded_pose_check.py.
FOLDED_STANCE = (HIP_LIMIT[1], THIGH_LIMIT[1], -2.62)
# Base origin height at which the folded body rests on its base mesh: the mesh
# spans z = 0.000..0.060 in the base frame, so z = 0 puts its bottom face on the
# ground. The margin guards against initial penetration.
FOLDED_REST_Z = 0.002
# Halfway between the folded pose and the nominal stance: a real mid-recovery
# crouch, with the wheels still carrying the robot.
CROUCH_STANCE = (0.4538, 1.0776, -1.9267)
CROUCH_SPAWN_Z = clearance(CROUCH_STANCE[1], CROUCH_STANCE[2], CROUCH_STANCE[0])
SPAWN_MARGIN = 0.002

# ---------------------------------------------------------------------------
# Left/right expansion: the single place the mirror convention lives.
#
# The model does *not* mirror the hip axes -- both ``*_hip_joint`` rotate about
# +X -- so one shared hip angle rolls both legs the same way in world space: the
# left wheel tucks up while the right one extends, and the body is pushed into a
# skew it cannot lie flat out of. A visually symmetric pose needs the right hip
# negated. The thigh and knee axes are both +Y, where a shared angle *is* mirror
# symmetric because that rotation does not involve the lateral offset, so those
# keep their sign. ``standing.leg_symmetry_error`` already tests the hip as a
# *sum* for exactly this reason; expanding a stance as ``pose * 2`` silently
# violated it and made the commanded folded pose itself asymmetric.
#
# The single-leg geometry above cannot see the mistake: mirroring flips y and
# leaves z exactly unchanged, so a clearance check passes either way. Only the
# lateral placement -- and therefore whether the body rests flat on the ground or
# is wedged on one edge -- is affected.
# ---------------------------------------------------------------------------
HIP_MIRROR = (1.0, -1.0)


def leg_joint_positions(pose) -> tuple[float, float, float, float, float, float]:
    """Expand a (hip, thigh, knee) leg pose into six left/right joint targets.

    Order matches ``standing._LEG_JOINTS``: left hip, thigh, knee, then right.
    """
    hip, thigh, knee = pose
    return (hip * HIP_MIRROR[0], thigh, knee, hip * HIP_MIRROR[1], thigh, knee)


# ---------------------------------------------------------------------------
# What one unit of leg action is worth, in radians.
#
# The command is ``target = offset + action * scale``, so the scale is how far
# the policy can move a joint. It used to be 0.125 rad for the hip and 0.25 for
# the thigh and knee, which with a policy standard deviation around 0.4 gave the
# knee a one-sigma excursion under 0.1 rad. Two things followed from that: the
# hips and knees visibly hardly moved, so the machine could not adapt to uneven
# ground or step sideways, and it could not hold its pitch against the reaction
# torque of accelerating, so it leaned back whenever it sped up.
#
# A quarter of the joint's whole travel per unit action lets a two-sigma action
# reach half the range, which is the motion the terrain and the lateral steps
# need while still leaving the policy a reason to keep its actions small.
# ---------------------------------------------------------------------------
ACTION_RANGE_FRACTION = 0.25

LEG_ACTION_SCALE = {
    "hip": ACTION_RANGE_FRACTION * (HIP_LIMIT[1] - HIP_LIMIT[0]),
    "thigh": ACTION_RANGE_FRACTION * (THIGH_LIMIT[1] - THIGH_LIMIT[0]),
    "knee": ACTION_RANGE_FRACTION * (KNEE_LIMIT[1] - KNEE_LIMIT[0]),
}
