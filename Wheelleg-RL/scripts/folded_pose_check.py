"""Check the "legs fully folded, body flat on the ground" pose.

Verifies, without MuJoCo:

1. every joint angle against the hard limits and the soft limit factor the
   articulation config applies;
2. where the wheel axles end up, in the base frame, at that pose;
3. whether the base mesh is actually the lowest part (i.e. whether the robot
   would rest on its body with the wheels clear of the ground).

Run: uv run python scripts/folded_pose_check.py
"""
from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src/wheelleg" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


S = _load("wheelleg_stance", "stance.py")

# The pose supplied in 完全趴下落地.txt. The hip and thigh values are described
# as "limit position", so they are clamped to the model limits for the checks.
REPORTED = {"hip": 0.91, "thigh": 1.31, "knee": -2.62, "wheel": 0.0}
# Applied by EntityArticulationInfoCfg(soft_joint_pos_limit_factor=...).
SOFT_LIMIT_FACTOR = 0.95


def stl_z_range(path: Path) -> tuple[float, float]:
    """min/max z of an ASCII or binary STL."""
    raw = path.read_bytes()
    if raw[:5].lower() == b"solid" and b"facet" in raw[:2048].lower():
        lo = hi = None
        for line in raw.decode("utf-8", "ignore").splitlines():
            parts = line.split()
            if len(parts) == 4 and parts[0].lower() == "vertex":
                z = float(parts[3])
                lo = z if lo is None else min(lo, z)
                hi = z if hi is None else max(hi, z)
        if lo is None:
            raise ValueError(f"{path} looks like ASCII STL but has no vertices")
        return lo, hi
    count = struct.unpack_from("<I", raw, 80)[0]
    lo = hi = None
    offset = 84
    for _ in range(count):
        values = struct.unpack_from("<12f", raw, offset)
        for vertex in range(3):
            z = values[3 + vertex * 3 + 2]
            lo = z if lo is None else min(lo, z)
            hi = z if hi is None else max(hi, z)
        offset += 50
    if lo is None:
        raise ValueError(f"{path} has no triangles")
    return lo, hi


def main() -> None:
    limits = {"hip": S.HIP_LIMIT, "thigh": S.THIGH_LIMIT, "knee": S.KNEE_LIMIT}
    print("joint limits (hard, and soft at factor 0.95)")
    for name, (low, high) in limits.items():
        value = REPORTED[name]
        margin = min(value - low, high - value)
        soft = (SOFT_LIMIT_FACTOR * low, SOFT_LIMIT_FACTOR * high) if low < 0 else (low, high)
        flag = "OK"
        if not (low <= value <= high):
            flag = "OUTSIDE HARD LIMIT"
        elif not (soft[0] <= value <= soft[1]):
            flag = "outside soft limit -> joint_pos_limits penalty"
        print(f"  {name:6s} reported {value:+.4f}  hard [{low:+.5f}, {high:+.5f}]  "
              f"soft [{soft[0]:+.5f}, {soft[1]:+.5f}]  {flag}")

    hip = min(max(REPORTED["hip"], S.HIP_LIMIT[0]), S.HIP_LIMIT[1])
    thigh = min(max(REPORTED["thigh"], S.THIGH_LIMIT[0]), S.THIGH_LIMIT[1])
    knee = min(max(REPORTED["knee"], S.KNEE_LIMIT[0]), S.KNEE_LIMIT[1])

    nominal = S.NOMINAL_STANCE
    print()
    print("wheel axle position in the base frame (left leg, hip clamped to the limit)")
    for label, (h, t, k) in (
        ("folded (reported)", (hip, thigh, knee)),
        ("nominal stance   ", nominal),
        ("reference stand  ", S.REFERENCE_STANCE),
    ):
        pose = S.leg_pose(t, k, h)
        print(f"  {label}: axle x={pose.axle_x:+.4f}  z={pose.axle[2]:+.4f}  "
              f"clearance-if-wheel-supported={pose.base_clearance:+.4f}")

    mesh = ROOT / "mjcf" / "meshes" / "base_link.STL"
    z_lo, z_hi = stl_z_range(mesh)
    print()
    print(f"base_link mesh z range in the base frame: {z_lo:+.4f} .. {z_hi:+.4f} m")

    pose = S.leg_pose(thigh, knee, hip)
    body_rest_height = -z_lo
    axle_world_z = body_rest_height + pose.axle[2]
    wheel_bottom = axle_world_z - S.WHEEL_RADIUS
    print()
    print("if the robot rests on its base mesh:")
    print(f"  base origin height            = {body_rest_height:.4f} m")
    print(f"  wheel axle height             = {axle_world_z:+.4f} m")
    print(f"  wheel lowest rim point        = {wheel_bottom:+.4f} m "
          f"({'CLEAR of the ground' if wheel_bottom > 0 else 'still touching the ground'})")
    print()
    print(f"for comparison the standing clearance is {S.STANDING_CLEARANCE:.3f} m, "
          f"so the folded body sits {S.STANDING_CLEARANCE - body_rest_height:+.4f} m lower")


if __name__ == "__main__":
    main()
