"""Report the wheel-leg stance geometry used by training.

Run: uv run python scripts/stance_kinematics.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Loaded by path so the script runs without the mjlab runtime installed.
def _load(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[1] / "src/wheelleg" / filename
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


S = _load("wheelleg_stance", "stance.py")
ACT = _load("wheelleg_actuator_spec", "actuator_spec.py")


def main() -> None:
    ref = S.leg_pose(S.REFERENCE_STANCE[1], S.REFERENCE_STANCE[2])
    print("supplied reference stand (hip 0, thigh 1.02, knee -1.57)")
    print(f"  axle z          = {ref.axle[2]:+.6f} m")
    print(f"  axle x          = {ref.axle_x:+.6f} m")
    print(f"  base clearance  = {ref.base_clearance:.6f} m")
    print(f"  -> {'above' if ref.base_clearance >= S.MIN_CLEARANCE else 'BELOW'} "
          f"required {S.MIN_CLEARANCE:.3f} m")
    print()

    hip, thigh, knee = S.NOMINAL_STANCE
    nom = S.leg_pose(thigh, knee, hip)
    print("nominal trained stance (solved: wheels under the base origin)")
    print(f"  hip={hip:.4f} thigh={thigh:.4f} knee={knee:.4f} rad")
    print(f"  base clearance  = {nom.base_clearance:.6f} m "
          f"(margin {nom.base_clearance - S.MIN_CLEARANCE:+.4f} m)")
    print(f"  axle x          = {nom.axle_x:+.4f} m")
    print(f"  shank tilt      = {nom.shank_angle:+.4f} rad "
          f"(reference {S.REFERENCE_STANCE[1] + S.REFERENCE_STANCE[2]:+.4f})")
    print(f"  centre of mass x= {S.com_x(thigh, knee, hip):+.5f} m")
    print()

    print(f"total moving mass = {S.TOTAL_MASS:.2f} kg (weight {S.TOTAL_MASS * S.GRAVITY:.1f} N, "
          f"{S.TOTAL_MASS * S.GRAVITY / 2:.1f} N per leg)")
    x = S.joint_x(thigh, knee)
    torques = S.static_joint_torques(thigh, knee)
    print("static holding torque at the nominal stance:")
    for name in ("hip", "thigh", "knee"):
        print(f"  {name:6s} arm={abs(x['contact'] - x[name]):.4f} m  "
              f"torque={torques[name]:.3f} N*m")
    print(f"  configured leg torque limit = {ACT.LEG_TORQUE_LIMIT:g} N*m "
          f"({ACT.LEG_TORQUE_LIMIT / max(torques.values()):.1f}x the static peak)")
    print(f"  configured wheel torque limit = {ACT.WHEEL_TORQUE_LIMIT:g} N*m")
    print()

    print(f"  collapse termination below {S.COLLAPSE_CLEARANCE:.3f} m")
    print(f"  low-height barrier starts below {S.MIN_CLEARANCE:.3f} m")
    print()

    reach = [
        (S.clearance(t / 1000, k / 1000), t / 1000, k / 1000)
        for t in range(int(S.THIGH_LIMIT[0] * 1000), int(S.THIGH_LIMIT[1] * 1000) + 1, 5)
        for k in range(int(S.KNEE_LIMIT[0] * 1000), 1, 5)
    ]
    lo = min(reach)
    hi = max(reach)
    print(f"reachable clearance range: {lo[0]:+.4f} m (thigh={lo[1]:+.2f} knee={lo[2]:+.2f}) "
          f".. {hi[0]:+.4f} m (thigh={hi[1]:+.2f} knee={hi[2]:+.2f})")

    print()
    print("solved stances at other clearances (wheels centring under the base):")
    for target in (0.13, 0.135, 0.145, 0.155, 0.165):
        _, t, k = S.solve_stance(target)
        pose = S.leg_pose(t, k)
        print(f"  clearance {target:.3f} m -> thigh={t:.4f} knee={k:.4f} "
              f"axle_x={pose.axle_x:+.4f} (actual {pose.base_clearance:.4f})")


if __name__ == "__main__":
    main()
