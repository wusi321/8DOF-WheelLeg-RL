"""Actuator gains and torque limits — the single source of truth.

Provenance
----------
The URDF carries no actuator data: every joint is exported with
``<limit ... effort="0" velocity="0" />``, so nothing here is a measured motor
spec. These values are engineering choices inherited from the 16DOF reference
machine (which used 17 N*m and is a much larger robot), scaled down by hand.

The generated MJCF embeds the same numbers so that the file can still be opened
in a standalone MuJoCo viewer. That embedded block is *not* what training uses:
``robot_cfg.get_spec`` deletes every actuator in the file and the articulation
config below rebuilds them from this module. Change the values here, regenerate
the MJCF, and both copies stay in step.
"""
from __future__ import annotations

# Leg joints: MuJoCo <position> actuators, torque = kp * (target - q) - kd * qd,
# clamped to +/- LEG_TORQUE_LIMIT.
LEG_KP = 35.0
LEG_KD = 1.0
LEG_TORQUE_LIMIT = 4.0  # N*m

# Wheel joints: MuJoCo <velocity> actuators, torque = kd * (target - qd),
# clamped to +/- WHEEL_TORQUE_LIMIT.
WHEEL_KD = 0.5
WHEEL_TORQUE_LIMIT = 2.0  # N*m

# Standalone-viewer control range only. The runtime velocity actuator sets
# ctrllimited=False, so this does not bound the policy's wheel speed target.
WHEEL_MAX_SPEED = 13.0  # rad/s
LEG_CTRL_RANGE = 3.14  # rad, standalone-viewer control range only
