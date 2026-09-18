from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg, BuiltinVelocityActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from .stance import NOMINAL_STANCE, STANDING_CLEARANCE, WHEEL_RADIUS, WHEEL_TRACK

ROOT = Path(__file__).resolve().parents[2]
ROBOT_XML = ROOT / "mjcf" / "8dof_wheelleg.xml"
LEG_EXPR = ("(left|right)_hip_joint", "(left|right)_thigh_joint", "(left|right)_knee_joint")
WHEEL_EXPR = ("(left|right)_wheel_joint",)

_HIP, _THIGH, _KNEE = NOMINAL_STANCE
INIT_JOINT_POS = {
    "(left|right)_hip_joint": _HIP,
    "(left|right)_thigh_joint": _THIGH,
    "(left|right)_knee_joint": _KNEE,
    "(left|right)_wheel_joint": 0.0,
}


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    return spec


LEG_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=LEG_EXPR, stiffness=35.0, damping=1.0, effort_limit=4.0
)
WHEEL_ACTUATOR_CFG = BuiltinVelocityActuatorCfg(
    target_names_expr=WHEEL_EXPR, damping=0.5, effort_limit=2.0
)
ARTICULATION_CFG = EntityArticulationInfoCfg(
    actuators=(LEG_ACTUATOR_CFG, WHEEL_ACTUATOR_CFG), soft_joint_pos_limit_factor=0.95
)
# Only the *_collision geoms collide; the visual meshes stay non-colliding.
COLLISION_CFG = CollisionCfg(
    geom_names_expr=(".*_collision",),
    contype=1,
    conaffinity=1,
    condim={".*wheel.*": 6, ".*": 1},
    friction={".*wheel.*": (0.8, 0.05, 0.01)},
)
# Spawn with the wheels resting exactly on the ground in the nominal stance.
INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, STANDING_CLEARANCE), joint_pos=INIT_JOINT_POS, joint_vel={".*": 0.0}
)


def get_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(COLLISION_CFG,),
        spec_fn=get_spec,
        articulation=ARTICULATION_CFG,
    )


__all__ = [
    "ARTICULATION_CFG",
    "COLLISION_CFG",
    "INIT_JOINT_POS",
    "INIT_STATE",
    "ROBOT_XML",
    "WHEEL_RADIUS",
    "WHEEL_TRACK",
    "get_robot_cfg",
    "get_spec",
]
