from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).parents[1]

def test_mjcf_has_eight_dof_contract():
    root = ET.parse(ROOT / "mjcf" / "8dof_wheelleg.xml").getroot()
    joints = root.findall(".//joint")
    names = [j.get("name") for j in joints]
    assert len(joints) == 8
    assert set(names) == {"left_hip_joint", "left_thigh_joint", "left_knee_joint", "left_wheel_joint", "right_hip_joint", "right_thigh_joint", "right_knee_joint", "right_wheel_joint"}
    assert len(root.findall(".//freejoint")) == 1
    assert root.find("./worldbody/body/freejoint").get("name") == "floating_base"
    assert len(root.findall("./actuator/*")) == 8

def test_floating_base_compiles_and_falls():
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(ROOT / "mjcf/8dof_wheelleg.xml"))
    assert (model.nq, model.nv, model.nu) == (15, 14, 8)
    assert sum(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE) == 1
    data = mujoco.MjData(model)
    data.qpos[2] = 1.0
    initial_height = data.qpos[2]
    for _ in range(10):
        mujoco.mj_step(model, data)
    assert data.qpos[2] < initial_height


def test_training_contract():
    text = (ROOT / "configs" / "training.yaml").read_text(encoding="utf-8")
    assert "max_obstacle_height_m: 0.12" in text
    assert "policy_hz: 50" in text
