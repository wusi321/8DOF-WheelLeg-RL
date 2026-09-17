from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).parents[1]

def test_mjcf_has_eight_dof_contract():
    root = ET.parse(ROOT / "mjcf" / "8dof_wheelleg.xml").getroot()
    joints = root.findall(".//joint")
    names = [j.get("name") for j in joints]
    assert len(joints) == 8
    assert names == ["left_hip_joint", "left_thigh_joint", "left_knee_joint", "left_wheel_joint", "right_hip_joint", "right_thigh_joint", "right_knee_joint", "right_wheel_joint"]
    assert len(root.findall("./actuator/*")) == 8

def test_training_contract():
    text = (ROOT / "configs" / "training.yaml").read_text(encoding="utf-8")
    assert "max_obstacle_height_m: 0.12" in text
    assert "policy_hz: 50" in text
