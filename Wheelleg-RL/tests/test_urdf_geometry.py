"""Source geometry regressions; runnable without training dependencies."""
import hashlib
import importlib.util
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / "robot_description"
spec = importlib.util.spec_from_file_location("converter", ROOT / "scripts/urdf_to_mjcf.py")
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


def vertices(path):
    data = path.read_bytes()
    count = struct.unpack_from("<I", data, 80)[0]
    return set(struct.unpack_from("<3f", data, 84 + 50*i + 12 + 12*j)
               for i in range(count) for j in range(3))


class GeometryTest(unittest.TestCase):
    def test_reference_variants_and_meshes_match(self):
        self.assertEqual((REFERENCE / "urdf/8DOFROBOT2.urdf").read_bytes(),
                         (REFERENCE / "8DOFROBOT2/urdf/8DOFROBOT2.urdf").read_bytes())
        for side in ("left", "right"):
            name = side + "_wheel_link"
            source = REFERENCE / "meshes" / (name + ".STL")
            for copy in (REFERENCE / "8DOFROBOT2/meshes" / source.name,
                         ROOT / "mjcf/meshes" / source.name):
                self.assertEqual(source.read_bytes(), copy.read_bytes())
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
                             converter.WHEEL_MESH_SHA256[name])

    def test_wheel_rims_rotate_concentrically(self):
        model = ET.parse(ROOT / "mjcf/8dof_wheelleg.xml").getroot()
        for side in ("left", "right"):
            name = side + "_wheel_link"
            points = vertices(REFERENCE / "meshes" / (name + ".STL"))
            # Extreme-x rim samples independently locate the CAD circle center.
            extreme = max(x for x, y, z in points)
            center_z = {z for x, y, z in points if x == extreme}
            self.assertEqual(len(center_z), 1)
            center_z = center_z.pop()
            self.assertAlmostEqual(center_z, converter.WHEEL_RIM_CENTER_Z, places=12)
            radii = [math.hypot(x, z-center_z) for x, y, z in points]
            self.assertAlmostEqual(max(radii), 0.03, places=8)
            for kind in ("visual", "collision"):
                geom = model.find(f".//geom[@name='{name}_{kind}']")
                offset = list(map(float, geom.get("pos").split()))
                self.assertAlmostEqual(offset[2] + center_z, 0, places=12)
                for angle in (0, math.pi/2, math.pi, 3*math.pi/2):
                    cx, cz = offset[0], offset[2] + center_z
                    self.assertAlmostEqual(cx*math.cos(angle)+cz*math.sin(angle), 0, places=12)
                    self.assertAlmostEqual(-cx*math.sin(angle)+cz*math.cos(angle), 0, places=12)
                self.assertEqual(geom.get("group"), "1" if kind == "visual" else "2")

    def test_explicit_pose_is_authoritative(self):
        source = ET.fromstring('<visual><origin xyz="0.1 0.2 0.3" rpy="0.4 0.5 0.6"/><geometry><mesh filename="x"/></geometry></visual>')
        self.assertEqual(converter.mesh_pose("right_wheel_link", source, Path("unused")),
                         {"pos": "0.1 0.2 0.3", "euler": "0.4 0.5 0.6"})

    def test_regeneration_and_separate_collision_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "model.xml"
            command = [sys.executable, str(ROOT / "scripts/urdf_to_mjcf.py"),
                       str(REFERENCE / "urdf/8DOFROBOT2.urdf"), str(out),
                       "--mesh-dir", str(REFERENCE / "meshes")]
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
            self.assertEqual(out.read_bytes(), (ROOT / "mjcf/8dof_wheelleg.xml").read_bytes())
            urdf = ET.parse(REFERENCE / "urdf/8DOFROBOT2.urdf")
            collision = urdf.find(".//link[@name='right_wheel_link']/collision")
            collision.find("origin").set("xyz", "0.01 0.02 0.03")
            collision.find("origin").set("rpy", "0.1 0.2 0.3")
            collision.find("geometry/mesh").set("scale", "2 3 4")
            source = Path(tmp) / "source.urdf"
            urdf.write(source)
            command[2] = str(source)
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
            model = ET.parse(out)
            self.assertEqual(model.find("compiler").get("eulerseq"), "XYZ")
            geom = model.find(".//geom[@name='right_wheel_link_collision']")
            self.assertEqual(geom.get("pos"), "0.01 0.02 0.03")
            self.assertEqual(geom.get("euler"), "0.1 0.2 0.3")
            mesh = model.find(f"asset/mesh[@name='{geom.get('mesh')}']")
            self.assertEqual(mesh.get("scale"), "2 3 4")


if __name__ == "__main__":
    unittest.main()
