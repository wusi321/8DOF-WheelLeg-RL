"""Convert the canonical 8DOF URDF into a MJCF model for mjlab.

The converter intentionally uses URDF inertials and joint transforms, while adding
explicit wheel/leg actuators and stable collision defaults for RL. It also trims
whitespace from legacy SolidWorks joint names.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import xml.etree.ElementTree as ET
from pathlib import Path


def f(v: str) -> str:
    return " ".join(v.split())


def origin(el: ET.Element | None) -> tuple[str, str]:
    if el is None:
        return "0 0 0", "0 0 0"
    return f(el.get("xyz", "0 0 0")), f(el.get("rpy", "0 0 0"))


# Both original CAD wheel rims are centered at z=2.546479 mm, not on the
# URDF's y-axis axle. Only correct these exact, untransformed source meshes;
# replacement geometry or an explicit URDF mesh pose must remain authoritative.
WHEEL_MESH_SHA256 = {
    "left_wheel_link": "ae720743da9d1aebe70258cc0c9f1d02f61b6fb85ef833e1f19c8567b5d6f746",
    "right_wheel_link": "31586e53750a3dc269b9f0fa34e6b8f33a2436197926e25118392be0cbbea9c9",
}
WHEEL_RIM_CENTER_Z = 0.002546478994190693


def mesh_pose(name: str, source: ET.Element, mesh_path: Path) -> dict[str, str]:
    xyz, rpy = origin(source.find("origin"))
    mesh = source.find("geometry/mesh")
    scale = [float(v) for v in mesh.get("scale", "1 1 1").split()]
    if (
        name in WHEEL_MESH_SHA256
        and all(float(v) == 0 for v in (xyz + " " + rpy).split())
        and scale == [1.0, 1.0, 1.0]
        and hashlib.sha256(mesh_path.read_bytes()).hexdigest() == WHEEL_MESH_SHA256[name]
    ):
        xyz = f"0 0 {-WHEEL_RIM_CENTER_Z:.15g}"
    return {"pos": xyz, "euler": rpy}


def _positive_inertia(inertia: ET.Element) -> tuple[float, ...]:
    """Return a numerically positive-definite inertia matrix.

    Some SolidWorks URDF exports contain zero principal inertia or a product
    of inertia exactly equal to the positive-definiteness boundary. MuJoCo
    rejects those matrices during model compilation. A symmetric matrix that
    is strictly diagonally dominant with positive diagonal entries is positive
    definite, so we minimally increase diagonal terms when necessary.
    """
    values = [float(inertia.get(k, "0")) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
    ixx, iyy, izz, ixy, ixz, iyz = values
    scale = max(abs(ixx), abs(iyy), abs(izz), abs(ixy), abs(ixz), abs(iyz), 1.0e-6)
    margin = max(1.0e-8, scale * 1.0e-4)
    ixx = max(ixx, abs(ixy) + abs(ixz) + margin)
    iyy = max(iyy, abs(ixy) + abs(iyz) + margin)
    izz = max(izz, abs(ixz) + abs(iyz) + margin)
    return ixx, iyy, izz, ixy, ixz, iyz


def inertial(link: ET.Element) -> ET.Element | None:
    src = link.find("inertial")
    if src is None:
        return None
    out = ET.Element("inertial")
    xyz, rpy = origin(src.find("origin"))
    out.set("pos", xyz)
    if rpy != "0 0 0":
        out.set("euler", rpy)
    mass = src.find("mass")
    inertia = src.find("inertia")
    out.set("mass", mass.get("value", "0") if mass is not None else "0")
    if inertia is not None:
        values = _positive_inertia(inertia)
        out.set("fullinertia", " ".join(f"{value:.12g}" for value in values))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("urdf", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--mesh-dir", type=Path, required=True)
    args = ap.parse_args()
    root = ET.parse(args.urdf).getroot()
    links = {x.get("name"): x for x in root.findall("link")}
    joints = []
    children = set()
    for j in root.findall("joint"):
        name = (j.get("name") or "").strip()
        parent = j.find("parent").get("link")
        child = j.find("child").get("link")
        joints.append((name, j, parent, child))
        children.add(child)
    root_link = next(name for name in links if name not in children)
    mj = ET.Element("mujoco", {"model": "8dof_wheelleg"})
    ET.SubElement(mj, "compiler", {"angle": "radian", "eulerseq": "XYZ", "meshdir": "meshes"})
    ET.SubElement(mj, "option", {"timestep": "0.005", "integrator": "implicitfast", "gravity": "0 0 -9.81"})
    asset = ET.SubElement(mj, "asset")
    mesh_assets = {}
    for name, link in links.items():
        for kind in ("visual", "collision"):
            source = link.find(kind)
            mesh = link.find(f"{kind}/geometry/mesh")
            if mesh is None:
                continue
            file = Path(mesh.get("filename", "").split("/")[-1])
            mesh_path = args.mesh_dir / file
            if not mesh_path.is_file():
                raise FileNotFoundError(f"Missing {name} {kind} mesh: {mesh_path}")
            scale = f(mesh.get("scale", "1 1 1"))
            key = (file.name, scale)
            if key not in mesh_assets:
                asset_name = name if kind == "visual" else name + "_collision"
                ET.SubElement(asset, "mesh", {"name": asset_name, "file": file.name, "scale": scale})
                mesh_assets[key] = asset_name
    world = ET.SubElement(mj, "worldbody")
    def add_body(parent: ET.Element, name: str, pos: str = "0 0 0", rpy: str = "0 0 0") -> None:
        link = links[name]
        attrs = {"name": name, "pos": pos}
        if rpy != "0 0 0": attrs["euler"] = rpy
        body = ET.SubElement(parent, "body", attrs)
        if name == root_link:
            ET.SubElement(body, "freejoint", {"name": "floating_base"})
        inert = inertial(link)
        if inert is not None: body.append(inert)
        for kind in ("visual", "collision"):
            source = link.find(kind)
            mesh = link.find(f"{kind}/geometry/mesh")
            if mesh is None:
                continue
            file = Path(mesh.get("filename", "").split("/")[-1])
            key = (file.name, f(mesh.get("scale", "1 1 1")))
            attrs = {"name": name + "_" + kind, "type": "mesh", "mesh": mesh_assets[key]}
            attrs.update(mesh_pose(name, source, args.mesh_dir / file))
            if kind == "visual":
                attrs.update(contype="0", conaffinity="0", group="1")
            else:
                attrs.update(friction="0.8 0.05 0.01", group="2")
            ET.SubElement(body, "geom", attrs)
        for jname, j, p, child in joints:
            if p != name: continue
            xyz, rpy = origin(j.find("origin"))
            child_body_parent = body
            add_body(child_body_parent, child, xyz, rpy)
            cb = child_body_parent.findall("body")[-1]
            axis = f(j.find("axis").get("xyz", "0 0 1"))
            typ = j.get("type")
            ja = {"name": jname, "axis": axis, "damping": "0.02", "armature": "0.001"}
            lim = j.find("limit")
            if typ != "continuous" and lim is not None:
                ja["range"] = f'{lim.get("lower", "-3.14")} {lim.get("upper", "3.14")}'
            ET.SubElement(cb, "joint", ja)
    add_body(world, root_link)
    actuators = ET.SubElement(mj, "actuator")
    for jname, _, _, _ in joints:
        if "wheel" in jname:
            ET.SubElement(actuators, "velocity", {"name": jname, "joint": jname, "kv": "0.5", "ctrlrange": "-13 13", "forcelimited": "true", "forcerange": "-2 2"})
        else:
            ET.SubElement(actuators, "position", {"name": jname, "joint": jname, "kp": "35", "kv": "1.0", "ctrlrange": "-3.14 3.14", "forcelimited": "true", "forcerange": "-4 4"})
    ET.indent(mj, space="  ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(ET.tostring(mj, encoding="unicode"), encoding="utf-8")
    print(f"wrote {args.output} with {len(joints)} joints")


if __name__ == "__main__":
    main()
