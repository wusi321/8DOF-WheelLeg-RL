#!/usr/bin/env bash
set -euo pipefail
python scripts/urdf_to_mjcf.py ../robot_description/urdf/8DOFROBOT2.urdf mjcf/8dof_wheelleg.xml --mesh-dir ../robot_description/meshes
