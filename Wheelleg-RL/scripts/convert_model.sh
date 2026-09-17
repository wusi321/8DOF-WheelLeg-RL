#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
python scripts/urdf_to_mjcf.py ../robot_description/urdf/8DOFROBOT2.urdf mjcf/8dof_wheelleg.xml --mesh-dir ../robot_description/meshes
