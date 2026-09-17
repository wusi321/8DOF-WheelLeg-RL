#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
sudo apt-get update
sudo apt-get install -y build-essential git libgl1-mesa-glx libegl1
uv sync
uv run list-envs
