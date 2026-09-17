#!/usr/bin/env bash
set -euo pipefail
sudo apt-get update
sudo apt-get install -y build-essential git libgl1-mesa-glx libegl1
uv sync
uv run list-envs
