#!/usr/bin/env bash
set -euo pipefail
STAGE=${1:-flat}; NUM_ENVS=${NUM_ENVS:-2048}; ITERS=${ITERS:-5}
declare -A TASKS=([flat]=Wheelleg-Flat-v0 [rough]=Wheelleg-Rough-v0 [recovery]=Wheelleg-Recovery-v0)
uv run train "${TASKS[$STAGE]}" --env.scene.num-envs "$NUM_ENVS" --agent.max_iterations "$ITERS"
