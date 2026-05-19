#!/usr/bin/env bash
set -euo pipefail

FORGE_BIN="${FORGE_BIN:-forge}"
FORGE_IMAGE="${FORGE_IMAGE:-yunkwak/efficient-mcts:1.0}"
REPO_URL="${REPO_URL:-https://github.com/yun-kwak/efficient-mcts.git}"
GIT_REF="${GIT_REF:-main}"
JOB_NAME="${JOB_NAME:-efficient-mcts-smoke}"
CODE_ARCHIVE="${CODE_ARCHIVE:-}"
DISK_MOUNT="${DISK_MOUNT:-}"

command=(
  "${FORGE_BIN}" job submit
  --name "${JOB_NAME}" \
  --image "${FORGE_IMAGE}" \
  --gpu 1 \
  --num-nodes 1 \
  --max-duration 1 \
  --env "REPO_URL=${REPO_URL}" \
  --env "GIT_REF=${GIT_REF}" \
  --env "WANDB_MODE=offline" \
  --env "XLA_PYTHON_CLIENT_PREALLOCATE=false" \
  --entrypoint-file scripts/forge_smoke_entrypoint.sh
)

if [[ -n "${CODE_ARCHIVE}" ]]; then
  command+=(--env "CODE_ARCHIVE=${CODE_ARCHIVE}")
fi
if [[ -n "${DISK_MOUNT}" ]]; then
  command+=(--disk-mount "${DISK_MOUNT}")
fi

"${command[@]}"
