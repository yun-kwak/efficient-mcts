#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/yun-kwak/efficient-mcts.git}"
GIT_REF="${GIT_REF:-main}"
WORKDIR="${WORKDIR:-/tmp/efficient-mcts}"
CODE_ARCHIVE="${CODE_ARCHIVE:-}"

python --version
python - <<'PY'
import jax

print("jax", jax.__version__)
print("devices", jax.devices())
PY

rm -rf "${WORKDIR}"
if [[ -n "${CODE_ARCHIVE}" ]]; then
  mkdir -p "${WORKDIR}"
  if [[ -d "${CODE_ARCHIVE}" ]]; then
    cp -a "${CODE_ARCHIVE}/." "${WORKDIR}/"
  else
    tar -xzf "${CODE_ARCHIVE}" -C "${WORKDIR}"
  fi
else
  git clone --depth 1 "${REPO_URL}" "${WORKDIR}"
  cd "${WORKDIR}"
  git fetch --depth 1 origin "${GIT_REF}" || true
  git checkout "${GIT_REF}"
fi
cd "${WORKDIR}"

pip install -e pine
pytest -q pine/tests

export WANDB_MODE="${WANDB_MODE:-offline}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

python scripts/run_experiment.py \
  --debug True \
  --env_id MiniGrid-Color-DoorKey-Random-Goal-6x6-C2-v0 \
  --num_envs 1 \
  --evaluate_episodes 1 \
  --replay_min_size 2 \
  --replay_max_size 64 \
  --batch_size 1 \
  --total_frames 4 \
  --log_interval 1 \
  --unroll_steps 1 \
  --td_steps 1 \
  --num_simulations 1 \
  --max_search_depth 5 \
  --channels 8 \
  --num_bins 21 \
  --use_resnet_v2 True \
  --action_cardinalities "[3,2,3,3]" \
  --save_dir /tmp/efficient-mcts-smoke \
  --save_interval 1000000 \
  --exp_name forge-smoke \
  --run_name forge-smoke
