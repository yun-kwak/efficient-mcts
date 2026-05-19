#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/yun-kwak/efficient-mcts.git}"
GIT_REF="${GIT_REF:-main}"
WORKDIR="${WORKDIR:-/tmp/efficient-mcts}"
CODE_ARCHIVE="${CODE_ARCHIVE:-}"
TASK="${TASK:?TASK is required}"
METHOD="${METHOD:?METHOD is required}"
SEED="${SEED:-1}"
SAVE_DIR="${SAVE_DIR:-/data/efficient-mcts}"
METRICS_DIR="${METRICS_DIR:-/data/efficient-mcts/metrics}"
WANDB_MODE="${WANDB_MODE:-offline}"

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

extra_args=()
if [[ -n "${PAPER_TOTAL_FRAMES:-}" ]]; then
  extra_args+=(--total_frames "${PAPER_TOTAL_FRAMES}")
fi
if [[ -n "${PAPER_LOG_INTERVAL:-}" ]]; then
  extra_args+=(--log_interval "${PAPER_LOG_INTERVAL}")
fi
if [[ -n "${PAPER_EVALUATE_EPISODES:-}" ]]; then
  extra_args+=(--evaluate_episodes "${PAPER_EVALUATE_EPISODES}")
fi
if [[ -n "${PAPER_NUM_ENVS:-}" ]]; then
  extra_args+=(--num_envs "${PAPER_NUM_ENVS}")
fi

WANDB_MODE="${WANDB_MODE}" python scripts/run_paper_preset.py \
  --task "${TASK}" \
  --method "${METHOD}" \
  --seed "${SEED}" \
  --save_dir "${SAVE_DIR}" \
  --metrics_dir "${METRICS_DIR}" \
  --wandb_mode "${WANDB_MODE}" \
  "${extra_args[@]}"
