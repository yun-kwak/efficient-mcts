#!/usr/bin/env bash
set -euo pipefail

export WANDB_MODE="${WANDB_MODE:-offline}"
export JAX_PLATFORM_NAME="${JAX_PLATFORM_NAME:-cpu}"

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
  --exp_name smoke \
  --run_name smoke
