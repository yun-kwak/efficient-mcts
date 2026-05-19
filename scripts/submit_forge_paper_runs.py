#!/usr/bin/env python
"""Submit paper reproduction jobs to Forge."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRESETS_PATH = ROOT / "configs" / "paper_presets.json"


def load_presets() -> dict:
    with PRESETS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    presets = load_presets()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forge_bin", default="forge")
    parser.add_argument("--image", default="yunkwak/efficient-mcts:1.0")
    parser.add_argument("--repo_url", default="https://github.com/yun-kwak/efficient-mcts.git")
    parser.add_argument("--git_ref", default="main")
    parser.add_argument("--code_archive")
    parser.add_argument("--disk_mount", default="efficient-mcts-runs:/data")
    parser.add_argument("--max_duration", type=int, default=72)
    parser.add_argument("--wandb_mode", default="offline")
    parser.add_argument("--tasks", nargs="+", choices=sorted(presets["tasks"]), required=True)
    parser.add_argument("--methods", nargs="+", choices=sorted(presets["methods"]), default=["ours", "muzero"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--job_prefix", default="efficient-mcts")
    parser.add_argument("--total_frames", type=int)
    parser.add_argument("--log_interval", type=int)
    parser.add_argument("--evaluate_episodes", type=int)
    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    for task in args.tasks:
        for method in args.methods:
            for seed in args.seeds:
                job_name = f"{args.job_prefix}-{task}-{method}-s{seed}"
                command = [
                    args.forge_bin,
                    "job",
                    "submit",
                    "--name",
                    job_name,
                    "--image",
                    args.image,
                    "--gpu",
                    "1",
                    "--num-nodes",
                    "1",
                    "--max-duration",
                    str(args.max_duration),
                    "--env",
                    f"REPO_URL={args.repo_url}",
                    "--env",
                    f"GIT_REF={args.git_ref}",
                    "--env",
                    f"TASK={task}",
                    "--env",
                    f"METHOD={method}",
                    "--env",
                    f"SEED={seed}",
                    "--env",
                    f"WANDB_MODE={args.wandb_mode}",
                    "--env",
                    "XLA_PYTHON_CLIENT_PREALLOCATE=false",
                    "--entrypoint-file",
                    str(ROOT / "scripts" / "forge_paper_entrypoint.sh"),
                ]
                if args.code_archive:
                    command.extend(["--env", f"CODE_ARCHIVE={args.code_archive}"])
                if args.disk_mount:
                    command.extend(["--disk-mount", args.disk_mount])
                optional_envs = {
                    "PAPER_TOTAL_FRAMES": args.total_frames,
                    "PAPER_LOG_INTERVAL": args.log_interval,
                    "PAPER_EVALUATE_EPISODES": args.evaluate_episodes,
                    "PAPER_NUM_ENVS": args.num_envs,
                }
                for key, value in optional_envs.items():
                    if value is not None:
                        command.extend(["--env", f"{key}={value}"])

                print(" ".join(command))
                if not args.dry_run:
                    subprocess.check_call(command, cwd=ROOT)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
