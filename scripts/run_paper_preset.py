#!/usr/bin/env python
"""Run one paper reproduction preset."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRESETS_PATH = ROOT / "configs" / "paper_presets.json"


def load_presets() -> dict:
    with PRESETS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def cli_value(value):
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def build_config(presets: dict, task: str, method: str, seed: int) -> dict:
    config = {}
    config.update(presets["defaults"])
    config.update(presets["tasks"][task])
    config.update(presets["methods"][method])
    config["seed"] = seed
    config["env_kwargs"] = "{}"
    return config


def main() -> int:
    presets = load_presets()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(presets["tasks"]), required=True)
    parser.add_argument("--method", choices=sorted(presets["methods"]), required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save_dir", default="/data/efficient-mcts")
    parser.add_argument("--exp_name")
    parser.add_argument("--run_name")
    parser.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", "disabled"))
    parser.add_argument("--total_frames", type=int)
    parser.add_argument("--log_interval", type=int)
    parser.add_argument("--evaluate_episodes", type=int)
    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--save_interval", type=int)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    config = build_config(presets, args.task, args.method, args.seed)
    for key in [
        "total_frames",
        "log_interval",
        "evaluate_episodes",
        "num_envs",
        "save_interval",
    ]:
        value = getattr(args, key)
        if value is not None:
            config[key] = value

    exp_name = args.exp_name or f"{args.task}-{args.method}"
    run_name = args.run_name or f"{exp_name}-seed{args.seed}"

    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_experiment.py"),
        "--save_dir",
        args.save_dir,
        "--exp_name",
        exp_name,
        "--run_name",
        run_name,
    ]
    for key, value in config.items():
        command.extend([f"--{key}", cli_value(value)])

    env = os.environ.copy()
    env["WANDB_MODE"] = args.wandb_mode

    print(" ".join(shlex.quote(part) for part in command))
    if args.dry_run:
        return 0
    return subprocess.call(command, cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
