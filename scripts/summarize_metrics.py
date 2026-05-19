#!/usr/bin/env python
"""Summarize JSONL metrics written by run_experiment.py."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize(path: Path) -> dict:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"No metrics found in {path}")
    final = rows[-1]
    returns = [row["episode_return"] for row in rows if "episode_return" in row]
    return {
        "path": str(path),
        "num_points": len(rows),
        "final_num_updates": final.get("num_updates"),
        "final_num_frames": final.get("num_frames"),
        "final_episode_return": final.get("episode_return"),
        "best_episode_return": max(returns) if returns else None,
        "mean_episode_return": sum(returns) / len(returns) if returns else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--output")
    args = parser.parse_args()

    files = []
    for raw_path in args.paths:
        path = Path(raw_path)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.jsonl")))
        else:
            files.append(path)

    summaries = [summarize(path) for path in files]
    fieldnames = [
        "path",
        "num_points",
        "final_num_updates",
        "final_num_frames",
        "final_episode_return",
        "best_episode_return",
        "mean_episode_return",
    ]

    if args.output:
        out_f = open(args.output, "w", newline="", encoding="utf-8")
    else:
        out_f = None

    try:
        writer = csv.DictWriter(out_f or __import__("sys").stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    finally:
        if out_f is not None:
            out_f.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
