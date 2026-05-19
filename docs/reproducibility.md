# Reproducibility Notes

This page records the currently verified ways to run this repository.

## Verified on 2026-05-19

Local CPU smoke test:

- Machine: macOS arm64
- Python: 3.10.20
- JAX: 0.4.20, CPU
- Command: `python -m pytest -q pine/tests`
- Result: 60 tests passed
- Command: `bash scripts/smoke.sh`
- Result: completed 2 tiny MiniGrid training updates with W&B offline logging

Known local setup issue:

- `pygraphviz` needs Graphviz headers. With Homebrew Graphviz, install with `CFLAGS="-I/opt/homebrew/include" LDFLAGS="-L/opt/homebrew/lib" python -m pip install -r requirements/py310-cpu.txt`.
- `wandb==0.15.12` needs `setuptools==68.2.2`; newer setuptools releases may not provide `pkg_resources`.
- `jaxlib==0.4.16` is not available for current macOS arm64 Python 3.10, so local CPU uses JAX 0.4.20. The published Docker image remains the closest original GPU environment.

## Local CPU setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
CFLAGS="-I/opt/homebrew/include" LDFLAGS="-L/opt/homebrew/lib" \
  python -m pip install -e pine -r requirements/py310-cpu.txt
python -m pytest -q pine/tests
WANDB_MODE=offline bash scripts/smoke.sh
```

If you use `uv`:

```bash
uv venv --python 3.10 .venv
CFLAGS="-I/opt/homebrew/include" LDFLAGS="-L/opt/homebrew/lib" \
  uv pip install --python .venv/bin/python -e pine -r requirements/py310-cpu.txt
.venv/bin/python -m pytest -q pine/tests
PATH="$PWD/.venv/bin:$PATH" WANDB_MODE=offline bash scripts/smoke.sh
```

## Forge smoke test

The Forge smoke test uses the published Docker image and runs both unit tests and a tiny MiniGrid training job:

```bash
FORGE_BIN=/Users/yunhyeok.kwak/.local/bin/forge bash scripts/submit_forge_smoke.sh
```

Useful overrides:

```bash
JOB_NAME=efficient-mcts-smoke-mybranch \
GIT_REF=mybranch \
bash scripts/submit_forge_smoke.sh
```

`GIT_REF` must be available from `REPO_URL`, because the Forge job clones the repository inside the container.

## Paper result reproduction

Main paper results train Ours and MuZero for 100k gradient steps. The main matrix is:

- DoorKey Easy, Normal, Hard, 5 seeds per method.
- Sokoban Easy, Normal, Hard, 3 seeds per method.
- 48 training runs total.
- Evaluation every 2000 updates with 32 evaluation episodes.

Run one paper preset locally or in a container:

```bash
WANDB_MODE=offline python scripts/run_paper_preset.py \
  --task doorkey_easy \
  --method ours \
  --seed 1 \
  --save_dir /data/efficient-mcts \
  --metrics_dir /data/efficient-mcts/metrics
```

Available tasks are `doorkey_easy`, `doorkey_normal`, `doorkey_hard`, `sokoban_easy`, `sokoban_normal`, and `sokoban_hard`. Available methods are `ours` and `muzero`.

To submit a small Forge pilot with one seed for Ours and MuZero:

```bash
python scripts/submit_forge_paper_runs.py \
  --forge_bin /Users/yunhyeok.kwak/.local/bin/forge \
  --tasks doorkey_easy \
  --methods ours muzero \
  --seeds 1 \
  --git_ref main
```

To submit the full main matrix after validating the pilot:

```bash
python scripts/submit_forge_paper_runs.py \
  --forge_bin /Users/yunhyeok.kwak/.local/bin/forge \
  --tasks doorkey_easy doorkey_normal doorkey_hard sokoban_easy sokoban_normal sokoban_hard \
  --methods ours muzero \
  --seeds 1 2 3
```

Then submit DoorKey seeds 4 and 5:

```bash
python scripts/submit_forge_paper_runs.py \
  --forge_bin /Users/yunhyeok.kwak/.local/bin/forge \
  --tasks doorkey_easy doorkey_normal doorkey_hard \
  --methods ours muzero \
  --seeds 4 5
```

The batch submitter defaults to mounting `efficient-mcts-runs:/data`. Create the disk once if it does not exist:

```bash
forge disk create --name efficient-mcts-runs --size 100
```

If the local commit has not been pushed to GitHub, upload an archive to the disk and run from that archive:

```bash
git archive --format=tar.gz HEAD -o /tmp/efficient-mcts-head.tar.gz
forge file upload /tmp/efficient-mcts-head.tar.gz \
  --disk efficient-mcts-runs \
  --path /code/efficient-mcts-head.tar.gz

python scripts/submit_forge_paper_runs.py \
  --forge_bin /Users/yunhyeok.kwak/.local/bin/forge \
  --tasks doorkey_easy \
  --methods ours muzero \
  --seeds 1 \
  --code_archive /data/code/efficient-mcts-head.tar.gz
```

For shorter preflight runs that use the same paper presets but fewer updates:

```bash
python scripts/submit_forge_paper_runs.py \
  --forge_bin /Users/yunhyeok.kwak/.local/bin/forge \
  --tasks doorkey_easy \
  --methods ours muzero \
  --seeds 1 \
  --total_frames 64000 \
  --evaluate_episodes 4 \
  --job_prefix efficient-mcts-preflight
```

Preset details are stored in `configs/paper_presets.json`.

After jobs finish, summarize JSONL metrics:

```bash
python scripts/summarize_metrics.py /data/efficient-mcts/metrics \
  --output /data/efficient-mcts/summary.csv
```

## Maintenance recommendations

Recommended next steps, in priority order:

1. Add paper-level experiment presets, for example `configs/sokoban_c3.yaml` and `configs/minigrid_doorkey_c2.yaml`, so users do not need to reconstruct CLI flags.
2. Add a small results reproduction script that runs 3 seeds per environment and writes a CSV summary.
3. Publish a fresh Docker image with pinned requirements and a visible image digest.
4. Add GitHub Actions for `pytest -q pine/tests` on CPU.
5. Consider migrating from Gym to Gymnasium only after preserving the current Gym 0.25 behavior with regression tests.
