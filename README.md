# Efficient Monte Carlo Tree Search via On-the-Fly State-Conditioned Action Abstraction

[![UAI'24 Oral](https://img.shields.io/badge/UAI'24-Oral-331b1b.svg)](https://openreview.net/forum?id=UvDsWevxUI)
[![arXiv](https://img.shields.io/badge/arXiv-2406.00614-b31b1b.svg)](https://arxiv.org/abs/2406.00614)

This repository contains the official implementation of the publication:
Efficient Monte Carlo Tree Search via On-the-Fly State-Conditioned Action Abstraction, Yunhyeok Kwak, Inwoo Hwang, Dooyoung Kim, Sanghack Lee, Byoung-Tak Zhang, The 40th Conference on Uncertainty in Artificial Intelligence, 2024.

<div align="center">
    <img width="1600" alt="image" src="https://github.com/user-attachments/assets/9bd1957e-5621-400c-93ad-2bf60d107611">
</div>



## Current status

This repository has a small maintained smoke path again:

```bash
python -m pytest -q pine/tests
WANDB_MODE=offline bash scripts/smoke.sh
```

The smoke run uses a tiny MiniGrid DoorKey setup and is intended to verify that JAX, Haiku, W&B logging, environments, MCTS, replay, and one training update path all still work.

See [docs/reproducibility.md](docs/reproducibility.md) for the tested dependency set, Forge commands, and maintenance notes.

## 📦 Installation

- Use the Docker image (recommended): `yunkwak/efficient-mcts:1.0` ([Docker Hub](https://hub.docker.com/layers/yunkwak/efficient-mcts/1.0/images/sha256-b50c57d2d842b406affeee73413d9e926ed827c4e1ca4d699a1cfd658457a256))
- Or, install the local CPU dependency set:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
CFLAGS="-I/opt/homebrew/include" LDFLAGS="-L/opt/homebrew/lib" \
  python -m pip install -e pine -r requirements/py310-cpu.txt
```

JAX should still be selected intentionally for your platform. The local CPU path is tested with JAX 0.4.20 on macOS arm64. The published Docker image is the closest original GPU environment.

## 🚀 Quick Start

Run the unit tests:

```bash
python -m pytest -q pine/tests
```

Run the maintained smoke experiment:

```bash
WANDB_MODE=offline bash scripts/smoke.sh
```

Run the default Sokoban experiment:

```bash
python scripts/run_experiment.py \
  --env_id Sokoban-PushAndPull-7x7-B1-C3 \
  --exp_name sokoban-push-pull \
  --run_name seed1
```

Submit the Forge smoke job:

```bash
FORGE_BIN=/Users/yunhyeok.kwak/.local/bin/forge bash scripts/submit_forge_smoke.sh
```

Run one paper reproduction preset:

```bash
WANDB_MODE=offline python scripts/run_paper_preset.py \
  --task doorkey_easy \
  --method ours \
  --seed 1
```

Submit a Forge pilot for Ours and MuZero on DoorKey-Easy:

```bash
python scripts/submit_forge_paper_runs.py \
  --forge_bin /Users/yunhyeok.kwak/.local/bin/forge \
  --tasks doorkey_easy \
  --methods ours muzero \
  --seeds 1
```


## ✒️ Citation

If you use this code in your research, please cite the following paper:

```
@inproceedings{
    kwak2024efficient,
    title={Efficient Monte Carlo Tree Search via On-the-Fly State-Conditioned Action Abstraction},
    author={Yunhyeok Kwak and Inwoo Hwang and Dooyoung Kim and Sanghack Lee and Byoung-Tak Zhang},
    booktitle={The 40th Conference on Uncertainty in Artificial Intelligence},
    year={2024},
    url={https://openreview.net/forum?id=UvDsWevxUI}
}
```

## 📖 Credits

This repository is based on the following repositories:

- [jax_muzero](https://github.com/Hwhitetooth/jax_muzero)
- [Haiku](https://github.com/google-deepmind/dm-haiku)
- [Mctx](https://github.com/google-deepmind/mctx)
- [Flax](https://github.com/google/flax)
- [Monte Carlo Tree Search With Iteratively Refining State Abstractions, NeurIPS 2021, Sokota et al.](https://proceedings.neurips.cc/paper/2021/hash/9b0ead00a217ea2c12e06a72eec4923f-Abstract.html)
