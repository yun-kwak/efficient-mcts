# Efficient Monte Carlo Tree Search via On-the-Fly State-Conditioned Action Abstraction

[![UAI'24 Oral](https://img.shields.io/badge/UAI'24-Oral-331b1b.svg)](https://openreview.net/forum?id=UvDsWevxUI)
[![arXiv](https://img.shields.io/badge/arXiv-2406.00614-b31b1b.svg)](https://arxiv.org/abs/2406.00614)

This repository contains the official implementation of the publication:
Efficient Monte Carlo Tree Search via On-the-Fly State-Conditioned Action Abstraction, Yunhyeok Kwak, Inwoo Hwang, Dooyoung Kim, Sanghack Lee, Byoung-Tak Zhang, The 40th Conference on Uncertainty in Artificial Intelligence, 2024.

<div align="center">
    <img width="1600" alt="image" src="https://github.com/user-attachments/assets/9bd1957e-5621-400c-93ad-2bf60d107611">
</div>

## 📦 Installation

### Docker image

The recommended setup is the published Docker image:

```bash
docker pull yunkwak/efficient-mcts:1.0
```

For an exactly pinned image, use the digest:

```bash
docker pull yunkwak/efficient-mcts@sha256:b50c57d2d842b406affeee73413d9e926ed827c4e1ca4d699a1cfd658457a256
```

The image contains the Python, JAX, CUDA, and Python package dependencies. It does not bundle a checkout of this repository. Clone the repository on the host, mount it into the container, then install the local `pine` package in editable mode:

```bash
git clone https://github.com/yun-kwak/efficient-mcts.git
cd efficient-mcts

docker run --rm --gpus all -it \
  -v "$PWD":/workspace/efficient-mcts \
  -w /workspace/efficient-mcts \
  -e WANDB_MODE=disabled \
  yunkwak/efficient-mcts:1.0 \
  bash

pip install -e pine
python -m pytest -q pine/tests
```

## 🚀 Quick Start

Run the unit tests:

```bash
python -m pytest -q pine/tests
```

Run one paper preset:

```bash
WANDB_MODE=disabled python scripts/run_paper_preset.py \
  --task doorkey_easy \
  --method ours \
  --seed 1 \
  --save_dir ./outputs
```

Run the default Sokoban experiment:

```bash
python scripts/run_experiment.py \
  --env_id Sokoban-PushAndPull-7x7-B1-C3 \
  --exp_name sokoban-push-pull \
  --run_name seed1
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
