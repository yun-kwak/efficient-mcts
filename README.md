# Efficient Monte Carlo Tree Search via On-the-Fly State-Conditioned Action Abstraction

[UAI'24 Oral](https://openreview.net/forum?id=UvDsWevxUI) | [ArXiv](https://arxiv.org/abs/2406.00614)

This repository contains the official implementation of the publication:
Efficient Monte Carlo Tree Search via On-the-Fly State-Conditioned Action Abstraction, Yunhyeok Kwak, Inwoo Hwang, Dooyoung Kim, Sanghack Lee, Byoung-Tak Zhang, The 40th Conference on Uncertainty in Artificial Intelligence, 2024.

<div align="center">
    <img width="1600" alt="image" src="https://github.com/user-attachments/assets/9bd1957e-5621-400c-93ad-2bf60d107611">
</div>



## 📦 Installation

- Use the Docker image (recommended): `yunkwak/efficient-mcts:1.0` ([Docker Hub](https://hub.docker.com/layers/yunkwak/efficient-mcts/1.0/images/sha256-b50c57d2d842b406affeee73413d9e926ed827c4e1ca4d699a1cfd658457a256))
- Or, install the dependencies manually. JAX should be installed separately. (Tested on Python 3.10, jax==0.4.16, haiku==0.0.10): `.devcontainer/requirements.txt`

## 🚀 Quick Start

To run the experiments, use the following commands:

```bash
python scripts/run_experiment.py
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
