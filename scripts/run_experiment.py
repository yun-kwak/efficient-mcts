import argparse
import warnings

import wandb

from pine.algorithms.muzero import Experiment

warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")


def parse_bool(x):
    return x.capitalize() == "True"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for the script")
    parser.add_argument("--debug", default="False", type=parse_bool)

    # Parse environment arguments
    parser.add_argument("--env_id", default="Sokoban-PushAndPull-7x7-B1-C3", type=str)
    parser.add_argument("--env_kwargs", default="{}", type=str)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--num_envs", default=16, type=int)

    # Training and simulation parameters
    parser.add_argument("--unroll_steps", default=5, type=int)
    parser.add_argument("--td_steps", default=5, type=int)
    parser.add_argument("--max_search_depth", default=100, type=int)
    parser.add_argument("--num_bins", default=601, type=int)
    parser.add_argument("--channels", default=64, type=int)
    parser.add_argument("--use_resnet_v2", default="True", type=parse_bool)
    parser.add_argument("--output_init_scale", default=0.0, type=float)
    parser.add_argument("--discount_factor", default=0.997, type=float)
    parser.add_argument("--mcts_c1", default=1.25, type=float)
    parser.add_argument("--mcts_c2", default=19625, type=float)
    parser.add_argument("--alpha", default=0.3, type=float)
    parser.add_argument("--exploration_prob", default=0.25, type=float)
    parser.add_argument("--temperature_scheduling", default="staircase", type=str)
    parser.add_argument("--q_normalize_epsilon", default=0.01, type=float)
    parser.add_argument("--child_select_epsilon", default=1e-6, type=float)
    parser.add_argument("--num_simulations", default=50, type=int)

    # Replay buffer
    parser.add_argument("--replay_min_size", default=32_000, type=int)
    parser.add_argument("--replay_max_size", default=1_000_000, type=int)

    # Optimization parameters
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--value_coef", default=0.25, type=float)
    parser.add_argument("--policy_coef", default=1.0, type=float)
    parser.add_argument("--max_grad_norm", default=5.0, type=float)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--warmup_steps", default=1000, type=int)
    parser.add_argument("--learning_rate_decay", default=0.1, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--target_update_interval", default=200, type=int)

    # Evaluation parameters
    parser.add_argument("--evaluate_episodes", default=32, type=int)
    parser.add_argument("--log_interval", default=2000, type=int)
    parser.add_argument("--total_frames", default=1_632_000, type=int)

    # Masking parameters
    parser.add_argument("--use_action_mask", default="True", type=parse_bool)
    parser.add_argument("--use_abst_policy_learning", default="False", type=parse_bool)
    parser.add_argument(
        "--action_cardinalities",
        default=[5, 3, 3, 3],
        type=lambda x: [int(i) for i in x.strip("[]").split(",")],
    )
    parser.add_argument("--mask_temp", default=1.0, type=float)
    parser.add_argument("--mask_thres", default=0.5, type=float)
    parser.add_argument("--mcts_mask_thres", default=0.01, type=float)
    parser.add_argument("--mask_coef", default=0.0, type=float)

    # Additional parameters
    parser.add_argument("--causal_uct_start_step", default=0, type=int)
    parser.add_argument("--use_consistency_loss", default="False", type=parse_bool)
    parser.add_argument("--consistency_coef", default=0.0, type=float)
    parser.add_argument("--use_recon_loss", default="True", type=parse_bool)
    parser.add_argument("--recon_coef", default=0.1, type=float)
    parser.add_argument("--n_frame_stack", default=1, type=int)

    # Save and load parameters
    parser.add_argument("--save_dir", default="/data/", type=str)
    parser.add_argument("--save_interval", default=10000, type=int)

    parser.add_argument("--loglevel", default="WARNING", type=str)
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--run_name", type=str)

    args = parser.parse_args()

    import json

    args.env_kwargs = json.loads(args.env_kwargs)

    config = {}
    for x in vars(args):
        config[x] = getattr(args, x)
    wandb.init(
        project="efficient-mcts",
        group=args.exp_name,
        config=config,
        name=args.run_name,
    )

    import logging

    loglevel_dict = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    logging.basicConfig(level=loglevel_dict[args.loglevel])

    exp = Experiment(config)
    num_iters = (
        (args.total_frames - args.replay_min_size) // args.num_envs
    ) // args.log_interval
    exp.run(num_iters)
