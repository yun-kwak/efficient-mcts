"""MuZero: a MCTS agent that plans with a learned value-equivalent model."""
import logging
import os
# Performance optimization flags for NVIDIA GPUs.
# Refer to https://github.com/NVIDIA/JAX-Toolbox
os.environ["XLA_FALGS"] = "--xla_gpu_enable_latency_hiding_scheduler=true ----xla_gpu_enable_async_all_gather=true --xla_gpu_enable_async_reduce_scatter=true --xla_gpu_enable_triton_gemm=false"
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ["NCCL_IB_SL"] = "1"

import pickle
import time

import chex
import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import rlax
import seaborn as sns
import wandb

from pine import vec_env

from ..environments import minigrid, sokoban
from .actors import Actor, EvaluateActor
from .agents import Agent
from .replay_buffers import UniformBuffer
from .types import ActorOutput, Params
from .utils import logits_to_scalar, scalar_to_two_hot, hamming_distance, product_action_influence_jax, marginalize_policy_dist_flat


def generate_update_fn(
    agent: Agent,
    opt_update,
    unroll_steps: int,
    td_steps: int,
    discount_factor: float,
    value_coef: float,
    policy_coef: float,
    mask_coef: float,
    consistency_coef: float,
    recon_coef: float,
):
    def loss(
        params: Params,
        target_params: Params,
        trajectory: ActorOutput,
        rng_key: chex.PRNGKey,
        is_causal_uct_step: bool,
    ):
        freezed_mask_model_params = Params(
            encoder=params.encoder,
            transition=params.transition,
            prediction=hk.data_structures.map(
                lambda module_name, name, value: value
                if "mask" not in module_name
                else target_params.prediction[module_name][name],
                params.prediction,
            ),
            decoder=params.decoder,
        )
        # 1. Make predictions. Unroll the model from the first state.
        (
            rng_key,
            root_unroll_key,
            random_action_key,
            model_unroll_key,
        ) = jax.random.split(rng_key, 4)
        timestep = jax.tree_map(lambda t: t[:1], trajectory)
        learner_root = agent.root_unroll(
            root_unroll_key, freezed_mask_model_params, timestep
        )
        learner_root = jax.tree_map(lambda t: t[0], learner_root)

        # Fill the actions after the absorbing state with random actions.
        unroll_trajectory = jax.tree_map(lambda t: t[: unroll_steps + 1], trajectory)
        random_action_mask = jnp.cumprod(1.0 - unroll_trajectory.first[1:]) == 0.0
        action_sequence = unroll_trajectory.action_tm1[1:]
        num_actions = learner_root.logits.shape[-1]
        random_actions = jax.random.choice(
            random_action_key, num_actions, action_sequence.shape, replace=True
        )
        simulate_action_sequence = jax.lax.select(
            random_action_mask, random_actions, action_sequence
        )

        model_out = agent.model_unroll(
            model_unroll_key,
            freezed_mask_model_params,
            learner_root.state,
            learner_root.mask,
            simulate_action_sequence,
        )

        num_bins = learner_root.reward_logits.shape[-1]

        # 2. Construct targets.
        ## 2.1 Reward.
        rewards = trajectory.reward[1:]
        reward_target = jax.lax.select(
            random_action_mask,
            jnp.zeros_like(rewards[:unroll_steps]),
            rewards[:unroll_steps],
        )
        reward_target = agent.value_transform(reward_target)
        reward_logits_target = scalar_to_two_hot(reward_target, num_bins)

        ## 2.2 Policy
        rng_key, target_root_unroll_key, search_key = jax.random.split(rng_key, 3)
        target_roots = agent.root_unroll(
            target_root_unroll_key, target_params, trajectory
        )
        search_roots = jax.tree_map(lambda t: t[: unroll_steps + 1], target_roots)
        search_keys = jax.random.split(search_key, search_roots.state.shape[0])
        target_trees = jax.vmap(agent.mcts, (0, None, 0, None, None))(
            search_keys, target_params, search_roots, False, is_causal_uct_step
        )
        # The target distribution always uses a temperature of 1.
        policy_target = jax.vmap(agent.act_prob, (0, None, None, 0))(
            target_trees.visit_count[:, 0],
            1.0,
            is_causal_uct_step,
            target_trees.mcts_mask[:, 0],
        )
        def marginalize_and_mask(policy_dist, mask):
            policy_dist = marginalize_policy_dist_flat(
                policy_dist, mask, action_cardinalities=agent._action_cardinalities)
            mask = product_action_influence_jax(
                mask, np.array(agent._action_cardinalities))
            policy_dist = jnp.where(mask, policy_dist, 0.0)
            return policy_dist
        if agent._use_abst_policy_learning:
            target_mcts_mask = search_roots.mcts_mask
            random_mcts_mask = jnp.cumprod(1.0 - unroll_trajectory.last) == 0.0
            random_mcts_mask = jnp.broadcast_to(
                random_mcts_mask[:, None], target_mcts_mask.shape
            )
            target_mcts_mask = jax.lax.select(
                random_mcts_mask, jnp.ones_like(target_mcts_mask, dtype=jnp.bool_), target_mcts_mask
            )
            policy_target = jax.vmap(marginalize_and_mask)(policy_target, target_mcts_mask)
        # Set the policy targets for the absorbing state and the states after to uniform random.
        uniform_policy = jnp.ones_like(policy_target) / num_actions
        random_policy_mask = jnp.cumprod(1.0 - unroll_trajectory.last) == 0.0
        random_policy_mask = jnp.broadcast_to(
            random_policy_mask[:, None], policy_target.shape
        )
        policy_target = jax.lax.select(
            random_policy_mask, uniform_policy, policy_target
        )
        policy_target = jax.lax.stop_gradient(policy_target)

        ## 2.3 Value
        discounts = (1.0 - trajectory.last[1:]) * discount_factor

        def n_step_return(i):
            # According to the EfficientZero source code, it is unnecessary to use the search value for bootstrapping.
            # See: https://github.com/YeWR/EfficientZero/blob/main/main.py#L41
            #   and https://github.com/YeWR/EfficientZero/blob/main/core/reanalyze_worker.py#L325
            bootstrap_value = jax.tree_map(
                lambda t: t[i + td_steps], target_roots.value
            )
            _rewards = jnp.concatenate(
                [rewards[i : i + td_steps], bootstrap_value[None]], axis=0
            )
            _discounts = jnp.concatenate(
                [jnp.ones((1,)), jnp.cumprod(discounts[i : i + td_steps])], axis=0
            )
            return jnp.sum(_rewards * _discounts)

        returns = []
        for i in range(unroll_steps + 1):
            returns.append(n_step_return(i))
        returns = jnp.stack(returns)
        # Set the value targets for the absorbing state and the states after to 0.
        zero_return_mask = jnp.cumprod(1.0 - unroll_trajectory.last) == 0.0
        value_target = jax.lax.select(
            zero_return_mask, jnp.zeros_like(returns), returns
        )
        value_target = agent.value_transform(value_target)
        value_logits_target = scalar_to_two_hot(value_target, num_bins)
        value_logits_target = jax.lax.stop_gradient(value_logits_target)

        # 3. Compute the losses.
        _batch_categorical_cross_entropy = jax.vmap(rlax.categorical_cross_entropy)

        def categorical_cross_entropy(target_probs, probs):
            return -jnp.sum(target_probs * jnp.log(probs + 1e-8))
        _batch_policy_cross_entropy = jax.vmap(categorical_cross_entropy)
        reward_loss = jnp.mean(
            _batch_categorical_cross_entropy(
                reward_logits_target, model_out.reward_logits
            )
        )
        value_logits = jnp.concatenate(
            [learner_root.value_logits[None], model_out.value_logits], axis=0
        )
        value_loss = jnp.mean(
            _batch_categorical_cross_entropy(value_logits_target, value_logits)
        )
        logits = jnp.concatenate([learner_root.logits[None], model_out.logits], axis=0)
        policy_probs = distrax.Categorical(logits=logits).probs
        if agent._use_abst_policy_learning:
            policy_probs = jax.vmap(marginalize_and_mask)(policy_probs, target_mcts_mask)
        policy_loss = jnp.mean(_batch_policy_cross_entropy(policy_target, policy_probs))

        # Consistency loss
        # Consine similarity between the latent state of the next time step from the dynamics model
        # and the latent state of the next time step from the encoder which takes the observation of the next time step as input.

        def compute_consistency_loss(trajectory):
            # NOTE: The action masking is not used in the consistency loss for now.
            timestep = jax.tree_map(lambda t: t[:1], trajectory)
            next_timestep = jax.tree_map(lambda t: t[1:2], trajectory)
            states = jax.vmap(agent.encode, (None, 0))(params, timestep.observation)
            pred_next_states = jax.vmap(agent.transition_step, (None, 0, 0))(
                params, states, next_timestep.action_tm1
            )
            next_states_from_obs = jax.lax.stop_gradient(
                jax.vmap(agent.encode, (None, 0))(
                    target_params, next_timestep.observation
                )
            )
            # Flatten
            pred_next_states = jnp.reshape(pred_next_states, -1)
            next_states_from_obs = jnp.reshape(next_states_from_obs, -1)
            # Normalize the latent vectors.
            pred_next_states = pred_next_states / jnp.linalg.norm(
                pred_next_states, axis=-1, keepdims=True
            )
            next_states_from_obs = next_states_from_obs / jnp.linalg.norm(
                next_states_from_obs, axis=-1, keepdims=True
            )
            # Compute the cosine similarity.
            cossim = jnp.sum(pred_next_states * next_states_from_obs)
            # Mask the cossim for the absorbing state
            cossim = jax.lax.select(timestep.last[0] == 1.0, 0.0, cossim)
            return 1 - cossim

        consistency_loss = 0.0
        if agent._use_consistency_loss:
            consistency_loss = compute_consistency_loss(trajectory)

        scaled_consistency_loss = consistency_coef * consistency_loss

        # Reconstruction loss
        # Compute the reconstruction loss by rerolling the model from the first state.
        # This allows us to update the parameters of the mask model
        # based solely on the reconstruction loss.
        learner_root = agent.root_unroll(root_unroll_key, params, timestep)
        learner_root = jax.tree_map(lambda t: t[0], learner_root)
        model_out = agent.model_unroll(
            model_unroll_key,
            params,
            learner_root.state,
            learner_root.mask,
            simulate_action_sequence,
        )

        # Concatenate root state and states
        states = model_out.state
        root_state = jax.tree_map(lambda t: t[None], learner_root).state
        unroll_states = jnp.concatenate([root_state, states], axis=0)
        if not agent._use_recon_loss:
            unroll_states = jax.lax.stop_gradient(unroll_states)
        pred_obs = jax.vmap(agent.decode, (None, 0))(params, unroll_states)
        obs = unroll_trajectory.observation
        recon_loss = jnp.sum(jnp.square(pred_obs[1:] / 255. - obs[1:] / 255.), axis=(1, 2, 3))
        recon_loss = jax.lax.select(
            random_action_mask, jnp.zeros_like(recon_loss), recon_loss
        )
        root_recon_loss = jnp.sum(jnp.square(pred_obs[0:1] / 255. - obs[0:1] / 255.), axis=(1, 2, 3))
        recon_loss = jnp.mean(jnp.concatenate([root_recon_loss, recon_loss], axis=0))

        scaled_recon_loss = recon_coef * recon_loss

        # L1 mask loss
        mask_logits = jnp.concatenate(
            [learner_root.mask_logits[None], model_out.mask_logits[:-1]], axis=0
        )
        mask_probs = jax.nn.sigmoid(mask_logits)
        mask_loss = jnp.mean(jnp.sum(jnp.abs(mask_probs), axis=-1))
        scaled_mask_loss = mask_coef * mask_loss

        total_loss = reward_loss + value_coef * value_loss + policy_coef * policy_loss
        if agent._use_action_mask:
            total_loss += scaled_mask_loss
        if agent._use_consistency_loss:
            total_loss += scaled_consistency_loss
        if agent._use_recon_loss:
            total_loss += scaled_recon_loss
        policy_target_entropy = jax.vmap(
            lambda p: distrax.Categorical(probs=p).entropy()
        )(policy_target)
        log = {
            "reward_target": reward_target,
            "reward_prediction": model_out.reward,
            "value_target": value_target,
            "value_prediction": logits_to_scalar(value_logits),
            "policy_entropy": -rlax.entropy_loss(logits, jnp.ones(logits.shape[:-1])),
            "policy_target_entropy": policy_target_entropy,
            "reward_loss": reward_loss,
            "value_loss": value_loss,
            "scaled_value_loss": value_coef * value_loss,
            "policy_loss": policy_loss,
            "scaled_policy_loss": policy_coef * policy_loss,
            "mask_loss": mask_loss,
            "scaled_mask_loss": scaled_mask_loss,
            "consistency_loss": consistency_loss,
            "scaled_consistency_loss": scaled_consistency_loss,
            "recon_loss": recon_loss,
            "scaled_recon_loss": scaled_recon_loss,
            "obs": obs,
            "pred_obs": pred_obs,
            "total_loss": total_loss,
            "root_mask": learner_root.mask,
            "root_mcts_mask": learner_root.mcts_mask,
        }
        return total_loss, log

    def batch_loss(
        params: Params,
        target_params: Params,
        trajectories: ActorOutput,
        rng_key: chex.PRNGKey,
        is_causal_uct_step: bool,
    ):
        batch_size = trajectories.observation.shape[0]
        rng_keys = jax.random.split(rng_key, batch_size)
        losses, log = jax.vmap(loss, (None, None, 0, 0, None))(
            params, target_params, trajectories, rng_keys, is_causal_uct_step
        )

        # Compute false negative rates for the mask.
        root_gt_mask = trajectories.gt_mask[:, 0]
        gt_num_positives = jnp.sum(root_gt_mask, axis=0)
        gt_positive_rates = gt_num_positives / batch_size
        root_pred_mask = log["root_mask"]
        root_mcts_mask = log["root_mcts_mask"]
        num_false_negatives = jnp.sum(root_gt_mask * (1.0 - root_pred_mask), axis=0)
        num_false_negatives_mcts = jnp.sum(
            root_gt_mask * (1.0 - root_mcts_mask), axis=0
        )
        false_negative_rates = num_false_negatives / gt_num_positives
        false_negative_rates_mcts = num_false_negatives_mcts / gt_num_positives

        # Compute the hamming distance between the mask and the ground truth mask.
        mask_shd = jax.vmap(hamming_distance)(root_pred_mask, root_gt_mask)
        mcts_mask_shd = jax.vmap(hamming_distance)(root_mcts_mask, root_gt_mask)

        total_loss = jnp.mean(log["total_loss"])
        scaled_recon_loss = jnp.mean(log["scaled_recon_loss"])
        scaled_mask_loss = jnp.mean(log["scaled_mask_loss"])
        scaled_consistency_loss = jnp.mean(log["scaled_consistency_loss"])
        scaled_value_loss = jnp.mean(log["scaled_value_loss"])
        scaled_policy_loss = jnp.mean(log["scaled_policy_loss"])
        scaled_reward_loss = jnp.mean(log["reward_loss"])
        log.update(
            {
                "reward_target_mean": jnp.mean(log["reward_target"]),
                "reward_target_std": jnp.std(log["reward_target"]),
                "reward_prediction_mean": jnp.mean(log["reward_prediction"]),
                "reward_prediction_std": jnp.std(log["reward_prediction"]),
                "value_target_mean": jnp.mean(log["value_target"]),
                "value_target_std": jnp.std(log["value_target"]),
                "value_prediction_mean": jnp.mean(log["value_prediction"]),
                "value_prediction_std": jnp.std(log["value_prediction"]),
                "policy_entropy": jnp.mean(log["policy_entropy"]),
                "policy_target_entropy": jnp.mean(log["policy_target_entropy"]),
                "reward_loss": jnp.mean(log["reward_loss"]),
                "reward_loss_ratio": scaled_reward_loss / total_loss,
                "value_loss": jnp.mean(log["value_loss"]),
                "scaled_value_loss": jnp.mean(log["scaled_value_loss"]),
                "value_loss_ratio": scaled_value_loss / total_loss,
                "policy_loss": jnp.mean(log["policy_loss"]),
                "scaled_policy_loss": jnp.mean(log["scaled_policy_loss"]),
                "policy_loss_ratio": scaled_policy_loss / total_loss,
                "mask_loss": jnp.mean(log["mask_loss"]),
                "scaled_mask_loss": jnp.mean(log["scaled_mask_loss"]),
                "mask_loss_ratio": scaled_mask_loss / total_loss,
                "consistency_loss": jnp.mean(log["consistency_loss"]),
                "scaled_consistency_loss": jnp.mean(log["scaled_consistency_loss"]),
                "consistency_loss_ratio": scaled_consistency_loss / total_loss,
                "recon_loss": jnp.mean(log["recon_loss"]),
                "recon_loss_std": jnp.std(log["recon_loss"]),
                "scaled_recon_loss": jnp.mean(log["scaled_recon_loss"]),
                "recon_loss_ratio": scaled_recon_loss / total_loss,
                "total_loss": total_loss,
                "gt_positive_rates": gt_positive_rates,
                "false_negative_rates": false_negative_rates,
                "false_negative_rates_mcts": false_negative_rates_mcts,
                "mask_shd_mean": jnp.mean(mask_shd),
                "mask_shd_std": jnp.std(mask_shd),
                "mcts_mask_shd_mean": jnp.mean(mcts_mask_shd),
                "mcts_mask_shd_std": jnp.std(mcts_mask_shd),
            }
        )
        log.pop("reward_target")
        log.pop("reward_prediction")
        log.pop("value_target")
        log.pop("value_prediction")
        log.pop("root_mask")
        log.pop("root_mcts_mask")
        return jnp.mean(losses), log

    def update(
        rng_key: chex.PRNGKey,
        params: Params,
        target_params: Params,
        opt_state,
        trajectories: ActorOutput,
        is_causal_uct_step: bool,
    ):
        grads, log = jax.grad(batch_loss, has_aux=True)(
            params, target_params, trajectories, rng_key, is_causal_uct_step
        )
        grads = jax.lax.pmean(grads, axis_name="i")
        updates, opt_state = opt_update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        log.update(
            {
                "grad_norm": optax.global_norm(grads),
                "update_norm": optax.global_norm(updates),
                "param_norm": optax.global_norm(params),
                "is_causal_uct_step": jnp.float32(is_causal_uct_step),
            }
        )
        return params, opt_state, log

    return update


class Experiment:
    def __init__(self, config):
        self._config = config
        platform = jax.lib.xla_bridge.get_backend().platform
        self._num_devices = jax.lib.xla_bridge.get_backend().device_count()
        logging.warning("Running on %s %s(s)", self._num_devices, platform)

        seed = config["seed"]
        if "MiniGrid" in config["env_id"]:
            env_id = config["env_id"]
            self._envs = minigrid.make_minigrid_vec_env(
                env_id,
                num_env=config["num_envs"],
                seed=seed,
                env_kwargs=config["env_kwargs"],
                force_dummy=config["debug"]
            )
            self._evaluate_envs = minigrid.make_minigrid_vec_env(
                env_id=env_id,
                num_env=config["evaluate_episodes"],
                seed=seed,
                env_kwargs=config["env_kwargs"],
                force_dummy=config["debug"]
            )
        elif "Sokoban" in config["env_id"]:
            env_id = config["env_id"]
            self._envs = sokoban.make_sokoban_vec_env(
                env_id,
                num_env=config["num_envs"],
                seed=seed,
                env_kwargs=config["env_kwargs"],
                force_dummy=config["debug"]
            )
            self._evaluate_envs = sokoban.make_sokoban_vec_env(
                env_id=env_id,
                num_env=config["evaluate_episodes"],
                seed=seed,
                env_kwargs=config["env_kwargs"],
                force_dummy=config["debug"]
            )
        elif "Bandit" in config["env_id"]:
            env_id = config["env_id"]
            self._envs = bandit.make_bandit_vec_env(
                env_id,
                num_env=config["num_envs"],
                seed=seed,
                env_kwargs=config["env_kwargs"],
            )
            self._evaluate_envs = bandit.make_bandit_vec_env(
                env_id=env_id,
                num_env=config["evaluate_episodes"],
                seed=seed,
                env_kwargs=config["env_kwargs"],
            )
        elif "MinAtar" in config["env_id"]:
            env_id = config["env_id"]
            self._envs = minatar.make_minatar_vec_env(
                env_id,
                num_env=config["num_envs"],
                seed=seed,
                env_kwargs=config["env_kwargs"],
                force_dummy=config["debug"]
            )
            self._evaluate_envs = minatar.make_minatar_vec_env(
                env_id=env_id,
                num_env=config["evaluate_episodes"],
                seed=seed,
                env_kwargs=config["env_kwargs"],
                force_dummy=config["debug"]
            )
        else:
            env_id = config["env_id"] + "NoFrameskip-v4"
            self._envs = atari.make_vec_env(
                env_id,
                num_env=config["num_envs"],
                seed=seed,
                env_kwargs=config["env_kwargs"],
                force_dummy=config["debug"]
            )
            self._evaluate_envs = atari.make_vec_env(
                env_id=env_id,
                num_env=config["evaluate_episodes"],
                seed=config["seed"],
                env_kwargs=config["env_kwargs"],
                wrapper_kwargs={"episode_life": False},
                force_dummy=config["debug"]
            )
        if config["n_frame_stack"] > 1:
            self._envs = vec_env.VecFrameStack(self._envs, config["n_frame_stack"])
            self._evaluate_envs = vec_env.VecFrameStack(
                self._evaluate_envs, config["n_frame_stack"]
            )
        self._agent = Agent(
            self._envs.observation_space,
            self._envs.action_space,
            num_bins=config["num_bins"],
            channels=config["channels"],
            use_v2=config["use_resnet_v2"],
            output_init_scale=config["output_init_scale"],
            discount_factor=config["discount_factor"],
            num_simulations=config["num_simulations"],
            max_search_depth=config["max_search_depth"],
            mcts_c1=config["mcts_c1"],
            mcts_c2=config["mcts_c2"],
            alpha=config["alpha"],
            exploration_prob=config["exploration_prob"],
            q_normalize_epsilon=config["q_normalize_epsilon"],
            child_select_epsilon=config["child_select_epsilon"],
            use_action_mask=config["use_action_mask"],
            use_abst_policy_learning=config["use_abst_policy_learning"],
            action_cardinalities=config["action_cardinalities"],
            mask_temp=config["mask_temp"],
            mask_thres=config["mask_thres"],
            mcts_mask_thres=config["mcts_mask_thres"],
            causal_uct_start_step=config["causal_uct_start_step"],
            use_consistency_loss=config["use_consistency_loss"],
            use_recon_loss=config["use_recon_loss"],
            n_frame_stack=config["n_frame_stack"],
        )
        self._actor = Actor(self._envs, self._agent)
        self._evaluate_actor = EvaluateActor(self._evaluate_envs, self._agent)

        if config["temperature_scheduling"] == "staircase":

            def temperature_fn(num_frames: int):
                frac = num_frames / config["total_frames"]
                if frac < 0.5:
                    return 1.0
                elif frac < 0.75:
                    return 0.5
                else:
                    return 0.25

        elif config["temperature_scheduling"] == "constant":

            def temperature_fn(num_frames: int):
                return 1.0

        else:
            raise KeyError

        self._temperature_fn = temperature_fn

        self._rng_key = jax.random.PRNGKey(seed)
        self._rng_key, init_key = jax.random.split(self._rng_key)
        self._params = self._agent.init(init_key)
        self._target_params = self._params
        self._target_update_interval = config["target_update_interval"]

        # Only apply weight decay to the weights in Dense layers and Conv layers.
        # Do NOT apply to the biases and the scales and offsets in normalization layers.
        weight_decay_mask = Params(
            encoder=hk.data_structures.map(
                lambda module_name, name, value: True if name == "w" else False,
                self._params.encoder,
            ),
            transition=hk.data_structures.map(
                lambda module_name, name, value: True if name == "w" else False,
                self._params.transition,
            ),
            prediction=hk.data_structures.map(
                lambda module_name, name, value: True if name == "w" else False,
                self._params.prediction,
            ),
            decoder=hk.data_structures.map(
                lambda module_name, name, value: True if name == "w" else False,
                self._params.decoder,
            ),
        )
        learning_rate = optax.warmup_exponential_decay_schedule(
            init_value=0.0,
            peak_value=config["learning_rate"],
            warmup_steps=config["warmup_steps"],
            transition_steps=100_000,
            decay_rate=config["learning_rate_decay"],
            staircase=True,
        )
        # Apply the decoupled weight decay. Ref: https://arxiv.org/abs/1711.05101.
        self._opt = optax.adamw(
            learning_rate=learning_rate,
            weight_decay=config["weight_decay"],
            mask=weight_decay_mask,
        )
        if config["max_grad_norm"]:
            self._opt = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                self._opt,
            )
        self._opt_state = self._opt.init(self._params)

        self._params = jax.device_put_replicated(self._params, jax.local_devices())
        self._target_params = self._params
        self._opt_state = jax.device_put_replicated(
            self._opt_state, jax.local_devices()
        )
        self._is_causal_uct_step = jax.device_put_replicated(
            jnp.array(False), jax.local_devices()
        )

        self._update_fn = generate_update_fn(
            self._agent,
            self._opt.update,
            unroll_steps=config["unroll_steps"],
            td_steps=config["td_steps"],
            discount_factor=config["discount_factor"],
            value_coef=config["value_coef"],
            policy_coef=config["policy_coef"],
            mask_coef=config["mask_coef"],
            consistency_coef=config["consistency_coef"],
            recon_coef=config["recon_coef"],
        )
        self._update_fn = jax.pmap(self._update_fn, axis_name="i")

        self._replay_buffer = UniformBuffer(
            min_size=config["replay_min_size"],
            max_size=config["replay_max_size"],
            traj_len=config["unroll_steps"] + config["td_steps"],
        )
        self._batch_size = config["batch_size"]
        self._causal_uct_start_step = config["causal_uct_start_step"]
        assert self._batch_size % self._num_devices == 0

        self._log_interval = config["log_interval"]
        self._save_interval = config["save_interval"]
        self._save_dir = config["save_dir"]
        self._num_frames = 0
        self._total_frames = config["total_frames"]
        self._num_updates = 0

        init_timestep = self._actor.initial_timestep()
        self._replay_buffer.extend(init_timestep)
        self._num_frames += init_timestep.observation.shape[0]
        act_params = jax.tree_map(lambda t: t[0], self._params)
        while not self._replay_buffer.ready():
            self._rng_key, timesteps, epinfos = self._actor.step(
                self._rng_key, act_params, random=True
            )
            self._replay_buffer.extend(timesteps)
            self._num_frames += timesteps.observation.shape[0]
        self._trajectories = [
            self._replay_buffer.sample(self._batch_size // self._num_devices)
            for _ in range(self._num_devices)
        ]
        self._trajectories = jax.device_put_sharded(
            self._trajectories, jax.local_devices()
        )

    def _save_ckpt(self):
        """Save the model parameters, optimizer state"""
        save_dict = {
            "params": self._params,
            "target_params": self._target_params,
            "opt_state": self._opt_state,
        }
        save_path = os.path.join(
            self._save_dir, wandb.run.name, f"ckpt_{self._num_updates}.pkl"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(save_dict, f)
        print(f"Saved to {save_path}.")

    def step(self):
        t0 = time.time()
        for _ in range(self._log_interval):
            # There are essentially 3 operations in each iteration: sampling, training, and acting.
            # The typical order of execution is: acting -> sampling -> training. But this order does not allow any
            # parallelization. Here we use a different order: training -> sampling -> acting. Due to the asynchronous
            # dispatching of JAX, the call to self._update_fn returns immediately before the computation completes.
            # This enables overlap between training, which only needs the GPU, and sampling, which only needs the CPU.
            # Acting needs both the CPU, the GPU, and synchronization between these two, so it is like a barrier and
            # cannot overlap with either of the other two operations.
            self._rng_key, update_key = jax.random.split(self._rng_key)
            update_keys = jax.random.split(update_key, self._num_devices)
            self._params, self._opt_state, log = self._update_fn(
                update_keys,
                self._params,
                self._target_params,
                self._opt_state,
                self._trajectories,
                self._is_causal_uct_step,
            )
            self._num_updates += 1
            if self._num_updates % self._target_update_interval == 0:
                self._target_params = self._params

            # Save the model, optimizer state
            if self._num_updates % self._save_interval == 0:
                self._save_ckpt()

            self._trajectories = [
                self._replay_buffer.sample(self._batch_size // self._num_devices)
                for _ in range(self._num_devices)
            ]
            self._trajectories = jax.device_put_sharded(
                self._trajectories, jax.local_devices()
            )
            self._is_causal_uct_step = jax.device_put_replicated(
                jnp.array(self._causal_uct_start_step <= self._num_updates),
                jax.local_devices(),
            )

            if self._num_frames < self._total_frames:
                act_params = jax.tree_map(lambda t: t[0], self._params)
                is_causal_uct_step = self._causal_uct_start_step <= self._num_updates
                temperature = self._temperature_fn(self._num_frames)
                self._rng_key, timesteps, epinfos = self._actor.step(
                    self._rng_key,
                    act_params,
                    random=False,
                    temperature=temperature,
                    is_causal_uct_step=is_causal_uct_step,
                )
                self._replay_buffer.extend(timesteps)
                self._num_frames += timesteps.observation.shape[0]

        is_causal_uct_step = self._causal_uct_start_step <= self._num_updates
        act_params = jax.tree_map(lambda t: t[0], self._params)
        self._rng_key, epinfos, eval_stats = self._evaluate_actor.evaluate(
            self._rng_key, act_params, is_causal_uct_step
        )
        self._plot_mask_info(eval_stats.final_step_mask_info)
        self._plot_policy_info(eval_stats.final_step_policy_info)

        log = jax.tree_map(lambda t: t[0], log)
        log = jax.device_get(log)
        num_action_var = log["gt_positive_rates"].shape[-1]
        log.update(
            {
                "ups": self._log_interval / (time.time() - t0),
                "num_frames": self._num_frames,
                "num_updates": self._num_updates,
                "episode_return": np.mean([epinfo["r"] for epinfo in epinfos]),
                "episode_length": np.mean([epinfo["l"] for epinfo in epinfos]),
                "mean_shd": eval_stats.mean_shd,
                "mean_tree_depth": eval_stats.mean_tree_depth,
                "mean_mean_nodes_depth": eval_stats.mean_mean_nodes_depth,
                # "mean_median_nodes_depth": eval_stats.mean_median_nodes_depth,
                "mean_false_neg": eval_stats.mean_false_neg,
                **{
                    f"gt_positive_rates/{i}": log["gt_positive_rates"][i]
                    for i in range(num_action_var)
                },
                **{
                    f"false_negative_rates/{i}": log["false_negative_rates"][i]
                    for i in range(num_action_var)
                },
                **{
                    f"false_negative_rates_mcts/{i}": log["false_negative_rates_mcts"][
                        i
                    ]
                    for i in range(num_action_var)
                },
            }
        )
        log.update()
        self._plot_observations(log["obs"], log["pred_obs"])
        log.pop("obs")
        log.pop("pred_obs")
        log.pop("gt_positive_rates")
        log.pop("false_negative_rates")
        log.pop("false_negative_rates_mcts")
        wandb.log(log)
        return log

    def cleanup(self):
        self._envs.close()
        self._evaluate_envs.close()

    def _plot_observations(self, obs, pred_obs):
        for i in range(min(len(obs), 12)):
            for j in range(self._config["unroll_steps"] + 1):
                fig, ax = plt.subplots()
                ax.imshow(obs[i][j][:, :, -3:])
                wandb.log(
                    {f"observations/{j:02d}/{i:02d}": wandb.Image(fig)}, commit=False
                )
                fig, ax = plt.subplots()
                ax.imshow(pred_obs[i][j][:, :, -3:].astype(np.uint8))
                wandb.log(
                    {f"pred_observations/{j:02d}/{i:02d}": wandb.Image(fig)},
                    commit=False,
                )
                plt.close("all")

    def _plot_mask_info(self, mask_info):
        mask_probs = jax.nn.sigmoid(mask_info.mask_logits)
        for i in range(min(len(mask_info.observations), 12)):
            fig, ax = plt.subplots()
            ax.imshow(mask_info.observations[i][:, :, -3:])
            wandb.log({f"samples/observations/{i:02d}": wandb.Image(fig)}, commit=False)
            fig = plot_bar(mask_info.masks[i])
            wandb.log({f"samples/masks/{i:02d}": wandb.Image(fig)}, commit=False)
            fig = plot_bar(mask_info.mcts_masks[i])
            wandb.log({f"samples/mcts_masks/{i:02d}": wandb.Image(fig)}, commit=False)
            fig = plot_bar(mask_info.gt_masks[i])
            wandb.log({f"samples/gt_masks/{i:02d}": wandb.Image(fig)}, commit=False)
            fig = plot_bar(mask_probs[i])
            wandb.log({f"samples/mask_probs/{i:02d}": wandb.Image(fig)}, commit=False)
            plt.close("all")

    def _plot_policy_info(self, policy_info):
        for i in range(min(len(policy_info.act_prob), 12)):
            fig = plot_bar(policy_info.act_prob[i], figsize=(15, 5))
            wandb.log({f"samples/act_prob/{i:02d}": wandb.Image(fig)}, commit=False)
            fig = plot_bar(policy_info.greedy_prob[i], figsize=(15, 5))
            wandb.log({f"samples/greedy_prob/{i:02d}": wandb.Image(fig)}, commit=False)
            fig = plot_bar(policy_info.prior_act_prob[i], figsize=(15, 5))
            wandb.log(
                {f"samples/prior_act_prob/{i:02d}": wandb.Image(fig)}, commit=False
            )
            plt.close("all")

    def run(self, num_steps: int):
        for _ in range(num_steps):
            self.step()
        self.cleanup()


def plot_bar(p: np.array, figsize=(4, 4), ymin=0.0, ymax=1.0):
    # Check that p is 1-dimensional
    if len(p.shape) != 1:
        raise ValueError("Input array must be 1-dimensional.")

    # Check that the values of p are between 0 and 1
    if not np.all((0 <= p) & (p <= 1)):
        raise ValueError(
            "Values in the input array must be between 0 and 1 (inclusive)."
        )

    # Create array of indices
    indices = np.arange(len(p))

    # Create the figure and the axes
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the bar plot on the axes
    sns.barplot(x=indices, y=p, color="b", ax=ax)

    ax.set_ylim(ymin, ymax)

    # Return the figure
    return fig


if __name__ == "__main__":
    config = {
        "env_id": "Qbert",
        "env_kwargs": {},
        "seed": 42,
        "num_envs": 1,
        "unroll_steps": 5,
        "td_steps": 5,
        "max_search_depth": None,
        "channels": 64,
        "num_bins": 601,
        "use_resnet_v2": True,
        "output_init_scale": 0.0,
        "discount_factor": 0.997**4,
        "mcts_c1": 1.25,
        "mcts_c2": 19625,
        "alpha": 0.3,
        "exploration_prob": 0.25,
        "temperature_scheduling": "staircase",
        "q_normalize_epsilon": 0.01,
        "child_select_epsilon": 1e-6,
        "num_simulations": 50,
        "replay_min_size": 2_000,
        "replay_max_size": 100_000,
        "batch_size": 256,
        "value_coef": 0.25,
        "policy_coef": 1.0,
        "max_grad_norm": 5.0,
        "learning_rate": 7e-4,
        "warmup_steps": 1_000,
        "learning_rate_decay": 0.1,
        "weight_decay": 1e-4,
        "target_update_interval": 200,
        "evaluate_episodes": 32,
        "log_interval": 4_000,
        "total_frames": 100_000,
    }
    analysis = tune.run(
        Experiment,
        config=config,
        stop={
            "num_updates": 120_000,
        },
        resources_per_trial={
            "gpu": 4,
        },
    )
