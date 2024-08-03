"""Actors for generating trajectories."""
from typing import Optional, Tuple, List

import chex
import jax
import numpy as np
from .types import ActorOutput, Params
from .utils import false_negative, hamming_distance

class Actor(object):
    def __init__(self, envs, agent):
        self._envs = envs
        self._agent_step = jax.jit(agent.batch_step)
        num_envs = self._envs.num_envs
        self._timestep = ActorOutput(
            action_tm1=np.zeros((num_envs,), dtype=np.int32),
            reward=np.zeros((num_envs,), dtype=np.float32),
            observation=self._envs.reset(),
            first=np.ones((num_envs,), dtype=np.float32),
            last=np.zeros((num_envs,), dtype=np.float32),
            truncated=np.zeros((num_envs,), dtype=np.float32),
            gt_mask=self._envs.get_action_influence(),
        )

    def initial_timestep(self):
        return self._timestep

    def step(
        self,
        rng_key: chex.PRNGKey,
        params: Params,
        random: bool,
        temperature: Optional[float] = None,
        is_causal_uct_step: bool = False,
    ):
        if random:
            action = np.array(
                [self._envs.action_space.sample() for _ in range(self._envs.num_envs)]
            )
        else:
            rng_key, action, agent_out, search_stats = self._agent_step(
                rng_key, params, jax.device_put(self._timestep), temperature, False, is_causal_uct_step
            )
            action = jax.device_get(action)
        observation, reward, done, info = self._envs.step(action)
        self._timestep = ActorOutput(
            action_tm1=action,
            reward=reward,
            observation=observation,
            first=self._timestep.last,  # If the previous timestep is the last, this is the first.
            last=done.astype(np.float32),
            truncated=np.array([i['truncated'] for i in info], dtype=np.float32),
            gt_mask=np.stack([i['action_influence'] for i in info]),
        )
        epinfos = []
        for i in info:
            maybeepinfo = i.get("episode")
            if maybeepinfo:
                epinfos.append(maybeepinfo)
        return rng_key, self._timestep, epinfos


class EvaluateActor(object):
    def __init__(self, envs, agent):
        self._envs = envs
        self._agent_step = jax.jit(agent.batch_step)

    def evaluate(self, rng_key: chex.PRNGKey, params, is_causal_uct_step) -> Tuple[chex.Array, List[dict], "EvalStats"]:
        num_envs = self._envs.num_envs
        next_timestep = ActorOutput(
            action_tm1=np.zeros((num_envs,), dtype=np.int32),
            reward=np.zeros((num_envs,), dtype=np.float32),
            observation=self._envs.reset(),
            first=np.ones((num_envs,), dtype=np.float32),
            last=np.zeros((num_envs,), dtype=np.float32),
            truncated=np.zeros((num_envs,), dtype=np.float32),
            gt_mask=self._envs.get_action_influence(),
        )
        epinfos = [None] * num_envs
        shd_sum = 0.0
        false_neg_sum = 0.0  # false negative
        count = 0
        step_count = 0
        tree_depth_sum = 0
        mean_nodes_depth_sum = 0
        # median_nodes_depth_sum = 0
        while count < num_envs:
            timestep = next_timestep
            rng_key, action, agent_out, search_stats = self._agent_step(
                rng_key, params, jax.device_put(timestep), 1.0, True, is_causal_uct_step
            )
            shd_sum += jax.jit(hamming_distance)(agent_out.mcts_mask, timestep.gt_mask)
            false_neg_sum += jax.jit(false_negative)(agent_out.mcts_mask, timestep.gt_mask)

            tree_depth_sum += search_stats.tree_depth.mean()
            mean_nodes_depth_sum += search_stats.mean_nodes_depth.mean()
            # median_nodes_depth_sum += search_stats.median_nodes_depth.mean()
            action = jax.device_get(action)
            observation, reward, done, info = self._envs.step(action)
            next_timestep = ActorOutput(
                action_tm1=action,
                reward=reward,
                observation=observation,
                first=timestep.last,  # If the previous timestep is the last, this is the first.
                last=done.astype(np.float32),
                truncated=np.array([i['truncated'] for i in info], dtype=np.float32),
                gt_mask=np.stack([i['action_influence'] for i in info]),
            )
            step_count += 1
            for k, i in enumerate(info):
                maybeepinfo = i.get("episode")
                if maybeepinfo and epinfos[k] is None:
                    epinfos[k] = maybeepinfo
                    count += 1
        eval_stats = EvalStats(
            mean_shd=shd_sum / step_count,
            mean_false_neg=false_neg_sum / step_count,
            mean_tree_depth=tree_depth_sum / step_count,
            mean_mean_nodes_depth=mean_nodes_depth_sum / step_count,
            # mean_median_nodes_depth=median_nodes_depth_sum / step_count,
            final_step_policy_info=PolicyInfo(
                act_prob=search_stats.act_prob,
                greedy_prob=search_stats.greedy_prob,
                prior_act_prob=search_stats.prior_act_prob,
            ),
            final_step_mask_info=MaskInfo(
                observations=timestep.observation,
                gt_masks=timestep.gt_mask,
                masks=agent_out.mask,
                mcts_masks=agent_out.mcts_mask,
                mask_logits=agent_out.mask_logits,
                )
            )
        return rng_key, epinfos, eval_stats


@chex.dataclass(frozen=True)
class MaskInfo:
    observations: chex.Array
    gt_masks: chex.Array
    masks: chex.Array
    mcts_masks: chex.Array
    mask_logits: chex.Array

@chex.dataclass(frozen=True)
class PolicyInfo:
    act_prob: chex.Array
    greedy_prob: chex.Array
    prior_act_prob: chex.Array

@chex.dataclass(frozen=True)
class EvalStats:
    mean_shd: float
    mean_false_neg: float
    mean_tree_depth: float
    mean_mean_nodes_depth: float  # Batch mean of mean nodes depth
    # mean_median_nodes_depth: float  # Batch mean of median nodes depth
    final_step_policy_info: PolicyInfo
    final_step_mask_info: MaskInfo 