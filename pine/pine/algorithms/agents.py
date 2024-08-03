from __future__ import annotations

import chex
import gym
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import rlax

from .haiku_nets import (
    EZPrediction,
    EZStateDecoder,
    EZStateEncoder,
    EZTransition,
    gumbel_sigmoid,
    straight_through_estimator,
)
from .types import ActorOutput, AgentOutput, Params, SearchStats, Tree
from .utils import (
    divide_act_prob_no_influence_actions_flat,
    factorize_action_jax,
    inv_value_transform,
    logits_to_scalar,
    marginalize_policy_dist_flat,
    product_action_influence_jax,
    scale_gradient,
    value_transform,
)


class Agent(object):
    """A MCTS agent."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
        num_bins: int,
        channels: int,
        use_v2: bool,
        output_init_scale: float,
        discount_factor: float,
        num_simulations: int,
        max_search_depth: int,
        mcts_c1: float,
        mcts_c2: float,
        alpha: float,
        exploration_prob: float,
        q_normalize_epsilon: float,
        child_select_epsilon: float,
        use_action_mask: bool,
        use_abst_policy_learning: bool,
        action_cardinalities: list[int],
        mask_temp: float,
        mask_thres: float,
        mcts_mask_thres: float,
        causal_uct_start_step: int,
        use_consistency_loss: bool,
        use_recon_loss: bool,
        n_frame_stack: int = 1,
    ):
        self._observation_space = observation_space
        self._action_space = action_space
        self._discount_factor = discount_factor
        self._num_simulations = num_simulations
        self._max_search_depth = (
            num_simulations if max_search_depth is None else max_search_depth
        )
        self._mcts_c1 = mcts_c1
        self._mcts_c2 = mcts_c2
        self._alpha = alpha
        self._exploration_prob = exploration_prob
        self._q_normalize_epsilon = q_normalize_epsilon
        self._child_select_epsilon = child_select_epsilon
        self._use_action_mask = use_action_mask
        self._use_abst_policy_learning = use_abst_policy_learning
        self._action_cardinalities = action_cardinalities
        self._num_action_variables = num_action_variables = len(action_cardinalities)
        self._mask_temp = mask_temp
        self._mask_thres = mask_thres
        self._mcts_mask_thres = mcts_mask_thres
        self._causal_uct_start_step = causal_uct_start_step
        self._use_consistency_loss = use_consistency_loss
        self._use_recon_loss = use_recon_loss
        self._n_frame_stack = n_frame_stack
        self.value_transform = value_transform
        self.inv_value_transform = inv_value_transform
        self._encode_fn = hk.without_apply_rng(
            hk.transform(
                lambda observations: EZStateEncoder(channels, use_v2)(observations)
            )
        )
        num_actions = self._action_space.n
        self._predict_fn = hk.without_apply_rng(
            hk.transform(
                lambda states: EZPrediction(
                    num_actions,
                    num_bins,
                    output_init_scale,
                    use_v2,
                    use_action_mask,
                    num_action_variables,
                )(states)
            )
        )
        self._transit_fn = hk.without_apply_rng(
            hk.transform(lambda action, state: EZTransition(use_v2)(action, state))
        )
        self._decode_fn = hk.without_apply_rng(
            hk.transform(
                lambda states: EZStateDecoder(
                    channels, use_v2, channels_out=3 * n_frame_stack
                )(states)
            )
        )

    def init(self, rng_key: chex.PRNGKey):
        encoder_key, prediction_key, decoder_key, transition_key = jax.random.split(
            rng_key, 4
        )
        dummy_observation = self._observation_space.sample()
        encoder_params = self._encode_fn.init(encoder_key, dummy_observation)
        dummy_state = self._encode_fn.apply(encoder_params, dummy_observation)
        prediction_params = self._predict_fn.init(prediction_key, dummy_state)
        decoder_params = self._decode_fn.init(decoder_key, dummy_state)
        dummy_action = jnp.zeros((sum(self._action_cardinalities),))
        transition_params = self._transit_fn.init(
            transition_key, dummy_action, dummy_state
        )
        params = Params(
            encoder=encoder_params,
            prediction=prediction_params,
            transition=transition_params,
            decoder=decoder_params,
        )
        return params

    def batch_step(
        self,
        rng_key: chex.PRNGKey,
        params: Params,
        timesteps: ActorOutput,
        temperature: float,
        is_eval: bool,
        is_causal_uct_step: bool,
    ):
        batch_size = timesteps.reward.shape[0]
        rng_key, step_key = jax.random.split(rng_key)
        step_keys = jax.random.split(step_key, batch_size)
        batch_root_step = jax.vmap(self._root_step, (0, None, 0, None, None, None))
        actions, agent_out, search_stats = batch_root_step(
            step_keys, params, timesteps, temperature, is_eval, is_causal_uct_step
        )
        return rng_key, actions, agent_out, search_stats

    def _root_step(
        self,
        rng_key: chex.PRNGKey,
        params: Params,
        timesteps: ActorOutput,
        temperature: float,
        is_eval: bool,
        is_causal_uct_step: bool,
    ):
        """The input `timesteps` is assumed to be [input_dim]."""
        trajectories = jax.tree_map(
            lambda t: t[None], timesteps
        )  # Add a dummy time dimension.
        unroll_key, search_key, sample_key, greedy_key = jax.random.split(rng_key, 4)
        agent_out = self.root_unroll(unroll_key, params, trajectories)
        agent_out = jax.tree_map(
            lambda t: t.squeeze(axis=0), agent_out
        )  # Squeeze the dummy time dimension.
        tree = self.mcts(search_key, params, agent_out, is_eval, is_causal_uct_step)
        act_prob = self.act_prob(
            tree.visit_count[0], temperature, is_causal_uct_step, tree.mcts_mask[0]
        )
        sampled_action = rlax.categorical_sample(sample_key, act_prob)
        visit_count = jax.lax.select(
            is_causal_uct_step,
            divide_act_prob_no_influence_actions_flat(
                tree.visit_count[0].astype(jnp.float32),
                tree.mcts_mask[0],
                action_cardinalities=np.array(self._action_cardinalities),
            ),
            tree.visit_count[0].astype(jnp.float32),
        )
        greedy_actions = (visit_count == visit_count.max()).astype(jnp.float32)
        greedy_prob = greedy_actions / greedy_actions.sum()
        greedy_action = rlax.categorical_sample(greedy_key, greedy_prob)
        # Choose the greedy action during evaluation.
        action = jax.lax.select(is_eval, greedy_action, sampled_action)
        search_stats = SearchStats(
            tree_depth=jnp.max(tree.depth) + 1,
            mean_nodes_depth=jnp.mean(tree.depth) + 1,
            # median_nodes_depth=jnp.median(tree.depth) + 1,
            act_prob=act_prob,
            greedy_prob=greedy_prob,
            prior_act_prob=jax.nn.softmax(agent_out.logits),
        )
        return action, agent_out, search_stats

    def root_unroll(
            self, rng_key: chex.PRNGKey, params: Params, trajectory: ActorOutput
        ):
        """
        Encode observations and predict policy, value, reward, and mask of the root node.

        Args:
            rng_key (chex.PRNGKey): The random number generator key.
            params (Params): The parameters for the agent.
            trajectory (ActorOutput): The trajectory of observations.

        Returns:
            AgentOutput: The output of the agent after unrolling the root node.
        """
        state = self._encode_fn.apply(params.encoder, trajectory.observation)  # [T, S]
        logits, reward_logits, value_logits, mask_logits = self._predict_fn.apply(
            params.prediction, state
        )
        reward = logits_to_scalar(reward_logits)
        reward = self.inv_value_transform(reward)
        value = logits_to_scalar(value_logits)
        value = self.inv_value_transform(value)
        mask_gumbel_sigmoid_probs = gumbel_sigmoid(
            rng_key, mask_logits, self._mask_temp
        )
        mask = straight_through_estimator(
            mask_gumbel_sigmoid_probs, mask_gumbel_sigmoid_probs > self._mask_thres
        )
        mcts_mask = jax.nn.sigmoid(mask_logits) > self._mcts_mask_thres

        return AgentOutput(
            state=state,
            logits=logits,
            reward_logits=reward_logits,
            reward=reward,
            value_logits=value_logits,
            value=value,
            mask_logits=mask_logits,
            mask=mask,
            mcts_mask=mcts_mask,
        )

    def transition_step(
        self,
        params: Params,
        state: chex.Array,
        action: chex.Array,
    ):
        """The input `state` and `action` are assumed to be [S] and []."""
        factored_action = factorize_action_jax(
            action, np.array(self._action_cardinalities)
        )
        one_hot_action = self._one_hot(factored_action)
        next_state = self._transit_fn.apply(params.transition, one_hot_action, state)
        return next_state

    def encode(
        self,
        params: Params,
        observation: chex.Array,
    ):
        return self._encode_fn.apply(params.encoder, observation)

    def decode(
        self,
        params: Params,
        state: chex.Array,
    ):
        return self._decode_fn.apply(params.decoder, state)

    def model_step(
        self,
        rng_key: chex.PRNGKey,
        params: Params,
        state: chex.Array,
        action: chex.Array,
        mask: chex.Array,
    ):
        """The input `state` and `action` are assumed to be [S] and []."""
        factored_action = factorize_action_jax(
            action, np.array(self._action_cardinalities)
        )
        one_hot_action = self._one_hot(factored_action)
        one_hot_action = self._mask_one_hot_action(one_hot_action, mask)
        next_state = self._transit_fn.apply(params.transition, one_hot_action, state)
        next_state = scale_gradient(next_state, 0.5)
        logits, reward_logits, value_logits, mask_logits = self._predict_fn.apply(
            params.prediction, next_state
        )
        reward = logits_to_scalar(reward_logits)
        reward = self.inv_value_transform(reward)
        value = logits_to_scalar(value_logits)
        value = self.inv_value_transform(value)
        mask_gumbel_sigmoid_probs = gumbel_sigmoid(
            rng_key, mask_logits, self._mask_temp
        )
        mask = straight_through_estimator(
            mask_gumbel_sigmoid_probs, mask_gumbel_sigmoid_probs > self._mask_thres
        )
        mcts_mask = jax.nn.sigmoid(mask_logits) > self._mcts_mask_thres
        return AgentOutput(
            state=next_state,
            logits=logits,
            reward_logits=reward_logits,
            reward=reward,
            value_logits=value_logits,
            value=value,
            mask_logits=mask_logits,
            mask=mask,
            mcts_mask=mcts_mask,
        )

    def _one_hot(self, factored_action):
        one_hot_action = jnp.empty((sum(self._action_cardinalities),))
        for i in range(len(self._action_cardinalities)):
            one_hot_action = one_hot_action.at[
                sum(self._action_cardinalities[:i]) : sum(
                    self._action_cardinalities[: i + 1]
                )
            ].set(jax.nn.one_hot(factored_action[i], self._action_cardinalities[i]))
        return one_hot_action

    def _mask_one_hot_action(self, one_hot_action, factored_mask):
        mask = jnp.repeat(factored_mask, np.array(self._action_cardinalities))
        return one_hot_action * mask

    def model_unroll(
        self,
        rng_key: chex.PRNGKey,
        params: Params,
        state: chex.Array,
        mask: chex.Array,
        action_sequence: chex.Array,
    ):
        """The input `state` and `action` are assumed to be [S] and [T]."""

        def fn(carry: chex.Array, action):
            rng_key, state, mask = carry
            next_rng_key, rng_key = jax.random.split(rng_key)
            factored_action = factorize_action_jax(
                action, np.array(self._action_cardinalities)
            )
            one_hot_action = self._one_hot(factored_action)
            one_hot_action = self._mask_one_hot_action(one_hot_action, mask)
            next_state = self._transit_fn.apply(
                params.transition, one_hot_action, state
            )
            next_state = scale_gradient(next_state, 0.5)
            logits, reward_logits, value_logits, mask_logits = self._predict_fn.apply(
                params.prediction, next_state
            )
            mask_gumbel_sigmoid_probs = gumbel_sigmoid(
                rng_key, mask_logits, self._mask_temp
            )
            next_mask = straight_through_estimator(
                mask_gumbel_sigmoid_probs,
                mask_gumbel_sigmoid_probs > self._mask_thres,
            )
            return (next_rng_key, next_state, next_mask), (
                next_state,
                logits,
                reward_logits,
                value_logits,
                mask_logits,
            )

        _, (
            state_sequence,
            logits,
            reward_logits,
            value_logits,
            mask_logits,
        ) = jax.lax.scan(fn, (rng_key, state, mask), action_sequence)
        reward = logits_to_scalar(reward_logits)
        reward = self.inv_value_transform(reward)
        value = logits_to_scalar(value_logits)
        value = self.inv_value_transform(value)
        mask_gumbel_sigmoid_probs = gumbel_sigmoid(
            rng_key, mask_logits, self._mask_temp
        )
        mask = straight_through_estimator(
            mask_gumbel_sigmoid_probs, mask_gumbel_sigmoid_probs > self._mask_thres
        )
        mcts_mask = jax.nn.sigmoid(mask_logits) > self._mcts_mask_thres
        return AgentOutput(
            state=state_sequence,
            logits=logits,
            reward_logits=reward_logits,
            reward=reward,
            value_logits=value_logits,
            value=value,
            mask_logits=mask_logits,
            mask=mask,
            mcts_mask=mcts_mask,
        )

    def init_tree(self, rng_key: chex.PRNGKey, root: AgentOutput, is_eval: bool):
        num_nodes = self._num_simulations + 1
        num_actions = self._action_space.n
        state = jnp.zeros((num_nodes,) + root.state.shape)
        logits = jnp.zeros((num_nodes,) + root.logits.shape)
        prob = jnp.zeros((num_nodes,) + root.logits.shape)
        reward_logits = jnp.zeros((num_nodes,) + root.reward_logits.shape)
        reward = jnp.zeros((num_nodes,) + root.reward.shape)
        value_logits = jnp.zeros((num_nodes,) + root.value_logits.shape)
        value = jnp.zeros((num_nodes,) + root.value.shape)
        mask_logits = jnp.zeros((num_nodes,) + root.mask_logits.shape)
        mask = jnp.zeros((num_nodes,) + root.mask.shape, dtype=jnp.bool_)
        mcts_mask = jnp.zeros((num_nodes,) + root.mcts_mask.shape, dtype=jnp.bool_)
        action_value = jnp.zeros((num_nodes, num_actions) + root.value.shape)
        depth = jnp.zeros((num_nodes,), dtype=jnp.int32)
        parent = jnp.zeros((num_nodes,), dtype=jnp.int32)
        parent_action = jnp.zeros((num_nodes,), dtype=jnp.int32)
        child = jnp.zeros((num_nodes, num_actions), dtype=jnp.int32)
        visit_count = jnp.zeros((num_nodes, num_actions), dtype=jnp.int32)
        state = state.at[0].set(root.state)
        logits = logits.at[0].set(root.logits)
        noise = jax.random.dirichlet(rng_key, jnp.full((num_actions,), self._alpha))
        # Do not apply the Dirichlet noise during evaluation.
        exploration_prob = jax.lax.select(is_eval, 0.0, self._exploration_prob)
        root_prob = (
            jax.nn.softmax(root.logits) * (1 - exploration_prob)
            + exploration_prob * noise
        )
        prob = prob.at[0].set(root_prob)
        reward_logits = reward_logits.at[0].set(root.reward_logits)
        reward = reward.at[0].set(root.reward)
        value_logits = value_logits.at[0].set(root.value_logits)
        value = value.at[0].set(root.value)
        mask_logits = mask_logits.at[0].set(root.mask_logits)
        mask = mask.at[0].set(root.mask)
        mcts_mask = mcts_mask.at[0].set(root.mcts_mask)

        parent = parent.at[0].set(-1)
        parent_action = parent_action.at[0].set(-1)
        tree = Tree(
            state=state,
            logits=logits,
            prob=prob,
            reward_logits=reward_logits,
            reward=reward,
            value_logits=value_logits,
            value=value,
            mask_logits=mask_logits,
            mask=mask,
            mcts_mask=mcts_mask,
            action_value=action_value,
            depth=depth,
            parent=parent,
            parent_action=parent_action,
            child=child,
            visit_count=visit_count,
        )
        return tree

    def mcts(
        self,
        rng_key: chex.PRNGKey,
        params: Params,
        root: AgentOutput,
        is_eval: bool,
        is_causal_uct_step: bool,
    ):
        num_actions = self._action_space.n
        max_search_depth = self._max_search_depth
        c1 = self._mcts_c1
        c2 = self._mcts_c2
        discount_factor = self._discount_factor

        def simulate(rng_key: chex.PRNGKey, tree: Tree):
            # First compute the minimum and the maximum action-value in the current tree.
            # Note that these statistics are hard to maintain incrementally because they are non-monotonic.
            is_valid = jnp.clip(tree.visit_count, 0, 1)
            action_value = tree.action_value
            q_min = jnp.min(
                jnp.where(is_valid, action_value, jnp.full_like(action_value, jnp.inf))
            )
            q_max = jnp.max(
                jnp.where(is_valid, action_value, jnp.full_like(action_value, -jnp.inf))
            )
            q_min = jax.lax.select(is_valid.sum() == 0, 0.0, q_min)
            q_max = jax.lax.select(is_valid.sum() == 0, 0.0, q_max)

            def _select_action(rng_key: chex.PRNGKey, t, q_mean):
                # Assign an estimated value to the unvisited nodes.
                # See Eq. (8) in https://arxiv.org/pdf/2111.00210.pdf
                # and https://github.com/YeWR/EfficientZero/blob/main/core/ctree/cnode.cpp#L96.
                q = action_value[t]
                q = jax.lax.select(tree.visit_count[t] > 0, q, jnp.full_like(q, q_mean))
                # Normalize the action-values of the current node so that they are in [0, 1].
                # This is required for the pUCT rule.
                # See Eq. (5) in https://www.nature.com/articles/s41586-020-03051-4.pdf
                q = (q - q_min) / jnp.maximum(q_max - q_min, self._q_normalize_epsilon)
                p = tree.prob[t]
                n = tree.visit_count[t]
                # The action scores are computed by the pUCT rule.
                # See Eq. (2) in https://www.nature.com/articles/s41586-020-03051-4.pdf.
                p = jax.lax.cond(
                    is_causal_uct_step,
                    lambda p: marginalize_policy_dist_flat(
                        p,
                        tree.mcts_mask[t],
                        action_cardinalities=np.array(self._action_cardinalities),
                    ),
                    lambda p: p,
                    p,
                )
                score = q + p * jnp.sqrt(n.sum()) / (1 + n) * (
                    c1 + jnp.log((n.sum() + c2 + 1) / c2)
                )

                # Mask out invalid actions.
                def mask_score(score):
                    mask = product_action_influence_jax(
                        tree.mcts_mask[t], np.array(self._action_cardinalities)
                    )
                    score = jnp.where(mask, score, jnp.full_like(score, -jnp.inf))
                    return score

                score = jax.lax.cond(
                    is_causal_uct_step,
                    lambda score: mask_score(score),
                    lambda score: score,
                    score,
                )
                best_actions = score >= score.max() - self._child_select_epsilon
                tie_breaking_prob = best_actions / best_actions.sum()
                return jax.random.choice(rng_key, num_actions, p=tie_breaking_prob)

            def _cond(loop_state):
                rng_key, p, a, q_mean = loop_state
                return jnp.logical_and(
                    tree.depth[p] + 1 < max_search_depth, tree.visit_count[p, a] > 0
                )

            def _body(loop_state):
                rng_key, p, a, q_mean = loop_state
                p = tree.child[p, a]
                is_valid_child = jnp.clip(tree.visit_count[p], 0, 1)
                q_mean = (q_mean + jnp.sum(tree.action_value[p] * is_valid_child)) / (
                    jnp.sum(is_valid_child) + 1
                )
                rng_key, sub_key = jax.random.split(rng_key)
                a = _select_action(sub_key, p, q_mean)
                return rng_key, p, a, q_mean

            is_valid_child = jnp.clip(tree.visit_count[0], 0, 1)
            q_mean = jnp.sum(tree.action_value[0] * is_valid_child) / jnp.maximum(
                jnp.sum(is_valid_child), 1
            )
            rng_key, sub_key = jax.random.split(rng_key)
            a = _select_action(sub_key, 0, q_mean)
            _, p, a, _ = jax.lax.while_loop(
                _cond,
                _body,
                (rng_key, 0, a, q_mean),
            )
            return p, a

        def expand(rng_key, tree: Tree, p, a, c):
            p_state = tree.state[p]
            p_mcts_mask = tree.mcts_mask[p]
            model_out = self.model_step(rng_key, params, p_state, a, p_mcts_mask)
            tree = tree._replace(
                state=tree.state.at[c].set(model_out.state),
                logits=tree.logits.at[c].set(model_out.logits),
                prob=tree.prob.at[c].set(jax.nn.softmax(model_out.logits)),
                reward_logits=tree.reward_logits.at[c].set(model_out.reward_logits),
                reward=tree.reward.at[c].set(model_out.reward),
                value_logits=tree.value_logits.at[c].set(model_out.value_logits),
                value=tree.value.at[c].set(model_out.value),
                mask_logits=tree.mask_logits.at[c].set(model_out.mask_logits),
                mask=tree.mask.at[c].set(model_out.mask),
                mcts_mask=tree.mcts_mask.at[c].set(model_out.mcts_mask),
                depth=tree.depth.at[c].set(tree.depth[p] + 1),
                parent=tree.parent.at[c].set(p),
                parent_action=tree.parent_action.at[c].set(a),
                child=tree.child.at[p, a].set(c),
            )
            return tree

        def backup(tree: Tree, c):
            def _update(tree, c, g):
                g = tree.reward[c] + discount_factor * g
                p = tree.parent[c]
                a = tree.parent_action[c]
                new_n = tree.visit_count[p, a] + 1
                new_q = (tree.action_value[p, a] * tree.visit_count[p, a] + g) / new_n
                tree = tree._replace(
                    visit_count=tree.visit_count.at[p, a].add(1),
                    action_value=tree.action_value.at[p, a].set(new_q),
                )
                return tree, p, g

            tree, _, _ = jax.lax.while_loop(
                lambda t: t[1] > 0,
                lambda t: _update(t[0], t[1], t[2]),
                (tree, c, tree.value[c]),
            )
            return tree

        def body_fn(sim, loop_state):
            rng_key, tree = loop_state
            rng_key, simulate_key, expand_key = jax.random.split(rng_key, 3)
            p, a = simulate(simulate_key, tree)
            c = sim + 1
            tree = expand(expand_key, tree, p, a, c)
            tree = backup(tree, c)
            return rng_key, tree

        rng_key, init_key = jax.random.split(rng_key)
        tree = self.init_tree(init_key, root, is_eval)
        rng_key, tree = jax.lax.fori_loop(
            0, self._num_simulations, body_fn, (rng_key, tree)
        )

        return tree

    def act_prob(
        self,
        visit_count: chex.Array,
        temperature: float,
        is_causal_uct_step: bool,
        mcts_mask: chex.Array,
    ):
        # , is_causal_uct_step: bool, mcts_mask: chex.Array):
        """Compute the final policy recommended by MCTS for acting."""
        unnormalized = jnp.power(visit_count, 1.0 / temperature)
        act_prob = unnormalized / unnormalized.sum(axis=-1, keepdims=True)
        act_prob = jax.lax.select(
            is_causal_uct_step,
            divide_act_prob_no_influence_actions_flat(
                act_prob,
                mcts_mask,
                action_cardinalities=np.array(self._action_cardinalities),
            ),
            act_prob,
        )
        return act_prob

    def value(
        self, value: chex.Array, action_value: chex.Array, visit_count: chex.Array
    ):
        """Compute the improved value estimation recommended by MCTS."""
        total_value = value + jnp.sum(action_value * visit_count, axis=-1)
        total_count = 1 + visit_count.sum(axis=-1)
        return total_value / total_count
