import functools

import chex
import jax
import jax.numpy as jnp
import numpy as np
import tree as tree_util


def uniform_sample_no_influence_actions_with_single_action(
    rng_key, action, mcts_mask, action_cardinalities
):
    action = jnp.array(factorize_action(action, action_cardinalities))
    action = uniform_sample_no_influence_actions(
        rng_key, action, mcts_mask, action_cardinalities
    )
    return product_action_jax(action, action_cardinalities)


def uniform_sample_no_influence_actions(
    rng_key, action, mcts_mask, action_cardinalities
):
    """
    Sample actions uniformly from the set of actions that have no influence.

    Examples:
        >>> uniform_sample_no_influence_actions(
                jnp.array([1, 2, 0, 1]),
                jnp.array([True, False, False, True])
            )
        array([1, 1, 2, 1])
        >>> uniform_sample_no_influence_actions(
                jnp.array([1, 2, 0, 1]),
                jnp.array([True, False, False, True])
            )
        array([1, 3, 1, 1])
    """
    rand_action = jax.random.randint(
        rng_key,
        [4],
        minval=np.array([0, 0, 0, 0]),
        maxval=np.array(action_cardinalities),
    )
    return jax.lax.select(mcts_mask, action, rand_action)


def with_flat_policy_dist(func, policy_dist_arg_idx=0):
    @functools.wraps(func)
    def wrapper(*args, action_cardinalities):
        flat_policy_dist = args[policy_dist_arg_idx]
        policy_dist_mat = flat_policy_dist.reshape(reversed(action_cardinalities)).T
        new_args = list(args)
        new_args[policy_dist_arg_idx] = policy_dist_mat
        return func(*new_args).T.flatten()

    return wrapper


def divide_act_prob_no_influence_actions(p, action_influence):
    """Divide the action probability distribution over the set of actions that have no influence.

    Examples:
        >>> divide_act_prob_no_influence_actions(
                jnp.array([0.4, 0.6, 0.0, 0.0]),
                jnp.array([True, False]),
                jnp.array([2, 2])
            )
        array([0.2, 0.3, 0.2, 0.3])
        >>> divide_act_prob_no_influence_actions(
                jnp.array([0.3, 0.0, 0.7, 0.0]),
                jnp.array([False, True]),
                jnp.array([2, 2])
            )
        array([0.15, 0.15, 0.35, 0.35])
        >>> divide_act_prob_no_influence_actions(
                jnp.array([0.3, 0.1, 0.6, 0.0]),
                jnp.array([True, True]),
                jnp.array([2, 2])
            )
        array([0.3, 0.1, 0.6, 0.0])
        >>> divide_act_prob_no_influence_actions(
                jnp.array([0.3, 0.1, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                jnp.array([True, True, False]),
                jnp.array([2, 2, 3])
            )
        array([])
    """

    def distribute_prob(p):
        distribute_mat = jnp.zeros((p.shape[-1], p.shape[-1]))
        distribute_mat = distribute_mat.at[0].set(1 / p.shape[-1])
        return p @ distribute_mat

    for i in range(len(action_influence)):
        p = jnp.transpose(p, [j for j in range(1, len(action_influence))] + [0])
        p = jax.lax.cond(
            action_influence[i],
            lambda p: p,
            distribute_prob,
            p,
        )
    return p


divide_act_prob_no_influence_actions_flat = with_flat_policy_dist(
    divide_act_prob_no_influence_actions
)


def marginalize_policy_dist(p, action_influence):
    """Marginalize the policy distribution over a subset of the action space.

    Args:
        p: The policy distribution
        action_influence: The action_influence over the action space

    Returns:
        The marginalized policy distribution

    Examples:
        >>> marginalize_policy_dist(
                jnp.array([
                    [0.1, 0.2],
                    [0.3, 0.4],
                ]),
                jnp.array([True, False])
            )
        array(
            [
                [0.3, 0.3],
                [0.7, 0.7],
            ]
        )
        >>> marginalize_policy_dist(
            jnp.array(
                [
                    [
                        [0.1, 0.05],
                        [0.05, 0.1],
                    ],
                    [
                        [0.3, 0.15],
                        [0.15, 0.1],
                    ]
                ]
            ),
            jnp.array([True, False, False])
        )
        array(
            [
                [
                    [0.3, 0.3],
                    [0.3, 0.3],
                ],
            ],
            [
                [
                    [0.7, 0.7],
                    [0.7, 0.7],
                ],
            ]
        )
        >>> marginalize_policy_dist(
            jnp.array(
                [
                    [
                        [0.1, 0.05],
                        [0.05, 0.1],
                    ],
                    [
                        [0.3, 0.15],
                        [0.15, 0.1],
                    ]
                ]
            ),
            jnp.array([False, True, False])
        )
        array(
            [
                [
                    [0.6, 0.6],
                    [0.4, 0.4],
                ],
            ],
            [
                [
                    [0.6, 0.6],
                    [0.4, 0.4],
                ],
            ]
        )
        >>> marginalize_policy_dist(
            jnp.array(
                [
                    [
                        [0.1, 0.05],
                        [0.05, 0.1],
                    ],
                    [
                        [0.3, 0.15],
                        [0.15, 0.1],
                    ]
                ]
            ),
            jnp.array([False, True, True])
        )
        array(
            [
                [
                    [0.4, 0.2],
                    [0.2, 0.2],
                ],
            ],
            [
                [
                    [0.4, 0.2],
                    [0.2, 0.2],
                ],
            ]
        )
    """
    for i in range(len(action_influence)):
        p = jnp.transpose(p, [j for j in range(1, len(action_influence))] + [0])
        p = jax.lax.cond(
            action_influence[i],
            lambda p: p,
            lambda p: p @ jnp.ones((p.shape[-1], p.shape[-1])),
            p,
        )
    return p


marginalize_policy_dist_flat = with_flat_policy_dist(marginalize_policy_dist)


def false_negative(pred_mask, gt_mask):
    chex.assert_rank([pred_mask, gt_mask], 2)
    return jnp.mean(jnp.sum(jnp.logical_and(pred_mask == 0, gt_mask == 1), axis=-1))


def hamming_distance(x: chex.Array, y: chex.Array):
    return (x != y).astype(float).sum(-1).mean()


def pack_namedtuple_jnp(xs, axis=0):
    return jax.tree_map(lambda *xs: jnp.stack(xs, axis=axis), *xs)


def pack_namedtuple_np(xs, axis=0):
    return tree_util.map_structure(lambda *xs: np.stack(xs, axis=axis), *xs)


def unpack_namedtuple_jnp(structure, axis=0):
    transposed = jax.tree_map(lambda t: jnp.moveaxis(t, axis, 0), structure)
    flat = jax.tree_flatten(transposed)
    unpacked = list(map(lambda xs: jax.tree_unflatten(structure, xs), zip(*flat)))
    return unpacked


def unpack_namedtuple_np(structure, axis=0):
    transposed = tree_util.map_structure(lambda t: np.moveaxis(t, axis, 0), structure)
    flat = tree_util.flatten(transposed)
    unpacked = list(map(lambda xs: tree_util.unflatten_as(structure, xs), zip(*flat)))
    return unpacked


def scale_gradient(g, scale: float):
    return g * scale + jax.lax.stop_gradient(g) * (1.0 - scale)


def weighted_mean(x: chex.Array, w: chex.Array):
    return jnp.sum(x * w) / jnp.maximum(w.sum(), 1.0)


def weighted_std(x: chex.Array, w: chex.Array):
    mean = weighted_mean(x, w)
    return jnp.sqrt(jnp.sum((x - mean) ** 2 * w) / jnp.maximum(w.sum(), 1.0))


def scalar_to_two_hot(x: chex.Array, num_bins: int):
    """A categorical representation of real values. Ref: https://www.nature.com/articles/s41586-020-03051-4.pdf."""
    max_val = (num_bins - 1) // 2
    x = jnp.clip(x, -max_val, max_val)
    x_low = jnp.floor(x).astype(jnp.int32)
    x_high = jnp.ceil(x).astype(jnp.int32)
    p_high = x - x_low
    p_low = 1.0 - p_high
    idx_low = x_low + max_val
    idx_high = x_high + max_val
    cat_low = jax.nn.one_hot(idx_low, num_bins) * p_low[..., None]
    cat_high = jax.nn.one_hot(idx_high, num_bins) * p_high[..., None]
    return cat_low + cat_high


def logits_to_scalar(logits: chex.Array):
    """The inverse of the scalar_to_two_hot function above."""
    num_bins = logits.shape[-1]
    max_val = (num_bins - 1) // 2
    x = jnp.sum((jnp.arange(num_bins) - max_val) * jax.nn.softmax(logits), axis=-1)
    return x


def value_transform(x: chex.Array, epsilon: float = 1e-3):
    """A non-linear value transformation for variance reduction. Ref: https://arxiv.org/abs/1805.11593."""
    return jnp.sign(x) * (jnp.sqrt(jnp.abs(x) + 1) - 1) + epsilon * x


def inv_value_transform(x: chex.Array, epsilon: float = 1e-3):
    """The inverse of the non-linear value transformation above."""
    return jnp.sign(x) * (
        ((jnp.sqrt(1 + 4 * epsilon * (jnp.abs(x) + 1 + epsilon)) - 1) / (2 * epsilon))
        ** 2
        - 1
    )


def factorize_action(action, cardinalities):
    """Factorize a single action into a factored action.

    Args:
        action: The single action
        cardinalities: The cardinalities of the factored action

    Returns:
        The factored action
    """
    factored_action = []
    for cardinality in cardinalities:
        factored_action.append(action % cardinality)
        action = action // cardinality
    return factored_action


def factorize_action_jax(action, cardinalities):
    return jax.lax.scan(jnp.divmod, action, cardinalities)[1]


def product_action(factored_action, cardinalities):
    """Convert a factored action into a single action.

    Args:
        factored_action: The factored action
        cardinalities: The cardinalities of the factored action

    Returns:
        The single action
    """
    action = 0
    for i, cardinality in enumerate(cardinalities):
        action += factored_action[i] * np.prod(cardinalities[:i], dtype=np.int32)
    return action


def product_action_jax(factored_action, cardinalities):
    def _body(carry, xs):
        action, prod_cardi = carry
        factored_action, cardinality = xs
        action = action + factored_action * prod_cardi
        prod_cardi = prod_cardi * cardinality
        return (action, prod_cardi), None

    return jax.lax.scan(_body, (0, 1), (factored_action, cardinalities))[0][0]


def product_action_influence(factored_action_influence, cardinalities):
    # Convert the factored action influence into a producted? action candidates
    # e.g.) action_influence=[1, 0, 0], factored_action_space=[2, 2, 2]
    # => [1, 1, 0, 0, 0, 0, 0, 0]
    # e.g.) action_influence=[0, 1, 0], factored_action_space=[2, 2, 2]
    # => [1, 0, 1, 0, 0, 0, 0, 0]
    # e.g.) action_influence=[0, 0, 1], factored_action_space=[2, 2, 2]
    # => [1, 0, 0, 0, 1, 0, 0, 0]
    # e.g.) action_influence=[1, 1, 1], factored_action_space=[2, 2, 2]
    # => [1, 1, 1, 1, 1, 1, 1, 1]
    # e.g.) action_influence=[0, 1, 1], factored_action_space=[2, 2, 2]
    # => [1, 0, 1, 0, 1, 0, 1, 0]

    action_influence = np.array(
        [False] * np.prod(cardinalities)
    )  # Initialize action candidates
    for idx in range(len(action_influence)):
        factored_action = factorize_action(idx, cardinalities)
        if all(
            [
                is_influencer or action_i == 0
                for is_influencer, action_i in zip(
                    factored_action_influence, factored_action
                )
            ]
        ):
            action_influence[idx] = True
    return action_influence


def product_action_influence_jax(factored_action_influence, cardinalities):
    def _body(idx):
        factored_action = factorize_action_jax(idx, cardinalities)

        def _is_valid(factored_action_influence_i, factored_action_i):
            return jnp.logical_or(factored_action_influence_i, factored_action_i == 0)

        return jnp.all(jax.vmap(_is_valid)(factored_action_influence, factored_action))

    range_ = np.arange(np.prod(cardinalities))
    action_influence = jax.vmap(_body)(range_)
    return action_influence


if __name__ == "__main__":
    cardinalities = [2, 2, 2]
    factored_action_influence = [1, 0, 1]
    action_influence = product_action_influence(
        factored_action_influence, cardinalities
    )
    print(action_influence)

    action = 5
    factored_action = factorize_action(action, cardinalities)
    print(factored_action)

    action = product_action(factored_action, cardinalities)
    print(action)

    action = product_action_jax(jnp.array(factored_action), jnp.array(cardinalities))
    print(action)

    action = 5
    factored_action = factorize_action_jax(action, jnp.array(cardinalities))
    print(factored_action)

    action_influence = product_action_influence_jax(
        jnp.array(factored_action_influence), jnp.array(cardinalities)
    )
    print(action_influence)
