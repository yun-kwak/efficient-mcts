import jax.numpy as jnp
import numpy as np
import pytest

from pine.algorithms.utils import (
    divide_act_prob_no_influence_actions,
    divide_act_prob_no_influence_actions_flat,
    factorize_action,
    factorize_action_jax,
    false_negative,
    marginalize_policy_dist,
    marginalize_policy_dist_flat,
    product_action,
    product_action_jax,
)


def test_false_negative():
    # Test case 1
    pred_mask = jnp.array([[0, 1, 0], [1, 0, 1]])
    gt_mask = jnp.array([[1, 0, 1], [0, 1, 0]])
    expected_output = 1.5
    assert np.isclose(false_negative(pred_mask, gt_mask), expected_output)

    # Test case 2
    pred_mask = jnp.array([[1, 0], [0, 1]])
    gt_mask = jnp.array([[1, 0], [1, 0]])
    expected_output = 0.5
    assert np.isclose(false_negative(pred_mask, gt_mask), expected_output)

    # Test case 3
    pred_mask = jnp.array([[0, 0], [0, 0]])
    gt_mask = jnp.array([[1, 1], [1, 1]])
    expected_output = 2.0
    assert np.isclose(false_negative(pred_mask, gt_mask), expected_output)

    # Test case 4
    pred_mask = jnp.array([[1, 1], [1, 1]])
    gt_mask = jnp.array([[0, 0], [0, 0]])
    expected_output = 0.0
    assert np.isclose(false_negative(pred_mask, gt_mask), expected_output)


@pytest.mark.parametrize(
    "p, action_influence, expected_result",
    [
        (
            jnp.array([[0.1, 0.2], [0.3, 0.4]]),
            jnp.array([True, False]),
            jnp.array([[0.3, 0.3], [0.7, 0.7]]),
        ),
        (
            jnp.array([[0.1, 0.05, 0.05], [0.3, 0.4, 0.1]]),
            jnp.array([False, True]),
            jnp.array([[0.4, 0.45, 0.15], [0.4, 0.45, 0.15]]),
        ),
        (
            jnp.array([[[0.1, 0.05], [0.05, 0.1]], [[0.3, 0.15], [0.15, 0.1]]]),
            jnp.array([True, False, False]),
            jnp.array([[[0.3, 0.3], [0.3, 0.3]], [[0.7, 0.7], [0.7, 0.7]]]),
        ),
        (
            jnp.array([[[0.1, 0.05], [0.05, 0.1]], [[0.3, 0.15], [0.15, 0.1]]]),
            jnp.array([False, True, False]),
            jnp.array([[[0.6, 0.6], [0.4, 0.4]], [[0.6, 0.6], [0.4, 0.4]]]),
        ),
        (
            jnp.array([[[0.1, 0.05], [0.05, 0.1]], [[0.3, 0.15], [0.15, 0.1]]]),
            jnp.array([False, False, False]),
            jnp.array([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]),
        ),
        (
            jnp.array(
                [
                    [
                        [0.1, 0.01, 0.03, 0.06],
                        [0.03, 0.02, 0.03, 0.04],
                    ],
                    [
                        [0.01, 0.01, 0.02, 0.05],
                        [0.12, 0.03, 0.02, 0.03],
                    ],
                    [
                        [0.01, 0.12, 0.05, 0.03],
                        [0.02, 0.01, 0.12, 0.03],
                    ],
                ]
            ),
            jnp.array([False, True, False]),
            jnp.array(
                [
                    [
                        [0.5, 0.5, 0.5, 0.5],
                        [0.5, 0.5, 0.5, 0.5],
                    ],
                    [
                        [0.5, 0.5, 0.5, 0.5],
                        [0.5, 0.5, 0.5, 0.5],
                    ],
                    [
                        [0.5, 0.5, 0.5, 0.5],
                        [0.5, 0.5, 0.5, 0.5],
                    ],
                ]
            ),
        ),
        (
            jnp.array(
                [
                    [
                        [0.1, 0.01, 0.03, 0.06],
                        [0.03, 0.02, 0.03, 0.04],
                    ],
                    [
                        [0.01, 0.01, 0.02, 0.05],
                        [0.12, 0.03, 0.02, 0.03],
                    ],
                    [
                        [0.01, 0.12, 0.05, 0.03],
                        [0.02, 0.01, 0.12, 0.03],
                    ],
                ]
            ),
            jnp.array([False, True, True]),
            jnp.array(
                [
                    [
                        [0.12, 0.14, 0.1, 0.14],
                        [0.17, 0.06, 0.17, 0.1],
                    ],
                    [
                        [0.12, 0.14, 0.1, 0.14],
                        [0.17, 0.06, 0.17, 0.1],
                    ],
                    [
                        [0.12, 0.14, 0.1, 0.14],
                        [0.17, 0.06, 0.17, 0.1],
                    ],
                ]
            ),
        ),
        (
            jnp.array(
                [
                    [
                        [0.1, 0.01, 0.03, 0.06],
                        [0.03, 0.02, 0.03, 0.04],
                    ],
                    [
                        [0.01, 0.01, 0.02, 0.05],
                        [0.12, 0.03, 0.02, 0.03],
                    ],
                    [
                        [0.01, 0.12, 0.05, 0.03],
                        [0.02, 0.01, 0.12, 0.03],
                    ],
                ]
            ),
            jnp.array([True, False, True]),
            jnp.array(
                [
                    [
                        [0.13, 0.03, 0.06, 0.1],
                        [0.13, 0.03, 0.06, 0.1],
                    ],
                    [
                        [0.13, 0.04, 0.04, 0.08],
                        [0.13, 0.04, 0.04, 0.08],
                    ],
                    [
                        [0.03, 0.13, 0.17, 0.06],
                        [0.03, 0.13, 0.17, 0.06],
                    ],
                ]
            ),
        ),
    ],
)
def test_marginalize_policy_dist(p, action_influence, expected_result):
    assert jnp.allclose(marginalize_policy_dist(p, action_influence), expected_result)


@pytest.mark.parametrize(
    "p, action_influence, action_cardinalities, expected_result",
    [
        (
            jnp.array([0.1, 0.3, 0.2, 0.4]),
            jnp.array([True, False]),
            jnp.array([2, 2]),
            jnp.array([0.3, 0.7, 0.3, 0.7]),
        ),
        (
            jnp.array([0.1, 0.3, 0.05, 0.4, 0.05, 0.1]),
            jnp.array([False, True]),
            jnp.array([2, 3]),
            jnp.array([0.4, 0.4, 0.45, 0.45, 0.15, 0.15]),
        ),
        (
            jnp.array([0.1, 0.3, 0.05, 0.15, 0.05, 0.15, 0.1, 0.1]),
            jnp.array([True, False, False]),
            jnp.array([2, 2, 2]),
            jnp.array([0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7]),
        ),
        (
            jnp.array([0.1, 0.3, 0.05, 0.15, 0.05, 0.15, 0.1, 0.1]),
            jnp.array([False, True, False]),
            jnp.array([2, 2, 2]),
            jnp.array([0.6, 0.6, 0.4, 0.4, 0.6, 0.6, 0.4, 0.4]),
        ),
        (
            jnp.array([[[0.1, 0.05], [0.05, 0.1]], [[0.3, 0.15], [0.15, 0.1]]]),
            jnp.array([False, False, False]),
            jnp.array([2, 2, 2]),
            jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        ),
        (
            jnp.array(
                [
                    0.1,
                    0.01,
                    0.01,
                    0.03,
                    0.12,
                    0.02,
                    0.01,
                    0.01,
                    0.12,
                    0.02,
                    0.03,
                    0.01,
                    0.03,
                    0.02,
                    0.05,
                    0.03,
                    0.02,
                    0.12,
                    0.06,
                    0.05,
                    0.03,
                    0.04,
                    0.03,
                    0.03,
                ]
            ),
            jnp.array([False, True, False]),
            jnp.array([3, 2, 4]),
            jnp.array(
                [
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                ]
            ),
        ),
        (
            jnp.array(
                [
                    0.1,
                    0.01,
                    0.01,
                    0.03,
                    0.12,
                    0.02,
                    0.01,
                    0.01,
                    0.12,
                    0.02,
                    0.03,
                    0.01,
                    0.03,
                    0.02,
                    0.05,
                    0.03,
                    0.02,
                    0.12,
                    0.06,
                    0.05,
                    0.03,
                    0.04,
                    0.03,
                    0.03,
                ]
            ),
            jnp.array([False, True, True]),
            jnp.array([3, 2, 4]),
            jnp.array(
                [
                    0.12,
                    0.12,
                    0.12,
                    0.17,
                    0.17,
                    0.17,
                    0.14,
                    0.14,
                    0.14,
                    0.06,
                    0.06,
                    0.06,
                    0.1,
                    0.1,
                    0.1,
                    0.17,
                    0.17,
                    0.17,
                    0.14,
                    0.14,
                    0.14,
                    0.1,
                    0.1,
                    0.1,
                ]
            ),
        ),
        (
            jnp.array(
                [
                    0.1,
                    0.01,
                    0.01,
                    0.03,
                    0.12,
                    0.02,
                    0.01,
                    0.01,
                    0.12,
                    0.02,
                    0.03,
                    0.01,
                    0.03,
                    0.02,
                    0.05,
                    0.03,
                    0.02,
                    0.12,
                    0.06,
                    0.05,
                    0.03,
                    0.04,
                    0.03,
                    0.03,
                ]
            ),
            jnp.array([True, False, True]),
            jnp.array([3, 2, 4]),
            jnp.array(
                [
                    0.13,
                    0.13,
                    0.03,
                    0.13,
                    0.13,
                    0.03,
                    0.03,
                    0.04,
                    0.13,
                    0.03,
                    0.04,
                    0.13,
                    0.06,
                    0.04,
                    0.17,
                    0.06,
                    0.04,
                    0.17,
                    0.1,
                    0.08,
                    0.06,
                    0.1,
                    0.08,
                    0.06,
                ]
            ),
        ),
    ],
)
def test_marginalize_policy_dist_flat(
    p, action_influence, action_cardinalities, expected_result
):
    assert jnp.allclose(
        marginalize_policy_dist_flat(
            p, action_influence, action_cardinalities=action_cardinalities
        ),
        expected_result,
    )


@pytest.mark.parametrize(
    "p, action_influence, expected_result",
    [
        (
            jnp.array([[0.4, 0.0], [0.6, 0.0]]),
            jnp.array([True, False]),
            jnp.array([[0.2, 0.2], [0.3, 0.3]]),
        ),
        (
            jnp.array(
                [
                    [
                        [0.2, 0.0, 0.0, 0.0],
                        [0.8, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                ]
            ),
            jnp.array([False, True, False]),
            jnp.array(
                [
                    [
                        [0.2 / 12, 0.2 / 12, 0.2 / 12, 0.2 / 12],
                        [0.8 / 12, 0.8 / 12, 0.8 / 12, 0.8 / 12],
                    ],
                    [
                        [0.2 / 12, 0.2 / 12, 0.2 / 12, 0.2 / 12],
                        [0.8 / 12, 0.8 / 12, 0.8 / 12, 0.8 / 12],
                    ],
                    [
                        [0.2 / 12, 0.2 / 12, 0.2 / 12, 0.2 / 12],
                        [0.8 / 12, 0.8 / 12, 0.8 / 12, 0.8 / 12],
                    ],
                ]
            ),
        ),
        (
            jnp.array(
                [
                    [
                        [0.2, 0.0, 0.0, 0.0],
                        [0.8, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.5, 0.0, 0.0, 0.0],
                        [0.5, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.3, 0.0, 0.0, 0.0],
                        [0.1, 0.0, 0.0, 0.0],
                    ],
                ]
            ),
            jnp.array([False, True, False]),
            jnp.array(
                [
                    [
                        [0.2 / 12, 0.2 / 12, 0.2 / 12, 0.2 / 12],
                        [0.8 / 12, 0.8 / 12, 0.8 / 12, 0.8 / 12],
                    ],
                    [
                        [0.2 / 12, 0.2 / 12, 0.2 / 12, 0.2 / 12],
                        [0.8 / 12, 0.8 / 12, 0.8 / 12, 0.8 / 12],
                    ],
                    [
                        [0.2 / 12, 0.2 / 12, 0.2 / 12, 0.2 / 12],
                        [0.8 / 12, 0.8 / 12, 0.8 / 12, 0.8 / 12],
                    ],
                ]
            ),
        ),
        (
            jnp.array(
                [
                    [
                        [0.1, 0.03, 0.4, 0.02],
                        [0.2, 0.04, 0.2, 0.01],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                ]
            ),
            jnp.array([False, True, True]),
            jnp.array(
                [
                    [
                        [0.1 / 3, 0.03 / 3, 0.4 / 3, 0.02 / 3],
                        [0.2 / 3, 0.04 / 3, 0.2 / 3, 0.01 / 3],
                    ],
                    [
                        [0.1 / 3, 0.03 / 3, 0.4 / 3, 0.02 / 3],
                        [0.2 / 3, 0.04 / 3, 0.2 / 3, 0.01 / 3],
                    ],
                    [
                        [0.1 / 3, 0.03 / 3, 0.4 / 3, 0.02 / 3],
                        [0.2 / 3, 0.04 / 3, 0.2 / 3, 0.01 / 3],
                    ],
                ]
            ),
        ),
        (
            # for greedy prob
            jnp.array(
                [
                    [
                        [1, 3, 4, 2],
                        [2, 4, 2, 1],
                    ],
                    [
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                    [
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                ],
                dtype=jnp.float32,
            ),
            jnp.array([False, True, True]),
            jnp.array(
                [
                    [
                        [1 / 3, 3 / 3, 4 / 3, 2 / 3],
                        [2 / 3, 4 / 3, 2 / 3, 1 / 3],
                    ],
                    [
                        [1 / 3, 3 / 3, 4 / 3, 2 / 3],
                        [2 / 3, 4 / 3, 2 / 3, 1 / 3],
                    ],
                    [
                        [1 / 3, 3 / 3, 4 / 3, 2 / 3],
                        [2 / 3, 4 / 3, 2 / 3, 1 / 3],
                    ],
                ]
            ),
        ),
    ],
)
def test_divide_act_prob_no_influence_actions(p, action_influence, expected_result):
    assert jnp.allclose(
        divide_act_prob_no_influence_actions(p, action_influence), expected_result
    )


@pytest.mark.parametrize(
    "p, action_influence, cardinalities, expected_result",
    [
        (
            jnp.array([0.4, 0.6, 0.0, 0.0]),
            jnp.array([True, False]),
            jnp.array([2, 2]),
            jnp.array([0.2, 0.3, 0.2, 0.3]),
        ),
    ],
)
def test_divide_act_prob_no_influence_actions_flat(
    p, action_influence, cardinalities, expected_result
):
    assert jnp.allclose(
        divide_act_prob_no_influence_actions_flat(
            p, action_influence, action_cardinalities=cardinalities
        ),
        expected_result,
    )


@pytest.mark.parametrize(
    "action, cardinalities, expected_result",
    [
        (0, [2, 2, 3], [0, 0, 0]),
        (1, [2, 2, 3], [1, 0, 0]),
        (2, [2, 2, 3], [0, 1, 0]),
        (3, [2, 2, 3], [1, 1, 0]),
        (4, [2, 2, 3], [0, 0, 1]),
        (8, [2, 2, 3], [0, 0, 2]),
        (11, [2, 2, 3], [1, 1, 2]),
        (0, [3, 2, 3, 3], [0, 0, 0, 0]),
        (2, [3, 2, 3, 3], [2, 0, 0, 0]),
        (11, [3, 2, 3, 3], [2, 1, 1, 0]),
        (53, [3, 2, 3, 3], [2, 1, 2, 2]),
        (0, [3, 2, 7, 7], [0, 0, 0, 0]),
        (2, [3, 2, 7, 7], [2, 0, 0, 0]),
        (11, [3, 2, 7, 7], [2, 1, 1, 0]),
        (293, [3, 2, 7, 7], [2, 1, 6, 6]),
        (0, [3, 2, 3, 3, 2], [0, 0, 0, 0, 0]),
    ],
)
def test_factorize_action(action, cardinalities, expected_result):
    assert factorize_action(action, cardinalities) == expected_result
    assert (
        factorize_action_jax(jnp.array(action), np.array(cardinalities)).tolist()
        == expected_result
    )


@pytest.mark.parametrize(
    "factored_action, cardinalities, expected_result",
    [
        ([0, 0, 0], [2, 2, 3], 0),
        ([1, 0, 0], [2, 2, 3], 1),
        ([0, 1, 0], [2, 2, 3], 2),
        ([1, 1, 0], [2, 2, 3], 3),
        ([0, 0, 1], [2, 2, 3], 4),
        ([0, 0, 2], [2, 2, 3], 8),
        ([1, 1, 2], [2, 2, 3], 11),
        ([0, 0, 0, 0], [3, 2, 3, 3], 0),
        ([2, 0, 0, 0], [3, 2, 3, 3], 2),
        ([2, 1, 1, 0], [3, 2, 3, 3], 11),
        ([2, 1, 2, 2], [3, 2, 3, 3], 53),
        ([0, 0, 0, 0], [3, 2, 7, 7], 0),
        ([2, 0, 0, 0], [3, 2, 7, 7], 2),
        ([2, 1, 1, 0], [3, 2, 7, 7], 11),
        ([2, 1, 6, 6], [3, 2, 7, 7], 293),
        ([0, 0, 0, 0, 0], [3, 2, 3, 3, 2], 0),
    ],
)
def test_product_action(factored_action, cardinalities, expected_result):
    assert product_action(factored_action, cardinalities) == expected_result
    assert (
        product_action_jax(jnp.array(factored_action), np.array(cardinalities)).tolist()
        == expected_result
    )
