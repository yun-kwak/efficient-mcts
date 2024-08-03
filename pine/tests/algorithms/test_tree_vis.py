import jax
import jax.numpy as jnp

from pine.algorithms.tree_vis import convert_tree_to_graph
from pine.algorithms.types import Tree

num_nodes = 3
num_actions = 2
bins = 601

state = jnp.zeros((num_nodes, 1))
logits = jnp.zeros((num_nodes,) + (num_actions,))
prob = jnp.zeros((num_nodes,) + (num_actions,))
reward_logits = jnp.zeros((num_nodes,) + (bins,))
reward = jnp.zeros((num_nodes,))
value_logits = jnp.zeros((num_nodes,) + (bins,))
value = jnp.zeros((num_nodes,))
mask_logits = jnp.zeros((num_nodes, 4))
mask = jnp.zeros((num_nodes, 4), dtype=jnp.bool_)
mcts_mask = jnp.zeros((num_nodes, 4), dtype=jnp.bool_)
action_value = jnp.zeros((num_nodes,) + (num_actions,))
depth = jnp.zeros((num_nodes,), dtype=jnp.int32)
parent = jnp.zeros((num_nodes,), dtype=jnp.int32)
parent_action = jnp.zeros((num_nodes,), dtype=jnp.int32)
child = jnp.zeros((num_nodes, num_actions), dtype=jnp.int32)
visit_count = jnp.zeros((num_nodes, num_actions), dtype=jnp.int32)


child = child.at[0, 0].set(1)
child = child.at[1, 1].set(2)
parent = parent.at[1].set(0)
parent = parent.at[2].set(1)
parent_action = parent_action.at[1].set(0)
parent_action = parent_action.at[2].set(1)
visit_count = visit_count.at[0, 0].set(1)
visit_count = visit_count.at[1, 1].set(1)
prob = prob.at[0, 0].set(0.5)
prob = prob.at[1, 1].set(0.7)


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


def test_convert_tree_to_graph():
    graph = convert_tree_to_graph(tree)
    # Save graph to file
    assert graph.number_of_nodes() == num_nodes
