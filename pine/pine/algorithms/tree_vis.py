import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import pygraphviz

from .types import Tree
from .utils import factorize_action


def convert_tree_to_graph(
    tree: Tree,
    cardinalities = [3, 2, 10, 10],
) -> pygraphviz.AGraph:
    num_nodes = tree.state.shape[0]
    num_actions = tree.child.shape[1]

    def node_to_str(node_i):
        return (
            f"{node_i}\n"
            f"Reward: {tree.reward[node_i]:.2f}\n"
            f"Value: {tree.value[node_i]:.2f}\n"
            f"Mask: {tree.mask[node_i]}\n"
            f"MCTS Mask: {tree.mcts_mask[node_i]}\n"
        )

    def edge_to_str(node_i, a_i):
        return (
            f"{factorize_action(a_i, cardinalities)}\n"
            f"Q: {tree.action_value[node_i, a_i]:.2f}\n"
            f"p: {tree.prob[node_i, a_i]:.2f}\n"
            f"Visits: {tree.visit_count[node_i, a_i]}\n"
        )

    graph = pygraphviz.AGraph(directed=True)

    # Add root
    graph.add_node(0, label=node_to_str(node_i=0), color="green")
    # Add all other nodes and connect them up.
    for node_i in range(num_nodes):
        for a_i in range(num_actions):
            # Index of children, or 0 if not expanded
            children_i = tree.child[node_i, a_i]
            if children_i > 0:
                graph.add_node(
                    children_i, label=node_to_str(node_i=children_i), color="red"
                )
                graph.add_edge(node_i, children_i, label=edge_to_str(node_i, a_i))

    return graph
