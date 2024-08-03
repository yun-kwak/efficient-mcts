"""Customized data structures."""
import collections

ActorOutput = collections.namedtuple(
    "ActorOutput",
    [
        "action_tm1",
        "reward",
        "observation",
        "first",
        "last",
        "truncated",
        "gt_mask",
    ],
)


AgentOutput = collections.namedtuple(
    "AgentOutput",
    [
        "state",
        "logits",
        "value_logits",
        "value",
        "reward_logits",
        "reward",
        "mask_logits",
        "mask",
        "mcts_mask",
    ],
)


Params = collections.namedtuple(
    "Params",
    [
        "encoder",
        "prediction",
        "transition",
        "decoder",
    ],
)


Tree = collections.namedtuple(
    "Tree",
    [
        "state",
        "logits",
        "prob",
        "reward_logits",
        "reward",
        "value_logits",
        "value",
        "mask_logits",
        "mask",
        "mcts_mask",
        "action_value",
        "depth",
        "parent",
        "parent_action",
        "child",
        "visit_count",
    ],
)

SearchStats = collections.namedtuple(
    "SearchStats",
    [
        "tree_depth",
        "mean_nodes_depth",
        "act_prob",
        "greedy_prob",
        "prior_act_prob",
    ],
)