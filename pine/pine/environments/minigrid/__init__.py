from __future__ import annotations

import gym
import numpy as np
from gym_minigrid.minigrid import DIR_TO_VEC
from gym_minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper

from ...vec_env import DummyVecEnv, Monitor, ShmemVecEnv
from ..utils import WarpFrame
from .doorkey import *
from .empty import *


class FactorizeActionWrapper(gym.Wrapper):
    """Wrapper which factorizes action space into factored action space.

    Wrapper for the Unlock environment.
    Original action space is Discrete(7).

    0: turn left
    1: turn right
    2: move forward
    3: pick up an object
    4: drop
    5: toggle/activate an object
    6: done

    Factored action space is MultiDiscrete([3, 2, 3, 3]).
    Each action is a tuple of four integers, one for each action variable.
    0: turn left (1), turn right (2), do nothing (0)
    1: move forward (1), do nothing (0)
    2: pick up an object (1), drop (2), do nothing (0)
    3: toggle/activate an object (1), done (2), do nothing (0)
    """

    def __init__(self, env):
        """Initialize the wrapper.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)

        self.pick_key_cardinality = self.env.unwrapped.n_colors + 1  # +1 for no-op
        self.open_door_cardinality = self.env.unwrapped.n_colors + 1  # +1 for no-op
        self.action_space = gym.spaces.MultiDiscrete(
            [3, 2, self.pick_key_cardinality, self.open_door_cardinality]
        )

    def _to_single_action_sequence(self, factored_action: np.ndarray) -> list[int]:
        """Convert the factored action into a single action sequence.

        Variable temporal priority:
            1. Turn
            2. Move
            3. Pick up
            4. Toggle
        """

        assert factored_action.shape == (4,)
        turn, move, pick_up, toggle = factored_action
        # Todo: make sure the environment is deterministic
        # and fully observable when using this wrapper
        action_sequence: list[int] = []
        # Turn first
        if turn == 0:
            # no-op
            pass
        elif turn == 1:
            action_sequence.append(0)
        elif turn == 2:
            action_sequence.append(1)
        else:
            raise ValueError(f"Invalid turn action: {turn}")

        # Move
        if move == 0:
            # no-op
            pass
        elif move == 1:
            action_sequence.append(2)
        else:
            raise ValueError(f"Invalid move action: {move}")

        # Pick up
        if pick_up == 0:
            # no-op
            pass
        elif pick_up < self.pick_key_cardinality:
            if self.env.unwrapped.carrying:
                # no-op
                pass
                # c.f.) If there is no fwd_cell, then the agent will not drop the key.
            elif pick_up == self.env.unwrapped.color_act_idx:
                action_sequence.append(3)  # Pick up
            else:
                pass  # no-op
        else:
            raise ValueError(f"Invalid pick up action: {pick_up}")

        # Toggle
        if toggle == 0:
            # no-op
            pass
        elif toggle < self.open_door_cardinality:
            if toggle == self.env.unwrapped.color_act_idx:
                action_sequence.append(5)
            else:
                pass
        else:
            raise ValueError(f"Invalid toggle action: {toggle}")

        if len(action_sequence) == 0:
            # Do nothing
            # Todo: this is a hack, should be replaced with better no-op action
            action_sequence.append(6)

        return action_sequence

    def step(self, action):
        """Step through the environment with the factored action."""
        assert isinstance(action, (list, tuple, np.ndarray))
        action = np.array(action)
        # Convert the factored action into a single action
        action_sequence = self._to_single_action_sequence(action)
        prev_step_count = int(self.env.unwrapped.step_count)
        obs, cum_reward, done, info = None, 0.0, False, None
        for action in action_sequence:
            obs, reward, done, info = self.env.step(action)
            self.env.unwrapped.step_count = prev_step_count
            reward = float(reward)
            cum_reward = reward + cum_reward
            if done:
                break
        self.env.unwrapped.step_count = prev_step_count + 1
        return obs, cum_reward, done, info

    def reset(self, **kwargs):
        """Reset the environment."""
        self.env.unwrapped.step_count = 0
        return self.env.reset(**kwargs)


def _is_reachable(
    agent_pos: tuple[int, int], agent_dir: int, target_pos: tuple[int, int]
) -> bool:
    """Check if the target position is reachable from the agent position.

    Args:
        agent_pos: The agent position
        agent_dir: The agent direction
        target_pos: The target position

    Returns:
        True if the target position is reachable from the agent position
    """

    # If the the key is near enough (forward, left-forward, right-forward rechable),
    # then the agent can pick up the key
    # 'left' decreases the direction value. 'right' increases the direction value.
    # e.g.) dir=3, turn=right -> dir=0, turn=right -> dir=1, turn=left -> dir=0
    # Also, the direction value is modulo 4.
    # e.g.) pos=(1, 1), dir=1, move=forward => pos=(1, 2)
    # e.g.) pos=(1, 2), dir=3, move=forward => pos=(1, 1)
    fwd_dir_vec = DIR_TO_VEC[agent_dir]
    front_pos = agent_pos + fwd_dir_vec
    front_forward_pos = agent_pos + fwd_dir_vec + fwd_dir_vec
    left_dir_vec = DIR_TO_VEC[(agent_dir - 1) % 4]
    left_pos = agent_pos + left_dir_vec
    left_forward_pos = agent_pos + left_dir_vec + left_dir_vec
    right_dir_vec = DIR_TO_VEC[(agent_dir + 1) % 4]
    right_pos = agent_pos + right_dir_vec
    right_forward_pos = agent_pos + right_dir_vec + right_dir_vec

    rechable = any(
        np.all(target_pos == pos)
        for pos in [
            front_pos,
            left_pos,
            right_pos,
            front_forward_pos,
            left_forward_pos,
            right_forward_pos,
        ]
    )
    return rechable


def _action_influence(
    agent_pos: tuple[int, int],
    agent_dir: int,
    has_key: bool,
    key_pos_lst: list[tuple[int, int]],
    door_pos_lst: list[tuple[int, int]],
    ball_pos_lst: list[tuple[int, int]],
) -> tuple[bool, bool, bool, bool]:
    """Get the action influence information from the position information.

    Args:
        agent_pos: The agent position
        agent_dir: The agent direction
        key_pos_lst: The key position
        door_pos_lst: The door position
        ball_pos_lst: The ball position

    Returns:
        The action influence information
    """
    assert len(key_pos_lst) <= 1, "Multiple keys are not allowed for now"
    assert len(door_pos_lst) <= 1, "Multiple doors are not allowed for now"
    assert len(ball_pos_lst) <= 1, "Multiple balls are not allowed for now"
    # Todo: support multiple keys and doors correctly

    # The agent always can turn and move,
    # since we only consider the state for now.
    turn_influence = True
    move_influence = True
    key_influence = False
    door_influence = False
    ball_influence = False

    # if has_key:
    # The agent can drop the key
    # key_influence = True

    if key_pos_lst is not None:
        if any(_is_reachable(agent_pos, agent_dir, key_pos) for key_pos in key_pos_lst):
            # The agent can pick up the key
            key_influence = True

    if ball_pos_lst is not None:
        if any(
            _is_reachable(agent_pos, agent_dir, ball_pos) for ball_pos in ball_pos_lst
        ):
            # The agent can remove the ball
            ball_influence = True

    # If the door is near enough (forward, left-forward, right-forward rechable)
    # and the agent has the key, then the agent can open/close the door
    if door_pos_lst is not None:
        if has_key and any(
            _is_reachable(agent_pos, agent_dir, door_pos) for door_pos in door_pos_lst
        ):
            # The agent can open/close the door
            door_influence = True

    return turn_influence, move_influence, key_influence, door_influence, ball_influence


class StepRewardOverwriteWrapper(gym.Wrapper):
    def __init__(self, env, step_reward):
        super().__init__(env)
        assert step_reward < 0
        self.step_reward = step_reward

    def step(self, action):
        """Step through the environment."""
        state, reward, done, info = self.env.step(action)
        return state, self.step_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def get_action_influence(self):
        return self.env.get_action_influence()


class FactoredActionInfluenceInfoWrapper(gym.Wrapper):
    """Wrapper which adds factored action influence information to the info.

    Wrapper for the factored Unlock environment.
    Assumes that the environment is fully observable and the action space is
    factored.
    """

    def __init__(self, env):
        """Initialize the wrapper.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        assert isinstance(self.action_space, gym.spaces.MultiDiscrete)

    def _get_internal_info(self):
        info = {}
        env = self.env.unwrapped
        objects = [obj for obj in env.grid.grid if obj is not None]
        info["agent_pos"] = env.agent_pos
        info["agent_dir"] = env.agent_dir
        info["has_key"] = env.carrying is not None and env.carrying.type == "key"
        info["key_pos_lst"] = [obj.cur_pos for obj in objects if obj.type == "key"]
        info["door_pos_lst"] = [obj.cur_pos for obj in objects if obj.type == "door"]
        info["ball_pos_lst"] = [obj.cur_pos for obj in objects if obj.type == "ball"]
        return info

    def get_action_influence(self):
        """Get the action influence information.

        Args:
            obs: The observation

        Returns:
            The action influence information
        """
        info = self._get_internal_info()
        agent_pos = info["agent_pos"]
        agent_dir = info["agent_dir"]
        has_key = info["has_key"]
        key_pos_lst = info["key_pos_lst"]
        door_pos_lst = info["door_pos_lst"]
        ball_pos_lst = info["ball_pos_lst"]
        (
            turn_influence,
            move_influence,
            key_influence,
            door_influence,
            ball_influence,
        ) = _action_influence(
            agent_pos, agent_dir, has_key, key_pos_lst, door_pos_lst, ball_pos_lst
        )

        if self.env.unwrapped.blocked:
            return np.array(
                [
                    turn_influence,
                    move_influence,
                    key_influence,
                    door_influence,
                    ball_influence,
                ]
            )
        else:
            return np.array(
                [turn_influence, move_influence, key_influence, door_influence]
            )

    def step(self, action):
        """Step through the environment with the factored action."""
        obs, reward, done, info = self.env.step(action)
        info = {**info, **self._get_internal_info()}
        info["action_influence"] = self.get_action_influence()

        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset the environment."""
        return self.env.reset(**kwargs)


def make_minigrid_vec_env(
    env_id,
    num_env,
    seed,
    env_kwargs=None,
    wrapper_kwargs=None,
    start_index=0,
    force_dummy=False,
):
    """
    Create a wrapped, monitored SubprocVecEnv for MiniGrid.
    """
    wrapper_kwargs = wrapper_kwargs or {}
    seed = seed * 10000

    def make_thunk(rank):
        return lambda: make_minigrid_env(
            env_id=env_id,
            subrank=rank,
            seed=seed,
            env_kwargs=env_kwargs,
            wrapper_kwargs=wrapper_kwargs,
        )

    if not force_dummy and num_env > 1:
        return ShmemVecEnv([make_thunk(i + start_index) for i in range(num_env)])
    else:
        return DummyVecEnv([make_thunk(i + start_index) for i in range(num_env)])


def make_minigrid_env(
    env_id, subrank=0, seed=None, env_kwargs=None, wrapper_kwargs=None
):
    import warnings

    warnings.filterwarnings(
        action="ignore", category=DeprecationWarning, module=r".*gym"
    )
    del env_kwargs
    wrapper_kwargs = wrapper_kwargs or {}
    env = gym.make(env_id)
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)
    env.seed(seed + subrank if seed is not None else None)
    # env = FactorizeActionWrapper(env)
    env = FactoredActionInfluenceInfoWrapper(env)
    env = ProductFactoredActionWrapper(env)
    env = StepRewardOverwriteWrapper(env, -0.1)
    env = Monitor(env, allow_early_resets=True)
    env = WarpFrame(env, 96, 96, grayscale=False)
    return env


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


def product_action_influence(action_influence, cardinalities):
    # Convert the factored action mask into a single action mask
    # e.g.) action_influence=[0, 0, 1], factored_action_space=[2, 2, 3]
    # => [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    # e.g.) action_influence=[0, 1, 0], factored_action_space=[2, 3, 3]
    # => [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
    # e.g.) action_influence=[1, 0, 0], factored_action_space=[2, 3, 3]
    # => [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

    # convert action influence to action mask
    action_mask = np.array([False] * np.prod(cardinalities))
    for idx in range(len(action_mask)):
        factored_action = factorize_action(idx, cardinalities)
        if all(
            [
                is_influencer or action_i == 0
                for is_influencer, action_i in zip(action_influence, factored_action)
            ]
        ):
            action_mask[idx] = True
    return action_mask


class ScaleRewardWrapper(gym.Wrapper):
    """Wrapper which scales the reward."""

    def __init__(self, env, scale):
        """Scale the reward.

        Args:
            env: The environment to apply the wrapper
            scale: The scale factor
        """
        super().__init__(env)
        self.scale = scale

    def step(self, action):
        """Step through the environment."""
        state, reward, done, info = self.env.step(action)
        return state, self.scale * reward, done, info

    def reset(self, **kwargs):
        """Reset the environment."""
        return self.env.reset(**kwargs)


class ProductFactoredActionWrapper(gym.Wrapper):
    """Wrapper which produces cartesian product of factored action space."""

    def __init__(self, env):
        """Produce cartesian product of factored action space.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)

        assert isinstance(
            env.action_space, gym.spaces.MultiDiscrete
        ), "Action space must be MultiDiscrete."
        cardinalities = [space.n for space in env.action_space]
        assert all(
            cardinality > 1 for cardinality in cardinalities
        ), "All cardinalities must be greater than 1."
        self.action_space = gym.spaces.Discrete(np.prod(cardinalities))

    def _factorize_action(self, action):
        """Factorize a single action into a factored action.

        Args:
            action: The single action

        Returns:
            The factored action
        """
        cardinalities = [space.n for space in self.env.action_space]
        return factorize_action(action, cardinalities)

    def step(self, action):
        """Step through the environment with the single aciton."""
        factored_action = self._factorize_action(action)
        state, reward, done, info = self.env.step(factored_action)
        return state, reward, done, info

    def get_action_influence(self):
        return self.env.get_action_influence()

    def reset(self, **kwargs):
        """Reset the environment."""
        return self.env.reset(**kwargs)
