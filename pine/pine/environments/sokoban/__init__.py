import gym
from ..minigrid import ProductFactoredActionWrapper
from ..utils import WarpFrame
import gym_sokoban

from ...vec_env.dummy_vec_env import DummyVecEnv
from ...vec_env.shmem_vec_env import ShmemVecEnv
from ...vec_env.monitor import Monitor

from .pull import *


def make_sokoban_vec_env(
    env_id,
    num_env,
    seed,
    env_kwargs=None,
    wrapper_kwargs=None,
    start_index=0,
    force_dummy=False,
):
    """
    Create a wrapped, monitored SubprocVecEnv for Sokoban.
    """
    wrapper_kwargs = wrapper_kwargs or {}
    seed = seed * 10000

    def make_thunk(rank):
        return lambda: make_sokoban_env(
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


def make_sokoban_env(
    env_id, subrank=0, seed=None, env_kwargs=None, wrapper_kwargs=None
):
    import warnings

    warnings.filterwarnings(
        action="ignore", category=DeprecationWarning, module=r".*gym"
    )
    del env_kwargs
    wrapper_kwargs = wrapper_kwargs or {}
    env = gym.make(env_id)
    env.seed(seed + subrank if seed is not None else None)
    env = ProductFactoredActionWrapper(env)
    env = Monitor(env, allow_early_resets=True)
    env = WarpFrame(env, 96, 96, grayscale=False)
    return env
