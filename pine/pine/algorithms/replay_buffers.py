from typing import Any
import time

import numpy as np
import tree

from .types import ActorOutput

class UniformBuffer(object):
    def __init__(self, min_size: int, max_size: int, traj_len: int):
        self._min_size = min_size
        self._max_size = max_size
        self._traj_len = traj_len
        self._timestep_storage = None
        self._n = 0
        self._idx = 0
        self.temp_buffer = None
        self._is_future_truncated = np.empty((self._max_size,), np.bool_)

    def extend(self, timesteps: Any):
        if self._timestep_storage is None:
            sample_timestep = tree.map_structure(lambda t: t[0], timesteps)
            self._timestep_storage = self._preallocate(sample_timestep)
        num_env = timesteps.observation.shape[0]
        if self.temp_buffer is None:
            self.temp_buffer = [[] for _ in range(num_env)]
        for i in range(num_env):
            self.temp_buffer[i].append(
                ActorOutput(
                    observation=timesteps.observation[i],
                    action_tm1=timesteps.action_tm1[i],
                    reward=timesteps.reward[i],
                    first=timesteps.first[i],
                    last=timesteps.last[i],
                    truncated=timesteps.truncated[i],
                    gt_mask=timesteps.gt_mask[i],
                )
            )
            if timesteps.last[i]:
                self.add_trajectory(self.temp_buffer[i], timesteps.truncated[i])
                self.temp_buffer[i] = []

    def add_trajectory(self, trajectory, truncated):
        # Todo: Find a better way to do this
        num_steps = 1
        for idx, timesteps in enumerate(trajectory):
            indices = np.arange(self._idx, self._idx + num_steps) % self._max_size
            tree.map_structure(
                lambda a, x: assign(a, indices, x), self._timestep_storage, timesteps
            )
            self._idx = (self._idx + num_steps) % self._max_size
            self._n = min(self._n + num_steps, self._max_size)
            self._is_future_truncated[indices] = True if truncated and idx >= len(trajectory) - self._traj_len - 1 else False

    def sample(self, batch_size: int):
        if batch_size + self._traj_len > self._n:
            return None
        valid_indices = np.where(~self._is_future_truncated[:self._n - self._traj_len])[0]
        start_indices = np.random.choice(
            valid_indices, batch_size, replace=False
        )
        all_indices = start_indices[:, None] + np.arange(self._traj_len + 1)[None]
        base_idx = 0 if self._n < self._max_size else self._idx
        all_indices = (all_indices + base_idx) % self._max_size
        trajectories = tree.map_structure(
            lambda a: a[all_indices], self._timestep_storage
        )
        return trajectories

    def full(self):
        return self._n == self._max_size

    def ready(self):
        return self._n >= self._min_size

    @property
    def size(self):
        return self._n

    def _preallocate(self, item):
        return tree.map_structure(
            lambda t: np.empty((self._max_size,) + t.shape, t.dtype), item
        )


def assign(a, i, x):
    a[i] = x
