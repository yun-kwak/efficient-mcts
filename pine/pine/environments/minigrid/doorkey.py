from enum import IntEnum
import random

import numpy as np
import gym
from gym.envs.registration import register
from gym_minigrid.minigrid import (
    IDX_TO_COLOR,
    Door,
    Goal,
    Grid,
    Ball,
    Key,
    MiniGridEnv,
    MissionSpace,
)


class ColorDoorKeyEnv(MiniGridEnv):
    """
    Environment with a door and key, sparse reward
    """

    class Turn(IntEnum):
        no_turn = 0
        left = 1
        right = 2

    class Move(IntEnum):
        no_move = 0
        forward = 1

    def __init__(self, size=8, n_colors=6, n_key_placed=1,
                 blocked=False, n_ball_colors=None, **kwargs):
        if "max_steps" not in kwargs:
            kwargs["max_steps"] = 10 * size * size
        mission_space = MissionSpace(
            mission_func=lambda: "use the key to open the door and then get to the goal"
        )
        self.color_act_idx = None
        self.ball_color_act_idx = None
        self.n_colors = n_colors
        self.n_key_placed = n_key_placed
        self.blocked = blocked
        self.n_ball_colors = n_colors if n_ball_colors is None else n_ball_colors
        super().__init__(mission_space=mission_space, grid_size=size, **kwargs)

        if self.blocked:
            self.action_space = gym.spaces.MultiDiscrete(
                [3, 2, self.n_colors + 1, self.n_colors + 1, self.n_ball_colors + 1]  # + 1 for no-op
            )
        else:
            self.action_space = gym.spaces.MultiDiscrete(
                [3, 2, self.n_colors + 1, self.n_colors + 1]  # + 1 for no-op
            )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the corner
        # Sample a goal position (0, 1, 2, 3)
        goal_pos = self._rand_int(0, 4)
        if goal_pos == 0:
            self.put_obj(Goal(), 1, 1)
        elif goal_pos == 1:
            self.put_obj(Goal(), width - 2, 1)
        elif goal_pos == 2:
            self.put_obj(Goal(), 1, height - 2)
        elif goal_pos == 3:
            self.put_obj(Goal(), width - 2, height - 2)
        else:
            raise ValueError("Invalid goal position")

        # Create a vertical splitting wall
        splitIdx = self._rand_int(2, width - 2)
        self.grid.vert_wall(splitIdx, 0)

        # Generate a random color
        color_idx = self._rand_int(0, self.n_colors)
        color = IDX_TO_COLOR[color_idx]
        other_color_idxes = random.sample(
            [i for i in range(self.n_colors) if i != color_idx], self.n_key_placed - 1
        )
        self.color_act_idx = color_idx + 1

        # Place a door in the wall
        doorIdx = self._rand_int(1, width - 2)
        self.put_obj(Door(color, is_locked=True), splitIdx, doorIdx)

        # Place a ball in front of the door
        ball_color_idx = self._rand_int(0, self.n_ball_colors)
        ball_color = IDX_TO_COLOR[ball_color_idx]
        self.ball_color_act_idx = ball_color_idx + 1
        if self.blocked:
            if goal_pos == 1 or goal_pos == 3:
                self.put_obj(Ball(ball_color), splitIdx - 1, doorIdx)
            else:
                self.put_obj(Ball(ball_color), splitIdx + 1, doorIdx)

        # Place a color key on the opposite side of the splitting wall from the goal
        if goal_pos == 1 or goal_pos == 3:
            self.place_obj(obj=Key(color), top=(0, 0), size=(splitIdx, height))
            for i in range(self.n_key_placed - 1):
                self.place_obj(
                    obj=Key(IDX_TO_COLOR[other_color_idxes[i]]),
                    top=(splitIdx, 0),
                    size=(width - splitIdx, height),
                )
        else:
            self.place_obj(
                obj=Key(color), top=(splitIdx, 0), size=(width - splitIdx, height)
            )
            for i in range(self.n_key_placed - 1):
                self.place_obj(
                    obj=Key(IDX_TO_COLOR[other_color_idxes[i]]),
                    top=(0, 0),
                    size=(splitIdx, height),
                )

        # Place the agent at a random position and orientation
        # on the opposite side of the splitting wall from the goal
        if goal_pos == 1 or goal_pos == 3:
            self.place_agent(top=(0, 0), size=(splitIdx, height))
        else:
            self.place_agent(top=(splitIdx, 0), size=(width - splitIdx, height))

        self.mission = "use the key to open the door and then get to the goal"

    def step(self, action):
        self.step_count += 1
        if self.blocked:
            turn, move, pick_up_key, open_door, pick_up_ball = action
        else:
            turn, move, pick_up_key, open_door = action

        reward = 0
        done = False
        truncated = False

        # Turn first
        # Rotate left
        if turn == self.Turn.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif turn == self.Turn.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Move forward
        if move == self.Move.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell is not None and fwd_cell.type == "goal":
                done = True
                reward = self._reward()
            if fwd_cell is not None and fwd_cell.type == "lava":
                done = True

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Pick up an key
        if pick_up_key != 0:
            if pick_up_key == self.color_act_idx:
                if fwd_cell and fwd_cell.can_pickup() and fwd_cell.type == "key":
                    if self.carrying is None:
                        self.carrying = fwd_cell
                        self.carrying.cur_pos = np.array([-1, -1])
                        self.grid.set(*fwd_pos, None)

        # Toggle/activate an object
        if open_door != 0:
            if open_door == self.color_act_idx:
                if fwd_cell:
                    fwd_cell.toggle(self, fwd_pos)

        # Pick up a ball
        if self.blocked:
            if pick_up_ball != 0:
                if pick_up_ball == self.ball_color_act_idx:
                    if fwd_cell and fwd_cell.can_pickup() and fwd_cell.type == "ball":
                        fwd_cell.cur_pos = np.array([-1, -1])
                        self.grid.set(*fwd_pos, None)

        if self.step_count >= self.max_steps:
            done = True
            truncated = True

        obs = self.gen_obs()

        return obs, reward, done, {"truncated": truncated}


class ColorDoorKeyRandomGoalEnv8x8(ColorDoorKeyEnv):
    def __init__(self):
        super().__init__(size=8)


class ColorDoorKeyRandomGoalEnv12x12(ColorDoorKeyEnv):
    def __init__(self):
        super().__init__(size=12)


class ColorDoorKeyRandomGoalEnv16x16(ColorDoorKeyEnv):
    def __init__(self):
        super().__init__(size=16)


class ColorDoorKeyRandomGoalEnv6x6C2(ColorDoorKeyEnv):
    def __init__(self):
        super().__init__(size=6, n_colors=2)


class ColorDoorKeyRandomGoalEnv8x8C2(ColorDoorKeyEnv):
    def __init__(self):
        super().__init__(size=8, n_colors=2)


class ColorDoorKeyRandomGoalEnv12x12C2(ColorDoorKeyEnv):
    def __init__(self):
        super().__init__(size=12, n_colors=2)


class ColorDoorKeyRandomGoalEnv16x16C2(ColorDoorKeyEnv):
    def __init__(self):
        super().__init__(size=16, n_colors=2)


class ColorDoorKeyRandomGoalEnv6x6C3(ColorDoorKeyEnv):
    def __init__(self):
        super().__init__(size=6, n_colors=3)


class ColorDoorKeyRandomGoalEnv8x8C3(ColorDoorKeyEnv):
    def __init__(self):
        super().__init__(size=8, n_colors=3)


class ColorDoorKeyRandomGoalEnv12x12C3(ColorDoorKeyEnv):
    def __init__(self):
        super().__init__(size=12, n_colors=3)


class ColorDoorKeyRandomGoalEnv16x16C3(ColorDoorKeyEnv):
    def __init__(self):
        super().__init__(size=16, n_colors=3)


class ColorDoorKeyRandomGoalEnv6x6C5(ColorDoorKeyEnv):
    def __init__(self):
        super().__init__(size=6, n_colors=5)


class ColorDoorKeyRandomGoalEnv8x8C5(ColorDoorKeyEnv):
    def __init__(self):
        super().__init__(size=8, n_colors=5)


class ColorDoorKeyRandomGoalEnv12x12C4(ColorDoorKeyEnv):
    def __init__(self):
        super().__init__(size=12, n_colors=4)


class ColorDoorKeyRandomGoalEnv12x12C5(ColorDoorKeyEnv):
    def __init__(self):
        super().__init__(size=12, n_colors=5)


class ColorDoorKeyRandomGoalEnv16x16C5(ColorDoorKeyEnv):
    def __init__(self):
        super().__init__(size=16, n_colors=5)


class BlockedColorDoorKeyRandomGoalEnv6x6C5B5(ColorDoorKeyEnv):
    def __init__(self):
        super().__init__(size=6, n_colors=5, blocked=True, n_ball_colors=5)



class BlockedColorDoorKeyRandomGoalEnv8x8C5B5(ColorDoorKeyEnv):
    def __init__(self):
        super().__init__(size=8, n_colors=5, blocked=True, n_ball_colors=5)



register(
    id="MiniGrid-Color-DoorKey-Random-Goal-8x8-v0",
    entry_point="pine.environments.minigrid.doorkey:ColorDoorKeyRandomGoalEnv8x8",
)

register(
    id="MiniGrid-Color-DoorKey-Random-Goal-12x12-v0",
    entry_point="pine.environments.minigrid.doorkey:ColorDoorKeyRandomGoalEnv12x12",
)

register(
    id="MiniGrid-Color-DoorKey-Random-Goal-16x16-v0",
    entry_point="pine.environments.minigrid.doorkey:ColorDoorKeyRandomGoalEnv16x16",
)

register(
    id="MiniGrid-Color-DoorKey-Random-Goal-6x6-C2-v0",
    entry_point="pine.environments.minigrid.doorkey:ColorDoorKeyRandomGoalEnv6x6C2",
)

register(
    id="MiniGrid-Color-DoorKey-Random-Goal-8x8-C2-v0",
    entry_point="pine.environments.minigrid.doorkey:ColorDoorKeyRandomGoalEnv8x8C2",
)

register(
    id="MiniGrid-Color-DoorKey-Random-Goal-12x12-C2-v0",
    entry_point="pine.environments.minigrid.doorkey:ColorDoorKeyRandomGoalEnv12x12C2",
)

register(
    id="MiniGrid-Color-DoorKey-Random-Goal-16x16-C2-v0",
    entry_point="pine.environments.minigrid.doorkey:ColorDoorKeyRandomGoalEnv16x16C2",
)

register(
    id="MiniGrid-Color-DoorKey-Random-Goal-6x6-C3-v0",
    entry_point="pine.environments.minigrid.doorkey:ColorDoorKeyRandomGoalEnv6x6C3",
)

register(
    id="MiniGrid-Color-DoorKey-Random-Goal-8x8-C3-v0",
    entry_point="pine.environments.minigrid.doorkey:ColorDoorKeyRandomGoalEnv8x8C3",
)

register(
    id="MiniGrid-Color-DoorKey-Random-Goal-12x12-C3-v0",
    entry_point="pine.environments.minigrid.doorkey:ColorDoorKeyRandomGoalEnv12x12C3",
)

register(
    id="MiniGrid-Color-DoorKey-Random-Goal-16x16-C3-v0",
    entry_point="pine.environments.minigrid.doorkey:ColorDoorKeyRandomGoalEnv16x16C3",
)

register(
    id="MiniGrid-Color-DoorKey-Random-Goal-6x6-C5-v0",
    entry_point="pine.environments.minigrid.doorkey:ColorDoorKeyRandomGoalEnv6x6C5",
)

register(
    id="MiniGrid-Color-DoorKey-Random-Goal-8x8-C5-v0",
    entry_point="pine.environments.minigrid.doorkey:ColorDoorKeyRandomGoalEnv8x8C5",
)

register(
    id="MiniGrid-Color-DoorKey-Random-Goal-12x12-C4-v0",
    entry_point="pine.environments.minigrid.doorkey:ColorDoorKeyRandomGoalEnv12x12C4",
)

register(
    id="MiniGrid-Color-DoorKey-Random-Goal-12x12-C5-v0",
    entry_point="pine.environments.minigrid.doorkey:ColorDoorKeyRandomGoalEnv12x12C5",
)

register(
    id="MiniGrid-Color-DoorKey-Random-Goal-16x16-C5-v0",
    entry_point="pine.environments.minigrid.doorkey:ColorDoorKeyRandomGoalEnv16x16C5",
)

register(
    id="MiniGrid-Blocked-Color-DoorKey-Random-Goal-6x6-C5-B5-v0",
    entry_point="pine.environments.minigrid.doorkey:BlockedColorDoorKeyRandomGoalEnv6x6C5B5",
)

register(
    id="MiniGrid-Blocked-Color-DoorKey-Random-Goal-8x8-C5-B5-v0",
    entry_point="pine.environments.minigrid.doorkey:BlockedColorDoorKeyRandomGoalEnv8x8C5B5",
)
