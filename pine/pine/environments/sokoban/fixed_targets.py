from gym_sokoban.envs.sokoban_env import SokobanEnv, CHANGE_COORDINATES
from gym_sokoban.envs.render_utils import room_to_rgb_FT, room_to_tiny_world_rgb_FT
import gym
from gym.envs.registration import register
from gym.spaces import Box
import numpy as np

class FixedTargetsSokobanEnv(SokobanEnv):

    def __init__(self,
             dim_room=(10, 10),
             max_steps=120,
             num_boxes=3,
             num_gen_steps=None):

        super(FixedTargetsSokobanEnv, self).__init__(dim_room, max_steps, num_boxes, num_gen_steps)
        screen_height, screen_width = (dim_room[0] * 16, dim_room[1] * 16)
        self.observation_space = Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)
        self.boxes_are_on_target = [False] * num_boxes
        self.action_space = gym.spaces.MultiDiscrete(
            [5, num_boxes + 1]
        )
        
        _ = self.reset()

    def get_image(self, mode, scale=1):

        img = room_to_tiny_world_rgb_FT(self.room_state, self.box_mapping, self.room_fixed, scale=scale)

        return img

    def _check_right_push_action(self, move, act_idx):
        change = CHANGE_COORDINATES[(move - 1) % 4]
        new_position = self.player_position + change
        is_box = self.room_state[new_position[0], new_position[1]] in [3, 4]
        if not is_box:
            return False
        box_id = list(self.box_mapping.values()).index((new_position[0], new_position[1]))
        return act_idx == box_id

    def step(self, action, observation_mode='tiny_rgb_array'):
        move, push = action
        if move == 0:
            action = 0
        elif self._check_right_push_action(move, push - 1):
            action = move
        else:
            action = move + 4
        observation, self.reward_last, done, info = super(FixedTargetsSokobanEnv, self).step(action, observation_mode)

        info["action_influence"] = np.array([True, True])

        return observation, self.reward_last, done, info

    def _push(self, action):
        """
        Perform a push, if a box is adjacent in the right direction.
        If no box, can be pushed, try to move.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = CHANGE_COORDINATES[(action - 1) % 4]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        # No push, if the push would get the box out of the room's grid
        new_box_position = new_position + change
        if new_box_position[0] >= self.room_state.shape[0] \
                or new_box_position[1] >= self.room_state.shape[1]:
            return False, False


        can_push_box = self.room_state[new_position[0], new_position[1]] in [3, 4]
        can_push_box &= self.room_state[new_box_position[0], new_box_position[1]] in [1, 2]
        if can_push_box:

            self.new_box_position = tuple(new_box_position)
            self.old_box_position = tuple(new_position)

            # Move Player
            self.player_position = new_position
            self.room_state[(new_position[0], new_position[1])] = 5
            self.room_state[current_position[0], current_position[1]] = \
                self.room_fixed[current_position[0], current_position[1]]

            # Move Box
            box_type = 4
            if self.room_fixed[new_box_position[0], new_box_position[1]] == 2:
                box_type = 3
            self.room_state[new_box_position[0], new_box_position[1]] = box_type
            return True, True

        # Try to move if no box to push, available
        else:
            return self._move(action), False

    def _calc_reward(self):
        self._update_box_mapping()

        # Every step a small penalty is given, This ensures
        # that short solutions have a higher reward.
        self.reward_last = self.penalty_for_step

        for b in range(len(self.boxes_are_on_target)):

            previous_state = self.boxes_are_on_target[b]

            # Calculate new state
            box_id = list(self.box_mapping.keys())[b]
            new_state = self.box_mapping[box_id] == box_id

            if previous_state and not new_state:
                # Box was pushed of its target
                self.reward_last += self.penalty_box_off_target
            elif not previous_state and new_state:
                # box was pushed on its target
                self.reward_last += self.reward_box_on_target

            self.boxes_are_on_target[b] = new_state

    def _update_box_mapping(self):
        if self.new_box_position is not None:
            box_index = list(self.box_mapping.values()).index(self.old_box_position)
            box_id = list(self.box_mapping.keys())[box_index]
            self.box_mapping[box_id] = self.new_box_position

    def _check_if_all_boxes_on_target(self):

        for key in self.box_mapping.keys():
            if not key == self.box_mapping[key]:
                return False

        return True

    def _get_action_influence(self):
        return np.array([True, True])



class ActionInfluenceWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        """Step through the environment."""
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def _get_action_influence(self):
        return np.array([True, True])


class FixedTargets7x7C2(FixedTargetsSokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (7, 7))
        kwargs['max_steps'] = kwargs.get('max_steps', 150)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 2)
        kwargs['num_gen_steps'] = kwargs.get('num_gen_steps', 50)
        super().__init__(**kwargs)


class FixedTargets7x7C5(FixedTargetsSokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (7, 7))
        kwargs['max_steps'] = kwargs.get('max_steps', 150)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 5)
        kwargs['num_gen_steps'] = kwargs.get('num_gen_steps', 50)
        super().__init__(**kwargs)

register(
    id="Sokoban-FixedTargets-7x7-C2",
    entry_point="pine.environments.sokoban.fixed_targets:FixedTargets7x7C2",
)

register(
    id="Sokoban-FixedTargets-7x7-C5",
    entry_point="pine.environments.sokoban.fixed_targets:FixedTargets7x7C5",
)

