from gym_sokoban.envs.sokoban_env import SokobanEnv, CHANGE_COORDINATES
from gym_sokoban.envs.room_utils import generate_room
from gym.spaces import Box, MultiDiscrete
import gym
from gym.envs.registration import register
import numpy as np

BOX_COLORS = {
    "red": np.array([255, 0, 0]),
    "green": np.array([0, 255, 0]),
    "blue": np.array([0, 0, 255]),
    "purple": np.array([112, 39, 195]),
    "yellow": np.array([255, 255, 0]),
    "grey": np.array([100, 100, 100]),
}

COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5}

IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

def room_to_tiny_world_rgb(room, box_color_idx, room_structure=None, scale=1):

    room = np.array(room)
    if not room_structure is None:
        # Change the ID of a player on a target
        room[(room == 5) & (room_structure == 2)] = 6

    wall = [0, 0, 0]
    floor = [243, 248, 238]
    box_target = [254, 126, 125]
    box_on_target = [254, 95, 56]
    box = [142, 121, 56]
    player = [160, 212, 56]
    player_on_target = [219, 212, 56]

    surfaces = [wall, floor, box_target, box_on_target, box, player, player_on_target]

    # Assemble the new rgb_room, with all loaded images
    room_small_rgb = np.zeros(shape=(room.shape[0]*scale, room.shape[1]*scale, 3), dtype=np.uint8)
    for i in range(room.shape[0]):
        x_i = i * scale
        for j in range(room.shape[1]):
            y_j = j * scale
            surfaces_id = int(room[i, j])
            if surfaces_id == 4:  # Box
                room_small_rgb[x_i:(x_i+scale), y_j:(y_j+scale), :] = BOX_COLORS[IDX_TO_COLOR[box_color_idx]]
            else:
                room_small_rgb[x_i:(x_i+scale), y_j:(y_j+scale), :] = np.array(surfaces[surfaces_id])

    return room_small_rgb

class PushAndPullSokobanEnv(SokobanEnv):

    def __init__(self,
             dim_room=(10, 10),
             max_steps=120,
             num_boxes=3,
             num_colors=3,
             num_gen_steps=None):

        self.num_colors = num_colors
        super(PushAndPullSokobanEnv, self).__init__(dim_room, max_steps, num_boxes, num_gen_steps)
        screen_height, screen_width = (dim_room[0] * 16, dim_room[1] * 16)
        self.observation_space = Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)
        self.boxes_are_on_target = [False] * num_boxes
        self.action_space = MultiDiscrete(
            [5] + [3] * num_colors
        )
        
        _ = self.reset()

    def reset(self, second_player=False, render_mode='rgb_array'):
        self.box_color_idx = np.random.randint(0, self.num_colors)
        try:
            self.room_fixed, self.room_state, self.box_mapping = generate_room(
                dim=self.dim_room,
                num_steps=self.num_gen_steps,
                num_boxes=self.num_boxes,
                second_player=second_player
            )
        except (RuntimeError, RuntimeWarning) as e:
            print("[SOKOBAN] Runtime Error/Warning: {}".format(e))
            print("[SOKOBAN] Retry . . .")
            return self.reset(second_player=second_player)

        self.player_position = np.argwhere(self.room_state == 5)[0]
        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0

        starting_observation = self.render(render_mode)
        return starting_observation

    def get_image(self, mode, scale=1):
        
        img = room_to_tiny_world_rgb(self.room_state, self.box_color_idx, self.room_fixed, scale=scale)
        return img

    def step(self, action, observation_mode='rgb_array'):
        move, *box_act = action
        if move == 0:
            original_env_action = 0
        elif box_act[self.box_color_idx] == 0:  # Move
            original_env_action = move + 4
        elif box_act[self.box_color_idx] == 1:  # Push
            original_env_action = move
        else:  # Pull
            original_env_action = move + 8

        self.num_env_steps += 1

        self.new_box_position = None
        self.old_box_position = None

        moved_box = False
        if original_env_action == 0:
            moved_player = False

        # All push original_env_actions are in the range of [0, 3]
        if original_env_action < 5:
            moved_player, moved_box = self._push(original_env_action)

        elif original_env_action < 9:
            moved_player = self._move(original_env_action)

        else:
            moved_player, moved_box = self._pull(original_env_action)

        self._calc_reward()

        done = self._check_if_done()

        # Convert the observation to RGB frame
        observation = self.render(mode=observation_mode)

        info = {
            "action.name": ACTION_LOOKUP[original_env_action],
            "action.moved_player": moved_player,
            "action.moved_box": moved_box,
            "action_influence": self.get_action_influence(),
            "truncated": self._check_if_maxsteps(),
        }
        if done:
            info["all_boxes_on_target"] = self._check_if_all_boxes_on_target()

        return observation, self.reward_last, done, info

    def _pull(self, action):
        """
        Moves the player to the next field, if it is not occupied.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = CHANGE_COORDINATES[(action - 1) % 4]
        new_position = self.player_position + change
        current_position = self.player_position.copy()
        pull_content_position = self.player_position - change

        # Move player if the field in the moving direction is either
        # an empty field or an empty box target.
        if self.room_state[new_position[0], new_position[1]] in [1, 2]:
            self.player_position = new_position
            self.room_state[(new_position[0], new_position[1])] = 5
            self.room_state[current_position[0], current_position[1]] = \
                self.room_fixed[current_position[0], current_position[1]]

            box_next_to_player = self.room_state[pull_content_position[0], pull_content_position[1]] in [3, 4]
            if box_next_to_player:
                # Move Box
                box_type = 4
                if self.room_fixed[current_position[0], current_position[1]] == 2:
                    box_type = 3
                self.room_state[current_position[0], current_position[1]] = box_type
                self.room_state[pull_content_position[0], pull_content_position[1]] = \
                    self.room_fixed[pull_content_position[0], pull_content_position[1]]

            return True, box_next_to_player

        return False, False

    def get_action_lookup(self):
        return ACTION_LOOKUP

    def get_action_meanings(self):
        return ACTION_LOOKUP

    def get_action_influence(self):
        influence = np.array([False] * len(self.action_space.nvec))
        influence[0] = True
        # If the box is next to the player
        for pos_change in CHANGE_COORDINATES:
            if self.can_push_box(pos_change) or self.can_push_box(pos_change):
                influence[self.box_color_idx] = True
                return influence
        return influence

    def can_push_box(self, change):
        new_position = self.player_position + change
        # No push, if the push would get the box out of the room's grid
        new_box_position = new_position + change
        if new_box_position[0] >= self.room_state.shape[0] \
            or new_box_position[1] >= self.room_state.shape[1]:
            return False
        can_push_box = self.room_state[new_position[0], new_position[1]] in [3, 4]
        can_push_box &= self.room_state[new_box_position[0], new_box_position[1]] in [1, 2]
        return can_push_box

    def can_pull_box(self, change):
        new_position = self.player_position + change
        pull_content_position = self.player_position - change

        # Move player if the field in the moving direction is either
        # an empty field or an empty box target.
        if self.room_state[new_position[0], new_position[1]] in [1, 2]:
            box_next_to_player = self.room_state[pull_content_position[0], pull_content_position[1]] in [3, 4]
            return box_next_to_player
        return False


ACTION_LOOKUP = {
    0: 'no operation',
    1: 'push up',
    2: 'push down',
    3: 'push left',
    4: 'push right',
    5: 'move up',
    6: 'move down',
    7: 'move left',
    8: 'move right',
    9: 'pull up',
    10: 'pull down',
    11: 'pull left',
    12: 'pull right',
}


class PushAndPull7x7B1C2(PushAndPullSokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (7, 7))
        kwargs['max_steps'] = kwargs.get('max_steps', 150)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 1)
        kwargs['num_colors'] = kwargs.get('num_colors', 2)
        kwargs['num_gen_steps'] = kwargs.get('num_gen_steps', 50)
        super().__init__(**kwargs)



class PushAndPull7x7B1C3(PushAndPullSokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (7, 7))
        kwargs['max_steps'] = kwargs.get('max_steps', 150)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 1)
        kwargs['num_colors'] = kwargs.get('num_colors', 3)
        kwargs['num_gen_steps'] = kwargs.get('num_gen_steps', 50)
        super().__init__(**kwargs)

class PushAndPull7x7B1C4(PushAndPullSokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (7, 7))
        kwargs['max_steps'] = kwargs.get('max_steps', 150)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 1)
        kwargs['num_colors'] = kwargs.get('num_colors', 4)
        kwargs['num_gen_steps'] = kwargs.get('num_gen_steps', 50)
        super().__init__(**kwargs)

class PushAndPull7x7B1C5(PushAndPullSokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (7, 7))
        kwargs['max_steps'] = kwargs.get('max_steps', 150)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 1)
        kwargs['num_colors'] = kwargs.get('num_colors', 5)
        kwargs['num_gen_steps'] = kwargs.get('num_gen_steps', 50)
        super().__init__(**kwargs)

register(
    id="Sokoban-PushAndPull-7x7-B1-C2",
    entry_point="pine.environments.sokoban.pull:PushAndPull7x7B1C2",
)

register(
    id="Sokoban-PushAndPull-7x7-B1-C3",
    entry_point="pine.environments.sokoban.pull:PushAndPull7x7B1C3",
)

register(
    id="Sokoban-PushAndPull-7x7-B1-C4",
    entry_point="pine.environments.sokoban.pull:PushAndPull7x7B1C4",
)

register(
    id="Sokoban-PushAndPull-7x7-B1-C5",
    entry_point="pine.environments.sokoban.pull:PushAndPull7x7B1C5",
)
