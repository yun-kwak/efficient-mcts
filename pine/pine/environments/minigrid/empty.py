from gym_minigrid.minigrid import MiniGridEnv, Grid, Goal, MissionSpace
from gym.envs.registration import register

class EmptyRandomGoalEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=8,
        agent_start_pos=(1,1),
        agent_start_dir=0,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.color_act_idx = None
        self.n_colors = 6

        mission_space = MissionSpace(
            mission_func=lambda: "get to the green goal square"
        )

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True
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

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"

class EmptyRandomGoalEnv8x8(EmptyRandomGoalEnv):
    def __init__(self):
        super().__init__(size=8, agent_start_pos=None)

class EmptyRandomGoalEnv12x12(EmptyRandomGoalEnv):
    def __init__(self):
        super().__init__(size=12, agent_start_pos=None)

class EmptyRandomGoalEnv16x16(EmptyRandomGoalEnv):
    def __init__(self):
        super().__init__(size=16, agent_start_pos=None)

register(
    id='MiniGrid-Empty-Random-Goal-8x8-v0',
    entry_point='pine.environments.minigrid.empty:EmptyRandomGoalEnv8x8'
)

register(
    id='MiniGrid-Empty-Random-Goal-12x12-v0',
    entry_point='pine.environments.minigrid.empty:EmptyRandomGoalEnv12x12'
)

register(
    id='MiniGrid-Empty-Random-Goal-16x16-v0',
    entry_point='pine.environments.minigrid.empty:EmptyRandomGoalEnv16x16'
)
