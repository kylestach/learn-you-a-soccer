import numpy as np
from typing import Optional


class InitialConditionConfig:
    def __init__(self, fixed_ic: Optional[np.ndarray] = None, enable_scheduled_ic: bool = True):
        self.enable_scheduled_ic = enable_scheduled_ic

        if fixed_ic:
            self.fixed_ic = fixed_ic
        else:
            #                         x    y    h    bs   vx   vy   vh    bx   by   bvx  bvy
            self.fixed_ic = np.array([1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 0.0])


class RobocupBaseConfig:
    def __init__(self, initial_condition_config: InitialConditionConfig = InitialConditionConfig(),
                 max_timesteps: int = 300,
                 kicker_enabled: bool = True,
                 kick_power: float = 4.0,
                 kick_cooldown: int = 30,
                 num_robots: int = 1):
        self.initial_condition_config: InitialConditionConfig = initial_condition_config
        self.max_timesteps = max_timesteps
        self.kicker_enabled = kicker_enabled
        self.kick_power = kick_power
        self.kick_cooldown = kick_cooldown
        self.num_robots = num_robots


class BaseRewardConfig:
    def __init__(self,
                 move_reward: float = -0.3,
                 turn_penalty: float = 3.0):
        self.move_reward = move_reward
        self.turn_penalty = turn_penalty
