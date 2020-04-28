from robocup_env.envs.base.robocup import RoboCup
from typing import Tuple
from robocup_env.envs.base.constants import *
from robocup_env.envs.base.robocup_configs import RobocupBaseConfig, BaseRewardConfig


class PassConfig(BaseRewardConfig):
    def __init__(self, dribble_count_done: int = 100, dribbling_reward: float = 1.0, done_reward_additive: float = 0.0,
                 done_reward_coeff: float = 500.0, done_reward_exp_base: float = 0.998,
                 ball_out_of_bounds_reward: float = -100.0, distance_to_ball_coeff: float = -0.1,
                 enable_dribble_reward: bool = True, enable_distance_reward: bool = True,
                 pass_min_distance: float = 0.75
                 ):
        super().__init__()
        self.dribble_count_done = dribble_count_done
        self.dribbling_reward = dribbling_reward

        self.enable_dribble_reward = enable_dribble_reward
        self.enable_distance_reward = enable_distance_reward

        self.done_reward_additive = done_reward_additive
        self.done_reward_coeff = done_reward_coeff
        self.done_reward_exp_base = done_reward_exp_base

        self.ball_out_of_bounds_reward = ball_out_of_bounds_reward

        self.distance_to_ball_coeff = distance_to_ball_coeff
        self.pass_min_distance = pass_min_distance


class RoboCupPass(RoboCup):
    """
    Gym environment where the goal is to have one robot pass to another robot. The pass must be travel at least
    PassConfig.pass_min_distance units (default 0.4) to be counted as a pass.
    """

    def __init__(self, base_config: RobocupBaseConfig = RobocupBaseConfig(),
                 pass_config: PassConfig = PassConfig(),
                 verbose: bool = True):
        base_config.max_timesteps = 1200
        base_config.num_robots = 2
        super().__init__(base_config, verbose)
        self.pass_config = pass_config

        self.last_kick = None
        self.kick_ballpos = None

    def task_reset(self, t: float = 0):
        self.last_kick = None
        self.kick_ballpos = None

    def task_logic(self, action: np.ndarray) -> Tuple[float, bool, bool]:
        """
        Performs the task specific logic for calculating the reward and
        episode termination conditions
        @type action: np.ndarray Action passed in
        @return: Tuple of (step_reward, done, got_reward)
        """
        config = self.pass_config
        passed_ball = False
        for i, aux_state in enumerate(self.robot_aux_states):
            # kick_cooldown == 0 => this robot just kicked the ball
            if aux_state.kick_cooldown == 0:
                self.last_kick = i
                self.kick_ballpos = (self.ball.position[0], self.ball.position[1])
                print(f"{i} kicked it")
                continue

            # If the other robot is kicked the ball and this robot is dribbling it, then
            # a pass was made
            if aux_state.dribbling and self.last_kick is not None and self.last_kick != i:
                print(f"{i} is dribbling")
                # The pass must travel at least Config.pass_min_distance units before it is counted
                # as a pass
                dx = self.ball.position[0] - self.kick_ballpos[0]
                dy = self.ball.position[1] - self.kick_ballpos[1]
                traveled_dist = np.sqrt(dx ** 2 + dy ** 2)
                if traveled_dist >= config.pass_min_distance:
                    passed_ball = True
                else:
                    print(f"Passed but only {traveled_dist}")
                    self.last_kick = None
                    self.kick_ballpos = None

        done = passed_ball
        got_reward = passed_ball

        if done:
            step_reward = config.done_reward_additive + \
                          config.done_reward_coeff * config.done_reward_exp_base ** self.timestep
        else:
            step_reward = 0.0

        for i in range(self.config.num_robots):
            robot_action = np.array(action)[i * 4:(i + 1) * 4]
            step_reward += config.move_reward * (np.sum(robot_action[:2] ** 2) +
                                                 config.turn_penalty * robot_action[2] ** 2)

        # If the ball is out of bounds, and not in the goal, we're done but with low reward
        if not done and (self.ball.position[0] < FIELD_MIN_X or
                         self.ball.position[0] > FIELD_MAX_X or
                         self.ball.position[1] < FIELD_MIN_Y or
                         self.ball.position[1] > FIELD_MAX_Y):
            done = True
            step_reward = config.ball_out_of_bounds_reward

        # If the time limit has exceeded, we're done
        if self.timestep > self.config.max_timesteps:
            done = True
            step_reward = 0

        return step_reward, done, got_reward
