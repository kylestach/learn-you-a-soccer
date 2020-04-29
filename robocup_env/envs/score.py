from robocup_env.envs.base.robocup import RoboCup
from typing import Tuple
from robocup_env.envs.base.constants import *
from robocup_env.envs.base.robocup_configs import RobocupBaseConfig, BaseRewardConfig


class ScoreConfig(BaseRewardConfig):
    def __init__(self, dribble_count_done: int = 100, dribbling_reward: float = 1.0, done_reward_additive: float = 0.0,
                 done_reward_coeff: float = 500.0, done_reward_exp_base: float = 0.998,
                 ball_out_of_bounds_reward: float = -100.0, distance_to_ball_coeff: float = -0.1,
                 survival_reward: float = 1.0,
                 enable_dribble_reward: bool = True, enable_distance_reward: bool = True):
        super().__init__()
        self.dribble_count_done = dribble_count_done
        self.dribbling_reward = dribbling_reward

        self.enable_dribble_reward = enable_dribble_reward
        self.enable_distance_reward = enable_distance_reward

        self.done_reward_additive = done_reward_additive
        self.done_reward_coeff = done_reward_coeff
        self.done_reward_exp_base = done_reward_exp_base

        self.ball_out_of_bounds_reward = ball_out_of_bounds_reward
        self.survival_reward = survival_reward

        self.distance_to_ball_coeff = distance_to_ball_coeff


class RoboCupScore(RoboCup):
    def __init__(self, base_config: RobocupBaseConfig = RobocupBaseConfig(),
                 score_config: ScoreConfig = ScoreConfig(),
                 verbose: bool = True):
        base_config.max_timesteps = 600
        super().__init__(base_config, verbose)
        self.score_config = score_config

    def task_logic(self, action: np.ndarray) -> Tuple[float, bool, bool]:
        """
        Performs the task specific logic for calculating the reward and
        episode termination conditions
        @type action: np.ndarray Action passed in
        @return: Tuple of (step_reward, done, got_reward)
        """
        aux_state = self.robot_aux_states[0]
        robot = self.robot_bodies[0]

        config = self.score_config
        done_dribbled = aux_state.dribbling_count >= config.dribble_count_done

        ball_in_left_goal = (VIEW_MIN_X <= self.ball.position[0] <= FIELD_MIN_X) and (
                -GOAL_HEIGHT / 2 < self.ball.position[1] < GOAL_HEIGHT / 2)
        done = ball_in_left_goal and aux_state.kicked_ball
        got_reward = done

        dist = np.sqrt((robot.position[0] - self.ball.position[0]) ** 2 + (
                robot.position[1] - self.ball.position[1]) ** 2)

        step_reward = config.survival_reward  # Survival reward
        if done:
            step_reward += config.done_reward_additive + \
                           config.done_reward_coeff * config.done_reward_exp_base ** self.timestep
        elif config.enable_dribble_reward and aux_state.dribbling and not done_dribbled:
            step_reward += config.dribbling_reward
        elif config.enable_distance_reward:
            step_reward += config.distance_to_ball_coeff * dist
        else:
            step_reward += 0.0

        step_reward += config.move_reward * (np.sum(np.array(action)[:2] ** 2) + config.turn_penalty * action[2] ** 2)

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
