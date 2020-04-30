from robocup_env.envs.base.robocup import RoboCup, InitialConditionConfig, getstate, ROBOTSTATE_SIZE
from typing import Tuple
from robocup_env.envs.base.constants import *
from robocup_env.envs.base.robocup_configs import RobocupBaseConfig, BaseRewardConfig
from robocup_env.envs.base.robocup_types import *


class PassICConfig(InitialConditionConfig):
    """ Set up so that robots are facing each other with ball near the first robot
    """

    def __init__(self, r: float = 1.0, angle: float = 0.0, angle_offset: float = 0.0, swap: bool = False):
        r1_pos, r2_pos, ball_ic = self.get_pos(r, angle, angle_offset)
        #                     x     y    h      bs   vx   vy   vh
        robot1_ic = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        robot2_ic = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # ball_ic = np.array([0.0, 0.0, 0.0, 0.0])

        if swap:
            robot2_ic[0:3] = r1_pos
            robot1_ic[0:3] = r2_pos
        else:
            robot1_ic[0:3] = r1_pos
            robot2_ic[0:3] = r2_pos

        fixed_ic = np.concatenate((robot1_ic, robot2_ic, ball_ic))
        super().__init__(fixed_ic, enable_scheduled_ic=True)

    def get_pos(self, r: float, angle: float, angle_offset: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Position the robots in a circle around (0, 0) facing each other
        @param r: Radius of the circle
        @param angle: Angle of the first robot
        @param angle_offset: Angle offset of the second robot
        @return:
        """
        angle1, angle2 = angle, np.pi + angle + angle_offset
        ball_r = r - 0.15
        ball_v = 0.5

        robot1_pos = np.array([r * np.cos(angle1), r * np.sin(angle1), angle2])
        robot2_pos = np.array([r * np.cos(angle2), r * np.sin(angle2), angle1])
        ball_ic = np.array([ball_r * np.cos(angle1), ball_r * np.sin(angle1),
                            ball_v * np.cos(angle1), ball_v * np.sin(angle1)])

        return robot1_pos, robot2_pos, ball_ic


class PassConfig(BaseRewardConfig):
    def __init__(self, dribble_count_done: int = 50, dribbling_reward: float = 1.0, done_reward_additive: float = 0.0,
                 done_reward_coeff: float = 700.0, done_reward_exp_base: float = 0.998,
                 ball_out_of_bounds_reward: float = -100.0, distance_to_ball_coeff: float = -0.1,
                 survival_reward: float = 0.5,
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
        self.survival_reward = survival_reward

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
        base_config.max_timesteps = 600
        base_config.num_robots = 2
        base_config.initial_condition_config = PassICConfig()
        super().__init__(base_config, verbose)
        self.pass_config = pass_config

        self.last_kick = None
        self.kick_ballpos = None

    def task_reset(self, t: float = 0):
        self.last_kick = None
        self.kick_ballpos = None
        # 50% chance to swap robot1 and robot2 states (so that the ball isn't always next to robot1)
        # swap = np.random.uniformh(0, 1) > 0.5
        swap = False
        angle_offset = 0.05 * np.random.normal()
        self.config.initial_condition_config = PassICConfig(r=1.0, angle=np.random.uniform(0, 2 * np.pi),
                                                            angle_offset=angle_offset, swap=swap)

    def scale_states(self, scale_factor: float):
        conf: InitialConditionConfig = self.config.initial_condition_config
        for i in range(self.config.num_robots):
            # Robot pose
            self.state[getstate(i, ROBOT_X)] = scale_factor * self.state[getstate(i, ROBOT_X)] + (1 - scale_factor) * \
                                               conf.fixed_ic[getstate(i, ROBOT_X)]
            self.state[getstate(i, ROBOT_Y)] = scale_factor * self.state[getstate(i, ROBOT_Y)] + (1 - scale_factor) * \
                                               conf.fixed_ic[getstate(i, ROBOT_Y)]
            self.state[getstate(i, ROBOT_H)] = conf.fixed_ic[
                                                   getstate(i, ROBOT_H)] + 2 * np.random.randn() * scale_factor

            # Ball sense
            self.state[getstate(i, ROBOT_BALLSENSE)] = 0

            # Robot velocity
            self.state[getstate(i, ROBOT_DX)] = scale_factor * self.state[getstate(i, ROBOT_DX)] + (1 - scale_factor) * \
                                                conf.fixed_ic[getstate(i, ROBOT_DX)]
            self.state[getstate(i, ROBOT_DY)] = scale_factor * self.state[getstate(i, ROBOT_DY)] + (1 - scale_factor) * \
                                                conf.fixed_ic[getstate(i, ROBOT_DY)]
            self.state[getstate(i, ROBOT_DH)] = scale_factor * self.state[getstate(i, ROBOT_DH)] + (1 - scale_factor) * \
                                                conf.fixed_ic[getstate(i, ROBOT_DH)]

        # Ball position is always in front of the robot
        dist_ahead = 0.1
        ball_vel_damp = 0.8
        self.state[BALL_X] = self.state[ROBOT_X] + dist_ahead * np.cos(self.state[ROBOT_H])
        self.state[BALL_Y] = self.state[ROBOT_Y] + dist_ahead * np.sin(self.state[ROBOT_H])
        self.state[BALL_DX] = ball_vel_damp * self.state[ROBOT_DX]
        self.state[BALL_DY] = ball_vel_damp * self.state[ROBOT_DY]
        self.state[ROBOT_BALLSENSE] = 1

    def task_logic(self, action: np.ndarray) -> Tuple[float, bool, bool]:
        """
        Performs the task specific logic for calculating the reward and
        episode termination conditions
        @type action: np.ndarray Action passed in
        @return: Tuple of (step_reward, done, got_reward)
        """
        config = self.pass_config
        passed_ball = False
        aux_state0, aux_state1 = self.robot_aux_states[0], self.robot_aux_states[1]
        for i, aux_state in enumerate(self.robot_aux_states):
            # If the other robot is kicked the ball and this robot is dribbling it, then
            # a pass was made
            if aux_state.dribbling and self.last_kick is not None and self.last_kick != i:
                # print(f"{i} is dribbling")
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

            # kick_cooldown == 0 => this robot just kicked the ball
            if aux_state.kick_cooldown == 0:
                self.last_kick = i
                self.kick_ballpos = (self.ball.position[0], self.ball.position[1])
                print(f"{i} kicked it")
                continue

        done = passed_ball
        got_reward = passed_ball

        done_dribble0 = aux_state0.dribbling_count >= config.dribble_count_done
        done_dribble1 = aux_state1.dribbling_count >= config.dribble_count_done
        either_dribbling = (aux_state0.dribbling and not done_dribble0) or (aux_state1.dribbling and not done_dribble1)

        step_reward = config.survival_reward  # Survival reward
        if done:
            step_reward += config.done_reward_additive + \
                           config.done_reward_coeff * config.done_reward_exp_base ** self.timestep
        elif config.enable_dribble_reward and either_dribbling:
            step_reward += config.dribbling_reward
        else:
            step_reward += 0.0

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
