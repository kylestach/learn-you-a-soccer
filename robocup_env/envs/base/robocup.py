from typing import Optional
import Box2D
from Box2D import b2ContactListener, b2Contact, b2Body

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

from robocup_env.envs.base.constants import *
from robocup_env.envs.base.robocup_configs import *
import math

from robocup_env.envs.base.robocup_types import *


class CollisionDetector(b2ContactListener):
    """
    A collision detector to keep track of when the robot intercepts the ball
    """

    def __init__(self, env: "RoboCup"):
        b2ContactListener.__init__(self)
        self.env = env

    def has_ball(self, contact: b2Contact) -> bool:
        return contact.fixtureA.userData['type'] == 'ball' or contact.fixtureB.userData['type'] == 'ball'

    def has_kicker(self, contact: b2Contact) -> bool:
        return contact.fixtureA.userData['type'] == 'kicker' or contact.fixtureB.userData['type'] == 'kicker'

    def has_robot(self, contact: b2Contact) -> bool:
        return contact.fixtureA.body.userData['type'] == 'robot' or contact.fixtureB.body.userData['type'] == 'robot'

    def get_id(self, contact: b2Contact) -> bool:
        if contact.fixtureA.body.userData['type'] == 'robot':
            return contact.fixtureA.body.userData['robot_id']
        elif contact.fixtureB.body.userData['type'] == 'robot':
            return contact.fixtureB.body.userData['robot_id']
        else:
            raise Exception("[get_id] both fixtures didn't have robot_id")

    def BeginContact(self, contact: b2Contact):
        if self.has_ball(contact) and self.has_robot(contact):
            self.env.has_touched = True
            if self.has_kicker(contact):
                robot_id = self.get_id(contact)
                # print(f"{robot_id} is dribbling")
                self.env.robot_aux_states[robot_id].dribbling = True
                self.env.robot_aux_states[robot_id].can_kick = True

    def EndContact(self, contact):
        if 'type' not in contact.fixtureA.body.userData or 'type' not in contact.fixtureB.body.userData:
            return
        if self.has_ball(contact) and self.has_robot(contact):
            if self.has_kicker(contact):
                robot_id = self.get_id(contact)
                self.env.robot_aux_states[robot_id].dribbling = False
                self.env.robot_aux_states[robot_id].can_kick = False


class VelocitySpace(spaces.Space):
    """
    Sample a velocity in n-dimensional space. Sampling occurs from a normal
    distribution with given (independent) standard deviation
    """

    def __init__(self, shape, stdev):
        spaces.Space.__init__(self, shape, np.float32)
        self.low = np.array([-1e3] * shape[0])
        self.high = np.array([1e3] * shape[0])

        self.stdev = stdev
        self.shape = shape

    def sample(self):
        return self.stdev * np.random.randn(*self.shape)

    def contains(self, x):
        return True


class RoboCupStateSpace(spaces.Space):
    """
    Sample 2-dimensional position and velocity.
    rx, ry, rh, rdx, rdy, rdh, ball_sense
    bx, by, bdx, bdy
    """

    def __init__(self, num_robots):
        super().__init__()
        self.pose_space = spaces.Box(
            np.array([FIELD_MIN_X, FIELD_MIN_Y, -1, -1]),
            np.array([FIELD_MAX_X, FIELD_MAX_Y, 1, 1]),
            dtype=np.float32
        )
        self.actual_pose_space = spaces.Box(
            np.array([FIELD_MIN_X, FIELD_MIN_Y, 0]),
            np.array([FIELD_MAX_X, FIELD_MAX_Y, 2 * np.pi]),
            dtype=np.float32
        )

        self.position_space = spaces.Box(
            np.array([FIELD_MIN_X, FIELD_MIN_Y]),
            np.array([FIELD_MAX_X, FIELD_MAX_Y]),
            dtype=np.float32
        )

        self.num_robots = num_robots

        self.robot_velocity_space = VelocitySpace((3,), np.array([2.0, 2.0, 2.0]))
        self.ball_velocity_space = VelocitySpace((2,), np.array([1.0, 1.0]))

        self.shape = (num_robots * 8 + 4,)
        self.low = np.concatenate([
                                      np.concatenate([
                                          self.pose_space.low,
                                          np.array([0.0]),  # Ballsense
                                          self.robot_velocity_space.low]) for _ in range(self.num_robots)
                                  ] + [
                                      np.concatenate([
                                          self.position_space.low,
                                          self.ball_velocity_space.low
                                      ])
                                  ])
        self.high = np.concatenate([
                                       np.concatenate([
                                           self.pose_space.high,
                                           np.array([1.0]),  # Ballsense
                                           self.robot_velocity_space.high]) for _ in range(self.num_robots)
                                   ] + [
                                       np.concatenate([
                                           self.position_space.high,
                                           self.ball_velocity_space.high
                                       ])
                                   ])

    def sample(self):
        return np.concatenate([
                                  np.concatenate([
                                      self.pose_space.sample(),
                                      np.array([0.0]),
                                      self.robot_velocity_space.sample()]) for _ in range(self.num_robots)
                              ] + [
                                  np.concatenate([
                                      self.position_space.sample(),
                                      self.ball_velocity_space.sample()
                                  ])
                              ])

    def sample_state(self):
        return np.concatenate([
                                  np.concatenate([
                                      self.actual_pose_space.sample(),
                                      np.array([0.0]),
                                      self.robot_velocity_space.sample()]) for _ in range(self.num_robots)
                              ] + [
                                  np.concatenate([
                                      self.position_space.sample(),
                                      self.ball_velocity_space.sample()
                                  ])
                              ])

    def contains(self, x):
        return (self.pose_space.contains(x[:3]) and
                0 <= x[3] <= 1 and
                self.robot_velocity_space.contains(x[4:7]) and
                self.position_space.contains(x[7:9]) and
                self.ball_velocity_space.contains(x[9:]))


class CollectRewardConfig:
    def __init__(self,
                 dribbling_reward: float = 1.0,
                 done_reward_additive: float = 0.0,
                 done_reward_coeff: float = 500.0,
                 done_reward_exp_base: float = 0.998,
                 ball_out_of_bounds_reward: float = -30.0,
                 move_reward: float = -0.3,
                 # It's reward, not cost. Should be negative to reward lower distances
                 distance_to_ball_coeff: float = -0.1,
                 ):
        self.dribbling_reward = dribbling_reward

        self.done_reward_additive = done_reward_additive
        self.done_reward_coeff = done_reward_coeff
        self.done_reward_exp_base = done_reward_exp_base

        self.move_reward = move_reward

        self.ball_out_of_bounds_reward = ball_out_of_bounds_reward

        self.distance_to_ball_coeff = distance_to_ball_coeff


class CollectEnvConfig:
    def __init__(self, dribble_count_done: int = 100,
                 collect_reward_config: CollectRewardConfig = CollectRewardConfig(),
                 initial_condition_config: InitialConditionConfig = InitialConditionConfig()):
        self.dribble_count_done: int = dribble_count_done
        self.ball_dist_done: float = 0.1
        self.collect_reward_config: CollectRewardConfig = collect_reward_config
        self.initial_condition_config: InitialConditionConfig = initial_condition_config


class RobotAuxState:
    def __init__(self, dribbling: bool = False, can_kick: bool = False, kick_cooldown: int = 0,
                 dribbling_count: int = 0, has_touched: bool = False, kicked_ball: bool = False):
        self.dribbling = dribbling
        self.can_kick = can_kick
        self.kick_cooldown = kick_cooldown
        self.dribbling_count = dribbling_count
        self.has_touched = has_touched
        self.kicked_ball = kicked_ball


class RoboCup(gym.Env, EzPickle):
    def __init__(self, base_config: RobocupBaseConfig = RobocupBaseConfig(), verbose=1):
        EzPickle.__init__(self)
        self.seed()
        self.viewer = None

        self.world = None
        self.contactListener_keepref = None

        self.ball: Optional[b2Body] = None
        self.robot_bodies: Optional[List[b2Body]] = None
        self.config: RobocupBaseConfig = base_config

        self._max_episode_steps = base_config.max_timesteps

        self.timestep = 0

        self.robot_aux_states = [RobotAuxState() for _ in range(self.config.num_robots)]

        self.reward = 0.0
        self.prev_reward = 0.0
        self.verbose = verbose

        self.action_space = spaces.Box(
            np.array(self.config.num_robots * [-1.0, -1.0, -1.0, 0]),
            np.array(self.config.num_robots * [1.0, 1.0, 1.0, 1]),
            dtype=np.float32
        )

        self.observation_space = RoboCupStateSpace(self.config.num_robots)

        from gym.envs.classic_control import rendering
        self.ball_transform: Optional[rendering.Transform] = None

        self.robot_transforms: Optional[List[rendering.Transform]] = None

        self.state: Optional[np.ndarray] = None

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def state_list(self) -> np.ndarray:
        # Observation space is:
        # [ robot0_x robot0_y robot0_h robot0_vx robot0_vy robot0_vh ]
        # ,
        # [ ball_x ball_y ball_vx ball_vy ]
        obs = np.zeros(self.observation_space.shape)
        for i in range(self.config.num_robots):
            obs[getobsstate(i, 0): getobsstate(i + 1, 0)] = np.array([
                self.state[getstate(i, ROBOT_X)],  #
                self.state[getstate(i, ROBOT_Y)],  #
                np.sin(self.state[getstate(i, ROBOT_H)]),  #
                np.cos(self.state[getstate(i, ROBOT_H)]),  #
                self.state[getstate(i, ROBOT_BALLSENSE)],  #
                self.state[getstate(i, ROBOT_DX)],  #
                self.state[getstate(i, ROBOT_DY)],  #
                self.state[getstate(i, ROBOT_DH)],  #
            ])
        obs[OBS_BALL_X] = self.state[BALL_X]
        obs[OBS_BALL_Y] = self.state[BALL_Y]
        obs[OBS_BALL_DX] = self.state[BALL_DX]
        obs[OBS_BALL_DY] = self.state[BALL_DY]

        return obs

    def _destroy(self):
        if not self.world:
            return

        for body in [
            'ball',
            'top',
            'bottom',
            'left',
            'right',
            'goal',
        ]:
            if body in self.__dict__ and self.__dict__[body] is not None:
                self.world.DestroyBody(self.__dict__[body])
                self.__dict__[body] = None

        for idx, body in enumerate(self.robot_bodies):
            if body is not None:
                self.world.DestroyBody(body)
                self.robot_bodies[idx] = None

        self.contactListener_keepref = None
        self.world = None

    def _create(self, scale_factor: float = 1.0):
        self.contactListener_keepref = CollisionDetector(self)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)

        self.has_touched = False
        self.kicked_ball = False

        conf: InitialConditionConfig = self.config.initial_condition_config

        # Create the goal, robots, ball and field
        self.state: RoboCupState = self.observation_space.sample_state()

        # Robot pose
        self.state[ROBOT_X] = scale_factor * self.state[ROBOT_X] + (1 - scale_factor) * conf.fixed_ic[ROBOT_X]
        self.state[ROBOT_Y] = scale_factor * self.state[ROBOT_Y] + (1 - scale_factor) * conf.fixed_ic[ROBOT_Y]
        self.state[ROBOT_H] = np.pi + 2 * np.random.randn() * scale_factor

        # Ball sense
        self.state[ROBOT_BALLSENSE] = 0

        # Robot velocity
        self.state[ROBOT_DX] = scale_factor * self.state[ROBOT_DX] + (1 - scale_factor) * conf.fixed_ic[ROBOT_DX]
        self.state[ROBOT_DY] = scale_factor * self.state[ROBOT_DY] + (1 - scale_factor) * conf.fixed_ic[ROBOT_DY]
        self.state[ROBOT_DH] = scale_factor * self.state[ROBOT_DH] + (1 - scale_factor) * conf.fixed_ic[ROBOT_DH]

        # Ball position/velocity
        self.state[BALL_X] = scale_factor * self.state[BALL_X] + (1 - scale_factor) * conf.fixed_ic[BALL_X]
        self.state[BALL_Y] = scale_factor * self.state[BALL_Y] + (1 - scale_factor) * conf.fixed_ic[BALL_Y]
        self.state[BALL_DX] = scale_factor * self.state[BALL_DX] + (1 - scale_factor) * conf.fixed_ic[BALL_DX]
        self.state[BALL_DY] = scale_factor * self.state[BALL_DY] + (1 - scale_factor) * conf.fixed_ic[BALL_DY]

        # =========== Ball ====================================================
        self.ball = self.world.CreateDynamicBody(
            position=(float(self.state[BALL_X]), float(self.state[BALL_Y])),
            userData={"type": "ball"},
        )
        self.ball.CreateCircleFixture(
            radius=BALL_RADIUS,
            density=BALL_DENSITY,
            restitution=0.2,
            userData={"type": "ball"},
        )

        self.ball.linearVelocity[0] = float(self.state[BALL_DX])
        self.ball.linearVelocity[1] = float(self.state[BALL_DY])

        # =========== Robot ====================================================
        self.robot_bodies = []
        for i in range(self.config.num_robots):
            robot = self.world.CreateDynamicBody(
                position=(float(self.state[getstate(i, ROBOT_X)]), float(self.state[getstate(i, ROBOT_Y)])),
                angle=float(self.state[getstate(i, ROBOT_H)]),
                userData={"type": "robot", "robot_id": i},
            )
            robot.CreatePolygonFixture(
                vertices=robot_points,
                restitution=0.3,
                density=100.0,
                userData={"type": "robot", "robot_id": i},
            )
            # Kicker rectangle
            robot.CreatePolygonFixture(
                vertices=[
                    (ROBOT_RADIUS - 0.02, 0.06),
                    (ROBOT_RADIUS - 0.02, -0.06),
                    (0, -0.06),
                    (0, 0.06)
                ],
                restitution=0.15,
                userData={"type": "kicker", "robot_id": i}
            )

            robot.linearVelocity[0] = float(self.state[getstate(i, ROBOT_DX)])
            robot.linearVelocity[1] = float(self.state[getstate(i, ROBOT_DY)])
            robot.angularVelocity = float(self.state[getstate(i, ROBOT_DH)])
            robot.linearDamping = 3
            robot.angularDamping = 5
            robot.fixedRotation = False

            self.robot_bodies.append(robot)

        # =========== Walls ====================================================
        wall_data = {"type": "wall"}
        self.top: Box2D.b2Body = self.world.CreateStaticBody(position=(0, 0), userData=wall_data)
        self.top.CreateEdgeFixture(vertices=[
            (VIEW_MIN_X, VIEW_MIN_Y),
            (VIEW_MAX_X, VIEW_MIN_Y)
        ], restitution=0.7, userData=wall_data)

        self.bottom = self.world.CreateStaticBody(position=(0, 0), userData=wall_data)
        self.bottom.CreateEdgeFixture(vertices=[
            (VIEW_MIN_X, VIEW_MAX_Y),
            (VIEW_MAX_X, VIEW_MAX_Y),
        ], restitution=0.7, userData=wall_data)

        self.left = self.world.CreateStaticBody(position=(0, 0), userData=wall_data)
        self.left.CreateEdgeFixture(vertices=[
            (VIEW_MIN_X, VIEW_MIN_Y),
            (VIEW_MIN_X, VIEW_MAX_Y)
        ], restitution=0.7, userData=wall_data)

        self.right = self.world.CreateStaticBody(position=(0, 0), userData=wall_data)
        self.right.CreateEdgeFixture(vertices=[
            (VIEW_MAX_X, VIEW_MIN_Y),
            (VIEW_MAX_X, VIEW_MAX_Y)
        ], restitution=0.7, userData=wall_data)

        # =========== Goal ====================================================
        self.left_goal = self.world.CreateStaticBody(position=(FIELD_MIN_X, 0), userData={"type": "goal"})
        self.left_goal.CreatePolygonFixture(
            vertices=LEFT_GOAL_POINTS,
            isSensor=True,
            userData={"type": "goal"}
        )

    def _apply_ball_friction(self):
        # Apply friction to the ball
        ball_speed = np.sqrt(self.ball.linearVelocity[0] ** 2 + self.ball.linearVelocity[1] ** 2)
        friction_accel = -0.5 * self.ball.linearVelocity / FRICTION_LINEAR_REGION

        if ball_speed > FRICTION_LINEAR_REGION:
            friction_accel = friction_accel * FRICTION_LINEAR_REGION / ball_speed

        friction = self.ball.mass * friction_accel
        self.ball.ApplyForce(friction, self.ball.worldCenter, False)

    def task_logic(self, action: np.ndarray) -> Tuple[float, bool, bool]:
        """
        Performs the task specific logic for calculating the reward and
        episode termination conditions
        @type action: np.ndarray Action passed in
        @return: Tuple of (step_reward, done, got_reward)
        """
        ...

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        # Gather the entire state.

        for i in range(self.config.num_robots):
            action[4 * i:(i + 1) * 4] = np.clip(action[4 * i:(i + 1) * 4],
                                                np.array([-1.0, -1.0, -1.0, 0.0]), np.array([1.0, 1.0, 1.0, 1.0]))

        self.world.Step(1 / 60, 6 * 60, 6 * 60)

        self.state = np.zeros(self.observation_space.shape)
        for i in range(self.config.num_robots):
            self.state[i * ROBOTSTATE_SIZE:(i + 1) * ROBOTSTATE_SIZE] = np.array([
                self.robot_bodies[i].position[0],
                self.robot_bodies[i].position[1],
                self.robot_bodies[i].angle,
                1.0 * self.robot_aux_states[i].can_kick,
                self.robot_bodies[i].linearVelocity[0],
                self.robot_bodies[i].linearVelocity[1],
                self.robot_bodies[i].angularVelocity,
            ])
        self.state[OBS_BALL_X] = self.ball.position[0]
        self.state[OBS_BALL_Y] = self.ball.position[1]
        self.state[OBS_BALL_DX] = self.ball.linearVelocity[0]
        self.state[OBS_BALL_DY] = self.ball.linearVelocity[1]

        self._apply_ball_friction()

        kick_cooldown = self.config.kick_cooldown
        for i in range(self.config.num_robots):
            robot_action = action[i * 4:(i + 1) * 4]
            self.robot_bodies[i].ApplyForce(
                [5 * self.robot_bodies[i].mass * robot_action[0], 5 * self.robot_bodies[i].mass * robot_action[1]],
                self.robot_bodies[i].worldCenter, True)
            self.robot_bodies[i].ApplyTorque(self.robot_bodies[i].inertia * 40 * robot_action[2], True)

            if self.robot_aux_states[i].kick_cooldown <= kick_cooldown:
                self.robot_aux_states[i].kick_cooldown += 1
            if self.robot_aux_states[i].can_kick and self.robot_aux_states[i].kick_cooldown > kick_cooldown and (
                    3.0 * robot_action[3]) > KICK_THRESHOLD:
                self.robot_aux_states[i].kick_cooldown = 0
                shoot_magnitude = self.ball.mass * (self.config.kick_power * robot_action[3])
                shoot_impulse = [shoot_magnitude * np.cos(self.robot_bodies[i].angle),
                                 shoot_magnitude * np.sin(self.robot_bodies[i].angle)]
                self.ball.ApplyLinearImpulse(shoot_impulse, self.robot_bodies[i].worldCenter, True)
                self.robot_aux_states[i].kicked_ball = True

            if self.robot_aux_states[i].dribbling:
                # Apply force on the ball towards the center of the robot
                dribble_vec = self.robot_bodies[i].position - self.ball.position
                dribble_force = dribble_vec / np.sqrt(dribble_vec[0] ** 2 + dribble_vec[1] ** 2) * 5e-3
                self.ball.ApplyForce(dribble_force, self.ball.worldCenter, True)
                self.robot_bodies[i].ApplyForce(-dribble_force, self.robot_bodies[i].worldCenter, True)

                self.robot_aux_states[i].dribbling_count += 1
            else:
                self.robot_aux_states[i].dribbling_count = 0

        self.timestep += 1

        step_reward, done, got_reward = self.task_logic(action)
        if got_reward:
            print(f"Got reward! Episode Timestep: {self.timestep:4}, step_reward: {step_reward:6.2e}")

        return self.state_list(), step_reward, done, {}

    def task_reset(self, scale_factor: float = 0):
        """
        Performs the task specific logic of resetting any additional state
        that it might have
        """
        pass

    def reset(self, scale: float = 1.0) -> np.ndarray:
        """ Reset the env, returning the initial state
        @param scale: How much to scale the IC, for curriculum learning
        @return:
        """
        self.ball = None
        self.state = None
        self.robot_bodies = None
        self._create(scale)  # This sets self.state

        self.timestep = 0

        self.robot_aux_states = [RobotAuxState() for i in range(self.config.num_robots)]

        self.reward = 0.0
        self.prev_reward = 0.0
        self.task_reset(scale)

        return self.state_list()

    def getcolor(self, robot_num: int) -> Tuple[float, float, float]:
        if robot_num == 0:
            return 1.0, 0.3, 0.3
        elif robot_num == 1:
            return 0.3, 1.0, 0.3
        elif robot_num == 2:
            return 0.3, 0.3, 1.0
        else:
            return 0.6, 0.6, 0.6

    def render(self, mode="human", **kwargs):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(WINDOW_H, WINDOW_W)
            self.viewer.set_bounds(VIEW_MIN_X, VIEW_MAX_X,
                                   VIEW_MIN_Y, VIEW_MAX_Y)

            field_fill = rendering.make_polygon([
                (VIEW_MIN_X, VIEW_MIN_Y),
                (VIEW_MAX_X, VIEW_MIN_Y),
                (VIEW_MAX_X, VIEW_MAX_Y),
                (VIEW_MIN_X, VIEW_MAX_Y)
            ])
            field_fill.set_color(0, 0.5, 0)
            self.viewer.add_geom(field_fill)

            field_outline = rendering.make_polyline([
                (FIELD_MIN_X, FIELD_MIN_Y),
                (FIELD_MAX_X, FIELD_MIN_Y),
                (FIELD_MAX_X, FIELD_MAX_Y),
                (FIELD_MIN_X, FIELD_MAX_Y),
                (FIELD_MIN_X, FIELD_MIN_Y),
            ])
            field_outline.set_color(1, 1, 1)
            self.viewer.add_geom(field_outline)

            ball = rendering.make_circle(BALL_RADIUS)
            ball.set_color(0.8, 0.8, 0.3)
            self.ball_transform = rendering.Transform()
            ball.add_attr(self.ball_transform)
            self.viewer.add_geom(ball)

            self.robot_transforms = []
            for i in range(self.config.num_robots):
                robot = rendering.make_polygon(robot_points)
                robot.set_color(0.3, 0.3, 0.3)

                robot_transform = rendering.Transform()
                robot.add_attr(robot_transform)
                self.viewer.add_geom(robot)

                # Draw kicker
                kicker_width = 0.06
                kicker_depth = 0.02
                kicker_buffer = 0.001
                mouth_x = ROBOT_RADIUS * np.cos(ROBOT_MOUTH_ANGLE / 2) + kicker_buffer
                thing_points = [(mouth_x, kicker_width),
                                (mouth_x, -kicker_width),
                                (mouth_x - kicker_depth, -kicker_width),
                                (mouth_x - kicker_depth, kicker_width)]
                robot_thing = rendering.make_polygon(thing_points)
                robot_thing.set_color(*self.getcolor(i))
                robot_thing.add_attr(robot_transform)
                self.viewer.add_geom(robot_thing)

                self.robot_transforms.append(robot_transform)

            # Draw goal
            left_goal = rendering.make_polygon(LEFT_GOAL_POINTS)
            left_goal.set_color(0.6, 0.8, 0.6)
            self.viewer.add_geom(left_goal)

        self.ball_transform.set_translation(
            self.ball.position[0],
            self.ball.position[1]
        )

        for i in range(self.config.num_robots):
            self.robot_transforms[i].set_translation(
                self.robot_bodies[i].position[0],
                self.robot_bodies[i].position[1]
            )
            self.robot_transforms[i].set_rotation(
                math.fmod(math.fmod(self.robot_bodies[i].angle, 2 * math.pi) + 2 * math.pi, 2 * math.pi))

        return self.viewer.render()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
