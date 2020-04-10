from typing import Tuple, Optional
import Box2D
from Box2D import b2ContactListener, b2Contact, b2Body

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

import numpy as np
import math

from robocup_env.envs.robocup_types import *

WINDOW_W = 600
WINDOW_H = 900

FIELD_WIDTH = 8.0
FIELD_HEIGHT = 5.0

FIELD_SCALE = 1 / 2
FIELD_MIN_X = -4.0 * FIELD_SCALE
FIELD_MAX_X = 4.0 * FIELD_SCALE
FIELD_MIN_Y = -2.5 * FIELD_SCALE
FIELD_MAX_Y = 2.5 * FIELD_SCALE

VIEW_MIN_X = -4.5 * FIELD_SCALE
VIEW_MAX_X = 4.5 * FIELD_SCALE
VIEW_MIN_Y = -3.0 * FIELD_SCALE
VIEW_MAX_Y = 3.0 * FIELD_SCALE

BALL_RADIUS = 0.02
BALL_DENSITY = 0.7

NUM_ROBOTS = 1

ROBOT_RADIUS = 0.09
ROBOT_MOUTH_ANGLE = np.deg2rad(80)
robot_angles = list(np.linspace(
    ROBOT_MOUTH_ANGLE / 2,
    2 * np.pi - ROBOT_MOUTH_ANGLE / 2,
    15))

MAX_TIMESTEPS = 600

GOAL_HEIGHT = FIELD_HEIGHT / 6
GOAL_DEPTH = 0.3

LEFT_GOAL_POINTS = [
    (FIELD_MIN_X - GOAL_DEPTH, GOAL_HEIGHT / 2),
    (FIELD_MIN_X, GOAL_HEIGHT / 2),
    (FIELD_MIN_X, -GOAL_HEIGHT / 2),
    (FIELD_MIN_X - GOAL_DEPTH, -GOAL_HEIGHT / 2)
]

robot_points = [
    (ROBOT_RADIUS * np.cos(-a), ROBOT_RADIUS * np.sin(-a)) for a in robot_angles
]

KICK_THRESHOLD = 0.5
FRICTION_LINEAR_REGION = 0.1


class CollisionDetector(b2ContactListener):
    """
    A collision detector to keep track of when the robot intercepts the ball
    """

    def __init__(self, env):
        b2ContactListener.__init__(self)
        self.env = env

    def has_ball(self, contact: b2Contact) -> bool:
        return contact.fixtureA.userData['type'] == 'ball' or contact.fixtureB.userData['type'] == 'ball'

    def has_dribbler_sense(self, contact: b2Contact) -> bool:
        return contact.fixtureA.userData['type'] == 'dribbler' or contact.fixtureB.userData['type'] == 'dribbler'

    def has_kicker(self, contact: b2Contact) -> bool:
        return contact.fixtureA.userData['type'] == 'kicker' or contact.fixtureB.userData['type'] == 'kicker'

    def has_robot(self, contact: b2Contact) -> bool:
        return contact.fixtureA.body.userData['type'] == 'robot' or contact.fixtureB.body.userData['type'] == 'robot'

    def BeginContact(self, contact: b2Contact):
        if self.has_ball(contact) and self.has_robot(contact):
            self.env.has_touched = True
            if self.has_dribbler_sense(contact):
                self.env.dribbling = True
            elif self.has_kicker(contact):
                self.env.can_kick = True

    def EndContact(self, contact):
        if 'type' not in contact.fixtureA.body.userData or 'type' not in contact.fixtureB.body.userData:
            return
        if self.has_ball(contact) and self.has_robot(contact):
            if self.has_dribbler_sense(contact):
                self.env.dribbling = False
            elif self.has_kicker(contact):
                self.env.can_kick = False


class VelocitySpace(spaces.Space):
    """
    Sample a velocity in n-dimensional space. Sampling occurs from a normal
    distribution with given (independent) standard deviation
    """

    def __init__(self, shape, stdev):
        spaces.Space.__init__(self, shape, np.float32)
        self.stdev = stdev
        self.shape = shape

    def sample(self):
        return self.stdev * np.random.randn(*self.shape)

    def contains(self, x):
        return True


class RoboCupStateSpace(spaces.Space):
    """
    Sample 2-dimensional position and velocity.
    rx, ry, rh, rdx, rdy, rdh
    bx, by, bdx, bdy
    """

    def __init__(self, num_robots, sincos: bool = False):
        super().__init__()
        if sincos:
            self.pose_space = spaces.Box(
                np.array([FIELD_MIN_X, FIELD_MIN_Y, -1, -1]),
                np.array([FIELD_MAX_X, FIELD_MAX_Y, 1, 1]),
                dtype=np.float32
            )
        else:
            self.pose_space = spaces.Box(
                np.array([FIELD_MIN_X, FIELD_MIN_Y, 0]),
                np.array([FIELD_MAX_X, FIELD_MAX_Y, 2 * np.pi]),
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

        if sincos:
            self.shape = (num_robots * 7 + 4,)
        else:
            self.shape = (num_robots * 6 + 4,)

    def sample(self):
        return np.concatenate([
                                  np.concatenate([
                                      self.pose_space.sample(),
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
                                      self.robot_velocity_space.sample()]) for _ in range(self.num_robots)
                              ] + [
                                  np.concatenate([
                                      self.position_space.sample(),
                                      self.ball_velocity_space.sample()
                                  ])
                              ])

    def contains(self, x):
        return (self.pose_space.contains(x[:3]) and
                self.robot_velocity_space.contains(x[3:6]) and
                self.position_space.contains(x[6:8]) and
                self.ball_velocity_space.contains(x[8:]))


class CollectRewardConfig:
    def __init__(self,
                 dribbling_reward: float = 0.1,
                 done_reward_additive: float = 0.0,
                 done_reward_coeff: float = 500.0,
                 done_reward_exp_base: float = 0.998,
                 ball_out_of_bounds_reward: float = -30.0,
                 move_reward: float = -0.3,
                 # It's reward, not cost. Should be negative to reward lower distances
                 distance_to_ball_coeff: float = 0.0,
                 ):
        self.dribbling_reward = dribbling_reward

        self.done_reward_additive = done_reward_additive
        self.done_reward_coeff = done_reward_coeff
        self.done_reward_exp_base = done_reward_exp_base

        self.move_reward = move_reward

        self.ball_out_of_bounds_reward = ball_out_of_bounds_reward

        self.distance_to_ball_coeff = distance_to_ball_coeff


class InitialConditionConfig:
    def __init__(self, max_ball_velocity: float = 0.5):
        self.max_ball_velocity = max_ball_velocity


class CollectEnvConfig:
    def __init__(self, dribble_count_done: int = 100,
                 collect_reward_config: CollectRewardConfig = CollectRewardConfig(),
                 initial_condition_config: InitialConditionConfig = InitialConditionConfig()):
        self.dribble_count_done: int = dribble_count_done
        self.ball_dist_done: float = 0.1
        self.collect_reward_config: CollectRewardConfig = collect_reward_config
        self.initial_condition_config: InitialConditionConfig = initial_condition_config


class RoboCup(gym.Env, EzPickle):
    def __init__(self, collect_env_config: CollectEnvConfig = CollectEnvConfig(), verbose=1, sincos_embedding=False):
        EzPickle.__init__(self)
        self.seed()
        self.viewer = None

        self.world = None
        self.contactListener_keepref = None

        self.ball: Optional[b2Body] = None
        self.robot: Optional[b2Body] = None
        self.config: CollectEnvConfig = collect_env_config

        self.has_touched = None
        self.kicked_ball = False

        self._max_episode_steps = MAX_TIMESTEPS

        self.timestep = 0

        self.dribbling = False
        self.can_kick = False
        self.kick_cooldown = 0
        self.dribbling_count = 0

        self.reward = 0.0
        self.prev_reward = 0.0
        self.verbose = verbose

        self.action_space = spaces.Box(
            np.array([-1.0, -1.0, -1.0, 0]),
            np.array([1.0, 1.0, 1.0, 1]),
            dtype=np.float32
        )

        self.sincos = sincos_embedding
        self.observation_space = RoboCupStateSpace(1, sincos_embedding)

        from gym.envs.classic_control import rendering
        self.ball_transform: Optional[rendering.Transform] = None
        self.robot_transform: Optional[rendering.Transform] = None

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def state_list(self) -> np.ndarray:
        # Observation space is:
        # [ robot0_x robot0_y robot0_h robot0_vx robot0_vy robot0_vh ]
        # ,
        # [ ball_x ball_y ball_vx ball_vy ]

        # If the observations use a sin/cos embedding for the angle
        if self.sincos:
            obs = np.array([
                self.state[0], self.state[1], np.sin(self.state[2]), np.cos(self.state[2]),
                self.state[3], self.state[4], self.state[5], self.state[6], self.state[7],
                self.state[8], self.state[9]
            ])
            return obs
        else:
            return self.state

    def _destroy(self):
        if not self.world:
            return

        for body in [
            'ball',
            'top',
            'bottom',
            'left',
            'right',
            'robot',
            'goal',
        ]:
            if body in self.__dict__ and self.__dict__[body] is not None:
                self.world.DestroyBody(self.__dict__[body])
                self.__dict__[body] = None
        self.contactListener_keepref = None
        self.world = None

    def _create(self):
        self.contactListener_keepref = CollisionDetector(self)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)

        self.has_touched = False
        self.kicked_ball = False

        # Create the goal, robots, ball and field
        self.state: RoboCupState = self.observation_space.sample_state()
        self.state[6] *= 0.5
        self.state[7] *= 0.5

        conf: InitialConditionConfig = self.config.initial_condition_config
        self.state[BALL_DX] *= conf.max_ball_velocity
        self.state[BALL_DY] *= conf.max_ball_velocity

        # =========== Ball ====================================================
        self.ball = self.world.CreateDynamicBody(
            position=(float(self.state[6]), float(self.state[7])),
            userData={"type": "ball"},
        )
        self.ball.CreateCircleFixture(
            radius=BALL_RADIUS,
            density=BALL_DENSITY,
            restitution=0.2,
            userData={"type": "ball"},
        )

        self.ball.linearVelocity[0] = float(self.state[8])
        self.ball.linearVelocity[1] = float(self.state[9])

        # =========== Robot ====================================================
        self.robot = self.world.CreateDynamicBody(
            position=(float(self.state[0]), float(self.state[1])),
            angle=float(self.state[2]),
            userData={"type": "robot"},
        )
        self.robot.CreatePolygonFixture(
            vertices=robot_points,
            restitution=0.3,
            density=100.0,
            userData={"type": "robot"},
        )
        # Kicker rectangle
        self.robot.CreatePolygonFixture(
            vertices=[
                (ROBOT_RADIUS - 0.02, 0.06),
                (ROBOT_RADIUS - 0.02, -0.06),
                (0, -0.06),
                (0, 0.06)
            ],
            restitution=0.15,
            userData={"type": "kicker"}
        )

        self.robot.linearVelocity[0] = float(self.state[3])
        self.robot.linearVelocity[1] = float(self.state[4])
        self.robot.angularVelocity = float(self.state[5])
        self.robot.linearDamping = 3
        self.robot.angularDamping = 5
        self.robot.fixedRotation = False

        # =========== Dribbler Sense ====================================================
        self.dribbler_sense = self.robot.CreatePolygonFixture(
            vertices=[
                (ROBOT_RADIUS, 0.06),
                (ROBOT_RADIUS, -0.06),
                (0, -0.06),
                (0, 0.06)
            ],
            isSensor=True,
            userData={"type": "dribbler"}
        )

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

    def step(self, action: CollectAction) -> Tuple[np.ndarray, float, bool, dict]:
        # print([(b.userData['type'], [f.userData['type'] for f in b.fixtures]) for b in self.world.bodies])
        # print(self.dribbling, self.ball.position, self.robot.position)
        # Gather the entire state.
        self.state = np.array([
            self.robot.position[0],
            self.robot.position[1],
            self.robot.angle,
            self.robot.linearVelocity[0],
            self.robot.linearVelocity[1],
            self.robot.angularVelocity,
            self.ball.position[0],
            self.ball.position[1],
            self.ball.linearVelocity[0],
            self.ball.linearVelocity[1],
        ])

        self.world.Step(1 / 60, 6 * 60, 6 * 60)

        self._apply_ball_friction()

        self.robot.ApplyForce(
            [5 * self.robot.mass * action[0], 5 * self.robot.mass * action[1]],
            self.robot.worldCenter, True)
        self.robot.ApplyTorque(self.robot.inertia * 80 * action[2], True)

        self.kick_cooldown += 1
        if self.can_kick and self.kick_cooldown > 30 and (3.0 * action[3]) > KICK_THRESHOLD:
            self.kick_cooldown = 0
            shoot_magnitude = self.ball.mass * (3.0 * action[3])
            shoot_impulse = [shoot_magnitude * np.cos(self.robot.angle),
                             shoot_magnitude * np.sin(self.robot.angle)]
            self.ball.ApplyLinearImpulse(shoot_impulse, self.robot.worldCenter, True)
            self.kicked_ball = True

        if self.dribbling:
            # Apply force on the ball towards the center of the robot
            dribble_vec = self.robot.position - self.ball.position
            dribble_force = dribble_vec / np.sqrt(dribble_vec[0] ** 2 + dribble_vec[1] ** 2) * 5e-3
            self.ball.ApplyForce(dribble_force, self.ball.worldCenter, True)
            self.robot.ApplyForce(-dribble_force, self.robot.worldCenter, True)

            self.dribbling_count += 1
        else:
            self.dribbling_count = 0
        done_dribbled = self.dribbling_count >= self.config.dribble_count_done
        if self.dribbling_count == self.config.dribble_count_done:
            print("Dribbling!")

        self.timestep += 1

        reward_config = self.config.collect_reward_config

        ball_in_left_goal = (VIEW_MIN_X <= self.ball.position[0] <= FIELD_MIN_X) and (
                -GOAL_HEIGHT / 2 < self.ball.position[1] < GOAL_HEIGHT / 2)

        # done = ball_in_left_goal and self.kicked_ball
        done = done_dribbled

        dist = np.sqrt((self.robot.position[0] - self.ball.position[0]) ** 2 + (
                self.robot.position[1] - self.ball.position[1]) ** 2)
        if done:
            step_reward = reward_config.done_reward_additive + \
                          reward_config.done_reward_coeff * reward_config.done_reward_exp_base ** self.timestep
            print(f"Got done reward: {step_reward}")
        elif self.dribbling and not done_dribbled:
            step_reward = reward_config.dribbling_reward
        else:
            step_reward = reward_config.distance_to_ball_coeff * dist

        step_reward += reward_config.move_reward * np.sum(action[:3] ** 2)

        # ####################################
        # # TRANSFER LEARNING: Tracking ball #
        # ####################################
        # dist = np.sqrt((self.robot.position[0] - self.ball.position[0]) ** 2 + (
        #         self.robot.position[1] - self.ball.position[1]) ** 2)

        # done = dist < self.config.ball_dist_done or self.dribbling_count >= 1

        # if done:
        #     step_reward = reward_config.done_reward_additive + \
        #                   reward_config.done_reward_coeff * reward_config.done_reward_exp_base ** self.timestep
        # else:
        #     step_reward = reward_config.distance_to_ball_coeff * dist

        # If the ball is out of bounds, and not in the goal, we're done but with low reward
        if not done and (self.ball.position[0] < FIELD_MIN_X or
                         self.ball.position[0] > FIELD_MAX_X or
                         self.ball.position[1] < FIELD_MIN_Y or
                         self.ball.position[1] > FIELD_MAX_Y):
            done = True
            # if self.has_touched:
            #     step_reward = reward_config.ball_out_of_bounds_reward
            # else:
            #     step_reward = 0
            step_reward = reward_config.ball_out_of_bounds_reward

        # If time limit exceeded then we're done
        if self.timestep > MAX_TIMESTEPS:
            done = True
            step_reward = 0

        return self.state_list(), step_reward, done, {}

    def reset(self):
        # self._destroy()
        self.ball = None
        self.robot = None
        self.state = None
        self.dribbler_sense = None
        self.kicker = None
        self._create()  # This sets self.state

        self.timestep = 0

        self.dribbling = False
        self.can_kick = False
        self.kick_cooldown = 0
        self.dribbling_count = 0

        self.reward = 0.0
        self.prev_reward = 0.0

        return self.state_list()

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

            robot = rendering.make_polygon(robot_points)
            robot.set_color(0.3, 0.3, 0.3)
            self.robot_transform = rendering.Transform()
            robot.add_attr(self.robot_transform)
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
            robot_thing.set_color(1.0, 0.3, 0.3)
            robot_thing.add_attr(self.robot_transform)
            self.viewer.add_geom(robot_thing)

            # Draw goal
            left_goal = rendering.make_polygon(LEFT_GOAL_POINTS)
            left_goal.set_color(0.6, 0.8, 0.6)
            self.viewer.add_geom(left_goal)

        self.ball_transform.set_translation(
            self.ball.position[0],
            self.ball.position[1]
        )

        self.robot_transform.set_translation(
            self.robot.position[0],
            self.robot.position[1]
        )
        self.robot_transform.set_rotation(
            math.fmod(math.fmod(self.robot.angle, 2 * math.pi) + 2 * math.pi, 2 * math.pi))

        return self.viewer.render()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
