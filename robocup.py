from typing import Tuple, Optional, List
import Box2D
from Box2D import b2ContactListener, b2Contact, b2Body

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

import numpy as np

from robocup_types import CollectAction, RoboCupState

WINDOW_W = 600
WINDOW_H = 900

FIELD_WIDTH = 8.0
FIELD_HEIGHT = 5.0

FIELD_SCALE = 1 / 3
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

MAX_TIMESTEPS = 100

robot_points = [
    (ROBOT_RADIUS * np.cos(-a), ROBOT_RADIUS * np.sin(-a)) for a in robot_angles
]

KICK_THRESHOLD = 0.5e-3
FRICTION_LINEAR_REGION = 0.1


class CollisionDetector(b2ContactListener):
    """
    A collision detector to keep track of when the robot intercepts the ball
    """

    def __init__(self, env):
        b2ContactListener.__init__(self)
        self.env = env

    def has_ball(self, contact: b2Contact) -> bool:
        return contact.fixtureA.body == self.env.ball or contact.fixtureB.body == self.env.ball

    def has_dribbler_sense(self, contact: b2Contact) -> bool:
        return contact.fixtureA == self.env.dribbler_sense or contact.fixtureB == self.env.dribbler_sense

    def has_robot(self, contact: b2Contact) -> bool:
        return contact.fixtureA.body == self.env.robot or contact.fixtureB.body == self.env.robot

    def BeginContact(self, contact: b2Contact):
        if self.has_ball(contact) and self.has_robot(contact):
            if self.has_dribbler_sense(contact):
                self.env.dribbling = True
            else:
                self.env.can_kick = True

    def EndContact(self, contact):

        if self.has_ball(contact) and self.has_robot(contact):
            if self.has_dribbler_sense(contact):
                self.env.dribbling = False
            else:
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


class CollectRewardConfig:
    def __init__(self,
                 dribbling_reward: float = 1.0,
                 done_reward_additive: float = 0.0,
                 done_reward_coeff: float = 100.0,
                 done_reward_exp_base: float = 0.999,
                 ball_out_of_bounds_reward: float = 0.0,
                 ):
        self.dribbling_reward = dribbling_reward

        self.done_reward_additive = done_reward_additive
        self.done_reward_coeff = done_reward_coeff
        self.done_reward_exp_base = done_reward_exp_base

        self.ball_out_of_bounds_reward = ball_out_of_bounds_reward


class CollectEnvConfig:
    def __init__(self, dribble_count_done: int = 50,
                 collect_reward_config: CollectRewardConfig = CollectRewardConfig()):
        self.dribble_count_done: int = dribble_count_done
        self.collect_reward_config: CollectRewardConfig = collect_reward_config


class RoboCup(gym.Env, EzPickle):
    def __init__(self, collect_env_config: CollectEnvConfig = CollectEnvConfig(), verbose=1):
        EzPickle.__init__(self)
        self.seed()
        self.contactListener_keepref = CollisionDetector(self)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.viewer = None

        self.ball: Optional[b2Body] = None
        self.robot: Optional[b2Body] = None
        self.config: CollectEnvConfig = collect_env_config

        self.timestep = 0

        self.dribbling = False
        self.can_kick = False
        self.kick_cooldown = 0
        self.dribbling_count = 0

        self.reward = 0.0
        self.prev_reward = 0.0
        self.verbose = verbose

        self.action_space = spaces.Box(
            np.array([-3.0, -3.0, -3.0, 0]),
            np.array([3.0, 3.0, 3.0, 2.5e-3]),
            dtype=np.float32
        )

        self.state: RoboCupState = ((np.zeros(3), np.zeros(3)), (np.zeros(2), np.zeros(2)))

        # Observation space is:
        # [ robot0_x robot0_y robot0_h robot0_vx robot0_vy robot0_vh ]
        # ,
        # [ ball_x ball_y ball_vx ball_vy ]
        self.robot_space = spaces.Tuple([
            spaces.Box(
                np.array([FIELD_MIN_X, FIELD_MIN_Y, 0]),
                np.array([FIELD_MAX_X, FIELD_MAX_Y, np.pi * 2]),
                dtype=np.float32
            ),
            VelocitySpace((3,), np.array([0.5, 0.5, 3.0]))
        ])
        self.ball_space = spaces.Tuple([
            spaces.Box(
                np.array([FIELD_MIN_X, FIELD_MIN_Y]),
                np.array([FIELD_MAX_X, FIELD_MAX_Y]),
                dtype=np.float32
            ),
            VelocitySpace((2,), np.array([2.0, 2.0]))
        ])

        self.observation_space = spaces.Tuple([
            self.robot_space,
            self.ball_space
        ])

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
        return np.array(
            [self.state[0][0][0], self.state[0][0][1], self.state[0][0][2], self.state[0][1][0], self.state[0][1][0],
             self.state[0][1][2], self.state[1][0][0], self.state[1][0][1], self.state[1][1][0], self.state[1][1][1]])

    def _destroy(self):
        for body in [
            'ball',
            'top',
            'bottom',
            'left',
            'right',
            'robot',
        ]:
            if body in self.__dict__ and self.__dict__[body] is not None:
                self.world.DestroyBody(self.__dict__[body])
                self.__dict__[body] = None

    def _create(self):
        # Create the goal, robots, ball and field
        self.state = self.observation_space.sample()

        ball_data = {
            "id": 0
        }
        self.ball = self.world.CreateDynamicBody(
            position=(float(self.state[1][0][0]) * 0.3, float(self.state[1][0][1]) * 0.3),
            userData=ball_data
        )
        self.ball.CreateCircleFixture(
            radius=BALL_RADIUS,
            density=BALL_DENSITY,
            restitution=0.2
        )

        self.ball.linearVelocity[0] = float(self.state[1][1][0]) * 0.5
        self.ball.linearVelocity[1] = float(self.state[1][1][1]) * 0.5

        ball_data = {
            "id": 1
        }
        self.robot = self.world.CreateDynamicBody(
            position=(float(self.state[0][0][0]), float(self.state[0][0][1])),
            angle=float(self.state[0][0][2]),
            userData=ball_data
        )
        self.robot.CreatePolygonFixture(
            vertices=robot_points,
            restitution=0.3,
            density=100.0,
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
            userData={"kicker": True}
        )
        # Dribbler sense
        self.dribbler_sense = self.robot.CreatePolygonFixture(
            vertices=[
                (ROBOT_RADIUS + 0.00, 0.06),
                (ROBOT_RADIUS + 0.0, -0.06),
                (0, -0.06),
                (0, 0.06)
            ],
            isSensor=True
        )

        self.robot.linearVelocity[0] = float(self.state[0][1][0])
        self.robot.linearVelocity[1] = float(self.state[0][1][1])
        self.robot.angularVelocity = float(self.state[0][1][2])
        self.robot.linearDamping = 3
        self.robot.angularDamping = 5
        self.robot.fixedRotation = False

        wall_data = {
            "id": -1
        }
        self.top: Box2D.b2Body = self.world.CreateStaticBody(position=(0, 0), userData=wall_data)
        self.top.CreateEdgeFixture(vertices=[
            (VIEW_MIN_X, VIEW_MIN_Y),
            (VIEW_MAX_X, VIEW_MIN_Y)
        ], restitution=0.7)

        self.bottom = self.world.CreateStaticBody(position=(0, 0), userData=wall_data)
        self.bottom.CreateEdgeFixture(vertices=[
            (VIEW_MIN_X, VIEW_MAX_Y),
            (VIEW_MAX_X, VIEW_MAX_Y)
        ], restitution=0.7)

        self.left = self.world.CreateStaticBody(position=(0, 0), userData=wall_data)
        self.left.CreateEdgeFixture(vertices=[
            (VIEW_MIN_X, VIEW_MIN_Y),
            (VIEW_MIN_X, VIEW_MAX_Y)
        ], restitution=0.7)

        self.right = self.world.CreateStaticBody(position=(0, 0), userData=wall_data)
        self.right.CreateEdgeFixture(vertices=[
            (VIEW_MAX_X, VIEW_MIN_Y),
            (VIEW_MAX_X, VIEW_MAX_Y)
        ], restitution=0.7)

    def _apply_ball_friction(self):
        # Apply friction to the ball
        ball_speed = np.sqrt(self.ball.linearVelocity[0] ** 2 + self.ball.linearVelocity[1] ** 2)
        friction_accel = -0.5 * self.ball.linearVelocity / FRICTION_LINEAR_REGION

        if ball_speed > FRICTION_LINEAR_REGION:
            friction_accel = friction_accel * FRICTION_LINEAR_REGION / ball_speed

        friction = self.ball.mass * friction_accel
        self.ball.ApplyForce(friction, self.ball.worldCenter, False)

    def step(self, action: CollectAction) -> Tuple[np.ndarray, float, bool, dict]:
        # Gather the entire state.
        robot_state = (
            np.array([self.robot.position[0], self.robot.position[1], self.robot.angle]),
            np.array([
                self.robot.linearVelocity[0],
                self.robot.linearVelocity[1],
                self.robot.angularVelocity
            ]),
        )
        ball_state = (np.array([self.ball.position[0], self.ball.position[1]]),
                      np.array([self.ball.linearVelocity[0], self.ball.linearVelocity[1]]))
        self.state = (robot_state, ball_state)

        self.world.Step(1 / 60, 6 * 60, 6 * 60)

        self._apply_ball_friction()

        self.robot.ApplyForce(
            [self.robot.mass * action[0], self.robot.mass * action[1]],
            self.robot.worldCenter, True)
        self.robot.ApplyTorque(self.robot.inertia * action[2], True)

        self.kick_cooldown += 1
        if self.can_kick and self.kick_cooldown > 30 and action[3] > KICK_THRESHOLD:
            self.kick_cooldown = 0
            shoot_magnitude = action[3]
            shoot_impulse = [shoot_magnitude * np.cos(self.robot.angle), shoot_magnitude * np.sin(self.robot.angle)]
            self.ball.ApplyLinearImpulse(shoot_impulse, self.robot.worldCenter, True)

        if self.dribbling:
            # Apply force on the ball towards the center of the robot
            dribble_vec = self.robot.position - self.ball.position
            dribble_force = dribble_vec / np.sqrt(dribble_vec[0] ** 2 + dribble_vec[1] ** 2) * 5e-3
            self.ball.ApplyForce(dribble_force, self.ball.worldCenter, True)
            self.robot.ApplyForce(-dribble_force, self.robot.worldCenter, True)

            self.dribbling_count += 1
        else:
            self.dribbling_count = 0

        self.timestep += 1

        done = self.dribbling_count >= self.config.dribble_count_done

        reward_config = self.config.collect_reward_config
        if done:
            step_reward = reward_config.done_reward_additive + \
                          reward_config.done_reward_coeff * reward_config.done_reward_exp_base ** self.timestep
        elif self.dribbling:
            step_reward = reward_config.dribbling_reward
        else:
            step_reward = 0

        # If the ball is out of bounds, we're done but with low reward
        if (self.ball.position[0] < FIELD_MIN_X or
                self.ball.position[0] > FIELD_MAX_X or
                self.ball.position[1] < FIELD_MIN_Y or
                self.ball.position[1] > FIELD_MAX_Y):
            done = True
            step_reward = reward_config.ball_out_of_bounds_reward

        # If time limit exceeded then we're done
        if self.timestep > MAX_TIMESTEPS:
            done = True
            step_reward = 0

        return self.state_list(), step_reward, done, {}

    def reset(self):
        self._destroy()
        self.ball = None
        self.robot = None
        self.state = None
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

        self.ball_transform.set_translation(
            self.ball.position[0],
            self.ball.position[1]
        )

        self.robot_transform.set_translation(
            self.robot.position[0],
            self.robot.position[1]
        )
        self.robot_transform.set_rotation(self.robot.angle)

        return self.viewer.render()
