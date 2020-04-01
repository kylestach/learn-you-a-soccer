import pyglet
from pyglet import gl

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.envs.box2d.car_dynamics import Car
from gym.utils import colorize, seeding, EzPickle

import pyglet
from pyglet import gl

import numpy as np

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
    ROBOT_MOUTH_ANGLE/2,
    2 * np.pi - ROBOT_MOUTH_ANGLE/2,
    15))

robot_points = [
    (ROBOT_RADIUS * np.cos(-a), ROBOT_RADIUS * np.sin(-a)) for a in robot_angles
]

class CollisionDetector(contactListener):
    """
    A collision detector to keep track of when the robot intercepts the ball
    """
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        pass

    def EndContact(self, contact):
        pass

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

class RoboCup(gym.Env, EzPickle):
    def __init__(self, verbose=1):
        EzPickle.__init__(self)
        self.seed()
        self.contactListener_keepref = CollisionDetector(self)
        self.world = Box2D.b2World((0, 0))
        self.viewer = None

        self.ball = None
        self.robot = None

        self.reward = 0.0
        self.prev_reward = 0.0
        self.verbose = verbose

        self.action_space = spaces.Box(
            np.array([-1.0, -1.0, -1.0]),
            np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )

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

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

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

        self.ball = self.world.CreateDynamicBody(
            position=(float(self.state[1][0][0]), float(self.state[1][0][1]))
        )
        self.ball.CreateCircleFixture(
            radius=BALL_RADIUS,
            density=BALL_DENSITY,
            restitution=0.4
        )

        self.ball.linearVelocity[0] = float(self.state[1][1][0])
        self.ball.linearVelocity[1] = float(self.state[1][1][1])

        self.robot = self.world.CreateDynamicBody(
            position=(float(self.state[0][0][0]), float(self.state[0][0][1])),
            angle=float(self.state[0][0][2])
        )
        self.robot.CreatePolygonFixture(
            vertices=robot_points,
            restitution=0.3,
            density=10.0,
        )
        self.robot.CreatePolygonFixture(
            vertices=[
                (ROBOT_RADIUS-0.02, 0.06),
                (ROBOT_RADIUS-0.02, -0.06),
                (0, -0.06),
                (0, 0.06)
            ],
            restitution=0.1
        )
        self.robot.linearVelocity[0] = float(self.state[0][1][0])
        self.robot.linearVelocity[1] = float(self.state[0][1][1])
        self.robot.angularVelocity = float(self.state[0][1][2])
        self.robot.linearDamping = 4
        self.robot.angularDamping = 4
        self.robot.fixedRotation = False

        self.top = self.world.CreateStaticBody(position=(0, 0))
        self.top.CreateEdgeFixture(vertices=[
                (VIEW_MIN_X, VIEW_MIN_Y),
                (VIEW_MAX_X, VIEW_MIN_Y)
            ], restitution=1.0)

        self.bottom = self.world.CreateStaticBody(position=(0, 0))
        self.bottom.CreateEdgeFixture(vertices=[
                (VIEW_MIN_X, VIEW_MAX_Y),
                (VIEW_MAX_X, VIEW_MAX_Y)
            ], restitution=1.0)

        self.left = self.world.CreateStaticBody(position=(0, 0))
        self.left.CreateEdgeFixture(vertices=[
                (VIEW_MIN_X, VIEW_MIN_Y),
                (VIEW_MIN_X, VIEW_MAX_Y)
            ], restitution=1.0)

        self.right = self.world.CreateStaticBody(position=(0, 0))
        self.right.CreateEdgeFixture(vertices=[
                (VIEW_MAX_X, VIEW_MIN_Y),
                (VIEW_MAX_X, VIEW_MAX_Y)
            ], restitution=1.0)

    def _applyBallFriction(self):
        # Apply friction to the ball
        ball_speed = np.sqrt(self.ball.linearVelocity[0] ** 2 + self.ball.linearVelocity[1] ** 2)
        FRICTION_LINEAR_REGION = 0.1
        friction_accel = -0.5 * self.ball.linearVelocity / FRICTION_LINEAR_REGION

        if ball_speed > FRICTION_LINEAR_REGION:
            friction_accel = friction_accel * FRICTION_LINEAR_REGION / ball_speed

        friction = self.ball.mass * friction_accel
        self.ball.ApplyForce(friction, self.ball.worldCenter, False)

    def step(self, action):
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

        self.world.Step(1/60, 6*60, 6*60)

        self._applyBallFriction()

        step_reward = 0
        done = False
        return self.state, step_reward, done, {}

    def reset(self):
        self._destroy()
        self.ball = None
        self.state = None
        self._create()

    def render(self):
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

        self.ball_transform.set_translation(
            self.ball.position[0],
            self.ball.position[1]
        )
        print(self.ball.position)

        self.robot_transform.set_translation(
            self.robot.position[0],
            self.robot.position[1]
        )
        self.robot_transform.set_rotation(self.robot.angle)

        return self.viewer.render()

if __name__ == '__main__':
    from pyglet.window import key
    restart = False
    env = RoboCup()

    force = [0, 0, 0]

    def key_press(k, mod):
        global restart
        global force
        if k == key.SPACE:
            restart = True
        if k == key.UP:
            force[1] = 3
        if k == key.DOWN:
            force[1] = -3
        if k == key.LEFT:
            force[0] = -3
        if k == key.RIGHT:
            force[0] = 3
        if k == key.A:
            force[2] = -0.1
        if k == key.D:
            force[2] = 0.1
    def key_release(k, mod):
        global force
        if k == key.UP:
            force[1] = 0
        if k == key.DOWN:
            force[1] = 0
        if k == key.LEFT:
            force[0] = 0
        if k == key.RIGHT:
            force[0] = 0
        if k == key.A:
            force[2] = 0
        if k == key.D:
            force[2] = 0

    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

    is_open = True
    while is_open:
        env.reset()
        restart = False
        while True:
            a = None
            env.robot.ApplyForce(force[:2], env.robot.worldCenter, True)
            env.robot.ApplyTorque(force[2], True)
            print(env.robot.angularVelocity)
            s, r, done, info = env.step(a)
            is_open = env.render()
            if done or restart:
                break
