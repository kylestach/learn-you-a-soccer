from typing import Tuple, List
import numpy as np

# x, y, w, kick_power
CollectAction = List[float]

# [x, y, theta], [xdot, ydot, thetadot]
RobotState = Tuple[np.ndarray, np.ndarray]
# [x, y, theta], [xdot, ydot, thetadot]
BallState = Tuple[np.ndarray, np.ndarray]
RoboCupState = np.array

ROBOTSTATE_SIZE = 7

ROBOT_X = 0
ROBOT_Y = 1
ROBOT_H = 2
ROBOT_BALLSENSE = 3
ROBOT_DX = 4
ROBOT_DY = 5
ROBOT_DH = 6

BALL_X = -4
BALL_Y = -3
BALL_DX = -2
BALL_DY = -1

OBS_ROBOTSTATE_SIZE = 8

OBS_ROBOT_X = 0
OBS_ROBOT_Y = 1
OBS_ROBOT_SINH = 2
OBS_ROBOT_COSH = 3
OBS_ROBOT_BALLSENSE = 4
OBS_ROBOT_DX = 5
OBS_ROBOT_DY = 6
OBS_ROBOT_DH = 7

OBS_BALL_X = -4
OBS_BALL_Y = -3
OBS_BALL_DX = -2
OBS_BALL_DY = -1


def getstate(robot_num: int, property: int):
    return robot_num * ROBOTSTATE_SIZE + property


def getobsstate(robot_num: int, property: int):
    return robot_num * OBS_ROBOTSTATE_SIZE + property
