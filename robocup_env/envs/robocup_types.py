from typing import Tuple, List
import numpy as np

# x, y, w, kick_power
CollectAction = List[float]

# [x, y, theta], [xdot, ydot, thetadot]
RobotState = Tuple[np.ndarray, np.ndarray]
# [x, y, theta], [xdot, ydot, thetadot]
BallState = Tuple[np.ndarray, np.ndarray]
RoboCupState = np.array

ROBOT_X = 0
ROBOT_Y = 1
ROBOT_HX = 2
ROBOT_HY = 3
ROBOT_BALLSENSE = 4
ROBOT_DX = 5
ROBOT_DY = 6
ROBOT_DH = 7
BALL_X = 8
BALL_Y = 9
BALL_DX = 10
BALL_DY = 11
