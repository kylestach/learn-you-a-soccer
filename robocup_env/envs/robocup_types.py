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
ROBOT_H = 2
ROBOT_BALLSENSE = 3
ROBOT_DX = 4
ROBOT_DY = 5
ROBOT_DH = 6
BALL_X = 7
BALL_Y = 8
BALL_DX = 9
BALL_DY = 10
