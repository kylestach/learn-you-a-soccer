from typing import Tuple, List
import numpy as np

# x, y, w, kick_power
CollectAction = List[float]

# [x, y, theta], [xdot, ydot, thetadot]
RobotState = Tuple[np.ndarray, np.ndarray]
# [x, y, theta], [xdot, ydot, thetadot]
BallState = Tuple[np.ndarray, np.ndarray]
RoboCupState = np.array

BALL_DX = 8
BALL_DY = 9
