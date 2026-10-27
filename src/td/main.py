import numpy as np


# environment
gw = np.arange(0, 48).reshape(4, 12)
actions = {
    0: np.array([-1, 0]), # up
    1: np.array([0, 1]),  # right
    2: np.array([1, 0]),  # down
    3: np.array([0, -1]), # left
}
