import random

import numpy as np


# environment
gw = np.arange(0, 16).reshape(4, 4)
actions = {
    0: np.array([-1, 0]), # up
    1: np.array([0, 1]),  # right
    2: np.array([1, 0]),  # down
    3: np.array([0, -1]), # left
}


def move(s: np.ndarray, a: int):
    if a == 0 and s[0] == 0:
        return s
    elif a == 1 and s[1] == 3:
        return s
    elif a == 2 and s[0] == 3:
        return s
    elif a == 3 and s[1] == 0:
        return s
    return s + actions[a]
