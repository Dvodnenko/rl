import random

import numpy as np


def roll():
    return random.randint(1, 6)

Q_values = np.zeros(62).reshape(31, 2)
ROLL = 0
HOLD = 1
