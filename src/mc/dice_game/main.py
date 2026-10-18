import random

import numpy as np


def roll():
    return random.randint(1, 6)

Q_values = np.zeros(62).reshape(31, 2)
ROLL = 0
HOLD = 1

# behaviour policy, rolls with probability 80%,
# hold with probability 20%
b = lambda state: ROLL if random.random() < .8 else HOLD
# target policy, deterministic & greedy w.r.t. Q
p = lambda state: np.argmax(Q_values[state])
