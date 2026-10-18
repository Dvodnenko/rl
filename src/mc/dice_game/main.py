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


def episode():
    s = 0 # initial sum is 0
    while s <= 30:
        a = b(s)
        if a == HOLD: break
        r = roll()
        if r == 1:
            return 0
        s = min(s+r, 30)
    return s
