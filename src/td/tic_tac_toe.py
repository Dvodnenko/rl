import random

import numpy as np


# hyperparemeters
alpha: float = .5
gamma: float = 1 # undiscounted task
epsilon: float = .2


def A(s: str) -> list[int]:
    # returns actions available in state s
    a = []
    for i, v in enumerate(list(s)):
        if v == '-':
            a.append(i)
    return a

Q_x: dict[str, dict[int, float]] = {}
Q_o: dict[str, dict[int, float]] = {}

def update(Q, alpha, gamma, s, a, r, s_prime) -> float:
    return Q[s,a] + alpha*(
        r + gamma*max(Q[s], key=Q[s].get) - Q[s,a])
