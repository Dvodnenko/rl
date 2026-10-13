"""
implementation of monte carlo evaluation for grid
world from chapter 4. only includes prediction, no 
improvement yet
"""


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
    new = s + actions[a]

    if (new[0] < 0 or new[0] > 3) \
        or (new[1] < 0 or new[1] > 3): # if the action would take agent out of grid...
        return s # don't move

    return new


def episode(s: np.ndarray):
    G = 0

    while gw[s[0], s[1]] not in (0, 15): # while in non-terminal state
        s = move(s, random.choice([0, 1, 2, 3]))
        G -= 1
    return G


def evaluate(s: np.ndarray):
    """evaluates v_π(s) for π = equiprobable policy"""
    n_episodes = 100

    returns = np.zeros(n_episodes)

    for i in range(n_episodes):
        G = episode(s) # computes return of an episode 
        returns[i] = G

    avg = np.average(returns)
    return avg


if __name__ == "__main__":
    s = np.array([3, 0])
    v = evaluate(s)
    print(v)
