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


def episode(pi: list, s: np.ndarray, a: int):
    G = 0

    if gw[s[0], s[1]] in (0, 15):
        return G

    s = move(s, a)
    G -= 1

    while (v := gw[s[0], s[1]]) not in (0, 15): # while in non-terminal state
        s = move(s, random.choice(pi[v]))
        G -= 1
    return G


def q_pi(pi: list, s: np.ndarray, a: int):
    """
    evaluates q_π(s,a)
    """

    n_episodes = 1000
    returns = np.zeros(n_episodes)

    for i in range(n_episodes):
        G = episode(pi, s, a) # computes return of an episode 
        returns[i] = G

    avg = np.average(returns)
    return avg


def all_q_values(pi: list):
    q_values = np.zeros(64).reshape(16, 4)

    states = [np.array([y, x]) 
        for y in range(0, 4) for x in range(0, 4)]
    actions = (0, 1, 2, 3)

    for s in states:
        number: int = gw[s[0], s[1]]
        for a in actions:
            q_values[number, a] = q_pi(pi, s, a)
            print(f"state {number}, action {a} completed")

    return q_values


if __name__ == "__main__":
    random_policy = [[a for a in range (4)] for _ in range(16)]
    s = np.array([0, 2])
    q = q_pi(random_policy, s, 3)
    print(q)
