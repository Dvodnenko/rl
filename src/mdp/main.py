# Implementation of the grid world
# problem from S&B

import time

import numpy as np


class Policy:
    def __init__(self):
        ...

    def pi_a_s(self, action, state):
        "π(a|s) - probability of picking action a in state s"
        ...


class GridWorld:
    def __init__(
        self,
        states: np.ndarray, # 5x5 matrix
        policy: Policy,
        actions: dict = {
            1: np.array([-1, 0]), #  north/up
            2: np.array([1, 0]), #  south/down
            3: np.array([0, 1]), #  east/right
            4: np.array([0, -1]) #  west/left
        },
        current_state: np.ndarray = np.array([0,0])
    ):
        self.states = states
        self.actions = actions
        self.policy = policy
        self.current_state = current_state

    def move(self, action: int):
        self.current_state += self.actions[action]
