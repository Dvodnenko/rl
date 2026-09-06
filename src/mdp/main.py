# Implementation of the grid world
# problem from S&B

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
        actions: dict,
        policy: Policy,
        current_state: np.ndarray = np.zeros(2)
    ):
        self.states = states
        self.actions = actions
        self.policy = policy
        self.current_state = current_state
