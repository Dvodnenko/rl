# Implementation of the grid world
# problem from S&B

from enum import Enum

import numpy as np


class Policy:
    def __init__(self):
        ...

    def pi_a_s(self, action, state):
        "π(a|s) - probability of picking action a in state s"
        ...


class Action(Enum):
    UP = "up"
    RIGHT = "right"
    DOWN = "down"
    LEFT = "left"


class GridWorld:
    def __init__(
        self,
        states: np.ndarray, # 5x5 matrix
        policy: Policy,
        actions: dict = {
            Action.UP: np.array([-1, 0]), #  north/up
            Action.RIGHT: np.array([0, 1]), #  east/right
            Action.DOWN: np.array([1, 0]), #  south/down
            Action.LEFT: np.array([0, -1]), #  west/left
        },
        cs: np.ndarray = np.array([0,0])
    ):
        self.states = states
        self.actions = actions
        self.policy = policy
        self.cs = cs

        # special states A, A', B, B'
        self.A =  np.array([0, 1])
        self.Ap = np.array([4,1])
        self.B =  np.array([0, 3])
        self.Bp = np.array([2, 3])

    @property
    def value(self) -> int:
        return self.states[self.cs[0], self.cs[1]]

    def move(self, action: Action):
        self.cs += self.actions[action]
