# Implementation of the grid world
# problem from S&B

from enum import Enum
from typing import Callable

import numpy as np


class Action(Enum):
    UP = "up"
    RIGHT = "right"
    DOWN = "down"
    LEFT = "left"


class GridWorld:
    def __init__(
        self,
        states: np.ndarray, # 5x5 matrix
        policy: Callable,
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
        self.A  = np.array([0, 1])
        self.Ap = np.array([4, 1])
        self.B  = np.array([0, 3])
        self.Bp = np.array([2, 3])

    @property
    def value(self) -> int:
        return self.states[self.cs[0], self.cs[1]]

    def move(self, action: Action) -> int:
        """
        returns a reward for the chosen action
        
        return +10 for moving out of state A, +5 for moving out of B,
        -1 for actions that would take the agent out of the board, 0 otherwise
        """

        if self.cs[0] == self.A[0] and self.cs[1] == self.A[1]:
            self.cs = self.Ap
            return 10
        elif self.cs[0] == self.B[0] and self.cs[1] == self.B[1]:
            self.cs = self.Bp
            return 5

        if action == Action.UP and self.cs[0] == 0:
            return -1
        elif action == Action.DOWN and self.cs[0] == 4:
            return -1
        elif action == Action.RIGHT and self.cs[1] == 4:
            return -1
        elif action == Action.LEFT and self.cs[1] == 0:
            return -1

        self.cs += self.actions[action]
        return 0
