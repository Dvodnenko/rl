# Implementation of the grid world
# problem from S&B

import random
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
        policy: Callable[[Action, np.ndarray], float],
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

    def select_action(self) -> Action:
        point = random.uniform(0, 1)
        cumulative = 0.0
        for action in Action:
            cumulative += self.policy(action, self.cs)
            if point <= cumulative:
                return action

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

    def value_function(self, gamma: float = 0.9) -> np.ndarray:
        """
        Solves the Bellman equation for v_pi directly via linear algebra:
 
            v_pi = (I - gamma * P_pi)^-1 r_pi
 
        instead of iterative policy evaluation. States are indexed here by
        grid *position* (row-major, 0..24) rather than by the arbitrary
        values stored in self.states -- position is what determines
        transitions and rewards. The result is reshaped back to the same
        shape as self.states.
 
        Note: self.policy takes a single (action, state) pair, so building
        the per-state action-probability array needs a loop over the 25
        states x 4 actions (100 calls) -- unavoidable given that interface.
        This is not the iterative sweep-until-convergence loop of classic
        policy evaluation, it's just populating a matrix. The transition
        and reward computation itself is fully vectorized over all states
        at once, for every action.
        """
        n_rows, n_cols = self.states.shape
        n_states = n_rows * n_cols
 
        row_idx, col_idx = np.divmod(np.arange(n_states), n_cols)
        coords = np.stack([row_idx, col_idx], axis=1)  # shape (n_states, 2)
        state_idx = np.arange(n_states)
 
        A_idx = self.A[0] * n_cols + self.A[1]
        Ap_idx = self.Ap[0] * n_cols + self.Ap[1]
        B_idx = self.B[0] * n_cols + self.B[1]
        Bp_idx = self.Bp[0] * n_cols + self.Bp[1]
 
        P = np.zeros((n_states, n_states))
        R = np.zeros(n_states)
 
        for action, delta in self.actions.items():
            # policy only accepts one state at a time -> loop needed here
            probs = np.array([self.policy(action, coord) for coord in coords])
 
            new_rows = coords[:, 0] + delta[0]
            new_cols = coords[:, 1] + delta[1]
 
            off_grid = (
                (new_rows < 0) | (new_rows >= n_rows) |
                (new_cols < 0) | (new_cols >= n_cols)
            )
 
            new_rows_clipped = np.clip(new_rows, 0, n_rows - 1)
            new_cols_clipped = np.clip(new_cols, 0, n_cols - 1)
            next_state = new_rows_clipped * n_cols + new_cols_clipped
            next_state = np.where(off_grid, state_idx, next_state)  # bounce back
 
            reward = np.where(off_grid, -1.0, 0.0)
 
            # special states override the action entirely, regardless of
            # which action was chosen (mirrors the logic in move())
            is_A = state_idx == A_idx
            is_B = state_idx == B_idx
 
            next_state = np.where(is_A, Ap_idx, next_state)
            reward = np.where(is_A, 10.0, reward)
 
            next_state = np.where(is_B, Bp_idx, next_state)
            reward = np.where(is_B, 5.0, reward)
 
            P[state_idx, next_state] += probs
            R += probs * reward
 
        I = np.eye(n_states)
        v = np.linalg.solve(I - gamma * P, R)
 
        return v.reshape(n_rows, n_cols)

