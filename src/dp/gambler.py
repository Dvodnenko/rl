# implementation of the Gambler's Problem (Example 4.3) from S&B

import random


class Environment:
    def __init__(self, p_h: float = .5):
        self.p_h = p_h

    def bet(self, amount: int) -> int:
        point = random.uniform(0, 1)
        if point <= self.p_h: # if heads
            return amount
        return -amount


class Agent:
    def __init__(self, env: Environment, state: int = 10):
        self.env = env
        self.state = state

        self.policy: dict[int, int] = {
            s: s for s in range(1, self.state+1)}

    def select_bet(self):
        return self.policy[self.state]
