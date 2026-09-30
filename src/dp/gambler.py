# implementation of the Gambler's Problem (Example 4.3) from S&B

import random


class Environment:
    def __init__(self, p_h: float = .5):
        self.p_h = p_h

    def bet(self):
        point = random.uniform(0, 1)
        if point <= self.p_h: # if heads
            return 1
        return -1


class Agent:
    def __init__(self, env: Environment, state: int = 10, gamma: float = 1):
        self.env = env
        self.state = state
        self.gamma = gamma

        self.v = {s: 0 for s in range(0, 101)} # terminal states (0 & 100) included
        # initial policy says to bet all money the gambler has
        self.policy: dict[int, int] = {
            s: s for s in range(1, self.state+1)}

    def actions(self, s: int):
        return [a for a in range(0, min(s, 100-s)+1)]

    def pi_a_s(self, a: int, s: int):
        """
        self.policy is deterministic, but the Bellman
        equation uses stochastic one. this function 
        treats self.policy as a stochastic policy
        """

        return int(self.policy[s] == a)

    def select_bet(self):
        return self.policy[self.state]

    def bet(self, amount: int):
        coefficient = self.env.bet()
        self.state += coefficient*amount

    def step(self):
        amount = self.select_bet()
        self.bet(amount)
        return int(self.state == 100), self.state

    def transitions(self, s: int, a: int):
        win, lose = s + a, s - a

        return [
            (self.env.p_h, win, 1.0 if win == 100 else 0.0),
            (1 - self.env.p_h, lose, 0.0),
        ]


def episode():
    env = Environment()
    state = 10
    agent = Agent(env, state)

    return state


if __name__ == "__main__":
    episode()
