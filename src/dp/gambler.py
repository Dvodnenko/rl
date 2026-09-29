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
    def __init__(self, env: Environment, state: int = 10, gamma: float = .9):
        self.env = env
        self.state = state
        self.gamma = gamma

        self.actions: list[int] = []
        self._update_action_space()
        # initial policy says to bet all money the gambler has
        self.policy: dict[int, int] = {
            s: s for s in range(1, self.state+1)}

    def _update_action_space(self):
        self.actions = [
            a for a in range(
                0, min(self.state, 100-self.state)+1
            )]

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
        self._update_action_space()
        return int(self.state == 100), self.state

    def transitions(self, a: int):
        win, lose = self.state + a, self.state - a

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
