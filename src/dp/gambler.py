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
    def __init__(self, env: Environment, state: int = 10):
        self.env = env
        self.state = state

        self.policy: dict[int, int] = {
            s: s for s in range(1, self.state+1)}

    def select_bet(self):
        return self.policy[self.state]

    def bet(self, amount: int):
        coefficient = self.env.bet()
        self.state += coefficient*amount

        if self.state != 0: # if not lost
            self.policy = {
                s: s for s in range(1, self.state+1)}

    def step(self):
        amount = self.select_bet()
        self.bet(amount)
        return int(self.state == 100), self.state


def episode():
    env = Environment()
    state = 50
    agent = Agent(env, state)

    while state not in (0, 100):
        state = agent.step()[1]

    return state


if __name__ == "__main__":
    episode()
