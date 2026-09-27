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

    def bet(self, amount: int):
        reward = self.env.bet(amount)
        self.state += reward
        return reward

    def step(self):
        amount = self.select_bet()
        self.bet(amount)
        return self.state


def episode():
    env = Environment()
    agent = Agent(env, 50)

    while True:
        state = agent.step()
        print(state)
        if state == 0:
            print("Loss, bankrupt")
            break
        elif state == 100:
            print("Won 100 bucks")
            break


if __name__ == "__main__":
    episode()
