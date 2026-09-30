import sys
import json

from .gambler import Environment, Agent


def episode():
    env = Environment(.4)
    state = 10
    agent = Agent(env, state)

    for i in range(20):
        print(f"iteration {i}")
        for s in range(1, 100):
            agent.value_iteration(s)

    json.dump(agent.v, sys.stdout, indent=4)


if __name__ == "__main__":
    episode()
