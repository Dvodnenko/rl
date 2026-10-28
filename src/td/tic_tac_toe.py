import random

import numpy as np


# hyperparemeters
alpha: float = .5
gamma: float = 1 # undiscounted task
epsilon: float = .2


def A(s: str) -> list[int]:
    # returns actions available in state s
    a = []
    for i, v in enumerate(list(s)):
        if v == '-':
            a.append(i)
    return a

Q_x: dict[str, dict[int, float]] = {}
Q_o: dict[str, dict[int, float]] = {}

def update(Q, alpha, gamma, s, a, r, s_prime) -> float:
    return Q[s,a] + alpha*(
        r + gamma*max(Q[s], key=Q[s].get) - Q[s,a])

def is_terminal(s: str) -> bool:
    # checks whether state is terminal or not
    ...

def step(
    player: str, 
    Q: dict,
    s: str,
    alpha: float,
    gamma: float,
    epsilon: float,
):
    r = None

    a = None
    if not Q.get(s):
        Q[s] = {a: 0 for a in A(s)}
    if random.random() < epsilon:
        a = random.choice(A(s))
    else:
        a = max(Q[s], key=Q[s].get)

    s_prime = s
    s_prime[a] = player

    Q[s, a] = update(Q, alpha, gamma, s, a, r, s_prime)

    s = s_prime

    terminal = is_terminal(s)

    return Q, s, terminal
