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
    return Q[s][a] + alpha*(
        r + gamma*max(Q[s_prime], default=0) - Q[s][a])

terminal_states: dict[str, tuple[bool, str | None]] = {}

TERMINAL_COMBINATIONS = (
    ('x', 'x', 'x'),
    ('o', 'o', 'o'),
)

def is_terminal(s: str) -> tuple[bool, str]:
    if not terminal_states.get(s):
        entries = tuple(s)
        table = (
            tuple(s[0:3]),
            tuple(s[3:6]),
            tuple(s[6:9]),
        )

        # 1. check rows
        for row in table:
            if row in TERMINAL_COMBINATIONS:
                terminal_states[s] = (True, row[0])
                return terminal_states[s]
            
        # 2. check columns
        for col in range(3):
            column = []
            for row in table:
                column.append(row[col])
            if tuple(column) in TERMINAL_COMBINATIONS:
                terminal_states[s] = (True, table[0][col])
                return terminal_states[s]
        
        # 3. check diagonals
        if (s[0], s[4], s[8]) in TERMINAL_COMBINATIONS or \
            (s[2], s[4], s[6]) in TERMINAL_COMBINATIONS:
            terminal_states[s] = (True, s[4])
            return terminal_states[s]

        if not terminal_states.get(s):
            if s.count('-') == 0:
                terminal_states[s] = (True, None)
            else:
                terminal_states[s] = (False, None)
    
    return terminal_states[s]

def step(
    player: str, 
    Q: dict,
    s: str,
    alpha: float,
    gamma: float,
    epsilon: float,
):
    r = 0

    a = None
    if not Q.get(s):
        Q[s] = {a: 0 for a in A(s)}
    if random.random() < epsilon:
        a = random.choice(A(s))
    else:
        a = max(Q[s], key=Q[s].get)

    s_prime = list(s)
    s_prime[a] = player
    s_prime = "".join(s_prime)
    if not Q.get(s_prime):
        Q[s_prime] = {a: 0 for a in A(s_prime)}

    terminal, winner = is_terminal(s_prime)
    if terminal:
        if winner == player:
            r = 1
        elif winner == None:
            r = 0
        else:
            r = -1

    Q[s][a] = update(Q, alpha, gamma, s, a, r, s_prime)

    s = s_prime

    return Q, s, terminal, winner

def episode(
    Q_x: dict,
    Q_o: dict,
    s: str,
    alpha: float = .5,
    gamma: float = 1,
    epsilon: float = .1,
):
    terminal = False
    winner = None
    while not terminal:
        Q_x, s, terminal, winner = step('x', Q_x, s, alpha, gamma, epsilon)
        if terminal: break
        Q_o, s, terminal, winner = step('o', Q_o, s, alpha, gamma, epsilon)

    return Q_x, Q_o, winner
