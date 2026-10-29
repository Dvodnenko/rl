import random

import numpy as np


# hyperparemeters
alpha: float = .25
gamma: float = 1 # undiscounted task
epsilon: float = .2
PLAYER = 'x'


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
        r + gamma*max(Q[s_prime].values(), default=0) - Q[s][a])

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
    game: bool,
):
    if not Q.get(s):
        Q[s] = {a: 0 for a in A(s)}
    
    a = None
    if game and player == PLAYER:
        print(np.array(list(s)).reshape(3, 3))
        a = int(input(f"Enter cell to put '{PLAYER}':")) - 1
    else:
        if random.random() < epsilon:
            a = random.choice(A(s))
        else:
            a = max(Q[s], key=Q[s].get)

    s_prime = list(s)
    s_prime[a] = player
    s_prime = "".join(s_prime)
    if not Q.get(s_prime):
        Q[s_prime] = {a: 0 for a in A(s_prime)}

    return a, s_prime

def episode(
    Q_x: dict,
    Q_o: dict,
    s: str,
    game: bool = False,
    alpha: float = .5,
    gamma: float = 1,
    epsilon: float = .1,
):
    result = None
    psar: list[str, int] = [s, -1] # Previous State-Action Pair

    while True:
        # player X makes move
        a_x, s_prime = step('x', Q_x, s, alpha, gamma, epsilon, game)
        if is_terminal(s_prime)[0]:
            if is_terminal(s_prime)[1] == 'x':
                Q_x[s][a_x] = update(Q_x, alpha, gamma, s, a_x, 1, s_prime)
                if not Q_o.get(s_prime):
                    Q_o[s_prime] = {a: 0 for a in A(s_prime)}
                Q_o[psar[0]][psar[1]] = update(
                    Q_o, alpha, gamma, psar[0], psar[1], -1, s_prime)
                result = 'X Wins!'
                s = s_prime
                break
            else:
                Q_x[s][a_x] = update(Q_x, alpha, gamma, s, a_x, 0, s_prime)
                if not Q_o.get(s_prime):
                    Q_o[s_prime] = {a: 0 for a in A(s_prime)}
                Q_o[psar[0]][psar[1]] = update(
                    Q_o, alpha, gamma, psar[0], psar[1], 0, s_prime)
                result = 'Draw!'
                s = s_prime
                break
        else:
            Q_x[s][a_x] = update(Q_x, alpha, gamma, s, a_x, 0, s_prime)
            psar = [s, a_x]
            s = s_prime
        
        
        # player O makes move
        a_o, s_prime = step('o', Q_o, s, alpha, gamma, epsilon, game)
        if is_terminal(s_prime)[0]:
            if is_terminal(s_prime)[1] == 'o':
                Q_o[s][a_o] = update(Q_o, alpha, gamma, s, a_o, 1, s_prime)
                if not Q_x.get(s_prime):
                    Q_x[s_prime] = {a: 0 for a in A(s_prime)}
                Q_x[psar[0]][psar[1]] = update(
                    Q_x, alpha, gamma, psar[0], psar[1], -1, s_prime)
                result = 'O Wins!'
                s = s_prime
                break
            else:
                Q_o[s][a_o] = update(Q_o, alpha, gamma, s, a_o, 0, s_prime)
                if not Q_x.get(s_prime):
                    Q_x[s_prime] = {a: 0 for a in A(s_prime)}
                Q_x[psar[0]][psar[1]] = update(
                    Q_x, alpha, gamma, psar[0], psar[1], 0, s_prime)
                result = 'Draw!'
                s = s_prime
                break
        else:
            Q_o[s][a_o] = update(Q_o, alpha, gamma, s, a_o, 0, s_prime)
            psar = [s, a_o]
            s = s_prime

    return Q_x, Q_o, result, s


if __name__ == "__main__":
    print("learning...")
    for i in range(50_000):
        Q_x, Q_o, _, _ = episode(Q_x, Q_o, "---------", False, alpha, gamma, epsilon)

    while True:
        _, _, result = episode(Q_x, Q_o, "---------", True, alpha, gamma, 0)
        print(result)
        print()
