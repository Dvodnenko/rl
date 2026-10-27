import random

import numpy as np


# environment
gw = np.arange(0, 48).reshape(4, 12)
actions = {
    0: np.array([-1, 0]), # up
    1: np.array([0, 1]),  # right
    2: np.array([1, 0]),  # down
    3: np.array([0, -1]), # left
}

def move(s: np.ndarray, a: int):
    new = s + actions[a]

    if (new[0] < 0 or new[0] > 3) \
        or (new[1] < 0 or new[1] > 11): # if the action would take agent out of grid...
        return s # don't move

    return new

def episode(
    Q: np.ndarray,
    s: np.ndarray = np.array([3, 0]),
    alpha: float = .5,
    gamma: float = 1,
    epsilon: float = .1,
):
    # initially, Q(s,a) = 0 for all states and actions

    A = None
    if random.random() < epsilon:
        A = random.choice((0, 1, 2, 3))
    else:
        A = np.argmax(Q[gw[s[0], s[1]]])

    total_reward = 0
    while not (s[0] == 3 and s[1] == 11):

        # 1. take action A, observe S' and R
        s_prime = move(s, A)
        r = -1 # -1 by default, for each move
        if s_prime[0] == 3 and s_prime[1] not in (0, 11): # if in cliff
            r = -100
            s_prime = np.array([3, 0])

        # 2. select action A' from state S'
        A_prime = None
        if random.random() < epsilon:
            A_prime = random.choice((0, 1, 2, 3))
        else:
            A_prime = np.argmax(Q[gw[s_prime[0], s_prime[1]]])

        # 3. update Q(S,A)
        Q[gw[s[0], s[1]],A]+=alpha*(
            r+gamma*Q[gw[s_prime[0],s_prime[1]],A_prime]-Q[gw[s[0],s[1]],A])

        # 4. rewrite S <- S'; A <- A'
        s = s_prime
        A = A_prime

        total_reward += r

    return Q, total_reward


if __name__ == "__main__":
    Q = np.zeros(48*4).reshape(48, 4)
    for _ in range(500):
        Q, _ = episode(Q)
    learned_actions = np.argmax(Q, axis=1)
    
    arrs = {
        0: "↑",
        1: "→",
        2: "↓",
        3: "←",
    }

    arrows = [i for i in range(48)]
    for i, a in enumerate(learned_actions):
        arrows[i] = arrs[a]
    print(np.array(arrows).reshape(4, 12))
