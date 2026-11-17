import random

import numpy as np
import matplotlib.pyplot as plt


# hyperparameters
alpha: float = .34
gamma: float = 1
n: int = 8
n_states: int = 19

# states 0 and 6 are terminal; 1-5 correspond to A-E respectively
terminal_states = (0, n_states+1)


def move(state: int, action: int):
    next_state = state + action
    
    if next_state == n_states+1:
        reward = 1
    elif next_state == 0:
        reward = -1
    else:
        reward = 0
    
    return next_state, reward


def episode(
    V: dict[int, float],
    alpha: float,
    state: int = 3 # start at state C by default
):
    assert state not in terminal_states

    T = float("inf")
    t = 0

    rewards: dict[int, float] = {} # rewards at time steps
    states: dict[int, int] = {0: state} # states at time steps

    while True:
        if t < T:
            action = random.choice((-1, 1)) # -1 is left, 1 is right
            next_state, reward = move(state, action)
            if next_state in terminal_states:
                T = t + 1

            rewards[t+1] = reward
            states[t+1] = next_state

        tau = t - n + 1
        if tau >= 0:
            G = sum(pow(gamma, i-tau-1)*rewards[i]
                for i in range(tau+1, min(tau+n,T)+1))
            if tau+n < T:
                G = G + (gamma**n)*V[states[tau+n]]
            V[states[tau]] = V[states[tau]] + alpha*(G - V[states[tau]])

        t += 1
        state = next_state

        if tau == T - 1:
            break

    return V


if __name__ == "__main__":
    points = np.zeros(50)
    for numerator in range(50):
        alpha = numerator/50
        V_true = np.zeros(21)
        V_true[1:20] = np.linspace(-0.9, 0.9, 19)
        n_runs = 100
        errors = np.zeros(n_runs)
        for run in range(n_runs):
            V = np.zeros(21)
            ep_rms = []
            for ep in range(10):
                episode(V, alpha, 10)
                ep_rms.append(np.sqrt(np.mean((V[1:20] - V_true[1:20]) ** 2)))
            errors[run] = np.mean(ep_rms)
        points[numerator] = errors.mean()
        print(f"{numerator} done")

    plt.plot(points)
    plt.ylim(.2, .55)
    plt.show()
