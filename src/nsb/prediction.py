import random


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
