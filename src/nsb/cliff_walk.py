import random


ACTIONS = {
    0: (0, -1), # up
    1: (0, 1),  # down
    2: (1, 0),  # right
    3: (-1, 0), # left
}

def move(state: tuple[int], action: int) -> tuple[tuple, float]:
    "interaction with the environment. returns next state and reward"
    action = ACTIONS[action]
    next_state = (state[0]+action[0], state[1]+action[1])

    # if in the cliff
    if next_state[1] == 4 and next_state[0] in range(2, 12):
        return (1, 4), -100

    # if out of the grid
    if next_state[0] < 1 or next_state[0] > 12 or \
        next_state[1] < 1 or next_state[1] > 4:
        return state, -1

    return next_state, -1

TERMINAL_STATE = (12, 4)

def policy(
    epsilon: float,
    Q: dict[tuple, dict[int, float]],
    state: tuple,
) -> int:
    "epsilon greedy with respect to Q"
    if random.random() < epsilon:
        return random.choice((0, 1, 2, 3))
    max_val = max(Q[state].values())
    argmax_all = [a for a, v in Q[state].items() if v == max_val]
    return random.choice(argmax_all)

def episode(
    alpha: float,
    gamma: float,
    epsilon: float,
    n: int,
    Q: dict[tuple, dict[int, float]],
    policy,
):
    a = policy(epsilon, Q, (1, 4))
    T = float("inf")
    t = 0

    # states and actions history
    states = [(1, 4)] # (1, 4) is the initial state at all episodes
    actions = [a]
    rewards = [None, ]

    while True:
        if t < T:
            state = states[t]
            action = actions[t]
            next_state, reward = move(states[t], action)
            states.append(next_state)
            rewards.append(reward)

            if next_state == TERMINAL_STATE:
                T = t + 1
            else:
                actions.append(policy(epsilon, Q, states[t+1]))
        
        tau = t - n + 1

        if tau >= 0:
            G = sum(pow(gamma, i-tau-1)*rewards[i]
                    for i in range(tau+1, min(tau+n, T)+1))
            if tau+n < T:
                G = G + pow(gamma, n)*Q[states[tau+n]][actions[tau+n]]
            Q[states[tau]][actions[tau]] += alpha*(G - Q[states[tau]][actions[tau]])

        t += 1

        if tau == T - 1:
            break
    
    return Q


def init_Q():
    return {
        (x, y): {a: 0 for a in range(4)}
        for y in range(1, 5)
        for x in range(1, 13)
    }


if __name__ == "__main__":
    Q = init_Q()

    for i in range(100):
        Q = episode(.5, 1, .1, 1, Q, policy)
