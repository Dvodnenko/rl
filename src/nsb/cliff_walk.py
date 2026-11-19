import random


def move(state: tuple[int], action: tuple[int]) -> tuple[tuple, float]:
    "interaction with the environment. returns next state and reward"

    next_state = (state[0]+action[0], state[1]+action[1])

    # if in the cliff
    if next_state[1] == 4 and next_state[0] in range(2, 12):
        return (1, 4), -100

    # if out of the grid
    if next_state[0] < 1 or next_state[0] > 12 or \
        next_state[1] < 1 or next_state[1] > 4:
        return state, -1

    return next_state, -1


ACTIONS = {
    "up": (0, -1),
    "down": (0, 1),
    "right": (1, 0),
    "left": (-1, 0),
}
TERMINAL_STATE = (12, 4)

def policy(
    epsilon: float,
    Q: dict[tuple, dict[tuple, float]],
    state: tuple,
) -> tuple:
    "epsilon greedy with respect to Q"
    if random.random() < epsilon:
        return random.choice(ACTIONS.values())
    return max(Q[state].values(), key=Q.get)

def episode(
    alpha: float,
    gamma: float,
    epsilon: float,
    n: int,
    Q: dict[tuple, dict[tuple, float]],
    policy,
):
    a = policy(epsilon, Q, state)
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
            next_state, reward = move(s, action)
            states.append(next_state)
            rewards.append(reward)

            if next_state == TERMINAL_STATE:
                T = t + 1
            else:
                actions.append(policy(epsilon, Q, state))
        
        tau = t - n + 1

        if tau >= 0:
            G = sum(pow(gamma, i-tau-1)*rewards[i]
                    for i in range(tau+1, min(tau+n, T)+1))
            if tau+n < T:
                G = G + pow(gamma, n)*Q[states[tau+n]][actions[tau+n]]
            Q[states[tau+n]][actions[tau+n]] += alpha*(G - Q[states[tau+n]][actions[tau+n]])

        t += 1

        if tau == T - 1:
            break
