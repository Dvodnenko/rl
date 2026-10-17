import numpy as np

from .prediction import all_q_values


def control(policy: list):
    new_policy = policy

    # 1. prediction
    q_values = all_q_values(new_policy)

    # 2. find greedy action at each state
    qs = q_values.argmax(axis=1)

    # 3. improvement
    for state in range(16):
        new_policy[state] = [int(qs[state])]

    return new_policy


random_policy = [[a for a in range (4)] for _ in range(16)]
new_policy = control(random_policy)
new_policy = np.array([a[0] for a in new_policy]).reshape(4, 4)

arrs = {
    0: "↑",
    1: "→",
    2: "↓",
    3: "←",
}

arrows = [i for i in range(16)]
for i, a in enumerate(new_policy.reshape(16)):
    arrows[i] = arrs[a]

print(np.array(arrows).reshape(4, 4))
