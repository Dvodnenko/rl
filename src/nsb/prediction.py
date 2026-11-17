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
