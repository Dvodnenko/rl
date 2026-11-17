# hyperparameters
alpha: float = .34
gamma: float = 1
n: int = 8
n_states: int = 19

# states 0 and 6 are terminal; 1-5 correspond to A-E respectively
terminal_states = (0, n_states+1)
