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
