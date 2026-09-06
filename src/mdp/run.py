import numpy as np

from .main import GridWorld, Policy, Action



gw = GridWorld(
    np.arange(1, 26).reshape((5,5)),
    policy=Policy(),
    current_state=np.array([2, 2])
)

gw.move(Action.UP)
print(gw.states)
print(gw.current_state, gw.value)
