import numpy as np

from .main import GridWorld, Action



gw = GridWorld(
    np.arange(1, 26).reshape((5,5)),
    policy=lambda: None,
    cs=np.array([2, 2])
)

gw.move(Action.UP)
print(gw.states)
print(gw.cs, gw.value)
