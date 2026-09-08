import time

import numpy as np

from .main import GridWorld, Action
from .policies import random_policy


gw = GridWorld(
    np.arange(1, 26).reshape((5,5)),
    policy=random_policy,
    cs=np.array([0, 0])
)

gw.move(Action.UP)


t = 0
while True:
    time.sleep(1)
    
    S_t = gw.cs
    A_t = gw.select_action()
    print(f"t={t}, S_{t} = {S_t}, A_t {A_t.value}")
    R_t1 = gw.move(A_t)
    print(f"R_{t+1} = {"+" if R_t1 > 0 else ""}{R_t1}, S_{t+1} = {gw.cs}")
    print()

    t += 1
