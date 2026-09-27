# implementation of the Gambler's Problem (Example 4.3) from S&B

import random


class Environment:
    def __init__(self, p_h: float = .5):
        self.p_h = p_h
