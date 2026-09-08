"""
policy π(a|s) is a function that returns the probability 
of picking certain action "a" in a certain state "s"
"""

import numpy as np

from .main import Action


def random_policy(
    action: Action,
    state: np.ndarray
):
    # random policy returns equal probabilities 
    # of picking for all actions in all states
    return .25
