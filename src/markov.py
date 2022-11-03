"""Markov models."""


import numpy as np


class MarkovModel:
    """Representation of a Markov model."""

    init_probs: list[float]
    trans: list[list[float]]

    def __init__(self,
                 init_probs: list[float],
                 trans: list[list[float]]):
        """Create model from initial and transition probabilities."""
        # Sanity check...
        k = len(init_probs)
        assert k == len(trans)
        for row in trans:
            assert k == len(row)

        self.init_probs = init_probs
        self.trans = trans


def likelihood(x: list[int], mm: MarkovModel) -> float:
    """
    Compute the likelihood of mm given x.

    This is the same as the probability of x given mm,
    i.e., P(x ; mm).
    """
    def inner(x: list[int], mm: MarkovModel, pre_prob = None):
        pre_prob = mm.init_probs[x[0]] if not pre_prob else pre_prob
        match len(x):
            case 1:
                return pre_prob
            case 2:
                return pre_prob * mm.trans[x[0]][x[1]]
            case _:
                return inner(x[1:], mm, mm.init_probs[x[0]] * mm.trans[x[0]][x[1]])
    return inner(x, mm)
    
