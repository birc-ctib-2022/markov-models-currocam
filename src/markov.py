"""Markov models."""


from collections import Counter
import math
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
    if not x:
        return 1
    pre_prob, i = mm.init_probs[x[0]], 1
    while i < len(x):
        pre_prob *= mm.trans[x[i-1]][x[i]]
        i += 1
    return pre_prob

def log_likelihood(x: list[int], mm: MarkovModel) -> float:
    """
    Compute the likelihood of mm given x.

    This is the same as the probability of x given mm,
    i.e., P(x ; mm).
    """
    if not x:
        return math.log(1)
    pre_prob, i = math.log(mm.init_probs[x[0]]), 1
    while i < len(x):
        pre_prob += math.log(mm.trans[x[i-1]][x[i]])
        i += 1
    return pre_prob

def estimate_parameters(runs: list[list[int]]) -> MarkovModel:
    k = max(max(runs)) +1# number of possible states
    initial_states = np.zeros(k)
    for run in runs:
        initial_states[run[0]] += 1
    pi = initial_states/len(runs)

    transitions, states_counter  = np.zeros((k, k)), np.zeros(k)    
    for run in runs:
        run_iter = iter(run)
        previous = next(run_iter)
        for el in run_iter:
            transitions[previous][el] += 1 
            states_counter[previous] += 1
            previous = el
    for i, counter in enumerate(states_counter):
        if counter > 0:
            transitions[i] =  transitions[i] / counter
    return MarkovModel(pi, transitions)