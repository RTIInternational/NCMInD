
import numpy as np
from numba import njit


# ----------------------------------------------------------------------------------------------------------------------
# ------ JIT Functions
# ----------------------------------------------------------------------------------------------------------------------
@njit
def assign_conditions(age: np.array, randoms: np.array):
    conditions = np.zeros(len(age), dtype=np.int8)
    for i in range(len(age)):
        if age[i] == 1:
            if randoms[i] < .2374:
                conditions[i] = 1
        elif age[i] == 2:
            if randoms[i] < .5497:
                conditions[i] = 1
    return conditions


@njit
def update_community_probability(cp: np.array, age: np.array, cc: np.array):
    """
    If simulating risk, we can update hospital transitions based on concurrent conditions. Update an agents
    community_probability based on their concurrent conditions and their age.
    """
    for i in range(len(age)):
        if cc[i] == 1:
            if age[i] == 1:
                cp[i] = cp[i] * 55 / 23.74
            elif age[i] == 2:
                cp[i] = cp[i] * 79 / 54.97
        else:
            if age[i] == 1:
                cp[i] = cp[i] * 45 / 76.26
            elif age[i] == 2:
                cp[i] = cp[i] * 21 / 45.03
    return cp
