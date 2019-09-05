
import pytest
import sys
sys.path.append("")

from src.tests.fixtures import model
from src.location_functions import find_location_transitions


def test_basics(model):
    # --- Sum of transition probabilities should equal 1
    for i in range(model.population.shape[0]):
        if model.population.Start_Location[i] == 'COMMUNITY':
            row = model.data[i]
            transitions = find_location_transitions(model, row)
            # --- Sum of transitions should be 1
            assert sum(transitions) == 1
            # --- Transition to community should be 0 - as they are not allowed to move to the community from the community
            assert transitions[0] == 0


__all__ = ['pytest', 'model']
