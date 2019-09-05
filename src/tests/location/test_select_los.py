
import pytest
import sys
sys.path.append("")

from src.tests.fixtures import model
from src.location_functions import select_los


def test_community(model):
    # --- Moving to community should produce -1 LOS
    assert select_los(model, model.locations.values['COMMUNITY']) == -1


def test_non_community(model):
    # --- Moving to a hospital should select an integer greater than 0, but less than 100
    for i in range(100):
        los = select_los(model, model.locations.values['ST'])
        assert 0 < los < 100
        assert type(los) is int


def test_raises(model):
    # --- Providing a non-int should error
    with pytest.raises(KeyError):
        select_los(model, model.locations.ints[0])


__all__ = ['pytest', 'model']
