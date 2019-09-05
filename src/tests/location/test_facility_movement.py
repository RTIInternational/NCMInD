
from copy import deepcopy
import sys
sys.path.append("")

from src.tests.fixtures import model
from src.location_functions import facility_movement


def test_location_movement(model):

    # ----- Cannot pickle SQL connection, so close it first.
    model.collapse_sql_connection()
    m = deepcopy(model)
    m.create_sql_connection()

    # ----- Move everyone to an ST, and set their leave day to the current model time
    m.data[:, m.columns['location_status']] = m.locations.values['ST']
    m.data[:, m.columns['leave_facility_day']] = m.time

    facility_movement(m)
    new = m.data[1:, m.columns['location_status']]

    # ----- Most people should go to the community
    assert list(new).count(m.locations.values['COMMUNITY']) / len(new) > .75

    # ----- But not everyone
    assert list(new).count(m.locations.values['COMMUNITY']) / len(new) < 1

    # ----- If we run this again, only more people can go to the community. Community agents should not move
    old = deepcopy(new)
    facility_movement(m)
    assert list(new).count(m.locations.values['COMMUNITY']) >= list(old).count(m.locations.values['COMMUNITY'])


__all__ = ['model']
