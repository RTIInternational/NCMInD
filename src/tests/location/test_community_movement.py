
from copy import deepcopy
from numpy import alltrue
import sys
sys.path.append("")

from src.tests.fixtures import model
from src.location_functions import community_movement


def test_community_movement(model):

    # ----- Cannot pickle SQL connection, so close it first.
    model.collapse_sql_connection()
    m = deepcopy(model)
    m.create_sql_connection()

    # ----- Up everyone's probability so people have to move
    m.data[:, m.columns['community_probability']] = 1

    # ----- Kill one person to make sure they don't move
    m.data[0, m.columns['life_status']] = 0

    # ----- No one should be in the community anymore
    community_movement(m)
    assert m.locations.values['COMMUNITY'] not in m.data[1:, m.columns['location_status']]

    # ----- Except for the one agent who is dead
    assert m.locations.values['COMMUNITY'] == m.data[0, m.columns['location_status']]

    # ----- Since everyone was in the community, no one should move if we run this again
    old = deepcopy(m.data[:, m.columns['location_status']])
    community_movement(m)
    assert alltrue(old == m.data[:, m.columns['location_status']])


__all__ = ['model']
