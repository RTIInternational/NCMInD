
import sys
sys.path.append("")
from src.tests.fixtures import model, data_row
from src.location_functions import location_update


def test_update(model, data_row):

    location_update(model, data_row)

    # ----- Agent should now be in list of agents that moved
    assert data_row[model.columns['unique_id']] in model.agents_to_update

    # ----- Agent should not be in the community
    assert data_row[model.columns['location_status']] != model.locations.values['COMMUNITY']

    # ----- Agent should have a LOS
    assert data_row[model.columns['current_los']] > 0

    # ----- Agent should know when to leave a facility
    assert data_row[model.columns['leave_facility_day']] > model.time


__all__ = ['model', 'data_row']
