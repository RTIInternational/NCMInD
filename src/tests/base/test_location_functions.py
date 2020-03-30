
import numpy as np
import pandas as pd

from copy import deepcopy
from pytest import raises

from src.tests.fixtures import model
from src.state import LifeState


def test_locations_class(model):
    m = deepcopy(model)
    m.create_sql_connection()
    locations = m.location.locations

    # ----- Test Community
    assert locations.community == 0
    assert locations.facilities['COMMUNITY']['int'] == 0

    # ----- UNC has 10 hospitals
    assert len(locations.categories['UNC']['ints']) == 10
    assert len([item for item in locations.categories['UNC']['ids'] if 'UNC_' in item]) == 10

    # ----- Categories must be consistent
    for item in locations.facilities:
        assert locations.facilities[item]['category'] in locations.categories

    # ----- All facilities are accounted for
    items = locations.categories_list[1:]
    assert sum([len(locations.categories[item]['ids']) for item in items]) == locations.number_of_items - 1

    # ----- Ints and Value lengths match
    assert len(locations.categories['UNC']['ids']) == len(locations.categories['UNC']['ints'])
    assert len(locations.categories['NH']['ids']) == len(locations.categories['NH']['ints'])

    # ----- Add_Value works
    r = pd.DataFrame(data=['ID', 'Name', 'Large', 100, ]).T
    r.columns = ['ID', 'Name', 'Category', 'Beds']
    number_of_items = locations.number_of_items
    locations.add_value(r.loc[0], locations.category_enum.LARGE.value)
    assert locations.number_of_items == number_of_items + 1

    # ----- No Duplicate Keys
    with raises(ValueError, match=r"Cannot use*"):
        locations.add_value(r.loc[0], locations.category_enum.LARGE.value)


def test_select_los(model):
    m = deepcopy(model)

    # --- Move to different locations, making sure the LOS is within reasonable values
    for i in range(10):
        m.location.current_los[0] = 0
        m.location.assign_los(unique_id=0, new_location=m.location.locations.facilities['UNC_0']['int'])
        assert 0 < m.location.current_los[0] < 100
        m.location.current_los[0] = 0
        m.location.assign_los(unique_id=0, new_location=m.location.locations.facilities['nonUNC_0']['int'])
        assert 0 < m.location.current_los[0] < 100
        m.location.current_los[0] = 0
        m.location.assign_los(unique_id=0, new_location=m.location.locations.facilities['NH_0']['int'])
        assert 0 < m.location.current_los[0] < 2001

    # --- Providing a non-int should error
    with raises(KeyError):
        m.location.assign_los(0, 'UNC')
    with raises(KeyError):
        m.location.assign_los(0, 1000)


def test_find_location_transitions(model):
    county = model.county_codes[0]
    age = model.age_groups[0]
    transitions = model.location.find_location_transitions(county, age, model.location.location.values[0])
    # ----- Enough transitions to equal categories
    assert len(transitions) == len(model.location.locations.categories_list)

    # ----- All possible locations work
    for a_loc in model.location.locations.facilities:
        assert sum(model.location.find_location_transitions(
            135, 2, model.location.locations.facilities[a_loc]['int'])) > 0


def test_community_movement(model):
    # ----- Cannot pickle SQL connection, so close it first.
    m = deepcopy(model)
    m.create_sql_connection()
    # ----- Up everyone's probability so people have to move
    m.location.location.probabilities.fill(1)
    m.location.location.values.fill(0)
    # ----- No one should be in the community anymore
    m.location.community_movement()
    assert not any(m.location.location.values == 0)


def test_facility_movement(model):
    # ----- Cannot pickle SQL connection
    m = deepcopy(model)
    m.create_sql_connection()

    # ----- Move everyone to an ST, and set their leave day to the current model time
    m.location.location.values.fill(m.location.locations.facilities['UNC_0']['int'])
    for unique_id in m.unique_ids:
        m.location.current_los[unique_id] = 1
        m.location.leave_facility_day[unique_id] = m.time
    # --- Run Function
    m.location.facility_movement()

    # ----- Everyone should move
    e = m.make_events()
    assert len(e) == len(m.population)

    # ----- Most people should go to the community
    assert sum(m.location.location.values == 0) / len(m.population) > .75

    # ----- But not everyone
    assert sum(m.location.location.values == 0) < len(m.population)


def test_readmission_movement(model):
    # ----- Cannot pickle SQL connection
    m = deepcopy(model)
    m.create_sql_connection()

    # ----- Make sure everyone is at home with a readmission ready to occur
    m.location.current_los = dict()
    m.location.leave_facility_day = dict()
    for unique_id in m.unique_ids:
        m.location.location.values[unique_id] = m.location.locations.facilities['COMMUNITY']['int']
        m.location.readmission_date[unique_id] = m.time
        m.location.readmission_location[unique_id] = m.location.locations.facilities['UNC_0']['int']

    m.location.readmission_movement()

    # ----- Everyone should move
    e = m.make_events()
    assert len(e) == len(m.population)

    # ----- Everyone should go to UNC
    assert len(e.New.unique()) == 1
    assert e.New.unique()[0] == m.location.locations.facilities['UNC_0']['int']

    # ----- Everyone should be at a location
    assert sum(m.location.location.values != 0) == len(m.population)
    assert len(m.location.current_los) == len(m.population)

    # ----- No one should be readmitable (values are removed during the next time step)
    m.time = m.time + 1
    m.location.readmission_movement()
    assert len(m.location.readmission_location) == 0
    assert len(m.location.readmission_date) == 0


def test_select_unc_hospital(model):
    m = deepcopy(model)

    # ----- Make sure the UNC hospital is selected that matches the Transitions
    i = np.where(m.location.transitions['UNC'][:, 0] == 135)[0][0]
    m.location.transitions['UNC'][i] = [135, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    assert m.location.select_unc_hospital(m.county_codes[0], m.location.locations.facilities['UNC_0']['int']) == 'UNC_9'

    # ----- If the there are no other hospitals in the UNC system, they should still find a UNC hospital
    ids = m.location.locations.categories['UNC']['ids']
    m.location.transitions['UNC'][i] = [135, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert m.location.select_unc_hospital(m.county_codes[0], m.location.locations.facilities['UNC_0']['int']) in ids


def test_initialization(model):
    # ----- This could use expansion
    # ----- Unique ID
    assert model.unique_ids[0] == 0
    assert model.unique_ids[-1] == len(model.population) - 1

    # ----- There are only 3 age groups, 0, 1 and 2
    assert 0 <= min(model.age_groups)
    assert 2 >= max(model.age_groups)

    # ----- Community Probability should be small
    assert 0 <= model.location.location.probabilities.max() < .05
    assert 0 <= model.location.location.probabilities.min() < .05

    # ----- Agents should be alive
    for item in model.life.values:
        assert item == LifeState.ALIVE


# ----- TODO
# test_select_st_hospital
# update_location_transitions()
# select_location()

__all__ = ['model']
