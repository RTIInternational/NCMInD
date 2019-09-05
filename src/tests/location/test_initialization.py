
import sys
sys.path.append("")
from src.tests.fixtures import model, data_row


def test_initialization(model, data_row):
    # --- 0: Unique ID
    assert data_row[model.columns['unique_id']] == 0
    # --- 1: There are only 3 age groups, 0, 1 and 2
    assert 0 <= data_row[model.columns['age_group']] <= 2
    # --- 2: There are only 1800 demographic IDs. 100 counties by 3 races by 2 genders by 3 age groups
    assert 0 <= data_row[model.columns['age_group']] <= 2
    # --- 3: Community Probability should be a decimal
    assert 0 < data_row[model.columns['community_probability']] < .25
    # --- 4: Agent should be at home
    assert data_row[model.columns['location_status']] == 0
    # --- 5: Agent should be alive
    assert data_row[model.columns['life_status']] == 1
    # --- 6: LOS should not be assigned
    assert data_row[model.columns['current_los']] == -1
    # --- 7: Agent should not be in a nursing home
    assert data_row[model.columns['nh_patient']] == 0
    # --- 8: Agents leave facility day should not be set
    assert data_row[model.columns['leave_facility_day']] == -1
    # --- 9: Death probability should be a decimal between 0 and .25
    assert 0 < data_row[model.columns['death_probability']] < .25


__all__ = ['model', 'data_row']
