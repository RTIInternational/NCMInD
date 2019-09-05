
from numpy import where

from src.state import NameState
from src.jit_functions import select_agents
from src.misc_functions import random_selection, create_cdf


class Locations:
    """
    Given a set of unique values, create an int to value map in both directions
    """
    def __init__(self):
        self.ints = dict()
        self.values = dict()

    def add_values(self, values):
        for key in range(len(values)):
            self.ints[key] = values[key]
            self.values[values[key]] = key


def select_los(model, new_location):
    """ Given a location (new_location), select a LOS for a new patient
    """
    if new_location == model.locations.values['COMMUNITY']:
        return -1
    else:
        los = model.parameters.length_of_stay[new_location]
        # ----- Pick a random LOS based on the distribution matching the location
        if los['distribution'] == 'Gamma':
            selected_los = int(round(model.np_rng.gamma(los['shape'], los['support']), 0))
        elif los['distribution'] == 'Uniform':
            selected_los = int(round(model.np_rng.uniform(los['a'], los['b']), 0))
        else:
            raise ValueError("LOS distribution type % is not supported." % (los['distribution'],))
        # ----- LOS cannot be 0 days. They must stay at location at least one day
        if selected_los == 0:
            return 1
        return selected_los


def find_location_transitions(model, row):
    """ Look up the location transition probabilities for an agent, given their current location
    """
    return model.location_transitions[(row[model.columns['location_status']], row[model.columns['demo_id']])]


def community_movement(model):
    """ Find all agents in the community and see which ones are selected to move. Then move those agents
    """
    use_agents = (model.data[:, model.columns['life_status']] == 1) & \
                 (model.data[:, model.columns['location_status']] == model.locations.values['COMMUNITY'])

    values = model.data[:, model.columns['community_probability']][use_agents]
    move_agents = select_agents(values, model.np_rng.rand(values.shape[0], 1))

    move_ids = model.data[:, model.columns['unique_id']][use_agents][move_agents == 1]
    move_ids = list(move_ids.astype(int))
    for move_id in move_ids:
        location_update(model, model.data[move_id])


def facility_movement(model):
    """ Move all agents not in the community whose LOS ends today
    """
    use_agents = (model.data[:, model.columns['location_status']] != model.locations.values['COMMUNITY']) & \
                 (model.data[:, model.columns['leave_facility_day']] == model.time)

    for move_id in where(use_agents)[0]:
        location_update(model, model.data[move_id.astype(int)])


def location_update(model, row):
    # 78.5% of previous nursing home patients need to return to the NH
    # 78.5% because some will also randomly be selected to return (1.5%)
    if row[model.columns['nh_patient']] and (row[model.columns['location_status']] != model.locations.values['NH']) and \
            (model.np_rng.rand(1, 1)[0] < .785):
        new_location = model.locations.values['NH']
    else:
        p = find_location_transitions(model, row)
        new_location = random_selection(
            model.np_rng.rand(1, 1)[0], create_cdf(p), list(model.locations.values.values()))

    if new_location != model.locations.values['COMMUNITY']:

        if new_location == model.locations.values['NH']:
            row[model.columns['nh_patient']] = 1

        selected_los = select_los(model, new_location)

        model.record_state_change(row, NameState.LOCATION, row[model.columns['location_status']], new_location)
        row[model.columns['location_status']] = new_location
        row[model.columns['current_los']] = selected_los
        row[model.columns['leave_facility_day']] = model.time + selected_los
    else:
        row[model.columns['nh_patient']] = 0
        model.record_state_change(row, NameState.LOCATION, row[model.columns['location_status']],
                                  model.locations.values['COMMUNITY'])
        row[model.columns['current_los']] = -1
        row[model.columns['leave_facility_day']] = -1
        row[model.columns['location_status']] = model.locations.values['COMMUNITY']

    model.agents_to_update.append(row[model.columns['unique_id']])


def update_column(series, multiplier):
    l1 = [item * multiplier for item in series]
    return [1 if item > 1 else item for item in l1]


def update_location_transitions(model, lt):
    facilities = [item for item in model.locations.ints.values()]

    # ------------------------------------------------------
    # ----- Update Community to UNC and Community to Non-UNC
    lt2 = lt[lt['Location'] == 'COMMUNITY']
    lt2 = lt2[[item for item in model.locations.ints.values() if item not in ['NH', 'LT', 'COMMUNITY']]]

    total_pre = lt2.sum(axis=1)

    for name in model.parameters.tuning['home_to_hospital']:
        lt2[name] = lt2[name] * model.parameters.tuning['home_to_hospital'][name]

    total_post = lt2.sum(axis=1)

    for column in lt2.columns:
        lt2[column] = lt2[column].divide(total_post, axis=0)
        lt2[column] = lt2[column].multiply(total_pre, axis=0)

    lt.loc[lt2.index,
           [item for item in model.locations.ints.values() if item not in ['NH', 'LT', 'COMMUNITY']]] = lt2

    # ----------------------------------
    # ----- Update facility to NH for 65
    lt2 = lt[lt['Location'] != 'COMMUNITY'].copy()
    lt2 = lt2[lt2['Location'] != 'LT']
    # --- Only 65+
    lt2 = lt2[lt2['Age'] == 2]
    # --- Only those with probabilities
    lt2 = lt2[lt2['NH'] > 0]
    # --- Proportion values back to 0.
    total = lt2[facilities].sum(axis=1)
    for facility in [item for item in facilities if item != 'COMMUNITY']:
        lt2[[facility]] = lt2[[facility]].divide(total, axis=0)
    lt.loc[lt2.index] = lt2

    # ----- NH to Other values
    lt2 = lt[lt['Location'] == "NH"].copy()
    lt2 = lt2[lt2['Age'] == 2]
    # --- Set to new value
    val = model.parameters.tuning['nh_to_home']
    lt2['COMMUNITY'] = val
    #
    total = lt2[[item for item in facilities if item != 'COMMUNITY']].sum(axis=1)

    for facility in [item for item in facilities if item != 'COMMUNITY']:
        lt2[[facility]] = lt2[[facility]].divide(total, axis=0) * (1 - val)

    lt.loc[lt2.index] = lt2

    # ------------------------------------------------------
    # ----- Update Community to UNC and Community to Non-UNC
    h_names = ['CALDWELL', 'CHATHAM', 'HIGHPOINT', 'JOHNSTON', 'LENOIR', 'MARGARET', 'NASH', 'REX', 'UNC_CH', 'WAYNE']

    # --- Select rows where the community moves to UNC and to Non-UNC
    lt2 = lt[lt['Location'] == 'COMMUNITY'].copy()

    # --- Fix UNC
    for column in h_names:
        lt2[column] = update_column(lt2[column], model.parameters.tuning['COMMUNITY_UNC_adjustment'])
    # --- Reset
    lt2['ST'] = 1 - lt2[['COMMUNITY', 'LT', 'NH'] + h_names].sum(axis=1)

    lt.loc[lt2.index] = lt2

    # -------------------------
    # ----- Update UNC movement
    lt2 = lt[lt.Location.isin(h_names)].copy()
    # --- Fix NH
    lt2['NH'] = update_column(lt2['NH'], model.parameters.tuning['UNC_NH_adjustment'])
    # --- Reset
    left = 1 - lt2['NH']
    total = lt2[h_names + ['COMMUNITY', 'ST', 'LT']].sum(axis=1)
    for column in h_names + ['COMMUNITY', 'ST', 'LT']:
        x = lt2[column].divide(total).fillna(0)
        lt2[column] = x * left

    # --- Fix LT
    lt2['LT'] = update_column(lt2['LT'], model.parameters.tuning['UNC_LT_adjustment'])
    # --- Reset
    left = 1 - lt2['NH'] - lt2['LT']
    total = lt2[h_names + ['COMMUNITY', 'ST']].sum(axis=1)
    for column in h_names + ['COMMUNITY', 'ST']:
        x = lt2[column].divide(total).fillna(0)
        lt2[column] = x * left

    # --- Fix ST
    lt2['ST'] = update_column(lt2['ST'], model.parameters.tuning['UNC_ST_adjustment'])
    # --- Reset
    left = 1 - lt2['NH'] - lt2['LT'] - lt2['ST']
    total = lt2[h_names + ['COMMUNITY']].sum(axis=1)
    for column in h_names + ['COMMUNITY']:
        x = lt2[column].divide(total).fillna(0)
        lt2[column] = x * left

    # --- Fix UNC
    for column in h_names:
        lt2[column] = update_column(lt2[column], model.parameters.tuning['UNC_UNC_adjustment'])
    # --- Reset
    lt2['COMMUNITY'] = 1 - lt2[['ST', 'LT', 'NH'] + h_names].sum(axis=1)

    lt.loc[lt2.index] = lt2

    # ------------------------
    # ----- Update ST movement
    lt2 = lt[lt.Location == 'ST'].copy()
    # --- Fix NH
    lt2['NH'] = update_column(lt2['NH'], model.parameters.tuning['ST_NH_adjustment'])
    # --- Reset
    left = 1 - lt2['NH']
    total = lt2[h_names + ['COMMUNITY', 'ST', 'LT']].sum(axis=1)
    for column in h_names + ['COMMUNITY', 'ST', 'LT']:
        x = lt2[column].divide(total).fillna(0)
        lt2[column] = x * left

    # --- Fix LT
    lt2['LT'] = update_column(lt2['LT'], model.parameters.tuning['ST_LT_adjustment'])
    # --- Reset
    left = 1 - lt2['NH'] - lt2['LT']
    total = lt2[h_names + ['COMMUNITY', 'ST']].sum(axis=1)
    for column in h_names + ['COMMUNITY', 'ST']:
        x = lt2[column].divide(total).fillna(0)
        lt2[column] = x * left

    # --- Fix ST
    lt2['ST'] = update_column(lt2['ST'], model.parameters.tuning['ST_ST_adjustment'])
    # --- Reset
    left = 1 - lt2['NH'] - lt2['LT'] - lt2['ST']
    total = lt2[h_names + ['COMMUNITY']].sum(axis=1)
    for column in h_names + ['COMMUNITY']:
        x = lt2[column].divide(total).fillna(0)
        lt2[column] = x * left

    # --- Fix UNC
    for column in h_names:
        lt2[column] = update_column(lt2[column], model.parameters.tuning['ST_UNC_adjustment'])
    # --- Reset
    lt2['COMMUNITY'] = 1 - lt2[['ST', 'LT', 'NH'] + h_names].sum(axis=1)

    lt.loc[lt2.index] = lt2

    # ------------------------
    # ----- Update LT movement
    lt2 = lt[lt.Location == 'LT'].copy()
    # --- Fix NH
    lt2['NH'] = update_column(lt2['NH'], model.parameters.tuning['LT_NH_adjustment'])
    # --- Reset
    left = 1 - lt2['NH']
    total = lt2[h_names + ['COMMUNITY', 'ST', 'LT']].sum(axis=1)
    for column in h_names + ['COMMUNITY', 'ST', 'LT']:
        x = lt2[column].divide(total).fillna(0)
        lt2[column] = x * left

    # --- Fix ST
    lt2['ST'] = update_column(lt2['ST'], model.parameters.tuning['LT_ST_adjustment'])
    # --- Reset
    left = 1 - lt2['NH'] - lt2['LT'] - lt2['ST']
    total = lt2[h_names + ['COMMUNITY']].sum(axis=1)
    for column in h_names + ['COMMUNITY']:
        x = lt2[column].divide(total).fillna(0)
        lt2[column] = x * left

    # --- Fix UNC
    for column in h_names:
        lt2[column] = update_column(lt2[column], model.parameters.tuning['LT_UNC_adjustment'])
    # --- Reset
    lt2['COMMUNITY'] = 1 - lt2[['ST', 'LT', 'NH'] + h_names].sum(axis=1)

    # --- Save
    lt.loc[lt2.index] = lt2

    # ------------------------
    # ----- Update NH movement
    lt2 = lt[lt.Location == 'NH'].copy()
    # --- Fix ST
    lt2['ST'] = update_column(lt2['ST'], model.parameters.tuning['NH_ST_adjustment'])
    # --- Reset
    left = 1 - lt2['NH'] - lt2['LT'] - lt2['ST']
    total = lt2[h_names + ['COMMUNITY']].sum(axis=1)
    for column in h_names + ['COMMUNITY']:
        x = lt2[column].divide(total).fillna(0)
        lt2[column] = x * left

    # --- Fix UNC
    for column in h_names:
        lt2[column] = update_column(lt2[column], model.parameters.tuning['NH_UNC_adjustment'])
    # --- Reset
    lt2['COMMUNITY'] = 1 - lt2[['ST', 'LT', 'NH'] + h_names].sum(axis=1)

    # --- Save
    lt.loc[lt2.index] = lt2

    return lt
