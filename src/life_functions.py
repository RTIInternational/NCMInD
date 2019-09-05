
from pandas import DataFrame
from numpy import array
from src.jit_functions import select_agents, find_probability
from src.state import NameState


def life_movement(model):
    use_agents = model.data[:, model.columns['life_status']] == 1

    values = model.data[:, model.columns['death_probability']][use_agents]
    dying_agents = select_agents(values, model.np_rng.rand(values.shape[0], 1))

    move_ids = model.data[:, model.columns['unique_id']][use_agents][dying_agents == 1]
    if len(move_ids) > 0:
        for move_id in move_ids:
            life_update(model, model.data[move_id.astype(int)])


def life_update(model, row):
    """
    Perform a life update. The agent dies, returns to the community if necessary, and is removed from the scheduler.
    A new agent with similar base demographics is added to the population.
    """
    model.record_state_change(row, NameState.LIFE, 1, 0)
    row[model.columns['life_status']] = 0
    if row[model.columns['unique_id']] in model.agents_to_update:
        model.agents_to_update.remove(row[model.columns['unique_id']])
    # ----- Remove the agent from their current location
    if row[model.columns['location_status']] != model.locations.values['COMMUNITY']:
        # ----- Find their actual LOS
        model.cur.execute("SELECT * FROM event_tracker WHERE Unique_ID = ? and Time <> ? and State = ?",
                          (row[model.columns['unique_id']], model.time, NameState.LOCATION.value))
        events = model.cur.fetchall()

        # --- If they haven't moved, their LOS is the current model time (they started at facility on day 0 of model.
        if len(events) == 0:
            row[model.columns['current_los']] = model.time
        # --- If they have moved, their LOS is the current mode time - when they last moved
        else:
            row[model.columns['current_los']] = model.time - events[-1][model.event_tracker_columns['Time']]

        # ----- Record the state change
        model.record_state_change(row, NameState.LOCATION, row[model.columns['location_status']],
                                  model.locations.values['COMMUNITY'])
        row[model.columns['current_los']] = -1
        row[model.columns['leave_facility_day']] = -1
        row[model.columns['location_status']] = model.locations.values['COMMUNITY']

    # ----- Update the agents row in the model
    model.agents_to_recreate.append(row[model.columns['unique_id']])


def update_death_probabilities(model):
    ids = array(model.agents_to_update, dtype=int)
    death_probability = find_probability(model.death_probability,
                                         model.data[ids, model.columns['demo_id']].astype(int))
    location_status = model.data[:, model.columns['location_status']][ids]

    # ----- Find the right death multiplier and save the values
    death_multiplier = [model.parameters.death_multipliers[item] for item in location_status]
    death_probability = death_probability * death_multiplier
    model.data[:, model.columns['death_probability']][ids] = death_probability
