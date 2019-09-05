
from numpy import array, vstack, where, zeros, concatenate
from pandas import DataFrame
from math import exp

from src.state import AntibioticsState, CDIState, NameState
from src.jit_functions import *
from src.misc_functions import create_cdf, normalize_and_create_cdf, random_selection
from src.life_functions import life_update
import collections


def cdi_step(model):
    """ Run through a complete CDI step for all agents
    """

    # ----- Simulate Antibiotics
    update_antibiotics_probabilities(model)
    simulate_antibiotics(model)
    simulate_antibiotics_recovery(model)
    # ----- Simulate CDI
    update_force_of_colonization(model)
    simulate_cdi_movement(model)
    simulate_co_cdi(model)
    # ----- Make Variable Updates
    update_recent_cdi_variables(model)
    update_cdi_probabilities(model)
    # --- Reset the list to empty
    model.cdi['agents_to_update'] = list()


def prepare_cdi(model):
    """ Create the CDI array of data for the model
    """

    # ----- Setup the model variables
    model.cdi['cases'] = list()
    model.cdi['states'] = list()
    model.cdi['columns'] = dict()
    model.cdi['data'] = array(1)
    model.cdi['antibiotics_dictionary'] = dict()
    model.cdi['transition_lookup'] = dict()
    model.cdi['agents_to_update'] = list()
    model.cdi['antibiotic_classes'] = list(model.parameters.cdi['relative_risk']['antibiotics'].keys())

    # ----- Setup the column order
    model.cdi['columns']['cdi_state'] = 0
    model.cdi['columns']['antibiotics_status'] = 1
    model.cdi['columns']['recurrent_cdi_ends'] = 2
    model.cdi['columns']['antibiotics_ends'] = 3
    model.cdi['columns']['cdi_count'] = 4
    model.cdi['columns']['recent_cdi_count'] = 5
    model.cdi['columns']['most_recent_cdi'] = 6
    model.cdi['columns']['concurrent_conditions'] = 7
    model.cdi['columns']['antibiotics_probability'] = 8
    model.cdi['columns']['cdi_change_probability'] = 9
    model.cdi['columns']['antibiotics_risk_ratio'] = 10

    # ----- Create initial columns. All of these columns are not initialized (i.e. no one starts with antibiotics)
    cdi_state = zeros(model.population.shape[0])
    antibiotics_status = zeros(model.population.shape[0])
    recurrent_cdi_ends = zeros(model.population.shape[0])
    recurrent_cdi_ends.fill(-1)
    antibiotics_ends = zeros(model.population.shape[0])
    antibiotics_ends.fill(-1)
    cdi_count = zeros(model.population.shape[0])
    recent_cdi_count = zeros(model.population.shape[0])
    most_recent_cdi = zeros(model.population.shape[0])
    # --- Assign concurrent conditions
    concurrent_conditions = assign_conditions(model.population.Age.values,
                                              model.np_rng.rand(model.population.shape[0], 1))

    # ----- Create the CDI dictionary of lookup values for CDI risk
    model.cdi['antibiotics_dictionary'] = create_antibiotics_dictionary(model)
    model.cdi['transition_lookup'] = create_cdi_dictionary(model)
    model.cdi['states'] = [item.name for item in CDIState]
    # --- No one starts with antibiotics, so their antibiotic risk ratio is 1.0.
    antibiotics_risk_ratios = zeros(model.population.shape[0])
    antibiotics_risk_ratios.fill(model.parameters.cdi['relative_risk']['antibiotics']['default'])

    # ----- Find probability of being assigned antibiotics
    antibiotics_probability = find_antibiotics_probability(
        locations=model.data[:, model.columns['location_status']],
        age=model.data[:, model.columns['age_group']],
        antibiotics_dictionary=model.cdi['antibiotics_dictionary']
    )

    # ----- Find the CDI change probability (probability of changing CDI states)
    cdi_change_probability = find_cdi_change_probability(
        locations=model.data[:, model.columns['location_status']],
        concurrent_conditions=concurrent_conditions.astype(int),
        antibiotics_status=antibiotics_status.astype(int),
        recent_cdi_count=recent_cdi_count.astype(int),
        age=model.data[:, model.columns['age_group']],
        cdi_state=cdi_state.astype(int),
        antibiotics_risk_ratios=antibiotics_risk_ratios,
        cdi_dict=model.cdi['transition_lookup']
    )

    # ----- Update the community movement probability based on the concurrent conditions
    model.data[:, model.columns['community_probability']] = \
        update_community_probability(
            cp=model.data[:, model.columns['community_probability']],
            age=model.data[:, model.columns['age_group']],
            cc=concurrent_conditions)

    # ----- Make an array to store the data
    model.cdi['data'] = vstack(
        [cdi_state, antibiotics_status, recurrent_cdi_ends, antibiotics_ends, cdi_count,
         recent_cdi_count, most_recent_cdi, concurrent_conditions, antibiotics_probability,
         cdi_change_probability, antibiotics_risk_ratios]).T

    # ----- Initialize patients with colonization
    initialize_colonization(model)


def initialize_colonization(model):
    """ All agents start as SUSCEPTIBLE. Initialize a small percentage of agents to start with colonization
    """

    # ----- Find the colonization initialization chance for each person
    rates = []
    for location in model.data[:, model.columns['location_status']]:
        rates.append(model.parameters.cdi['colonization']['initialization'][location])

    # ----- Convert to an array and find random numbers to use with a JIT function
    rates = array(rates)
    randoms = model.np_rng.rand(rates.shape[0], 1)

    # ----- Assign initial colonization
    model.cdi['data'][:, model.cdi['columns']['cdi_state']] =\
        assign_colonization(rates, randoms, CDIState.COLONIZED.value)

    # ----- Find the ids for colonized patients
    col_patients = model.cdi['data'][:, model.cdi['columns']['cdi_state']] == CDIState.COLONIZED
    col_ids = model.data[:, model.columns['unique_id']][col_patients]
    # --- Add it to the update list, and update
    model.cdi['agents_to_update'].extend(list(col_ids.astype(int)))
    update_cdi_probabilities(model)


def update_cdi_probabilities(model):
    """ When an agent moves locations, or changes CDI states, we need to update their CDI probabilities
    """

    # ----- Find all agents who have moved locations or who have changed CDI this time step
    ids = list(set(model.agents_to_update + model.cdi['agents_to_update']))
    ids = array(ids, dtype=int)
    # --- Lookup their new CDI risk in the dictionary
    cdi_change_probability = find_cdi_change_probability(
        locations=model.data[:, model.columns['location_status']][ids],
        concurrent_conditions=model.cdi['data'][:, model.cdi['columns']['concurrent_conditions']][ids],
        antibiotics_status=model.cdi['data'][:, model.cdi['columns']['antibiotics_status']][ids],
        recent_cdi_count=model.cdi['data'][:, model.cdi['columns']['recent_cdi_count']][ids],
        age=model.data[:, model.columns['age_group']][ids],
        cdi_state=model.cdi['data'][:, model.cdi['columns']['cdi_state']][ids],
        antibiotics_risk_ratios=model.cdi['data'][:, model.cdi['columns']['antibiotics_risk_ratio']][ids],
        cdi_dict=model.cdi['transition_lookup']
    )

    model.cdi['data'][:, model.cdi['columns']['cdi_change_probability']][ids] = cdi_change_probability


def update_recent_cdi_variables(model):
    """ Check if any agents need their recent cdi variables updated
    """
    # ----- Any living agents who has not had CDI in X days, needs their recent cdi count reset to 0
    use_agents = (model.data[:, model.columns['life_status']] == 1) & \
                 (model.cdi['data'][:, model.cdi['columns']['recurrent_cdi_ends']] == model.time)
    # --- Set agents recent cdi count to 0
    update_ids = list(where(use_agents)[0])
    for update_id in update_ids:
        model.cdi['data'][update_id, model.cdi['columns']['recurrent_cdi_ends']] = -1
        model.cdi['data'][update_id, model.cdi['columns']['recent_cdi_count']] = 0
        # --- Add to list for updating cdi transitions
        model.cdi['agents_to_update'].append(update_id)


def simulate_cdi_movement(model):
    """ Update agents CDI status as needed """

    # ----- Only consider living agents
    use_agents = model.data[:, model.columns['life_status']] == 1

    # ----- Select agents to update, based on their chance of changing CDI states
    values = model.cdi['data'][:, model.cdi['columns']['cdi_change_probability']][use_agents]
    update_agents = select_agents(values, model.np_rng.rand(values.shape[0], 1))
    # --- First select agents being used (use_agents), and then select agents changing CDI states (update_agents)
    update_ids = model.data[:, model.columns['unique_id']][use_agents][update_agents == 1]
    update_ids = list(update_ids.astype(int))

    # ----- Perform a CDI update for each id in update_ids
    for update_id in update_ids:
        cdi_update(model, model.data[update_id], model.cdi['data'][update_id])


def simulate_co_cdi(model):
    """ For agents that have moved locations, force CO-CDI
    """
    for update_id in model.agents_to_update:
        update_id = int(update_id)

        # ----- Only consider STACH patients:
        if model.data[:, model.columns['location_status']][update_id] in model.stachs:
            # Find the rate at of forced CO-CDI for the agents locations
            location = model.data[:, model.columns['location_status']][update_id]
            rate = model.parameters.cdi['forced_co_cdi'][location]

            # Force a small percent of them to get CDI - ONLY if they came from the community
            if model.np_rng.rand(1, 1)[0] < rate:
                # --- Check if they came from the community
                model.cur.execute("SELECT * FROM event_tracker WHERE Unique_ID = ? and Time = ? and State = ?",
                                  (update_id, model.time, NameState.LOCATION.value))
                events = model.cur.fetchall()
                # --- If so, give CDI
                if events[-1][model.event_tracker_columns['Location']] == model.locations.values['COMMUNITY']:
                    cdi_update(model, model.data[int(update_id)], model.cdi['data'][int(update_id)], forced=True)


def cdi_update(model, row, cdi_row, forced=False):
    """ Change the agents cdi status
    """

    # ---- Find the agents new CDI state
    current_cdi_state = cdi_row[model.cdi['columns']['cdi_state']]
    if forced:
        new_cdi_state = CDIState.CDI.value
    elif current_cdi_state == CDIState.SUSCEPTIBLE:
        # --- The only state you can go to is colonized
        new_cdi_state = CDIState.COLONIZED.value
    else:
        # --- You are colonized or have CDI and we need to simulate if you progress or regress
        key = (row[model.columns['location_status']],
               cdi_row[model.cdi['columns']['concurrent_conditions']],
               cdi_row[model.cdi['columns']['antibiotics_status']],
               cdi_row[model.cdi['columns']['recent_cdi_count']],
               row[model.columns['age_group']],
               current_cdi_state,
               cdi_row[model.cdi['columns']['antibiotics_risk_ratio']])
        cdi_array = model.cdi['transition_lookup'][key]['movement_array'].copy()

        # ----- Agent cannot retain current cdi status - As they have already been selected to switch
        cdi_array[CDIState(current_cdi_state)] = 0
        cdf = normalize_and_create_cdf(cdi_array)

        # ----- Select a new random status for the agent to move to
        rs = random_selection(model.np_rng.rand(1, 1)[0], cdf, model.cdi['states'])
        new_cdi_state = CDIState[rs].value

    # ----- If the agent dies of CDI, record it, and perform a life update
    if new_cdi_state == CDIState.DEAD:
        model.record_state_change(row, NameState.CDI_RISK, current_cdi_state, new_cdi_state)
        cdi_row[model.cdi['columns']['cdi_state']] = new_cdi_state
        life_update(model, row)
        # No longer need to update a dead agent: DO NOT DELETE.
        if row[model.columns['unique_id']] in model.cdi['agents_to_update']:
            model.cdi['agents_to_update'].remove(row[model.columns['unique_id']])
        return

    # ----- If the Agent gets CDI
    if new_cdi_state == CDIState.CDI:
        # ----- NOTE: It was requested an exponential decay from day 30-90 occur for CDI risk. This would be
        #       extremely computationally expensive to calculate each persons new chance for changing risk each day
        #       Instead of updating a persons risk daily, we are seeing if they are assigned CDI first, and then
        #       reducing how often a person actually receives CDI based on the exponential decay

        # --- If not forced and on antibiotics: We apply the antibiotic risk decay equation:
        if (not forced) and (cdi_row[model.cdi['columns']['antibiotics_status']] == 1):
            # --- Find a new chance for receiving CDI and see if you actually get it
            if cdi_row[model.cdi['columns']['antibiotics_ends']] - model.time < 60:
                days_into_antibiotics = model.time - cdi_row[model.cdi['columns']['antibiotics_ends']] + 90
                new_chance = 2.4937 * exp(-.03 * days_into_antibiotics)
                # --- If this occurs, you should not have received CDI, and thus we stop your update
                if model.np_rng.rand() > new_chance:
                    return
        # --- END removal process

        # --- If no recent CDI cases, days since last cdi is set to 0
        if cdi_row[model.cdi['columns']['most_recent_cdi']] == 0:
            days_since_last_cdi = 0
        else:
            days_since_last_cdi = model.time - cdi_row[model.cdi['columns']['most_recent_cdi']]
        # If CDI is new, or it has been 2 weeks - Your CDI count goes up
        if (days_since_last_cdi > 13) or (days_since_last_cdi == 0):
            cdi_row[model.cdi['columns']['cdi_count']] += 1
            # If you have had less than 3 recent CDI cases, this also goes up
            if cdi_row[model.cdi['columns']['recent_cdi_count']] < 3:
                cdi_row[model.cdi['columns']['recent_cdi_count']] += 1

        # --- Attribute CDI to Incident, Recurrent, or Duplicate
        attribute_cdi(model, row, days_since_last_cdi)

        # --- Set the agents CDI variables
        cdi_row[model.cdi['columns']['most_recent_cdi']] = model.time
        cdi_row[model.cdi['columns']['recurrent_cdi_ends']] = \
            model.parameters.cdi['base']['maximum_length_of_recurring_CDI'] + model.time

        # --- 1: If you are not currently on antibiotics - you are now.
        # --- 2: If passed your initial dose, reassign antibiotics
        a_state = cdi_row[model.cdi['columns']['antibiotics_status']]
        day = model.time + model.parameters.cdi['base']['antibiotic_full_dose_length']
        if (a_state == 0) or ((a_state == 1) & (cdi_row[model.cdi['columns']['antibiotics_ends']] < day)):
            cdi_row[model.cdi['columns']['antibiotics_ends']] = assign_antibiotics_days(model)
            model.record_state_change(row, NameState.ANTIBIOTICS, a_state, 1)
            cdi_row[model.cdi['columns']['antibiotics_status']] = 1
            cdi_row[model.cdi['columns']['antibiotics_risk_ratio']] = assign_antibiotics_type(model, row)

        # ----- If you are in an STACH, extend your LOS by 3 days.
        if row[model.columns['location_status']] in model.stachs:
            row[model.columns['leave_facility_day']] += 3

    # ----- Record the CDI State Change
    model.record_state_change(row, NameState.CDI_RISK, current_cdi_state, new_cdi_state)
    cdi_row[model.cdi['columns']['cdi_state']] = new_cdi_state

    # ----- Add to list for assigning new risk change value
    model.cdi['agents_to_update'].append(row[model.columns['unique_id']])


def attribute_cdi(model, row, days_since_last_cdi):
    """ Each CDI case must be given a specific label.
    CDI occurring with 0-13 days is consider duplicate. Days 14-56 are considered recurrent. >56 is a new case.

    """

    cdi_type = 'Duplicate'
    if (days_since_last_cdi > 56) or (days_since_last_cdi == 0):
        cdi_type = 'Incident'
    elif days_since_last_cdi > 13:
        cdi_type = 'Recurrent'

    # ----- NHSN Definitions: Only non-duplicate CDI events count as new CDI cases
    if cdi_type != 'Duplicate':
        # Any CDI event that happened in the community, is labeled community
        if model.locations.ints[row[model.columns['location_status']]] == 'COMMUNITY':
            nhsn_description = 'Community'
        else:
            model.cur.execute("SELECT * FROM event_tracker WHERE Unique_ID=?", (row[model.columns['unique_id']],))
            events = model.cur.fetchall()
            events = [item for item in events if item[model.event_tracker_columns['State']] == NameState.LOCATION]

            # If no events, or if the last movement event happened more than3 days ago: HO CDI
            if (len(events) == 0) or (events[-1][model.event_tracker_columns['Time']] + 3 <= model.time):
                nhsn_description = 'HO CDI'
            # Those who went Hospital A ---> somewhere else ----> Hospital A, are eligible for CO-HCFA
            else:
                # Must be within 28 days
                hcfa_events = [item for item in events if item[model.event_tracker_columns['Time']] > (model.time - 28)]
                # Old location must match current location
                hcfa_events = [item for item in hcfa_events if
                               item[model.event_tracker_columns['Location']] == row[model.columns['location_status']]]
                if len(hcfa_events) > 0:
                    nhsn_description = 'CO-HCFA CDI'
                # Everyone else is CO CDI
                else:
                    nhsn_description = 'CO CDI'
    else:
        nhsn_description = 'N/A'

    county = int(model.demographic_table[int(row[model.columns['demo_id']])][1])
    # Time, Unique_ID, Location, CDI_Type, NHSN_Desc, County
    model.cdi['cases'].append(
        (model.time, row[model.columns['unique_id']], row[model.columns['location_status']],
         cdi_type, nhsn_description, county))


def find_cdi_change_probability(locations, concurrent_conditions, antibiotics_status,
                                recent_cdi_count, age, cdi_state, antibiotics_risk_ratios, cdi_dict):
    """ Given a list of agent features, return their chance of changing CDI states

    Note:
    -----
    Cannot currently JIT this function as it contains a call to a dictionary.

    Parameters:
    -----------
    locations : np_array
        an array of locations for where agents currently are
    concurrent_conditions : np_array
        an array of the agents concurrent conditions
    antibiotics_status : np_array
        an array of the agents current antibiotic statuses
    recent_cdi_count : np_array
        an array of the agents current recent cdi counts
    age : np_array
        an array of the agents current age
    cdi_state : np_array
        an array of the agents current cdi status
    antibiotics_risk_ratios : np_array
        an array of the agents current risk ratios

    """
    probability = zeros(len(cdi_state))
    for i in range(len(cdi_state)):
        key = (locations[i],
               concurrent_conditions[i].astype(int),
               antibiotics_status[i].astype(int),
               recent_cdi_count[i].astype(int),
               age[i],
               cdi_state[i],
               antibiotics_risk_ratios[i])
        probability[i] = cdi_dict[key]['chance_to_move']

    return probability


def find_antibiotics_probability(locations, age, antibiotics_dictionary):
    """ Given a list of agent features, return their chance of receiving antibiotics

    Note:
    -----
    Cannot currently JIT this function as it contains a call to a dictionary.

    Parameters:
    -----------
    locations : np_array
        an array of locations for where agents currently are
    age : np_array
        an array of the agents current age
    antibiotics_dictionary : dict
        dictionary of probabilities for receiving antibiotics

    """
    probability = zeros(len(locations))
    for i in range(len(locations)):
        key = (locations[i],
               age[i])
        probability[i] = antibiotics_dictionary[key]

    return probability


def regenerate_cdi_variables(model, agent_ids, new_data):
    """ When an agent dies, we regenerate them. This function will prepare a new agent with CDI values

    Parameters:
    -----------
    model : the model

    agent_ids : list
        List of the agents who need to be regenerated

    new_data : np_array
        All of the agents current data

    """

    # ----- Grab their current cdi data
    new_cdi_data = model.cdi['data'][agent_ids, :]
    new_cdi_data.fill(0)
    # --- Assign concurrent conditions
    new_cdi_data[:, model.cdi['columns']['concurrent_conditions']] = \
        assign_conditions(model.data[agent_ids, model.columns['age_group']],
                          model.np_rng.rand(len(agent_ids), 1))
    # --- Assign a default risk ratio
    new_cdi_data[:, model.cdi['columns']['antibiotics_risk_ratio']] = \
        model.parameters.cdi['relative_risk']['antibiotics']['default']
    # --- Reset their antibiotic use and recurrent cdi status
    new_cdi_data[:, model.cdi['columns']['antibiotics_ends']] = -1
    new_cdi_data[:, model.cdi['columns']['recurrent_cdi_ends']] = -1
    # --- Look up their CDI change probability
    cdi_change_probability = find_cdi_change_probability(
        locations=new_data[:, model.columns['location_status']],
        concurrent_conditions=new_cdi_data[:, model.cdi['columns']['concurrent_conditions']],
        antibiotics_status=zeros(len(agent_ids), dtype=int),
        recent_cdi_count=zeros(len(agent_ids), dtype=int),
        age=new_data[:, model.columns['age_group']],
        cdi_state=zeros(len(agent_ids), dtype=int),
        antibiotics_risk_ratios=new_cdi_data[:, model.cdi['columns']['antibiotics_risk_ratio']],
        cdi_dict=model.cdi['transition_lookup']
    )
    # --- Look up their Antibiotics change probability
    antibiotics_probability = find_antibiotics_probability(
        locations=new_data[:, model.columns['location_status']],
        age=new_data[:, model.columns['age_group']],
        antibiotics_dictionary=model.cdi['antibiotics_dictionary']
    )
    # ----- Add the CDI data to the model
    new_cdi_data[:, model.cdi['columns']['cdi_change_probability']] = cdi_change_probability
    new_cdi_data[:, model.cdi['columns']['antibiotics_probability']] = antibiotics_probability
    model.cdi['data'] = vstack((model.cdi['data'], new_cdi_data))


def assign_antibiotics_type(model, row):
    """ Agents can receive different classes of antibiotics. This will randomly select a class for them.
    """

    location = row[model.columns['location_status']]
    distribution = model.parameters.cdi['antibiotics']['distributions'][location]
    # Pick a random antibiotic
    a_class = random_selection(
        model.np_rng.rand(1, 1)[0], create_cdf(distribution), model.cdi['antibiotic_classes'][1:])

    return model.parameters.cdi['relative_risk']['antibiotics'][a_class]


def assign_antibiotics_days(model):
    """ Assign an end date for a persons antibiotics.
        This should be their initial dose (pulled from a distribution), plus a lasting effect
    """
    rp = model.parameters.cdi['base']
    initial = round(model.np_rng.normal(rp['antibiotic_administration_mean'], rp['antibiotic_administration_sd']))
    return initial + rp['antibiotic_full_dose_length'] + model.time


def simulate_antibiotics_recovery(model):
    # ----- Living agents whose antibiotics ends today, need antibiotics turned off, and CDI probabilities updated
    use_agents = (model.data[:, model.columns['life_status']] == 1) & \
                 (model.cdi['data'][:, model.cdi['columns']['antibiotics_ends']] == model.time)
    # --- Turn off their antibiotics
    ids = list(where(use_agents)[0])
    for agent_id in ids:
        model.record_state_change(model.data[agent_id], NameState.ANTIBIOTICS, 1, 0)
        model.cdi['data'][agent_id, model.cdi['columns']['antibiotics_ends']] = -1
        model.cdi['data'][agent_id, model.cdi['columns']['antibiotics_status']] = 0
        # --- Add to list for updating cdi transitions
        model.cdi['agents_to_update'].append(agent_id)


def simulate_facility_antibiotics(model):  # TODO Add this to tested functions
    """ Agents in facilities, on antibiotics, but off of their initial dose can be given more antibiotics.
    """
    p = model.parameters

    # ----- Find living agents, who are on antibiotics and at a facility
    living_agents = model.data[:, model.columns['life_status']] == 1
    agents_not_in_community = model.data[:, model.columns['location_status']] != model.locations.values['COMMUNITY']
    agents_on_antibiotics = model.cdi['data'][:, model.cdi['columns']['antibiotics_status']] == 1
    # --- Must be all 3
    use_agents = living_agents & agents_not_in_community & agents_on_antibiotics

    # ----- Find agents who are off their original administration of antibiotics
    day = model.time + p.cdi['base']['antibiotic_full_dose_length']
    agents_who_meet_criteria =\
        model.cdi['data'][:, model.cdi['columns']['antibiotics_ends']][use_agents] < day
    # --- Grab their unique IDS
    test_ids = model.data[:, model.columns['unique_id']][use_agents][agents_who_meet_criteria].astype(int)

    # ----- Find the name of their locations
    locations = model.data[:, model.columns['location_status']][test_ids].astype(int)

    # ----- Find the probability associated with the location
    probabilities = array([p.cdi['antibiotics']['facility'][item] for item in locations])

    # ----- Randomly select which agents get antibiotics
    update_agents = select_agents(probabilities, model.np_rng.rand(len(probabilities), 1))
    update_ids = list(test_ids[update_agents == 1])

    # ----- Give them a new dose of antibiotics
    for update_id in update_ids:
        cdi_row = model.cdi['data'][update_id]
        row = model.data[update_id]
        # --- Assign a new antibiotics end day
        cdi_row[model.cdi['columns']['antibiotics_ends']] = assign_antibiotics_days(model)
        # --- Record a state change: Even though you didn't change states: You were ON, now you are still ON
        model.record_state_change(row, NameState.ANTIBIOTICS, 1, 1)
        # --- Check new risk ratio
        new_risk_ratio = assign_antibiotics_type(model, row)
        # --- Only assign new risk if it is higher than the old risk
        if new_risk_ratio > cdi_row[model.cdi['columns']['antibiotics_risk_ratio']]:
            cdi_row[model.cdi['columns']['antibiotics_risk_ratio']] = new_risk_ratio
        # --- Add cdi update list
        model.cdi['agents_to_update'].append(update_id)


def simulate_antibiotics(model):
    """ Simulate agents receiving antibiotics
        1: Agents who are OFF of antibiotics
        2: Agents who are ON antibiotics, and whose initial dose of antibiotics has ended
    """

    # ----- PART #1: Agents who are OFF antibiotics
    living_agents = model.data[:, model.columns['life_status']] == 1
    agents_not_on_antibiotics = model.cdi['data'][:, model.cdi['columns']['antibiotics_status']] == AntibioticsState.OFF
    # --- Must be both
    use_agents = living_agents & agents_not_on_antibiotics

    # ----- Select agents to update, based on their chance of receiving antibiotics
    probabilities = model.cdi['data'][:, model.cdi['columns']['antibiotics_probability']][use_agents]
    update_agents = select_agents(probabilities, model.np_rng.rand(probabilities.shape[0], 1))
    # --- Find their unique IDS
    update_ids = model.data[:, model.columns['unique_id']][use_agents][update_agents == 1].astype(int)
    # --- Set their antibiotics status
    model.cdi['data'][update_ids, model.cdi['columns']['antibiotics_status']] = 1

    # ----- Give agents a course of antibiotics
    for update_id in update_ids:
        cdi_row = model.cdi['data'][update_id]
        row = model.data[update_id]
        # --- Assign antibiotics end day
        cdi_row[model.cdi['columns']['antibiotics_ends']] = assign_antibiotics_days(model)
        # --- Record a state change
        model.record_state_change(row, NameState.ANTIBIOTICS, 0, 1)
        # --- Assign Risk Ratio
        cdi_row[model.cdi['columns']['antibiotics_risk_ratio']] = assign_antibiotics_type(model, row)
        # --- Add cdi update list
        model.cdi['agents_to_update'].append(update_id)

    # ----- PART #2: Agents who are ON antibiotics
    cdi_params = model.parameters.cdi
    agents_on_antibiotics = model.cdi['data'][:, model.cdi['columns']['antibiotics_status']] == 1
    use_agents = living_agents & agents_on_antibiotics

    # ----- Find agents who are off their original administration of antibiotics
    day = model.time + cdi_params['base']['antibiotic_full_dose_length']
    agents_who_meet_criteria =\
        model.cdi['data'][:, model.cdi['columns']['antibiotics_ends']][use_agents] < day
    # --- Grab their unique IDS
    use_agents_temp = model.data[:, model.columns['unique_id']][use_agents][agents_who_meet_criteria].astype(int)

    # ----- Find the name of their locations
    probabilities = model.cdi['data'][:, model.cdi['columns']['antibiotics_probability']][use_agents_temp]
    # --- Randomly select which agents get antibiotics
    update_agents = select_agents(probabilities, model.np_rng.rand(len(probabilities), 1))
    update_ids = list(use_agents_temp[update_agents == 1])

    # ----- Give them a new dose of antibiotics
    for update_id in update_ids:
        cdi_row = model.cdi['data'][update_id]
        row = model.data[update_id]
        # --- Assign a new antibiotics end day
        cdi_row[model.cdi['columns']['antibiotics_ends']] = assign_antibiotics_days(model)
        # --- Record a state change: Even though you didn't change states: You were ON, now you are still ON
        model.record_state_change(row, NameState.ANTIBIOTICS, 1, 1)
        # --- Check new risk ratio
        new_risk_ratio = assign_antibiotics_type(model, row)
        # --- Only assign new risk if it is higher than the old risk
        if new_risk_ratio > cdi_row[model.cdi['columns']['antibiotics_risk_ratio']]:
            cdi_row[model.cdi['columns']['antibiotics_risk_ratio']] = new_risk_ratio
        # --- Add cdi update list
        model.cdi['agents_to_update'].append(update_id)


def update_antibiotics_probabilities(model):
    """ When an agent moves locations, update their antibiotic_probability """
    ids = model.agents_to_update
    ids = array(ids, dtype=int)
    temp_data = model.data[ids, :]
    antibiotics_probability = find_antibiotics_probability(
        locations=temp_data[:, model.columns['location_status']],
        age=temp_data[:, model.columns['age_group']],
        antibiotics_dictionary=model.cdi['antibiotics_dictionary']
    )
    model.cdi['data'][:, model.cdi['columns']['antibiotics_probability']][ids] = antibiotics_probability


def update_force_of_colonization(model):
    """ Colonization is based on the number of patients at each facility who are colonized or who have CDI.
        Below, we calculate the force of colonization based on facility and patients in that facility,
        based on an equation that comes from the Durham et al. paper.

    Note: We do not use this equation for the community. We use Durham's overall estimate

    Tasks:
        #1: Calculate the new force of colonization value for each facility
        #2: Update the cdi dictionary. Any susceptible (non-community) key needs to be updated
        #3: Update all non-community patients probability of CDI transition
    """

    # ----- Overall parameters
    # g: overall hospital hygiene
    g = 1
    # B_s = base CDI transition rate: same for all hospitals
    bs = model.parameters.cdi['CDI']['base_rate']['hospital']
    # B_a = base asymptomatic colonization transition rate: same for all hospitals
    ba = model.parameters.cdi['colonization']['base_rate']['hospital']
    # pi: probability that patient with CDI is identified & content precautions are used
    pi = model.parameters.cdi['contact_precautions']['identified']
    # epsilon: effectiveness of contact precautions employed
    epsilon = model.parameters.cdi['contact_precautions']['effectiveness']

    # ----- Task #1: Calculate the force equation for each facility
    force_values = dict()
    community = model.locations.values['COMMUNITY']
    # --- Find non-community agents (for speed purposes)
    non_community = model.data[:, model.columns['location_status']] != community
    temp_data = model.data[non_community, :]
    temp_cdi_data = model.cdi['data'][non_community, :]
    # --- Loop through locations and find the force of colonization value
    for location in model.locations.ints:
        if location != community:
            # Find the agents in the selected facility:
            use_agents = temp_data[:, model.columns['location_status']] == location
            agent_ids = temp_data[:, model.columns['unique_id']][use_agents].astype(int)
            if len(agent_ids) > 0:
                # Find the CDI Status
                status = model.cdi['data'][:, model.cdi['columns']['cdi_state']][agent_ids]
                # CDI_st = # of CDI cases in hospital / # of STACH patients
                cdi_st = collections.Counter(status)[CDIState.CDI.value] / len(agent_ids)
                # C_st = # of colonized cases in hospital / # of STACH patients
                c_st = collections.Counter(status)[CDIState.COLONIZED.value] / len(agent_ids)
            else:
                cdi_st = 0
                c_st = 0
            # lambda_st = equation
            lambda_st = g * (bs * (1 - pi) * cdi_st + ba * c_st) + pi * bs * cdi_st * (1 - epsilon)
            # Use parameters to correct colonization amount by location
            force_values[location] =\
                lambda_st * model.parameters.cdi['colonization']['tuning'][location]

    # ----- Task #2: Update the transition_lookup dictionary.
    for key in model.cdi['transition_lookup'].keys():
        if key[0] != community:
            if key[5] in [CDIState.SUSCEPTIBLE.value]:
                # Find the movement array
                movement_array = model.cdi['transition_lookup'][key]['movement_array']
                # Find force of colonization:
                force_of_col = force_values[key[0]]
                # Find out how much this has changed:
                change_of_col = movement_array[CDIState.COLONIZED.value] - force_of_col
                # Update movement_array:
                movement_array[0] = movement_array[0] + change_of_col
                movement_array[CDIState.COLONIZED.value] = force_of_col
                # Update chance to move:
                model.cdi['transition_lookup'][key]['chance_to_move'] = 1 - movement_array[0]

    # ----- Task #3: Agents not in the community need their CDI change probability updated
    cdi_change_probability = find_cdi_change_probability(
        locations=temp_data[:, model.columns['location_status']].astype(int),
        concurrent_conditions=temp_cdi_data[:, model.cdi['columns']['concurrent_conditions']].astype(int),
        antibiotics_status=temp_cdi_data[:, model.cdi['columns']['antibiotics_status']].astype(int),
        recent_cdi_count=temp_cdi_data[:, model.cdi['columns']['recent_cdi_count']].astype(int),
        age=model.data[:, model.columns['age_group']],
        cdi_state=temp_cdi_data[:, model.cdi['columns']['cdi_state']].astype(int),
        antibiotics_risk_ratios=temp_cdi_data[:, model.cdi['columns']['antibiotics_risk_ratio']].astype(int),
        cdi_dict=model.cdi['transition_lookup']
    )
    model.cdi['data'][non_community, model.cdi['columns']['cdi_change_probability']] = cdi_change_probability


def create_antibiotics_dictionary(model):
    params = model.parameters.cdi

    d = dict()
    for location in model.locations.ints.keys():
        for age in [0, 1, 2]:
            if model.locations.ints[location] == 'COMMUNITY':
                d[location, age] = params['antibiotics']['COMMUNITY']['age'][str(age)]
            else:
                d[location, age] = params['antibiotics']['facility'][location]
    return d


def create_cdi_dictionary(model):
    params = model.parameters.cdi
    antibiotics_risk_ratios = list(params['relative_risk']['antibiotics'].values())

    d_order = dict()
    d_order['location'] = 0
    d_order['concurrent_conditions'] = 1
    d_order['antibiotics_state'] = 2
    d_order['recent_cdi_count'] = 3
    d_order['age'] = 4
    d_order['cdi_state'] = 5
    d_order['antibiotics_risk_ratio'] = 6

    d = dict()
    for location in model.locations.ints.keys():
        for concurrent_conditions in [0, 1]:
            for antibiotics_state in [0, 1]:
                for recent_cdi_count in [0, 1, 2, 3]:
                    for age in [0, 1, 2]:
                        for cdi_state in [item.value for item in CDIState if item.name != 'DEAD']:
                            for risk_ratio in antibiotics_risk_ratios:
                                d[location, concurrent_conditions, antibiotics_state, recent_cdi_count,
                                  age, cdi_state, risk_ratio] = {
                                    'chance_to_move': 0,
                                    'movement_array': array(1)
                                }

    for key in d.keys():
        community_index = model.locations.values['COMMUNITY']
        nh_index = model.locations.values['NH']
        non_unc_index = model.locations.values['ST']
        if key[d_order['cdi_state']] == CDIState.SUSCEPTIBLE.value:
            # TO COLONIZED
            if key[d_order['location']] == community_index:
                to_col =\
                    params['colonization']['base_rate']['COMMUNITY'] * params['colonization']['tuning'][community_index]
            elif key[d_order['location']] == nh_index:
                to_col = params['colonization']['base_rate']['NH']
            else:
                to_col = params['colonization']['base_rate']['hospital']

            d[key]['movement_array'] = array([1 - to_col, to_col, 0, 0])
            d[key]['chance_to_move'] = to_col

        elif key[d_order['cdi_state']] == CDIState.COLONIZED.value:
            # TO SUSCEPTIBLE
            to_susceptible = params['colonization']['clearance']
            # TO CDI
            if key[d_order['recent_cdi_count']] > 0:
                to_cdi = params['CDI']['recurrence']
            else:
                # Add Base risk (AGE)
                rr = 1 * params['relative_risk']['age'][str(key[d_order['age']])]
                # Add Antibiotic Risk Ratio
                if key[d_order['antibiotics_state']] == 1:
                    rr *= key[d_order['antibiotics_risk_ratio']]
                # Add Concurrent Conditions
                rr *= params['relative_risk']['concurrent_conditions'][str(key[d_order['concurrent_conditions']])]
                # Add the tuning parameters by location
                if key[d_order['location']] == community_index:
                    to_cdi = params['CDI']['base_rate']['COMMUNITY'] * rr * params['CDI']['tuning']['COMMUNITY']
                elif key[d_order['location']] == nh_index:
                    to_cdi = params['CDI']['base_rate']['NH'] * rr * params['CDI']['tuning']['NH']
                elif key[d_order['location']] == non_unc_index:
                    to_cdi = params['CDI']['base_rate']['NH'] * rr * params['CDI']['tuning']['non_unc_hospital']
                else:
                    to_cdi = params['CDI']['base_rate']['hospital'] * rr * params['CDI']['tuning']['unc_hospital']

            chance_to_move = to_susceptible + to_cdi
            d[key]['movement_array'] = array([to_susceptible, 1 - chance_to_move, to_cdi, 0])
            d[key]['chance_to_move'] = chance_to_move

        elif key[d_order['cdi_state']] == CDIState.CDI.value:
            # To COLONIZED:
            to_col = params['CDI']['recovery'] * (
                params['colonization']['recurrence']['recent_CDI'][str(key[d_order['recent_cdi_count']])])
            # TO DEAD:
            to_dead = params['CDI']['recovery'] * params['death']['age'][str(key[d_order['age']])]
            # To SUSCEPTIBLE
            to_susceptible = params['CDI']['recovery'] - to_col - to_dead

            chance_to_move = to_susceptible + to_col + to_dead
            d[key]['movement_array'] = array([to_susceptible, to_col, 1 - chance_to_move, to_dead])
            d[key]['chance_to_move'] = chance_to_move
    return d


def collect_cdi_agents(model, initiate):
    if initiate:
        daily_count_index_list = []
        for an_anti in [0, 1]:
            for a_life in [0, 1]:
                for a_location in model.locations.values.values():
                    for a_cdi_state in CDIState:
                        daily_count_index_list.append((an_anti, a_life, a_location, a_cdi_state.value))
        daily_counts = DataFrame(daily_count_index_list)
        daily_counts.columns = ['Antibiotics', 'Life', 'Location', 'CDI']
        model.daily_counts = daily_counts.set_index(['Antibiotics', 'Life', 'Location', 'CDI'])

    d_array = concatenate(
        (model.cdi['data'][:, [model.cdi['columns']['antibiotics_status']]],
         model.data[:, [model.columns['life_status'], model.columns['location_status']]],
         model.cdi['data'][:, [model.cdi['columns']['cdi_state']]]), axis=1)

    df = DataFrame(d_array, columns=['Antibiotics', 'Life', 'Location', 'CDI'])
    df = df.groupby(by=['Antibiotics', 'Life', 'Location', 'CDI']).size()
    df.index.names = ['Antibiotics', 'Life', 'Location', 'CDI']
    model.daily_counts[model.time] = df
    model.daily_counts = model.daily_counts.fillna(0)
