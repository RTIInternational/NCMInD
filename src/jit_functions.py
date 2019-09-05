from numba import njit
from numpy import zeros


# ----------------------------------------------------------------------------------------------------------------------
# ------ JIT Functions
# ----------------------------------------------------------------------------------------------------------------------
@njit
def find_demo_id(dt, county_code, sex, age, race):
    demo_id = zeros(county_code.shape)
    for i in range(len(county_code)):
        a = dt[(dt[:, 1] == county_code[i]) & (dt[:, 2] == sex[i]) & (dt[:, 3] == age[i]) & (dt[:, 4] == race[i])]
        demo_id[i] = a[0][0]

    return demo_id


@njit
def find_probability(p_values, ids):
    probabilities = zeros(ids.shape)
    for i in range(len(ids)):
        probabilities[i] = p_values[ids[i]]
    return probabilities


@njit
def select_agents(probabilities, randoms):
    agents = zeros(probabilities.shape)
    for i in range(len(agents)):
        if probabilities[i] > randoms[i][0]:
            agents[i] = 1
    return agents


@njit
def assign_conditions(age, randoms):
    conditions = zeros(len(age))
    for i in range(len(age)):
        if age[i] == 1:
            if randoms[i][0] < .2374:
                conditions[i] = 1
        elif age[i] == 2:
            if randoms[i][0] < .5497:
                conditions[i] = 1
    return conditions


@njit
def assign_colonization(rates, randoms, value):
    cdi_status = zeros(rates.shape[0])
    for i in range(rates.shape[0]):
        if rates[i] > randoms[i][0]:
            cdi_status[i] = value
    return cdi_status


@njit
def update_community_probability(cp, age, cc):
    """
    If simulating risk, we can update hospital transitions based on concurrent conditions. Update an agents
    community_probability based on their concurrent conditions and their age.
    """
    for i in range(len(age)):
        if cc[i] == 1:
            if age[i] == 1:
                cp[i] = cp[i] * 55 / 23.74
            elif age[i] == 2:
                cp[i] = cp[i] * 79 / 54.97
        else:
            if age[i] == 1:
                cp[i] = cp[i] * 45 / 76.26
            elif age[i] == 2:
                cp[i] = cp[i] * 21 / 45.03
    return cp
