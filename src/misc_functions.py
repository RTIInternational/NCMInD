
import numpy as np
import pandas as pd
from bisect import bisect


def create_cdf(p: list) -> list:
    """ Convert a list of probabilities, into a cumulative distribution function

    Parameters
    ----------
    p : a list of probabilities that add to 1
    """
    s1 = sum(p)
    if (s1 < .999) or (s1 > 1.001):
        print(p)
        raise ValueError("You cannot give a list, p, that does not sum to 1.")
    cdf = list()
    cdf.append(p[0])
    for i in range(1, len(p)):
        cdf.append(cdf[-1] + p[i])
    cdf[len(cdf) - 1] = 1
    return cdf


def normalize_and_create_cdf(p: list) -> list:
    """ Normalize a list of probabilities and create the cdf for them

    Parameters
    ----------
    p : list of probabilities. May not add to 1, as one probability may have been removed. Thus, we normalize
    """
    total = sum(p)
    p = [item / total for item in p]
    return create_cdf(p)


def random_selection(random: float, cdf: list, options: list) -> object:
    """ Given cumulative distribution function and a list of options, make a random selection

    Parameters
    ----------
    random: a random number between 0 and 1
    cdf : a list containing the cumulative distribution values. Ex [0, .2, .3, .7, 1.0]
    options : a list containing the options that can be selected
    """
    return options[bisect(cdf, random)]


def dictionary_to_array(ldm, dictionary: dict, ids: list = None) -> np.array:
    """ Convert a dictionary of values into an array. Dictionaries do not have all values, so fill with 0s

    Parameters
    ----------
    ldm: LDM
    dictionary : A dictionary with key value pairs of agent ids and statuses
    ids : The list of ids to include
    """
    if ids:
        an_array = np.zeros(len(ids))
        for i, unique_id in enumerate(ids):
            if unique_id in dictionary:
                an_array[i] = dictionary[unique_id]
    else:
        an_array = np.zeros(len(ldm.unique_ids))
        for k, v in dictionary.items():
            an_array[k] = v
    return an_array.astype(int)


def int_to_category(locations, an_array: np.array) -> list:
    """ Convert an array of integers to their category

    Parameters
    ----------
    locations : a locations modeling object
    an_array : a numpy array containing integers representing locations
    """
    return [locations.convert_int(item, 'category') for item in an_array]


def row_to_df(m, u_id):
    df = pd.DataFrame(m.data[u_id]).T
    df.columns = m.columns
    return df


def to_df(m):
    df = pd.DataFrame(m.data)
    df.columns = m.columns
    df['Location'] = [m.current_locations[i] if i in m.current_locations else 0 for i in range(len(df))]
    df['LOS'] = [m.current_los[i] if i in m.current_los else 0 for i in range(len(df))]
    return df


sex_dictionary = {
    "M": 1,
    "F": 2
}
race_dictionary = {
    "White": 1,
    "Black": 2,
    "Other": 3
}
age_dictionary = {
    "L50": 0,
    "50-64": 1,
    "65+": 2
}
