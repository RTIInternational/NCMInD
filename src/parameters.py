
import json

from pathlib import Path
from pandas import read_csv, DataFrame
from numpy import random


class Parameters:

    def __init__(self, params_file):
        self.update_from_file(params_file)

    def update_from_file(self, params_file):
        params_file = Path(params_file)

        params = {}
        if params_file.exists():
            with open(params_file) as f:
                params = json.load(f)
        else:
            print('WARNING: Parameter file does not exist.')

        for key in params:
            setattr(self, key, params[key])


def update_parameters_file(parameters: json):
    """ Randomly select parameters from a distribution for a new parameters file
    """
    rng = random.RandomState(parameters['base']['seed'])

    d1 = read_csv("NCMIND/data/input/cdi/parameter_distributions.csv")

    # ------------------------------------------------------------------------------------------------------------------
    # ----- CDI Base Rates
    br = parameters['disease']['cdi']['base_rate']
    # --- Hospital
    br['COMMUNITY'] = pull_triangular(rng, d1[d1.Parameter == 'CDI Base Rate (COMMUNITY)'])
    st = pull_triangular(rng, d1[d1.Parameter == 'CDI Base Rate (STACH)'])
    br['SMALL'] = st
    br['LARGE'] = st
    br['UNC'] = st
    br['LT'] = st
    br['NH'] = pull_triangular(rng, d1[d1.Parameter == 'CDI Base Rate (NH)'])

    # ------------------------------------------------------------------------------------------------------------------
    # ----- CDI Revoery & Recurrence
    parameters['disease']['cdi']['recovery'] =\
        pull_triangular(rng, d1[d1.Parameter == 'CDI Recovery'])
    parameters['disease']['cdi']['recurrence']['base_rate'] =\
        pull_triangular(rng, d1[d1.Parameter == 'CDI Recurrence'])
    parameters['disease']['cdi']['recurrence']['probability_with_recent_CDI']['1'] =\
        pull_triangular(rng, d1[d1.Parameter == 'CDI Recurrence Probability 1'])
    parameters['disease']['cdi']['recurrence']['probability_with_recent_CDI']['2'] =\
        pull_triangular(rng, d1[d1.Parameter == 'CDI Recurrence Probability 2'])
    parameters['disease']['cdi']['recurrence']['probability_with_recent_CDI']['3'] =\
        pull_triangular(rng, d1[d1.Parameter == 'CDI Recurrence Probability 3'])

    # ------------------------------------------------------------------------------------------------------------------
    # ----- Colonization
    parameters['disease']['colonization']['clearance'] =\
        pull_triangular(rng, d1[d1.Parameter == 'Colonization Clearance'])
    br_colonization = parameters['disease']['colonization']['base_rate']
    br_colonization['COMMUNITY'] =\
        pull_triangular(rng, d1[d1.Parameter == 'Base Colonization Rate (COMMUNITY)'])
    st_colonization =\
        pull_triangular(rng, d1[d1.Parameter == 'Base Colonization Rate (STACHs/LTACHs)'])
    br_colonization['SMALL'] = st_colonization
    br_colonization['LARGE'] = st_colonization
    br_colonization['UNC'] = st_colonization
    br_colonization['LT'] = st_colonization
    br_colonization['NH'] =\
        pull_triangular(rng, d1[d1.Parameter == 'Base Colonization Rate (NH)'])
    # --- Force of Colonization Parameters
    parameters['disease']['cdi']['contact_precautions']['identified'] =\
        pull_triangular(rng, d1[d1.Parameter == 'Contact Precautions: Identified'])
    parameters['disease']['cdi']['contact_precautions']['effectiveness'] =\
        pull_triangular(rng, d1[d1.Parameter == 'Contact Precautions: Effectiveness'])

    # ------------------------------------------------------------------------------------------------------------------
    # ----- Relative Risks
    parameters['disease']['cdi']['relative_risk']['concurrent_conditions']['1'] =\
        pull_triangular(rng, d1[d1.Parameter == 'Relative Risk (Concurrent Conditions)'])
    parameters['disease']['cdi']['relative_risk']['age']['1'] =\
        pull_triangular(rng, d1[d1.Parameter == 'Relative Risk (Age, 50-64)'])
    parameters['disease']['cdi']['relative_risk']['age']['2'] =\
        pull_triangular(rng, d1[d1.Parameter == 'Relative Risk (Age, 65+)'])

    return parameters


def pull_triangular(rng: random.RandomState, row: DataFrame):
    return rng.triangular(row['Daily Low'], row.Daily, row['Daily High'])[0]
