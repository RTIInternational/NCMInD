
import pytest
import pandas as pd
from pathlib import Path
import json
from src.ldm import Ldm


# ----- Update the CDI model parameters: # TODO
# with open("NCMINDv3/cdi_calibration/default/parameters.json") as file:
#     params = json.load(file)
# params['base']['limit_pop'] = 100
# params['base']['population_file'] = "NCMINDv3/data/synthetic_population/synthetic_population_orange.csv"
# with open("NCMINDv3/demo/cdi_model/parameters.json", 'w') as outfile:
#     json.dump(params, outfile, indent=4)


@pytest.fixture(scope="session")
def model():
    # ----- Update the location model parameters
    with open("NCMIND/location_calibration/run_1/parameters.json") as file:
        params = json.load(file)
    params['base']['limit_pop'] = 100
    params['base']['population_file'] = "NCMIND/data/synthetic_population/synthetic_population_orange.csv"
    with open("NCMIND/location_demo/run_1/parameters.json", 'w') as outfile:
        json.dump(params, outfile, indent=4)
    # ----- Make the model
    model = Ldm(experiment='NCMIND', scenario='location_demo', run='run_1')
    model.collapse_sql_connection()
    return model


@pytest.fixture(scope="session")
def cdi_model():
    Path("NCMIND/cdi_demo/").mkdir(exist_ok=True)
    Path("NCMIND/cdi_demo/run_1").mkdir(exist_ok=True)

    # ----- Update the location model parameters
    with open("NCMIND/cdi_calibration/run_1/parameters.json") as file:
        params = json.load(file)
    params['base']['limit_pop'] = 100
    params['base']['population_file'] = "NCMIND/data/synthetic_population/synthetic_population_orange.csv"
    with open("NCMIND/cdi_demo/run_1/parameters.json", 'w') as outfile:
        json.dump(params, outfile, indent=4)
    # ----- Make the model
    model = Ldm(experiment='NCMIND', scenario='cdi_demo', run='run_1')
    model.collapse_sql_connection()
    return model


@pytest.fixture()
def synthetic_population_orange():
    return pd.read_csv('NCMIND/data/synthetic_population/synthetic_population_orange.csv')


@pytest.fixture()
def transition_files():
    d = "NCMIND/data/input/"
    t_files = dict()
    t_files['community'] = pd.read_csv(Path(d, 'community_transitions.csv'))
    t_files['LARGE'] = pd.read_csv(Path(d, "large_discharge_transitions.csv"))
    t_files['SMALL'] = pd.read_csv(Path(d, "small_discharge_transitions.csv"))
    t_files['location'] = pd.read_csv(Path(d, "location_transitions.csv"))
    t_files['logrecnos'] = pd.read_csv(Path(d, 'logrecnos.csv'))
    t_files['UNC'] = pd.read_csv(Path(d, 'unc_to_unc_transitions.csv'))
    return t_files


@pytest.fixture()
def id_files():
    d = 'NCMIND/data/IDs/'
    id_files = dict()
    id_files['hospitals'] = pd.read_csv(Path(d, 'hospital_ids.csv'))
    id_files['nursing_homes'] = pd.read_csv(Path(d, 'nh_ids.csv'))
    id_files['ltachs'] = pd.read_csv(Path(d, 'lt_ids.csv'))
    return id_files
