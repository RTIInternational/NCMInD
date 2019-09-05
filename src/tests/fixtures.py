import pytest
from pandas import read_csv
from pathlib import Path
import json
import sys
sys.path.append("")
from src.ldm import LDM


with open("NCMIND/cdi_calibration/default/parameters.json") as file:
    params = json.load(file)

params['base']['limit_pop'] = 100
params['base']['population_file'] = 'synthetic_population_orange.csv'
with open("NCMIND/demo/unit_tests/parameters.json", 'w') as outfile:
    json.dump(params, outfile, indent=4)


@pytest.fixture(scope="session")
def model():
    return LDM(exp_dir='NCMIND/demo', scenario='unit_tests')


@pytest.fixture(scope="session")
def data_row(model):
    return model.data[0]


@pytest.fixture(scope="session")
def cdi_row(model):
    return model.cdi['data'][0]


@pytest.fixture()
def synthetic_population_orange():
    return read_csv('NCMIND/data/synthetic_population/synthetic_population_orange.csv')


@pytest.fixture()
def transition_files():
    d = "NCMIND/data/raw/"
    t_files = dict()
    t_files['unc'] = read_csv(Path(d, 'location_transitions/UNC.csv'))
    t_files['non_unc'] = read_csv(Path(d, 'location_transitions/Non-UNC.csv'))
    t_files['lt'] = read_csv(Path(d, 'location_transitions/LT.csv'))
    t_files['nh'] = read_csv(Path(d, 'location_transitions/NH.csv'))
    t_files['community'] = read_csv(Path(d, 'location_transitions/Community.csv'))
    t_files['unc_to_unc'] = read_csv(Path(d, 'location_transitions/UNC-to-UNC.csv'))

    return t_files


@pytest.fixture()
def name_lists():
    name_lists = dict()
    names = ['County', 'County Code', 'From', 'To', 'Race', 'Sex', 'Age Group', 'Transition Probability',
             'Caldwell', 'Chatham', 'High Point', 'Johnston', 'Lenoir', 'Margaret', 'Nash', 'Rex', 'UNC-CH', 'Wayne']
    unc_names = ['County', 'County Code', 'Hospital', 'Caldwell', 'Chatham', 'High Point', 'Johnston',
                 'Lenoir', 'Margaret', 'Nash', 'Rex', 'UNC-CH', 'Wayne']
    hospitals = ['Caldwell', 'Chatham', 'High Point', 'Johnston', 'Lenoir', 'Margaret', 'Nash', 'Rex', 'UNC-CH',
                 'Wayne']
    name_lists['names'] = names
    name_lists['unc_names'] = unc_names
    name_lists['hospitals'] = hospitals

    return name_lists


@pytest.fixture()
def initial_files():
    d = "NCMIND/data/raw/"
    initial_files = dict()
    # Initial UNC File
    initial_files['unc'] = read_csv(Path(d, 'initial_population/Initial_UNC_Population.csv'))
    # Initial Non-UNC File
    initial_files['non_unc'] = read_csv(Path(d, 'initial_population/Initial_Non-UNC_Population.csv'))
    # Initial NH File
    initial_files['nh'] = read_csv(Path(d, 'initial_population/Initial_NH_Population.csv'))
    # Initial LT File
    initial_files['lt'] = read_csv(Path(d, 'initial_population/Initial_LT_Population.csv'))

    return initial_files
