"""
We need to figure out how much to reduce the daily probability of prescribing antibiotics in order to achieve an actual
    10%, or 20% drop in prescribed antibiotics for each facility type
"""
import json
import pandas as pd
from pathlib import Path
from copy import deepcopy

from NCMIND.src.analyze import Analyze
from src.state import NameState, AntibioticsState
from src.misc_functions import int_to_category
from src.ldm import Ldm


if __name__ == "__main__":
    # ----- Set up and run a baseline model
    experiment = "NCMIND/"
    scenario = "stewardship_paper/testing_directory"
    scenario_dir = Path(experiment, scenario)
    scenario_dir.mkdir(exist_ok=True)
    run_dir = Path(experiment, scenario, 'run_1')
    run_dir.mkdir(exist_ok=True)

    with open("NCMIND/cdi_calibration/run_1/parameters.json") as file:
        params = json.load(file)

    params['base']['limit_pop'] = 250000

    p = deepcopy(params)
    with open(Path(run_dir, 'parameters.json'), 'w') as outfile:
        json.dump(p, outfile, indent=4)

    # ----- The baseline
    model = Ldm(experiment=experiment, scenario=scenario, run='run_1')
    model.run_model()
    a = Analyze(experiment=experiment, scenario=scenario, run='run_1')
    # --- How many total antibiotics were administered in NHs and STACHs
    baseline = [0]
    locations = ['NH', 'UNC', 'SMALL', 'LARGE']
    state = NameState.ANTIBIOTICS.value
    to_state = AntibioticsState.ON.value

    events = a.events
    events['Category'] = int_to_category(a, events.Location)
    for location in locations:
        count = a.events[(a.events.Category == location) & (a.events.State == state) & (a.events.New == to_state)].shape[0]
        baseline.append(count)

    # ----- Now Run Scenarios with 10%-40% lower daily probability until we see 10%, and 20% drops at each location
    results = []
    for x in range(10, 40, 2):
        print(x)
        x = x / 100
        # ----- Lower all antibiotic use by x percent.
        p = deepcopy(params)

        for item in p['facilities']['COMMUNITY']['age']:
            p['facilities']['COMMUNITY']['age'][item] *= (1 - x)
        for item in p['facilities']:
            if item != 'COMMUNITY':
                p['facilities'][item]['antibiotics'] *= (1 - x)

        with open(Path(run_dir, 'parameters.json'), 'w') as outfile:
            json.dump(p, outfile, indent=4)

        # ----- Run the model
        model_x = Ldm(experiment=experiment, scenario=scenario, run='run_1')
        model_x.run_model()
        a = Analyze(experiment=experiment, scenario=scenario, run='run_1')

        events = a.events
        events['Category'] = int_to_category(a, events.Location)

        # ----- How many total antibiotics were administered at each place
        total = [x]
        for location in locations:
            l1 = a.events.Category == location
            e1 = a.events.State == state
            e2 = a.events.New == to_state
            count = a.events[l1 & e1 & e2].shape[0]
            total.append(count)
        results.append(total)

    df = pd.DataFrame([baseline] + results)
    df.columns = ['Percent Drop', 'Community', 'NH', 'LT', 'Hospitals']

    df.to_csv("NCMIND/stewardship_paper/output/percent_drop.csv")
