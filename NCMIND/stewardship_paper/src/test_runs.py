
import json
from pathlib import Path
from src.ldm import Ldm
from copy import deepcopy
from NCMIND.cdi_calibration.src.test_targets import Analyze
from NCMIND.stewardship_paper.src.scenario_runs import reduce_unc_antibiotics


# ----------------------------------------------------------------------------------------------------------------------
# ----- Run a normal run
exp_dir = "NCMIND"
scenario_dir = "cdi_calibration"
run = "run_1"
full_dir = Path(exp_dir, scenario_dir, run)
ldm = Ldm(exp_dir, scenario_dir, run)
ldm.run_model()

a = Analyze(experiment=exp_dir, scenario=scenario_dir, run=run)
t1 = a.onset_cdi()
t1

# ----------------------------------------------------------------------------------------------------------------------
# 1: Lower antibiotic prescriptions at UNC by ~75%

experiment_name = "NCMIND"
base_scenario = "stewardship_paper/"
base_scenario_path = Path(experiment_name, base_scenario)
with open("NCMIND/cdi_calibration/run_1/parameters.json") as file:
    params = json.load(file)

p = deepcopy(params)
p = reduce_unc_antibiotics(p, .75)
print(params['location']['facilities']['UNC_8']['antibiotics'])
print(p['location']['facilities']['UNC_8']['antibiotics'])

experiment = "NCMIND"
scenario = "stewardship_paper/normal_run/"
run = "run_0"
run_dir = "NCMIND/stewardship_paper/normal_run/run_0"
with open(Path(run_dir, 'parameters.json'), 'w') as outfile:
    json.dump(p, outfile, indent=4)

# ----- Run the model
model = Ldm(experiment=experiment, scenario=scenario, run=run)
model.run_model()
# --- see the output
a = Analyze(experiment=experiment, scenario=scenario, run=run)
t2 = a.onset_cdi()
print(t2)

# ----------------------------------------------------------------------------------------------------------------------
# 2: Lower high-risk antibiotic prescriptions at UNC to none
p = deepcopy(params)
print(params['antibiotics']['distributions']['UNC'])
p['antibiotics']['distributions']['UNC'] = [0.4, 0.6, 0.0]

with open(Path(run_dir, 'parameters.json'), 'w') as outfile:
    json.dump(p, outfile, indent=4)

# ----- Run the model
model = Ldm(experiment=experiment, scenario=scenario, run=run)
model.run_model()

a = Analyze(experiment=experiment, scenario=scenario, run=run)
t3 = a.onset_cdi()
print(t3)
