
from copy import deepcopy
import sys
sys.path.append("")

from src.analyze import *
from src.ldm import *

"""
We need to figure out how much to reduce the daily probability of prescribing antibiotics in order to achieve an actual
    10%, 20%, and 30% drop in prescribed antibiotics for each facility.
"""

if __name__ == "__main__":

    # ----- Set up and run a baseline model
    folder = "NCMIND/stewardship_paper"
    scenario = "testing_directory"
    scenario_dir = Path(folder, scenario)
    scenario_dir.mkdir(exist_ok=True)

    with open(Path(folder, "parameters.json")) as file:
        params = json.load(file)

    params['base']['limit_pop'] = 250000

    p = deepcopy(params)
    with open(Path(scenario_dir, 'parameters.json'), 'w') as outfile:
        json.dump(p, outfile, indent=4)

    # ----- The baseline
    model = LDM(exp_dir=folder, scenario_dir=scenario)
    model.run_model()
    a = Analyze(exp_dir=folder, scenario_dir=scenario)
    # --- How many total antibiotics were administered in the community?
    baseline = [0]
    locations = ['COMMUNITY', 'NH', 'LT']
    state = NameState.ANTIBIOTICS.value
    to_state = 1
    # --- Community, NH, and LT
    for location in locations:
        count = a.events[(a.events.Location == model.locations.ints[location]) &
                         (a.events.State == state) &
                         (a.events.New == to_state)].shape[0]
        baseline.append(count)
    # --- Hospital Baseline:
    count = a.events[(a.events.State == state) & (a.events.New == to_state)].shape[0]
    baseline.append(count - sum(baseline))

    # ----- Now Run Scenarios with 10%-50% lower daily probability until we see 10%, 20%, and 30% drops at each location
    results = []
    for x in range(10, 50):
        x = x / 100
        # ----- Lower all antibiotic use by x percent.
        p = deepcopy(params)

        for item in p['cdi_transition']['antibiotics']['COMMUNITY']['age']:
            p['cdi_transition']['antibiotics']['COMMUNITY']['age'][item] *= (1 - x)
        for item in p['cdi_transition']['antibiotics']['facility']:
            p['cdi_transition']['antibiotics']['facility'][item] *= (1 - x)

        with open(Path(scenario_dir, 'parameters.json'), 'w') as outfile:
            json.dump(p, outfile, indent=4)

        # ----- Run the model
        model_x = LDM(exp_dir=folder, scenario_dir=scenario)
        model_x.run_model()
        a_x = Analyze(exp_dir=folder, scenario_dir=scenario)

        # ----- How many total antibiotics were administered at each place
        total = [x]
        # --- Community, NH, and LT
        for location in locations:
            count = a_x.events[(a_x.events.Location == model.locations.ints[location]) &
                               (a_x.events.State == state) &
                               (a_x.events.New == to_state)].shape[0]
            total.append(count)
        # --- Hospital Baseline:
        count = a_x.events[(a_x.events.State == state) & (a_x.events.New == to_state)].shape[0]
        total.append(count - sum(total[1:]))
        results.append(total)

    df = pd.DataFrame([baseline] + results)
    df.columns = ['Percent Drop', 'Community', 'NH', 'LT', 'Hospitals']

    df.to_csv("NCMIND/stewardship_paper/output/percent_drop.csv")

# # ----- OK so this only worked for Community and NH. Let's try something else for STACHs and LTACHs.
#
# # STACHs have an average daily probability of 33%
# base = .33
#
# for i in range(10, 70):
#     adjusted = base * (1-i/100)
#
#     base_results = []
#     adjusted_results = []
#     for j in range(10000):
#         LOS = max(round(np.random.gamma(.4, 12.9)), 1)
#         base_final = (1 - base) ** LOS
#         adjusted_final = (1 - adjusted) ** LOS
#
#         r = np.random.random()
#
#         if base_final < r:
#             base_results.append(1)
#         else:
#             base_results.append(0)
#
#         if adjusted_final < r:
#             adjusted_results.append(1)
#         else:
#             adjusted_results.append(0)
#
#     br = sum(base_results) / 10000
#     ar = sum(adjusted_results) / 10000
#     print(br)
#     print(ar)
#
#     print('Drop: {}'.format((br - ar) / br))
