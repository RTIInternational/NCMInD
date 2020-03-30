
import argparse
import multiprocessing
import os
import json
import pandas as pd
import numpy as np
from copy import deepcopy
from pathlib import Path

from NCMIND.src.analyze import Analyze
from src.ldm import Ldm
from src.parameters import update_parameters_file


# ----------------------------------------------------------------------------------------------------------------------
# ----- Functions for running models
def multi_process_runs(run_count: int, parameters: dict, experiment: str, scenario_x: str, cpus: int = 20):
    """ Multiprocess the model runs. We want multiple runs per scenario - to run at the same time.
        Do not do more than 5 (on singularity), or 20 (on baldur)
    """
    scenario_x = base_scenario + scenario_x
    Path(experiment, scenario_x).mkdir(exist_ok=True)
    runs = list()
    for r in range(run_count):
        runs.append((r, parameters, experiment, scenario_x))

    with multiprocessing.Pool(cpus) as pool:
        pool.starmap(run_model, runs)


def run_model(run_number: int, parameters: dict, experiment: str, scenario_x: str):
    """ Run a model with a specific seed: randomly draw from parameters using this seed """
    # ----- Create the run directory
    run = "run_" + str(run_number)
    run_dir = Path(experiment, scenario_x, run)
    run_dir.mkdir(exist_ok=True)

    # ----- Update the seed
    run_parameters = deepcopy(parameters)
    run_parameters['base']['seed'] = run_number
    # --- Draw Random Parameters
    run_parameters = update_parameters_file(run_parameters)

    with open(Path(run_dir, 'parameters.json'), 'w') as outfile:
        json.dump(run_parameters, outfile, indent=4)

    # ----- Run the model
    model = Ldm(experiment=experiment, scenario=scenario_x, run=run)
    model.run_model()


# ----------------------------------------------------------------------------------------------------------------------
# ----- Functions for analyzing models
def analyze_results(experiment: str, scenario_x: str, run_x: str):
    """ Create a dataframe of the results. """
    print('Working on scenario: {}, run: {}'.format(scenario_x, run_x))
    analysis = Analyze(experiment=experiment, scenario=scenario_x, run=run_x)

    output = [scenario_x + "___" + run_x]
    columns = ['Scenario']

    # ----- Associated CDI: All
    types, values = analysis.associated_cdi()
    output.extend(values)
    columns.extend(types)

    # ----- Associated CDI: Catchment Only
    types, values = analysis.associated_cdi(unc_only=True)
    types = [item + ": UNC Only" for item in types]
    output.extend(values)
    columns.extend(types)

    # ----- Onset CDI
    onset_cdi = analysis.onset_cdi(unc_only=True)
    # CO
    for i, item_x in enumerate(onset_cdi[0]):
        output.append(onset_cdi[0][item_x])
        columns.append(item_x + ": CO-CDI")
    # HO
    for i, item_x in enumerate(onset_cdi[1]):
        output.append(onset_cdi[1][item_x])
        columns.append(item_x + ": HO-CDI")

    # ----- Make the DataFrame
    dframe = pd.DataFrame(output).T
    dframe.columns = columns

    return dframe


def multi_process_analysis(experiment: str, scenario_x: str, runs: str, cpus: int = 20) -> pd.DataFrame:
    """ Run function analyze_results on multiple cores. Analyzes all runs at the same time. """
    scenario_x = base_scenario + scenario_x
    # Make a list of the runs
    analyze_runs = list()
    for r in runs:
        analyze_runs.append((experiment, scenario_x, r))
    # Analyze using multiple cores
    with multiprocessing.Pool(cpus) as pool:
        r = pool.starmap(analyze_results, analyze_runs)
    # Create a standard dataframe
    dframe = r[0]
    for i in range(1, len(r)):
        dframe = dframe.append(r[i])

    return dframe


# ----------------------------------------------------------------------------------------------------------------------
# ----- Functions for run setup
def reduce_unc_antibiotics(p: dict, reduction: float) -> dict:
    for item in p['location']['facilities']:
        if item[0:4] == 'UNC_':
            p['location']['facilities'][item]['antibiotics'] =\
                p['location']['facilities'][item]['antibiotics'] * (1 - reduction)
    return p


def reduce_highrisk(p: dict, value: int, category: str) -> dict:
    if category in ['UNC', 'LTACH', 'SMALL', 'LARGE']:
        # Original is [.4, .3, .3]
        if value == 20:
            p['antibiotics']['distributions'][category] = [0.4, 0.4, 0.2]
        if value == 10:
            p['antibiotics']['distributions'][category] = [0.4, 0.5, 0.1]
    if category in ['NH', 'COMMUNITY']:
        # Original is [.1, .6, .3]
        if value == 20:
            p['antibiotics']['distributions'][category] = [0.1, 0.7, 0.2]
        if value == 10:
            p['antibiotics']['distributions'][category] = [0.1, 0.8, 0.1]
    return p


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        '--num_agents',
        type=int,
        default=1000,
        help='number of agents for each scenario (default: %(default)s)'
    )
    parser.add_argument(
        '--runs_per',
        type=int,
        default=2,
        help='number of runs per scenario (default: %(default)s)'
    )
    args = parser.parse_args()
    print(args)

    experiment_name = "NCMIND"
    base_scenario = "stewardship_paper"
    base_scenario_path = Path(experiment_name, base_scenario)
    with open("NCMIND/cdi_calibration/run_1/parameters.json") as file:
        params = json.load(file)
    params['base']['limit_pop'] = args.num_agents
    runs_per_scenario = args.runs_per

    # ------------------------------------------------------------------------------------------------------------------
    # ----- The Normal Scenario
    scenario = "normal_run"
    Path(experiment_name, base_scenario, scenario).mkdir(exist_ok=True)
    Path(experiment_name, base_scenario, "output").mkdir(exist_ok=True)

    print("Starting normal runs.")
    multi_process_runs(runs_per_scenario, params, experiment_name, scenario)

    # ------------------------------------------------------------------------------------------------------------------
    # ----- Scenario Group #1: Reduction in antibiotics administration: At NH, STACHs, and NH/STACHs
    # ------------------------------------------------------------------------------------------------------------------
    # Review readme to see how formulas were calculated.
    lower = dict()
    # --- NHs
    lower['NH'], lower['HOSPITAL'], lower['COMMUNITY'], lower['LTACH'] = dict(), dict(), dict(), dict()
    lower['NH'][10] = (.1 - .0457) / .5366
    lower['NH'][20] = (.2 - .0457) / .5366
    lower['NH'][30] = (.3 - .0457) / .5366
    # --- STACHs
    lower['HOSPITAL'][10] = (.1 + .0363) / .5924
    lower['HOSPITAL'][20] = (.2 + .0363) / .5924
    lower['HOSPITAL'][30] = (.3 + .0363) / .5924
    # --- Community
    lower['COMMUNITY'][10] = (.1 + .0196) / .9333
    lower['COMMUNITY'][20] = (.2 + .0196) / .9333
    lower['COMMUNITY'][30] = (.3 + .0196) / .9333
    # --- LTACHs
    lower['LTACH'][10] = (.1 + .0244) / .6484
    lower['LTACH'][20] = (.2 + .0244) / .6484
    lower['LTACH'][30] = (.3 + .0244) / .6484

    # ------------------------------------------------------------------------------------------------------------------
    # ----- Scenario #1.1: Reduce the total # of antibiotic exposures initiated at UNC STACHs
    for value in [10, 20, 30]:
        print("Starting Scenario 1.1: {}".format(value))
        # Make the directory
        scenario = "1.1_UNC_" + str(value)

        # Update the file
        p = deepcopy(params)
        p = reduce_unc_antibiotics(p, lower['HOSPITAL'][value])

        multi_process_runs(runs_per_scenario, p, experiment_name, scenario)

    # ------------------------------------------------------------------------------------------------------------------
    # ----- Scenario #1.2: Reduce the total # of antibiotic exposures initiated in NHs
    for value in [10, 20, 30]:
        print("Starting Scenario 1.2: {}".format(value))
        # Make the directory
        scenario = "1.2_NHs_" + str(value)

        # Update the file
        p = deepcopy(params)
        reduction = lower['NH'][value]
        p['location']['facilities']['NH']['antibiotics'] =\
            p['location']['facilities']['NH']['antibiotics'] * (1 - reduction)

        multi_process_runs(runs_per_scenario, p, experiment_name, scenario)

    # ------------------------------------------------------------------------------------------------------------------
    # ----- Scenario #1.3: Reduce the total # of antibiotic exposures initiated in UNC STACHs and NHs
    for value in [10, 20, 30]:
        print("Starting Scenario 1.3: {}".format(value))
        # Make the directory
        scenario = "1.3_UNC_NH_" + str(value)

        # Update the file
        p = deepcopy(params)
        # STACHs
        p = reduce_unc_antibiotics(p, lower['HOSPITAL'][value])
        # NHs
        reduction = lower['NH'][value]
        p['location']['facilities']['NH']['antibiotics'] =\
            p['location']['facilities']['NH']['antibiotics'] * (1 - reduction)

        multi_process_runs(runs_per_scenario, p, experiment_name, scenario)

    # ------------------------------------------------------------------------------------------------------------------
    # ----- Scenario #1.4: Reduce the total # of antibiotic exposures in the community
    for value in [10, 20, 30]:
        print("Starting Scenario 1.4: {}".format(value))
        # Make the directory
        scenario = "1.4_OUTPATIENT_" + str(value)

        # Update the file
        p = deepcopy(params)
        # Community
        reduction = lower['COMMUNITY'][value]
        for item in p['location']['facilities']['COMMUNITY']['age']:
            p['location']['facilities']['COMMUNITY']['age'][item] =\
                p['location']['facilities']['COMMUNITY']['age'][item] * (1 - reduction)

        multi_process_runs(runs_per_scenario, p, experiment_name, scenario)

    # ------------------------------------------------------------------------------------------------------------------
    # ----- Scenario #1.5: Reduce the total # of antibiotic exposures everywhere
    for value in [10, 20, 30]:
        print("Starting Scenario 1.5: {}".format(value))
        # Make the directory
        scenario = "1.5_ALL_" + str(value)

        # Update the file
        p = deepcopy(params)
        # STACHs
        p = reduce_unc_antibiotics(p, lower['HOSPITAL'][value])
        # NHs
        r1 = lower['NH'][value]
        p['location']['facilities']['NH']['antibiotics'] =\
            p['location']['facilities']['NH']['antibiotics'] * (1 - r1)
        # Community
        r2 = lower['COMMUNITY'][value]
        for item in p['location']['facilities']['COMMUNITY']['age']:
            p['location']['facilities']['COMMUNITY']['age'][item] =\
                p['location']['facilities']['COMMUNITY']['age'][item] * (1 - r2)
        # LTACHs
        r3 = lower['LTACH'][value]
        p['location']['facilities']['LT']['antibiotics'] =\
            p['location']['facilities']['LT']['antibiotics'] * (1 - r3)

        multi_process_runs(runs_per_scenario, p, experiment_name, scenario)

    # ------------------------------------------------------------------------------------------------------------------
    # ----- Scenario Group #2: Adjust the percent of high risk antibiotics prescribed
    # ------------------------------------------------------------------------------------------------------------------
    # ----- Scenario #2.1: UNC STACHs
    for value in [20, 10]:
        print("Starting Scenario 2.1: {}".format(value))
        # Make the directory
        scenario = "2.1_UNC_" + str(value)

        # Update the file
        p = deepcopy(params)
        p = reduce_highrisk(p, value, category='UNC')

        multi_process_runs(runs_per_scenario, p, experiment_name, scenario)

    # ------------------------------------------------------------------------------------------------------------------
    # ----- Scenario #2.2: NH
    for value in [20, 10]:
        print("Starting Scenario 2.2: {}".format(value))
        # Make the directory
        scenario = "2.2_NHs_" + str(value)

        # Update the file
        p = deepcopy(params)
        p = reduce_highrisk(p, value, category='NH')

        multi_process_runs(runs_per_scenario, p, experiment_name, scenario)

    # ------------------------------------------------------------------------------------------------------------------
    # ----- Scenario #2.3: UNC STACHs and NHs
    for value in [20, 10]:
        print("Starting Scenario 2.3: {}".format(value))
        # Make the directory
        scenario = "2.3_UNC_NH_" + str(value)
        # Update the file
        p = deepcopy(params)
        p = reduce_highrisk(p, value, category='UNC')
        p = reduce_highrisk(p, value, category='NH')

        multi_process_runs(runs_per_scenario, p, experiment_name, scenario)

    # ------------------------------------------------------------------------------------------------------------------
    # ----- Scenario #2.4: Outpatient
    for value in [20, 10]:
        print("Starting Scenario 2.4: {}".format(value))
        # Make the directory
        scenario = "2.4_OUTPATIENT_" + str(value)
        # Update the file
        p = deepcopy(params)
        p = reduce_highrisk(p, value, category='COMMUNITY')

        multi_process_runs(runs_per_scenario, p, experiment_name, scenario)

    # ------------------------------------------------------------------------------------------------------------------
    # ----- Scenario #2.5: ALL
    for value in [20, 10]:
        print("Starting Scenario 2.5: {}".format(value))
        # Make the directory
        scenario = "2.5_ALL_" + str(value)
        # Update the file
        p = deepcopy(params)
        p = reduce_highrisk(p, value, category='UNC')
        p = reduce_highrisk(p, value, category='NH')
        p = reduce_highrisk(p, value, category='COMMUNITY')

        multi_process_runs(runs_per_scenario, p, experiment_name, scenario)

    # ------------------------------------------------------------------------------------------------------------------
    #  ----- Grab the Results
    df = pd.DataFrame()
    df_temp = pd.DataFrame()

    for a_scenario in os.listdir(base_scenario_path):
        if ('1.' in a_scenario) or ('2.' in a_scenario) or ('normal_run' in a_scenario):
            run_list = []
            for run_y in os.listdir(Path(base_scenario_path, a_scenario)):
                if 'run' in run_y:
                    run_list.append(run_y)
            df_temp = multi_process_analysis(experiment_name, a_scenario, run_list)

            if df.shape[0] > 0:
                df = df.append(df_temp)
            else:
                df = df_temp

    df.Scenario = [item.replace(base_scenario, "") for item in df.Scenario]

    df.to_csv(Path(base_scenario_path, "output", "scenario_outputs.csv"), index=False)

    # ----- Find the run id
    cutoffs = df.Scenario.str.find("__run_") + 6 
    df['run_id'] = [item[cutoffs[i]:] for i, item in enumerate(df.Scenario)]
    df['run_id'] = df['run_id'].astype(int)

    def grab_scenario(text):
        s_df = df[df.Scenario.str.contains(text)]
        s_df = s_df.sort_values(by=['run_id']).reset_index(drop=True)
        return s_df.drop(['Scenario', 'run_id'], axis=1)

    # --- Create scenario DFs
    scenario_dfs = dict()
    scenarios = list(set([item[0:item.find("__")] for item in df.Scenario.unique()]))
    scenarios.sort()
    ts = scenarios[0:-1]
    ts.sort(key=lambda x: int(x[2]))
    scenarios = [scenarios[-1]] + ts

    for scenario in scenarios:
        scenario_dfs[scenario] = grab_scenario(scenario)

    # ----- Calculate Percent Change
    normal = scenario_dfs['normal_run']

    final_df = pd.DataFrame()
    final_df['normal_run_means'] = normal.mean()
    for scenario in scenarios[1:]:
        temp_df = scenario_dfs[scenario]

        percent_change = (temp_df - normal) / normal * 100

        final_df[scenario + '_\u03BC'] = temp_df.mean()
        final_df[scenario + '_pc_\u03BC'] = percent_change.mean()
        final_df[scenario + '_pc_\u03C3'] = percent_change.std()
        final_df[scenario + '_pc_SE'] = final_df[scenario + '_pc_\u03C3'] / np.sqrt(percent_change.shape[0])

    final_df.to_excel(Path(base_scenario_path, "output", "results.xlsx"))

    # ----- Aggregate the runs and find the means.
    aggregated_df = pd.DataFrame()
    for scenario in scenarios:
        temp_df = scenario_dfs[scenario]
        means = temp_df.mean()
        sds = temp_df.std()

        aggregated_df[scenario + '_means'] = means
        aggregated_df[scenario + '_sds'] = sds
        aggregated_df[scenario + '_range'] = sds.apply(lambda x: 1.96 * x / np.sqrt(percent_change.shape[0]))
        aggregated_df[scenario + '_lower'] = means - aggregated_df[scenario + '_range']
        aggregated_df[scenario + '_upper'] = means + aggregated_df[scenario + '_range']

    aggregated_df.to_csv(Path(base_scenario_path, "output", "scenario_output_means.csv"))
