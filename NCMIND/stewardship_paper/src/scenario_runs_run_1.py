
import math
import multiprocessing
import os
import json
import pandas as pd
from copy import deepcopy
from pathlib import Path

from NCMIND.src.analyze import Analyze
from src.ldm import Ldm
from src.parameters import update_parameters_file


# ----------------------------------------------------------------------------------------------------------------------
# ----- Functions for running models
def multi_process_runs(run_count: int, parameters: dict, experiment: str, scenario_x: str):
    """ Multiprocess the model runs. We want multiple runs per scenario - to run at the same time.
        Do not do more than 5 (on singularity), or 20 (on baldur)
    """
    scenario_x = base_scenario + scenario_x
    Path(experiment, scenario_x).mkdir(exist_ok=True)
    runs = list()
    for r in range(run_count):
        runs.append((r, parameters, experiment, scenario_x))

    with multiprocessing.Pool() as pool:
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
    a_object = Analyze(experiment=experiment, scenario=scenario_x, run=run_x)

    output = [scenario_x + "___" + run_x]
    columns = ['Scenario']

    # ---- Calculate the Output Values
    types, values = a_object.associated_cdi()
    output.extend(values)
    columns.extend(types)

    onset_cdi = a_object.onset_cdi(unc_only=True)
    for i, item_x in enumerate(onset_cdi[0]):
        output.append(onset_cdi[0][item_x])
        columns.append(item_x + ": CO-CDI")

    for i, item_x in enumerate(onset_cdi[1]):
        output.append(onset_cdi[1][item_x])
        columns.append(item_x + ": HO-CDI")

    # ----- Make the DataFrame
    dframe = pd.DataFrame(output).T
    dframe.columns = columns

    return dframe


def multi_process_analysis(experiment: str, scenario_x: str, runs: str) -> pd.DataFrame:
    """ Run function analyze_results on multiple cores. Analyzes all runs at the same time. """
    scenario_x = base_scenario + scenario_x
    # Make a list of the runs
    analyze_runs = list()
    for r in runs:
        analyze_runs.append((experiment, scenario_x, r))
    # Analyze using multiple cores
    with multiprocessing.Pool() as pool:
        r = pool.starmap(analyze_results, analyze_runs)
    # Create a standard dataframe
    dframe = r[0]
    for i in range(1, len(r)):
        dframe = dframe.append(r[i])

    return dframe


# ----------------------------------------------------------------------------------------------------------------------
# ----- Functions for run setup
def reduce_antibiotics_stachs(p: dict, reduction: float) -> dict:
    for item in p['facilities']:
        if item not in ['NH', 'LT', 'COMMUNITY']:
            p['facilities'][item]['antibiotics'] = p['facilities'][item]['antibiotics'] * (1 - reduction)
    return p


def reduce_highrisk_stachs(p: dict, value: int) -> dict:
    for item in p['cdi']['antibiotics']['distributions']:
        if item not in ['COMMUNITY', 'NH', 'LT']:
            # Original is [.4, .3, .3]
            if value == 20:
                p['cdi']['antibiotics']['distributions'][item] = [0.4, 0.4, 0.2]
            if value == 10:
                p['cdi']['antibiotics']['distributions'][item] = [0.4, 0.5, 0.1]
    return p


def reduce_highrisk_nhs(p: dict, value: int) -> dict:
    # Original is [.1, .6, .3]
    if value == 20:
        p['cdi']['antibiotics']['distributions']['NH'] = [0.1, 0.7, 0.2]
    if value == 10:
        p['cdi']['antibiotics']['distributions']['NH'] = [0.1, 0.8, 0.1]
    return p


if __name__ == "__main__":

    experiment_name = "NCMIND"
    base_scenario = "stewardship_paper/"
    base_scenario_path = Path(experiment_name, base_scenario)
    with open("NCMIND/cdi_calibration/run_1/parameters.json") as file:
        params = json.load(file)
    params['base']['limit_pop'] = 2_000_000
    runs_per_scenario = 20

    # ------------------------------------------------------------------------------------------------------------------
    # ----- The Normal Scenario
    scenario = "normal_run"
    Path(experiment_name, base_scenario, scenario).mkdir(exist_ok=True)
    Path(experiment_name, base_scenario, "output").mkdir(exist_ok=True)

    multi_process_runs(runs_per_scenario, params, experiment_name, scenario)

    # ------------------------------------------------------------------------------------------------------------------
    # ----- Scenario Group #1: Reduction in antibiotics administration: At NH, STACHs, and NH/STACHs
    # ------------------------------------------------------------------------------------------------------------------
    lower = dict()
    # Review readme to see how formulas were calculated.
    # --- NHs
    lower['NH'] = dict()
    lower['NH'][10] = (.1 - .0457) / .5366    # Enables a 10% reduction to NH antibiotic use
    lower['NH'][20] = (.2 - .0457) / .5366
    # --- STACHs
    lower['HOSPITAL'] = dict()
    lower['HOSPITAL'][10] = (.1 + .0363) / .5924  # Enables a 10% reduction to STACH antibiotic use
    lower['HOSPITAL'][20] = (.2 + .0363) / .5924

    # ------------------------------------------------------------------------------------------------------------------
    # ----- Scenario #1.1: Reduce the total # of antibiotic exposures initiated at STACHs
    for value in [10, 20]:
        print("Starting Scenario 1.1: {}".format(value))
        # Make the directory
        scenario = "1.1_STACHs_" + str(value)

        # Update the file
        p = deepcopy(params)
        p = reduce_antibiotics_stachs(p, lower['HOSPITAL'][value])

        multi_process_runs(runs_per_scenario, p, experiment_name, scenario)

    # ------------------------------------------------------------------------------------------------------------------
    # ----- Scenario #1.2: Reduce the total # of antibiotic exposures initiated in NHs
    for value in [10, 20]:
        print("Starting Scenario 1.2: {}".format(value))
        # Make the directory
        scenario = "1.2_NHs_" + str(value)

        # Update the file
        p = deepcopy(params)
        reduction = lower['NH'][value]
        p['facilities']['NH']['antibiotics'] = p['facilities']['NH']['antibiotics'] * (1 - reduction)

        multi_process_runs(runs_per_scenario, p, experiment_name, scenario)

    # ------------------------------------------------------------------------------------------------------------------
    # ----- Scenario #1.3: Reduce the total # of antibiotic exposures initiated in STACHs and NHs
    for value in [10, 20]:
        print("Starting Scenario 1.3: {}".format(value))
        # Make the directory
        scenario = "1.3_STACH_NH_" + str(value)

        # Update the file
        p = deepcopy(params)
        # STACHs
        p = reduce_antibiotics_stachs(p, lower['HOSPITAL'][value])
        # NHs
        reduction = lower['NH'][value]
        p['facilities']['NH']['antibiotics'] = p['facilities']['NH']['antibiotics'] * (1 - reduction)

        multi_process_runs(runs_per_scenario, p, experiment_name, scenario)

    # ------------------------------------------------------------------------------------------------------------------
    # ----- Scenario Group #2: Adjust the percent of high risk antibiotics prescribed
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # ----- Scenario #2.1: STACHs
    for value in [20, 10]:
        print("Starting Scenario 2.1: {}".format(value))
        # Make the directory
        scenario = "2.1_STACHs_" + str(value)

        # Update the file
        p = deepcopy(params)
        p = reduce_highrisk_stachs(p, value)

        multi_process_runs(runs_per_scenario, p, experiment_name, scenario)

    # ------------------------------------------------------------------------------------------------------------------
    # ----- Scenario #2.2: NH
    for value in [20, 10]:
        print("Starting Scenario 2.2: {}".format(value))
        # Make the directory
        scenario = "2.2_NHs_" + str(value)

        # Update the file
        p = deepcopy(params)
        p = reduce_highrisk_nhs(p, value)

        multi_process_runs(runs_per_scenario, p, experiment_name, scenario)

    # ------------------------------------------------------------------------------------------------------------------
    # ----- Scenario #2.3: STACHs and NHs
    for value in [20, 10]:
        print("Starting Scenario 2.3: {}".format(value))
        # Make the directory
        scenario = "2.3_STACH_NH_" + str(value)

        # Update the file
        p = deepcopy(params)
        p = reduce_highrisk_stachs(p, value)
        p = reduce_highrisk_nhs(p, value)

        multi_process_runs(runs_per_scenario, p, experiment_name, scenario)

    # ------------------------------------------------------------------------------------------------------------------
    # ----- Scenario Group #3: Run combinations of both 1 & 2
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # ----- Scenario #3.1: STACHs
    for anti_value in [10, 20]:
        for highrisk_value in [20, 10]:
            print("Starting Scenario 3.1: {}".format(value))
            # Make the directory
            scenario = "3.1_STACHs_" + str(anti_value) + "_" + str(highrisk_value)

            # Update the file
            p = deepcopy(params)

            # Antibiotics Reductions
            p = reduce_antibiotics_stachs(p, lower['HOSPITAL'][anti_value])
            # Lower Highrisk
            p = reduce_highrisk_stachs(p, highrisk_value)

            multi_process_runs(runs_per_scenario, p, experiment_name, scenario)

    # ------------------------------------------------------------------------------------------------------------------
    # ----- Scenario #3.2: NHs
    for anti_value in [10, 20]:
        for highrisk_value in [20, 10]:
            print("Starting Scenario 3.1: {}".format(value))
            # Make the directory
            scenario = "3.2_NHs_" + str(anti_value) + "_" + str(highrisk_value)

            # Update the file
            p = deepcopy(params)

            # Antibiotics Reductions
            reduction = lower['NH'][anti_value]
            p['facilities']['NH']['antibiotics'] = p['facilities']['NH']['antibiotics'] * (1 - reduction)
            # Lower Highrisk
            p = reduce_highrisk_nhs(p, highrisk_value)

            multi_process_runs(runs_per_scenario, p, experiment_name, scenario)

    # ------------------------------------------------------------------------------------------------------------------
    # ----- Scenario #3.3: BOTH
    for anti_value in [10, 20]:
        for highrisk_value in [20, 10]:
            print("Starting Scenario 3.3: {}".format(value))
            # Make the directory
            scenario = "3.3_STACH_NH_" + str(anti_value) + "_" + str(highrisk_value)

            # Update the file
            p = deepcopy(params)

            # Antibiotics Reductions
            p = reduce_antibiotics_stachs(p, lower['HOSPITAL'][anti_value])
            reduction = lower['NH'][anti_value]
            p['facilities']['NH']['antibiotics'] = p['facilities']['NH']['antibiotics'] * (1 - reduction)
            # Lower Highrisk
            p = reduce_highrisk_stachs(p, highrisk_value)
            p = reduce_highrisk_nhs(p, highrisk_value)

            multi_process_runs(runs_per_scenario, p, experiment_name, scenario)

    # ------------------------------------------------------------------------------------------------------------------
    #  ----- Grab the Results
    df = pd.DataFrame()
    df_temp = pd.DataFrame()

    for a_scenario in os.listdir(base_scenario_path):
        if ('1.' in a_scenario) or ('2.' in a_scenario) or ('3.' in a_scenario) or ('normal_run' in a_scenario):
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

    # ----- Aggregate the runs and find the means.
    aggregated_means = pd.DataFrame()
    aggregated_stds = pd.DataFrame()
    scenarios = list(set([item[0:item.find("__")] for item in df.Scenario.unique()]))
    scenarios.sort()
    for scenario in scenarios:
        temp_df = df[[scenario in item for item in df.Scenario.values]]
        temp_df = temp_df.drop('Scenario', axis='columns')
        means = temp_df.mean()
        sds = temp_df.std()

        aggregated_means[scenario + '_means'] = means
        aggregated_stds[scenario + '_sds'] = sds

    aggregated_means = aggregated_means.reset_index().rename(columns={'index': 'Measure'})
    aggregated_stds = aggregated_stds.reset_index().rename(columns={'index': 'Measure'})

    intervals = aggregated_stds.copy()
    intervals.columns = [item.replace("_sds", "") for item in intervals.columns]
    sample_size = len([item for item in df.Scenario if 'normal' in item])
    for column in intervals.columns[1:]:
        intervals[column] = intervals[column].apply(lambda x: 1.96 * x / math.sqrt(sample_size))

    final_df = pd.DataFrame()
    final_df['Measure'] = aggregated_means.Measure

    for scenario in scenarios:
        means_col = [item for item in aggregated_means.columns if scenario in item]
        final_df[means_col] = aggregated_means[means_col]

        std_col = [item for item in aggregated_stds.columns if scenario in item]
        final_df[std_col] = aggregated_stds[std_col]

        interval_col = [item for item in intervals.columns if scenario in item]
        final_df[scenario + '_lower'] = aggregated_means[means_col].values - intervals[interval_col].values
        final_df[scenario + '_upper'] = aggregated_means[means_col].values + intervals[interval_col].values

    aggregated_means.to_csv(Path(base_scenario_path, "output", "scenario_output_means.csv"), index=False)
    final_df.to_csv(Path(base_scenario_path, "output", "scenario_output_aggregated.csv"), index=False)
    intervals.to_csv(Path(base_scenario_path, "output", "intervals.csv"), index=False)
