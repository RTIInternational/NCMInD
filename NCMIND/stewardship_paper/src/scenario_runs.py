
import math
from copy import deepcopy
import multiprocessing
import os
import sys
sys.path.append('')

try:
    from src.analyze import *
    from src.ldm import *
except Exception as E:
    raise


def run_model(run_number, parameters, exp_dir, scenario_x):
    """ Run a model with a specific seed

    Parameters
    ----------
    run_number : int
        The seed value for the run number

    parameters : dict
        A dictionary of parameters from the normal scenario

    exp_dir : str
        Location of the base experiment

    scenario_x : str
        The scenario name

    """
    # Create the run directory
    run = "run_" + str(run_number)
    run_dir = Path(folder, scenario_x, run)
    run_dir.mkdir(exist_ok=True)
    # Update and save parameters
    run_parameters = deepcopy(parameters)
    run_parameters['base']['seed'] = run_number
    with open(Path(run_dir, 'parameters.json'), 'w') as outfile:
        json.dump(run_parameters, outfile, indent=4)
    # Run the model
    model = LDM(exp_dir=exp_dir, scenario=scenario_x, run=run)
    model.run_model()


def multi_process_runs(run_count, parameters, exp_dir, scenario_x):
    """ Multiprocess the model runs. We want ~10 runs per scenario - So do they at the same time.
        Do not do more than 5 (on singularity), or 10 (on baldur)

    Parameters
    ----------
    run_count : int
        Number of runs to do per scenario. 5 is suggested

    parameters : see run_model function

    exp_dir : see run_model function

    scenario_x : see run_model function
    """
    runs = list()
    for r in range(run_count):
        runs.append((r, parameters, exp_dir, scenario_x))

    with multiprocessing.Pool() as pool:
        pool.starmap(run_model, runs)


def analyze_results(exp_dir, scenario_x, run_x):
    """ Create a dataframe of the results.
    """
    print('Working on scenario: {}, run: {}'.format(scenario_x, run_x))
    a_object = Analyze(exp_dir=exp_dir, scenario=scenario_x, run=run_x)

    output = [scenario + "___" + run_x]
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
    dframe = DataFrame(output).T
    dframe.columns = columns

    return dframe


def multi_process_analysis(exp_dir, scenario_x, runs):
    """ Run function analyze_results on multiple cores. We need to analyze all runs at the same time.
    """

    # Make a list of the runs
    analyze_runs = list()
    for r in runs:
        analyze_runs.append((exp_dir, scenario_x, r))
    # Analyze using multiple cores
    with multiprocessing.Pool() as pool:
        r = pool.starmap(analyze_results, analyze_runs)
    # Create a standard dataframe
    dframe = r[0]
    for i in range(1, len(r)):
        dframe = dframe.append(r[i])

    return dframe


if __name__ == "__main__":

    folder = "NCMIND/stewardship_paper/"
    with open(Path(folder, "parameters.json")) as file:
        params = json.load(file)
    total_runs = 20

    # ------------------------------------------------------------------------------------------------------------------
    # ----- The Normal Scenario
    scenario = "normal_run"
    Path(folder, scenario).mkdir(exist_ok=True)

    multi_process_runs(total_runs, params, folder, scenario)

    # ------------------------------------------------------------------------------------------------------------------
    # ----- Scenario Group #1: We are lowering antibiotic daily rates
    # ------------------------------------------------------------------------------------------------------------------
    lower = dict()
    # --- Hospitals
    lower['HOSPITAL'] = dict()
    lower['HOSPITAL']['10'] = 0.1907
    lower['HOSPITAL']['20'] = 0.3643
    lower['HOSPITAL']['30'] = 0.5379
    # --- COMMUNITY
    lower['COMMUNITY'] = dict()
    lower['COMMUNITY']['10'] = 0.1243
    lower['COMMUNITY']['20'] = 0.2323
    lower['COMMUNITY']['30'] = 0.3402
    # --- LTCF
    lower['LTCF'] = dict()
    lower['LTCF']['10'] = 0.1239
    lower['LTCF']['20'] = 0.2256
    lower['LTCF']['30'] = 0.3272

    # ------------------------------------------------------------------------------------------------------------------
    # ----- Scenario #1.1: Reduce the total # of antibiotic exposures initiated in 10 UNC STACHs
    for value in ['10', '20', '30']:
        print("Starting Scenario 1.1: {}".format(value))
        # Make the directory
        scenario = "1.1_UNC_" + str(value)
        Path(folder, scenario).mkdir(exist_ok=True)

        # Update the file
        p = deepcopy(params)
        reduction = lower['HOSPITAL'][value]
        for item in p['cdi']['antibiotics']['facility']:
            if item not in ['NH', 'LT', 'ST']:
                p['cdi']['antibiotics']['facility'][item] = \
                    p['cdi']['antibiotics']['facility'][item] * (1 - reduction)

        multi_process_runs(total_runs, p, folder, scenario)

    # ------------------------------------------------------------------------------------------------------------------
    # ----- Scenario #1.2: Reduce the total # of antibiotic exposures initiated in LTCFs
    for value in ['10', '20', '30']:
        print("Starting Scenario 1.2: {}".format(value))
        # Make the directory
        scenario = "1.2_LTCF_" + str(value)
        Path(folder, scenario).mkdir(exist_ok=True)

        # Update the file
        p = deepcopy(params)
        reduction = lower['LTCF'][value]
        p['cdi']['antibiotics']['facility']['NH'] = \
            p['cdi']['antibiotics']['facility']['NH'] * (1 - reduction)
        p['cdi']['antibiotics']['facility']['LT'] = \
            p['cdi']['antibiotics']['facility']['LT'] * (1 - reduction)

        multi_process_runs(total_runs, p, folder, scenario)

    # ------------------------------------------------------------------------------------------------------------------
    # ----- Scenario #1.3: Reduce the total # of antibiotic exposures initiated in UNC STACHS and LTCFs
    for value in ['10', '20', '30']:
        print("Starting Scenario 1.3: {}".format(value))
        # Make the directory
        scenario = "1.3_UNC_LTCF_" + str(value)
        Path(folder, scenario).mkdir(exist_ok=True)

        # Update the file
        p = deepcopy(params)
        # LTCFs
        reduction = lower['LTCF'][value]
        p['cdi']['antibiotics']['facility']['NH'] = \
            p['cdi']['antibiotics']['facility']['NH'] * (1 - reduction)
        p['cdi']['antibiotics']['facility']['LT'] = \
            p['cdi']['antibiotics']['facility']['LT'] * (1 - reduction)
        # UNC STACHs
        reduction = lower['HOSPITAL'][value]
        for item in p['cdi']['antibiotics']['facility']:
            if item not in ['NH', 'LT', 'ST']:
                p['cdi']['antibiotics']['facility'][item] = \
                    p['cdi']['antibiotics']['facility'][item] * (1 - reduction)

        multi_process_runs(total_runs, p, folder, scenario)

    # ------------------------------------------------------------------------------------------------------------------
    # ----- Scenario #1.4: Reduce the total # of antibiotic exposures initiated in the community
    for value in ['10', '20', '30']:
        print("Starting Scenario 1.4: {}".format(value))
        # Make the directory
        scenario = "1.4_COMMUNITY_" + str(value)
        Path(folder, scenario).mkdir(exist_ok=True)

        # Update the file
        p = deepcopy(params)
        reduction = lower['COMMUNITY'][value]
        p['cdi']['antibiotics']['COMMUNITY']['age']['0'] = \
            p['cdi']['antibiotics']['COMMUNITY']['age']['0'] * (1 - reduction)
        p['cdi']['antibiotics']['COMMUNITY']['age']['1'] = \
            p['cdi']['antibiotics']['COMMUNITY']['age']['1'] * (1 - reduction)
        p['cdi']['antibiotics']['COMMUNITY']['age']['2'] = \
            p['cdi']['antibiotics']['COMMUNITY']['age']['2'] * (1 - reduction)

        multi_process_runs(total_runs, p, folder, scenario)

    # ------------------------------------------------------------------------------------------------------------------
    # ----- Scenario #1.5: Reduce the total # of antibiotic exposures initiated everywhere
    for value in ['10', '20', '30']:
        print("Starting Scenario 1.5: {}".format(value))
        # Make the directory
        scenario = "1.5_ALL_" + str(value)
        Path(folder, scenario).mkdir(exist_ok=True)

        # Update the file
        p = deepcopy(params)

        # LTCFs
        reduction = lower['LTCF'][value]
        p['cdi']['antibiotics']['facility']['NH'] = \
            p['cdi']['antibiotics']['facility']['NH'] * (1 - reduction)
        p['cdi']['antibiotics']['facility']['LT'] = \
            p['cdi']['antibiotics']['facility']['LT'] * (1 - reduction)
        # UNC STACHs
        reduction = lower['HOSPITAL'][value]
        for item in p['cdi']['antibiotics']['facility']:
            if item not in ['NH', 'LT', 'ST']:
                p['cdi']['antibiotics']['facility'][item] = \
                    p['cdi']['antibiotics']['facility'][item] * (1 - reduction)
        # --- COMMUNITY
        reduction = lower['COMMUNITY'][value]
        p['cdi']['antibiotics']['COMMUNITY']['age']['0'] = \
            p['cdi']['antibiotics']['COMMUNITY']['age']['0'] * (1 - reduction)
        p['cdi']['antibiotics']['COMMUNITY']['age']['1'] = \
            p['cdi']['antibiotics']['COMMUNITY']['age']['1'] * (1 - reduction)
        p['cdi']['antibiotics']['COMMUNITY']['age']['2'] = \
            p['cdi']['antibiotics']['COMMUNITY']['age']['2'] * (1 - reduction)

        multi_process_runs(total_runs, p, folder, scenario)

    # ------------------------------------------------------------------------------------------------------------------
    # ----- Scenario Group #2: Adjust the percent of high risk antibiotics prescribed
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # ----- Scenario #2.1: UNC STACHs Only
    # ----- Adjust all 10 UNC STACHs to: HR=0.2, MR=0.4 (keep LR=0.4) AND HR=0.1, MR=0.5 (keep LR=0.4)
    for value in [20, 10]:
        print("Starting Scenario 2.1: {}".format(value))
        # Make the directory
        scenario = "2.1_UNC_" + str(value)
        Path(folder, scenario).mkdir(exist_ok=True)

        # Update the file
        p = deepcopy(params)

        for item in p['cdi']['antibiotics']['distributions']:
            if item not in ['COMMUNITY', 'NH', 'LT', 'ST']:
                # Original is [.4, .3, .3]
                if value == 20:
                    p['cdi']['antibiotics']['distributions'][item] = [0.4, 0.4, 0.2]
                if value == 10:
                    p['cdi']['antibiotics']['distributions'][item] = [0.4, 0.5, 0.1]

        multi_process_runs(total_runs, p, folder, scenario)

    # ------------------------------------------------------------------------------------------------------------------
    # ----- Scenario #2.2: LTCF
    # ----- Adjust proportion in NH + LTs
    for value in [20, 10]:
        print("Starting Scenario 2.2: {}".format(value))
        # Make the directory
        scenario = "2.2_LTCF_" + str(value)
        Path(folder, scenario).mkdir(exist_ok=True)

        # Update the file
        p = deepcopy(params)

        for item in p['cdi']['antibiotics']['distributions']:
            if item in ['NH']:
                # Original is [.1, .6, .3]
                if value == 20:
                    p['cdi']['antibiotics']['distributions'][item] = [0.1, 0.7, 0.2]
                if value == 10:
                    p['cdi']['antibiotics']['distributions'][item] = [0.1, 0.8, 0.1]
            if item in ['LT']:
                # Original is [.4, .3, .3]
                if value == 20:
                    p['cdi']['antibiotics']['distributions'][item] = [0.4, 0.4, 0.2]
                if value == 10:
                    p['cdi']['antibiotics']['distributions'][item] = [0.4, 0.5, 0.1]

        multi_process_runs(total_runs, p, folder, scenario)

    # ------------------------------------------------------------------------------------------------------------------
    #  ----- Grab the Results
    d = Path(folder)
    df = DataFrame()
    df_temp = DataFrame()

    for scenario in os.listdir(d):
        if ('1.' in scenario) or ('2.' in scenario) or ('normal_run' in scenario):
            run_list = []
            for run_y in os.listdir(Path(d, scenario)):
                if 'run' in run_y:
                    run_list.append(run_y)
            df_temp = multi_process_analysis(folder, scenario, run_list)

            if df.shape[0] > 0:
                df = df.append(df_temp)
            else:
                df = df_temp

    df.to_csv(Path(folder, "scenario_output.csv"), index=False)

    # ----- Aggregate the runs and find the means.
    aggregated_means = DataFrame()
    aggregated_stds = DataFrame()
    scenarios = list(set([item[0:item.find("__")] for item in df.Scenario.unique()]))
    scenarios.sort()
    for scenario in scenarios:
        temp_df = df[[scenario in item for item in df.Scenario.values]]
        temp_df = temp_df.drop('Scenario',axis='columns')
        means = temp_df.mean()
        sds = temp_df.std()

        aggregated_means[scenario + '_means'] = means
        aggregated_stds[scenario + '_sds'] = sds

    aggregated_means = aggregated_means.reset_index().rename(columns={'index': 'Measure'})
    aggregated_stds = aggregated_stds.reset_index().rename(columns={'index': 'Measure'})

    row_names = ['Community Associated', 'Healthcare Associated',
                 'ST: HO-CDI', 'LT: HO-CDI', 'NH: HO-CDI', 'UNC Overall: HO-CDI', 'LTCF: HO-CDI',
                 'ST: CO-CDI', 'LT: CO-CDI', 'NH: CO-CDI', 'UNC Overall: CO-CDI', 'UNC_Regional: HO-CDI']
    means_df = aggregated_means[aggregated_means.Measure.isin(row_names)].reset_index(drop=True)
    std_df = aggregated_stds[aggregated_stds.Measure.isin(row_names)].reset_index(drop=True)

    intervals = std_df.copy()
    for column in intervals.columns[1:]:
        intervals[column] = intervals[column].apply(lambda x: 1.96 * x / math.sqrt(20))

    final_df = pd.DataFrame()
    final_df['Measure'] = means_df.Measure

    for scenario in scenarios:
        means_col = [item for item in means_df.columns if scenario in item]
        final_df[means_col] = means_df[means_col]

        std_col = [item for item in std_df.columns if scenario in item]
        final_df[std_col] = std_df[std_col]

        interval_col = [item for item in intervals.columns if scenario in item]
        final_df[scenario + '_lower'] = means_df[means_col].values - intervals[interval_col].values
        final_df[scenario + '_upper'] = means_df[means_col].values + intervals[interval_col].values

    means_df.to_csv(Path(folder, "scenario_output_means.csv"), index=False)
    final_df.to_csv(Path(folder, "scenario_output_aggregated.csv"), index=False)
