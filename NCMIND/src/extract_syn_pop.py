
import pandas as pd
import argparse as arg
from pathlib import Path
import numpy as np


def rule_of_x(df, x=3):
    """
    The rule of x states that if a demographic is missing from our population, adding "x" individuals of that
    demographic will have a minimal impact on our results

    Parameters
    ----------
    df : pandas DataFrame
        a dataframe containing the full synthetic population
    x : int
        integer representing the minimum number of agents needed at each demographic

    Returns
    -------
    df : the synthetic population with new agents appended
    """

    gb = df.groupby(['County_Code', 'Sex', 'Age', 'Race'])
    gb = pd.DataFrame(gb.size())
    gb = gb.reset_index()

    # ----- Go through each combination of county, sex, age, and race. Create a list containing the demographic
    #       combination and the number of agents that are needed.
    agents_needed = []
    for county in range(1, 201, 2):
        for sex in [1, 2]:
            for age_group in [0, 1, 2]:
                for race in [1, 2, 3]:
                    temp_gb =\
                        gb[(gb.County_Code == county) & (gb.Sex == sex) & (gb.Age == age_group) & (gb.Race == race)]
                    if temp_gb.shape[0] == 0:
                        agents_needed.append([county, sex, age_group, race, x])
                    elif temp_gb[0].values[0] < x:
                        agents_needed.append([county, sex, age_group, race, x - temp_gb[0].values[0]])

    # ----- Add agents to the synthetic population based on which demographics were missing
    new_df = pd.DataFrame()
    for row in agents_needed:
        # ----- Find a random latitude/longitude for that county
        df_temp = df[df['County_Code'] == row[0]].sample(row[4]).reset_index()
        # ----- row[4] contains the number of agents to append.
        for i in range(row[4]):
            # ----- Assign a random age in years
            if row[2] == 0:
                age_years = 25
            elif row[2] == 1:
                age_years = 55
            else:
                age_years = 70
            # --- put the row together
            values = row[0:4] + [age_years] + [df_temp.loc[i].latitude] + [df_temp.loc[i].longitude]
            values = pd.DataFrame(values).T
            values.columns = df.columns
            new_df = new_df.append(values, ignore_index=True)

    return df.append(new_df, ignore_index=True)


def print_summary(exp_dir, df):
    """ Statisticians need a summary of the final synthetic population. This will group people by
        County, Sex, Age, and Race and output a CSV of the summary.

    Parameters
    ----------
    exp_dir : str
        The directory of the experiment
    df : pandas Dataframe
        The final synthetic population

    Returns
    -------
    gb : pandas Dataframe
        Group totals by County_Code, Sex, Age, Race
    """

    gb = df.groupby(['County_Code', 'Sex', 'Age', 'Race'])
    gb = pd.DataFrame(gb.size())
    gb = gb.reset_index()

    # ----- Must reverse the values to match what statistician would have
    race = dict()
    race[1] = 'White'
    race[2] = 'Black'
    race[3] = 'Other'

    age = dict()
    age[0] = 'L50'
    age[1] = '5064'
    age[2] = 'G65'

    sex = dict()
    sex[1] = 'M'
    sex[2] = 'F'

    # ----- Reverse values
    gb = gb.rename(columns={'Age': 'Age Group', 0: 'Count'})
    gb = gb.replace({'Age Group': age, 'Sex': sex, 'Race': race})

    gb.to_csv(Path(exp_dir, "data/synthetic_population/synthetic_population_summary.csv"), index=False)


def initialize_syn_pop(exp_dir, df, print_warnings=True):
    """

    Parameters
    ----------
    exp_dir : str
        The directory of the experiment
    df : pandas DataFrame
        The final synthetic population
    print_warnings : Boolean
        True to print messages for each demographic combination that did not have enough people to initialize correctly

    Returns
    -------
    df : pandas DataFrame
        The original input DataFrame with additional initialization variables
    """

    population = df.set_index(['County_Code', 'Sex', 'Age', 'Race']).sort_index()

    # ----- Set default values for start location, length of stay, and for persons unique id
    community = 'COMMUNITY'
    population['Start_Location'] = community
    population['p_id'] = range(population.shape[0])

    # ----- Since the p_id in order - we can use the p_id as an index for the numpy array
    population2 = population.reset_index()
    column_names = population2.columns
    start_location_index = list(column_names).index('Start_Location')
    population2 = population2.values

    # ----- Read initial population
    initial = pd.read_csv(Path(exp_dir, 'data/input/initial_population/all_initial.csv'))

    # ----- Define the Possible Locations
    locations = [item for item in initial.columns if item not in ['County', 'County_Code', 'Sex', 'Age', 'Race']]

    # ----- Go through each combination of county code, sex, age, and race and assign people to hospitals
    failed_count = 0
    # ----- Go through each combination of county code, sex, age, and race and assign people to hospitals
    for index, row in initial.iterrows():
        total_count = int(sum(row[locations]))

        if total_count > 0:
            try:
                pop = population.loc[row.County_Code, row.Sex, row.Age, row.Race]

                if total_count > pop.shape[0]:
                    if len(pop.shape) == 1:
                        number = 1
                    else:
                        number = pop.shape[0]
                    print('Trying to assign %s people. Only %s agent(s) fit this criteria' % (total_count, number))
                    print([row.County_Code, row.Sex, row.Age, row.Race])
                    pop = pop[pop['Start_Location'] == community].sample()
                else:
                    pop = pop[pop['Start_Location'] == community].sample(total_count)

                # --- for each of the locations, assign the appropriate number of people
                count = 0
                for j in locations:
                    for k in range(int(row[j])):
                        p_id = pop['p_id'].values[count:(count + 1)][0]
                        population2[p_id][start_location_index] = j
                        count = count + 1

            except Exception as E:
                failed_count += total_count
                if print_warnings:
                    print(E)
                    print('Population did not have combination of: ')
                    print(str(row[['County_Code', 'Sex', 'Age', 'Race']]))
                    print("I would like to assign %s people to this population" % str(total_count))

    df = pd.DataFrame(population2)
    df.columns = population.reset_index().columns
    df = df.drop(['p_id'], axis=1)

    return df


def extract_syn_pop(exp_dir):
    """ Extract the necessary columns from the 2013 synthetic population to use for NCMiND

    Parameters
    ----------
    exp_dir : str
        The directory of the experiment

    """

    # ----- Read the 2013 Synthetic Persons and Households files
    df = pd.read_csv('base_files/syn_pop_NC/NC2013Persons.csv')[['p_id', 'hh_id', 'age', 'sex', 'race', 'co']]
    df_household = pd.read_csv('base_files/syn_pop_NC/NC2013Households.csv')[['hh_id', 'latitude', 'longitude']]
    df = df.merge(df_household)

    df = df.rename(columns={'age': 'Age', 'sex': 'Sex', 'race': 'Race', 'co': 'County_Code'})

    # ----- Correct the Age
    df['Age_Years'] = df['Age']
    df['Age'] = -1
    df.loc[df['Age_Years'] < 50, 'Age'] = 0
    df.loc[df['Age_Years'] > 64, 'Age'] = 2
    df.loc[df['Age'] == -1, 'Age'] = 1
    # --- Correct the Race
    df.loc[df['Race'] > 2, 'Race'] = 3

    df = df.drop(['p_id', 'hh_id'], axis=1)

    df = df[['County_Code', 'Sex', 'Age', 'Race', 'Age_Years', 'latitude', 'longitude']]

    # ----- Make sure every demographic has at least 3 agents.
    df = rule_of_x(df, x=3)

    # ----- Pad the population to 10,042,802 people
    number_to_add = 10_042_802 - df.shape[0]
    new_people = df.sample(number_to_add)
    df = df.append(new_people)

    df = initialize_syn_pop(exp_dir, df, print_warnings=True)
    df.to_csv(Path(exp_dir,  "data/synthetic_population/synthetic_population.csv"), index=False)

    print_summary(exp_dir, df)

    # Now limit to orange county, and create another file
    df_orange = df[df['County_Code'] == 135]
    df_orange.to_csv(Path(exp_dir,  "data/synthetic_population/synthetic_population_orange.csv"), index=False)


if __name__ == '__main__':
    parser = arg.ArgumentParser(description='None')

    parser.add_argument(
        '--seed',
        type=str,
        default=1111,
        help='Seed to use for the model'
    )

    args = parser.parse_args()

    # Make the directory if it doesn't exist
    d = Path("NCMIND/data/synthetic_population")
    d.mkdir(exist_ok=True)

    np.random.seed(args.seed)

    extract_syn_pop(exp_dir='NCMIND')
